#!/usr/bin/env python3
"""One-time re-embed of finer_products_omega with OpenAI text-embedding-3-small.

Overwrites product_embedding (previously a sparse 768-dim fashionSigLIP set,
~9% coverage) with 1536-dim text embeddings on every row, so the
match_products_semantic RPC (migrations/004_product_semantic_search.sql) can
rank the full catalog and the API server can embed queries with a cheap
OpenAI call instead of a local torch model.

Embedded text per product: title + slot + color + tags + image caption +
description. ~45 tokens/row average → full catalog ≈ 1.1M tokens ≈ $0.02.

Usage:
    python scripts/reembed_omega_openai.py --dry-run      # show composed text, no writes
    python scripts/reembed_omega_openai.py --limit 50     # small live test
    python scripts/reembed_omega_openai.py                # full run (resumable; rerun is idempotent)

Env: SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY (reads .env via dotenv).
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reembed_omega")

EMBED_MODEL = os.getenv("SEARCH_EMBED_MODEL", "text-embedding-3-small")
EXPECTED_DIM = 1536
PAGE_SIZE = 500          # rows per select + per embeddings API call
UPDATE_WORKERS = 8       # parallel per-row PATCH writes
MAX_TEXT_CHARS = 4000    # guard against pathological descriptions

TEXT_FIELDS = (
    "product_id,product_title,product_slot,product_color,product_tags,"
    "product_image_caption,product_description"
)


def compose_text(row: dict) -> str:
    parts = [row.get("product_title") or ""]
    for key in ("product_slot", "product_color"):
        if row.get(key):
            parts.append(row[key])
    tags = row.get("product_tags") or []
    if tags:
        parts.append(" ".join(tags))
    if row.get("product_image_caption"):
        parts.append(row["product_image_caption"])
    if row.get("product_description"):
        parts.append(row["product_description"])
    text = " ".join(p for p in parts if p).strip()
    return text[:MAX_TEXT_CHARS] or (row.get("product_id") or "unknown product")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process N rows")
    parser.add_argument("--dry-run", action="store_true", help="no API calls, no writes")
    args = parser.parse_args()

    from supabase import create_client

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    openai_client = None
    if not args.dry_run:
        from openai import OpenAI

        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    total = (
        supabase.table("finer_products_omega")
        .select("product_id", count="exact")
        .limit(1)
        .execute()
        .count
    )
    logger.info("rows in finer_products_omega: %s (model=%s)", total, EMBED_MODEL)

    processed = 0
    failed_ids: list = []
    start = time.time()
    offset = 0

    while True:
        if args.limit is not None and processed >= args.limit:
            break
        page = (
            supabase.table("finer_products_omega")
            .select(TEXT_FIELDS)
            .order("product_id")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
            .data
        )
        if not page:
            break
        offset += PAGE_SIZE
        if args.limit is not None:
            page = page[: args.limit - processed]

        texts = [compose_text(r) for r in page]

        if args.dry_run:
            for row, text in zip(page[:5], texts[:5]):
                logger.info("DRY %s :: %s", row["product_id"], text[:140])
            processed += len(page)
            continue

        try:
            response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        except Exception as exc:
            logger.error("embedding batch failed at offset %s: %s", offset, exc)
            failed_ids.extend(r["product_id"] for r in page)
            continue

        embeddings = [d.embedding for d in response.data]
        bad_dims = [len(e) for e in embeddings if len(e) != EXPECTED_DIM]
        if bad_dims:
            logger.error("unexpected dims %s at offset %s; skipping batch", set(bad_dims), offset)
            failed_ids.extend(r["product_id"] for r in page)
            continue

        now = datetime.now(timezone.utc).isoformat()

        def write(pid_emb):
            pid, emb = pid_emb
            # Parallel PATCHes can transiently exhaust local sockets
            # (EAGAIN); back off and retry before declaring failure.
            for attempt in range(4):
                try:
                    supabase.table("finer_products_omega").update(
                        {"product_embedding": emb, "last_embedded": now}
                    ).eq("product_id", pid).execute()
                    return None
                except Exception as exc:
                    if attempt == 3:
                        logger.warning("update failed for %s: %s", pid, exc)
                        return pid
                    time.sleep(0.5 * (2**attempt))

        with ThreadPoolExecutor(max_workers=UPDATE_WORKERS) as pool:
            futures = [
                pool.submit(write, (row["product_id"], emb))
                for row, emb in zip(page, embeddings)
            ]
            for fut in as_completed(futures):
                pid = fut.result()
                if pid:
                    failed_ids.append(pid)

        processed += len(page)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed else 0
        logger.info(
            "progress %s/%s (%.0f rows/s, %.1f min remaining)",
            processed,
            args.limit or total,
            rate,
            ((args.limit or total) - processed) / rate / 60 if rate else 0,
        )

    logger.info("done: %s rows in %.1f min, %s failures", processed, (time.time() - start) / 60, len(failed_ids))
    if failed_ids:
        logger.error("failed product_ids (rerun to retry — script is idempotent): %s", failed_ids[:50])
        sys.exit(1)


if __name__ == "__main__":
    main()
