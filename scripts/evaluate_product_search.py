#!/usr/bin/env python3
"""Run a small, repeatable relevance gate against GET /products/search.

The cases target the failure modes the Search tab must not regress:

* parent category recall ("pants" includes jeans and other pants families)
* exact brand retrieval
* exact color retrieval
* combined brand + color + category intent

Examples:
    python scripts/evaluate_product_search.py
    python scripts/evaluate_product_search.py \
        --base-url https://finer-stylist-backend.onrender.com
    python scripts/evaluate_product_search.py --json
"""

import argparse
from dataclasses import dataclass
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class RelevanceCase:
    name: str
    query: str
    top_n: int
    min_results: int
    allowed_categories: Tuple[str, ...] = ()
    required_categories: Tuple[str, ...] = ()
    allowed_colors: Tuple[str, ...] = ()
    brand_prefix: Optional[str] = None
    required_title_groups: Tuple[Tuple[str, ...], ...] = ()


CASES: Tuple[RelevanceCase, ...] = (
    RelevanceCase(
        name="pants_parent_recall",
        query="pants",
        top_n=20,
        min_results=10,
        allowed_categories=("pants", "jeans", "leggings"),
        required_categories=("pants", "jeans"),
        required_title_groups=(
            ("jean",),
            ("trouser", "slack", "chino"),
            ("pant", "jogger", "legging"),
        ),
    ),
    RelevanceCase(
        name="trousers_category",
        query="trousers",
        top_n=10,
        min_results=10,
        allowed_categories=("pants",),
    ),
    RelevanceCase(
        name="brand_only",
        query="Kowtow",
        top_n=10,
        min_results=10,
        brand_prefix="kowtow",
    ),
    RelevanceCase(
        name="color_only",
        query="red",
        top_n=10,
        min_results=10,
        allowed_colors=("red",),
    ),
    RelevanceCase(
        name="color_and_category",
        query="black jeans",
        top_n=10,
        min_results=10,
        allowed_categories=("jeans",),
        allowed_colors=("black",),
    ),
    RelevanceCase(
        name="brand_and_category",
        query="Stussy hoodies",
        top_n=5,
        min_results=5,
        allowed_categories=("sweatshirt_hoodie",),
        brand_prefix="stussy",
    ),
    RelevanceCase(
        name="brand_color_category",
        query="black Kowtow jeans",
        top_n=5,
        min_results=5,
        allowed_categories=("jeans",),
        allowed_colors=("black",),
        brand_prefix="kowtow",
    ),
    RelevanceCase(
        name="parent_category_and_color",
        query="blue pants",
        top_n=10,
        min_results=10,
        allowed_categories=("pants", "jeans", "leggings"),
        required_categories=("pants", "jeans"),
        allowed_colors=("blue",),
    ),
)


def _normalized(value: Any) -> str:
    return str(value or "").casefold().strip(" .")


def _brand_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalized(value))


def fetch_results(
    base_url: str,
    query: str,
    *,
    gender: Optional[str],
    limit: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    params = {"q": query, "limit": str(limit)}
    if gender:
        params["gender"] = gender
    url = f"{base_url.rstrip('/')}/products/search?{urlencode(params)}"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    if not payload.get("success"):
        raise RuntimeError(payload.get("error") or "search returned success=false")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("search response has no results array")
    return results


def evaluate_case(case: RelevanceCase, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    top = list(results[: case.top_n])
    failures: List[str] = []

    if len(top) < case.min_results:
        failures.append(f"expected at least {case.min_results} results, got {len(top)}")

    categories = [_normalized(item.get("category")) for item in top]
    colors = [_normalized(item.get("product_color")) for item in top]
    brands = [_brand_key(item.get("product_brand")) for item in top]
    titles = [_normalized(item.get("product_title")) for item in top]

    if case.allowed_categories:
        missing = sum(not value for value in categories)
        outside = sorted(
            {value for value in categories if value and value not in case.allowed_categories}
        )
        if missing:
            failures.append(f"{missing} results are missing category")
        if outside:
            failures.append(f"unexpected categories: {', '.join(outside)}")

    if case.required_categories:
        absent = sorted(set(case.required_categories) - set(categories))
        if absent:
            failures.append(f"missing category coverage: {', '.join(absent)}")

    if case.allowed_colors:
        missing = sum(not value for value in colors)
        outside = sorted(
            {value for value in colors if value and value not in case.allowed_colors}
        )
        if missing:
            failures.append(f"{missing} results are missing product_color")
        if outside:
            failures.append(f"unexpected colors: {', '.join(outside)}")

    if case.brand_prefix:
        expected = _brand_key(case.brand_prefix)
        mismatches = sum(not brand.startswith(expected) for brand in brands)
        if mismatches:
            failures.append(
                f"{mismatches} results do not match brand prefix {case.brand_prefix}"
            )

    for token_group in case.required_title_groups:
        if not any(any(token in title for token in token_group) for title in titles):
            failures.append(
                "missing title family: " + " / ".join(token_group)
            )

    return {
        "name": case.name,
        "query": case.query,
        "passed": not failures,
        "result_count": len(results),
        "checked_count": len(top),
        "categories": sorted(set(filter(None, categories))),
        "colors": sorted(set(filter(None, colors))),
        "brands": sorted(set(filter(None, brands))),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("SEARCH_EVAL_BASE_URL", "http://localhost:8000"),
        help="API origin (default: SEARCH_EVAL_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--gender",
        default="feminine",
        help="catalog gender query parameter; pass an empty string for none",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    reports: List[Dict[str, Any]] = []
    for case in CASES:
        try:
            results = fetch_results(
                args.base_url,
                case.query,
                gender=args.gender or None,
                limit=max(20, case.top_n),
                timeout=args.timeout,
            )
            report = evaluate_case(case, results)
        except Exception as exc:  # one broken request should not hide later cases
            report = {
                "name": case.name,
                "query": case.query,
                "passed": False,
                "result_count": 0,
                "checked_count": 0,
                "categories": [],
                "colors": [],
                "brands": [],
                "failures": [str(exc)],
            }
        reports.append(report)

    passed = sum(report["passed"] for report in reports)
    if args.json:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "passed": passed,
                    "total": len(reports),
                    "cases": reports,
                },
                indent=2,
            )
        )
    else:
        for report in reports:
            status = "PASS" if report["passed"] else "FAIL"
            print(
                f"{status:4} {report['name']}: {report['query']!r} "
                f"({report['checked_count']}/{report['result_count']} checked)"
            )
            for failure in report["failures"]:
                print(f"     - {failure}")
        print(f"\n{passed}/{len(reports)} relevance cases passed")

    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
