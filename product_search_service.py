#!/usr/bin/env python3
"""Semantic product search service.

Embeds the user's query with OpenAI text-embedding-3-small — the same model
that embedded the catalog (scripts/reembed_omega_openai.py, 1536-dim) — and
ranks products in-database via the match_products_semantic RPC (pgvector
cosine over finer_products_omega, migrations/004_product_semantic_search.sql).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import supabase
from openai import OpenAI

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Must match the model used by scripts/reembed_omega_openai.py — query and
# catalog vectors only compare meaningfully inside one embedding space.
EMBED_MODEL = os.getenv("SEARCH_EMBED_MODEL", "text-embedding-3-small")
SEARCH_RPC = os.getenv("SEARCH_RPC", "match_products_semantic")


class ProductSearchService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)

    def embed_query(self, query: str) -> List[float]:
        """1536-dim text embedding of the search query."""
        response = self.openai.embeddings.create(model=EMBED_MODEL, input=[query])
        return response.data[0].embedding

    def search(
        self, query: str, limit: int = 20, gender: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Embed the query and rank products by cosine similarity in-database.

        gender must already be RPC-normalized (outfit_builder._rpc_gender):
        'masculine'/'feminine' filter to that label + genderless, None means
        no filter."""
        embedding = self.embed_query(query)
        params: Dict[str, Any] = {"query_embedding": embedding, "match_count": limit}
        if gender:
            params["p_gender"] = gender
        result = self.supabase.rpc(SEARCH_RPC, params).execute()
        return [self._serialize(row) for row in (result.data or [])]

    @staticmethod
    def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
        """Match the product shape the outfit endpoints return, plus
        slot/category and similarity. Fields the catalog doesn't have yet
        (sustainability_score, quality_score, materials) are omitted — iOS
        renders defaults for missing fields (finer-ios docs/backend-blanks.md)."""
        price = row.get("product_price_amount")
        similarity = row.get("similarity")
        item: Dict[str, Any] = {
            "product_id": row.get("product_id"),
            "product_title": row.get("product_title"),
            "product_brand": row.get("product_brand"),
            "product_price_amount": float(price) if price is not None else None,
            "product_img_link": row.get("product_img_link"),
            "product_link": row.get("product_link"),
            "similarity": round(float(similarity), 4) if similarity is not None else None,
        }
        slot = row.get("product_slot")
        if slot:
            item["slot"] = slot
            item["category"] = slot
        description = row.get("product_description")
        if description:
            item["description"] = description
        return item
