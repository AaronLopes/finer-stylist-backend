#!/usr/bin/env python3
"""Hybrid product search service.

Search combines two kinds of evidence:

* OpenAI embeddings handle fuzzy fashion language and style concepts.
* Catalog fields handle intent that should be exact, especially product
  category, color, and brand.

The hybrid database function is defined in
``migrations/010_hybrid_product_search.sql``. During a staged rollout the
service falls back to the legacy semantic RPC if Supabase has not loaded the
new function yet.
"""

from dataclasses import dataclass
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import supabase
from openai import OpenAI

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Must match the model used by scripts/reembed_omega_openai.py — query and
# catalog vectors only compare meaningfully inside one embedding space.
EMBED_MODEL = os.getenv("SEARCH_EMBED_MODEL", "text-embedding-3-small")
SEARCH_RPC = os.getenv("SEARCH_RPC", "match_products_hybrid")
LEGACY_SEARCH_RPC = os.getenv("LEGACY_SEARCH_RPC", "match_products_semantic")


@dataclass(frozen=True)
class _CategoryRule:
    aliases: Tuple[str, ...]
    categories: Tuple[str, ...]
    semantic_terms: Tuple[str, ...]
    diversify: bool = False


# Rules are ordered from specific phrases to broad parent concepts. Matching
# consumes the phrase, so "dress pants" does not also become a dress search.
_CATEGORY_RULES: Tuple[_CategoryRule, ...] = (
    _CategoryRule(
        ("dress pants",),
        ("pants",),
        ("dress pants", "trousers", "slacks"),
    ),
    _CategoryRule(
        ("cargo pants", "cargo trousers", "cargos"),
        ("pants",),
        ("cargo pants", "utility trousers"),
    ),
    _CategoryRule(
        ("sweatpants", "track pants", "joggers", "jogger"),
        ("pants",),
        ("sweatpants", "track pants", "joggers"),
    ),
    _CategoryRule(
        ("t shirt", "t shirts", "tee", "tees"),
        ("t_shirt",),
        ("t-shirt", "tee"),
    ),
    _CategoryRule(
        ("tank top", "tank tops"),
        ("tank_top",),
        ("tank top", "sleeveless top"),
    ),
    _CategoryRule(
        ("high heels",),
        ("heel",),
        ("heels", "pumps"),
    ),
    _CategoryRule(
        ("pants", "pant", "slacks"),
        ("pants", "jeans", "leggings"),
        (
            "pants",
            "trousers",
            "jeans",
            "chinos",
            "joggers",
            "sweatpants",
            "leggings",
        ),
        diversify=True,
    ),
    _CategoryRule(
        ("bottoms", "bottom"),
        ("pants", "jeans", "leggings", "shorts", "skirt_skort", "bottom_other"),
        ("pants", "jeans", "leggings", "shorts", "skirts"),
        diversify=True,
    ),
    _CategoryRule(
        ("trousers", "trouser", "chinos", "chino"),
        ("pants",),
        ("trousers", "pants", "slacks", "chinos"),
    ),
    _CategoryRule(
        ("jeans", "jean", "denim jeans"),
        ("jeans",),
        ("jeans", "denim"),
    ),
    _CategoryRule(
        ("leggings", "legging", "tights"),
        ("leggings",),
        ("leggings", "tights"),
    ),
    _CategoryRule(("shorts", "short"), ("shorts",), ("shorts",)),
    _CategoryRule(
        ("skirts", "skirt", "skorts", "skort"),
        ("skirt_skort",),
        ("skirts", "skorts"),
    ),
    _CategoryRule(("dresses", "dress"), ("dress",), ("dress", "gown")),
    _CategoryRule(
        ("tops", "top"),
        (
            "t_shirt",
            "tank_top",
            "sweater",
            "shirt",
            "blouse",
            "top_other",
            "polo",
            "bodysuit",
        ),
        ("tops", "shirts", "tees", "blouses", "sweaters"),
        diversify=True,
    ),
    _CategoryRule(
        ("shirts", "shirt", "button down", "button downs"),
        ("shirt",),
        ("shirt", "button-down"),
    ),
    _CategoryRule(("blouses", "blouse"), ("blouse",), ("blouse",)),
    _CategoryRule(("polos", "polo"), ("polo",), ("polo shirt",)),
    _CategoryRule(
        ("hoodies", "hoodie", "sweatshirts", "sweatshirt"),
        ("sweatshirt_hoodie",),
        ("hoodie", "sweatshirt"),
    ),
    _CategoryRule(
        ("sweaters", "sweater", "jumpers", "jumper", "knitwear"),
        ("sweater",),
        ("sweater", "jumper", "knitwear"),
    ),
    _CategoryRule(
        ("cardigans", "cardigan"),
        ("cardigan",),
        ("cardigan",),
    ),
    _CategoryRule(
        ("outerwear",),
        ("jacket", "coat", "blazer", "cardigan", "vest", "outerwear_other"),
        ("jackets", "coats", "blazers", "cardigans", "vests"),
        diversify=True,
    ),
    _CategoryRule(("jackets", "jacket"), ("jacket",), ("jacket",)),
    _CategoryRule(("coats", "coat"), ("coat",), ("coat",)),
    _CategoryRule(("blazers", "blazer"), ("blazer",), ("blazer",)),
    _CategoryRule(("vests", "vest"), ("vest",), ("vest",)),
    _CategoryRule(
        ("shoes", "shoe", "footwear"),
        ("sneaker", "boot", "sandal", "loafer_flat", "heel", "footwear_other"),
        ("shoes", "sneakers", "boots", "sandals", "loafers", "heels"),
        diversify=True,
    ),
    _CategoryRule(
        ("sneakers", "sneaker", "trainers", "trainer"),
        ("sneaker",),
        ("sneakers", "trainers"),
    ),
    _CategoryRule(("boots", "boot"), ("boot",), ("boots",)),
    _CategoryRule(("sandals", "sandal"), ("sandal",), ("sandals",)),
    _CategoryRule(
        ("loafers", "loafer", "flats", "flat"),
        ("loafer_flat",),
        ("loafers", "flats"),
    ),
    _CategoryRule(("heels", "heel", "pumps", "pump"), ("heel",), ("heels", "pumps")),
    _CategoryRule(
        ("accessories", "accessory"),
        ("hat", "bag", "eyewear", "scarf", "jewelry", "belt", "accessory_other"),
        ("accessories", "hats", "bags", "eyewear", "scarves", "jewelry", "belts"),
        diversify=True,
    ),
    _CategoryRule(("bags", "bag", "purses", "purse"), ("bag",), ("bags", "purses")),
    _CategoryRule(("hats", "hat", "caps", "cap"), ("hat",), ("hats", "caps")),
    _CategoryRule(("scarves", "scarf"), ("scarf",), ("scarves",)),
    _CategoryRule(("belts", "belt"), ("belt",), ("belts",)),
    _CategoryRule(
        ("jewelry", "jewellery"),
        ("jewelry",),
        ("jewelry", "jewellery"),
    ),
    _CategoryRule(
        ("sunglasses", "glasses", "eyewear"),
        ("eyewear",),
        ("sunglasses", "eyewear"),
    ),
    _CategoryRule(
        ("jumpsuits", "jumpsuit", "rompers", "romper"),
        ("jumpsuit_romper",),
        ("jumpsuit", "romper"),
    ),
    _CategoryRule(("bodysuits", "bodysuit"), ("bodysuit",), ("bodysuit",)),
    _CategoryRule(("underwear", "lingerie"), ("underwear",), ("underwear", "lingerie")),
)

# Values on the right match the normalized product_color values in Omega.
# Closely related shades intentionally widen to the catalog's coarser labels.
_COLOR_ALIASES: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("off white", "cream"), ("ivory", "white")),
    (("black",), ("black",)),
    (("white",), ("white",)),
    (("grey", "gray", "charcoal"), ("gray",)),
    (("navy", "indigo"), ("blue",)),
    (("blue",), ("blue",)),
    (("brown", "chocolate"), ("brown",)),
    (("beige",), ("beige",)),
    (("tan",), ("tan",)),
    (("ivory",), ("ivory",)),
    (("green", "olive"), ("green",)),
    (("pink", "rose"), ("pink",)),
    (("purple", "lilac", "violet"), ("purple",)),
    (("burgundy", "maroon", "wine"), ("burgundy", "red")),
    (("red", "crimson"), ("red",)),
    (("yellow",), ("yellow",)),
    (("orange",), ("orange",)),
    (("teal", "turquoise"), ("teal",)),
    (("silver",), ("silver",)),
    (("gold",), ("gold", "yellow")),
    (("multicolor", "multicolored", "multi color"), ("multicolor", "multicolored")),
)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "me",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "with",
}

# Useful semantic modifiers, but very unlikely to be a brand name on their
# own. Removing them keeps "linen shirts" from becoming a brand lookup.
_NON_BRAND_TERMS = {
    "all",
    "autumn",
    "boho",
    "bohemian",
    "casual",
    "cashmere",
    "cheap",
    "classic",
    "comfortable",
    "corduroy",
    "cotton",
    "denim",
    "designer",
    "elegant",
    "embroidered",
    "fall",
    "feminine",
    "fleece",
    "floral",
    "formal",
    "hemp",
    "leather",
    "linen",
    "loose",
    "luxury",
    "masculine",
    "minimal",
    "minimalist",
    "modal",
    "nylon",
    "organic",
    "oversized",
    "party",
    "patterned",
    "polyester",
    "rayon",
    "relaxed",
    "silk",
    "skinny",
    "slim",
    "smart",
    "spring",
    "streetwear",
    "striped",
    "suede",
    "summer",
    "sustainable",
    "tencel",
    "tight",
    "unisex",
    "velvet",
    "viscose",
    "winter",
    "women",
    "womens",
    "wool",
    "work",
}


@dataclass(frozen=True)
class SearchIntent:
    """Deterministic structured understanding of a search query."""

    query: str
    semantic_query: str
    categories: Tuple[str, ...]
    colors: Tuple[str, ...]
    query_terms: Tuple[str, ...]
    brand_query: Optional[str]
    brand_terms: Tuple[str, ...]
    require_brand: bool
    diversify_categories: bool

    def hybrid_rpc_params(
        self, embedding: Optional[Sequence[float]], limit: int
    ) -> Dict[str, Any]:
        return {
            "query_embedding": list(embedding) if embedding is not None else None,
            "match_count": limit,
            "p_query": self.query,
            "p_query_terms": list(self.query_terms) or None,
            "p_categories": list(self.categories) or None,
            "p_colors": list(self.colors) or None,
            "p_brand_query": self.brand_query,
            "p_brand_terms": list(self.brand_terms) or None,
            "p_require_brand": self.require_brand,
            "p_diversify_categories": self.diversify_categories,
        }


def _normalized_words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _append_unique(target: List[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def parse_search_intent(query: str) -> SearchIntent:
    """Extract only high-confidence catalog facets; embeddings handle the rest."""
    normalized = _normalized_words(query)
    remaining = f" {normalized} "
    categories: List[str] = []
    semantic_terms: List[str] = []
    colors: List[str] = []
    consumed_words = set()
    diversify = False

    for rule in _CATEGORY_RULES:
        matched = False
        for alias in sorted(rule.aliases, key=len, reverse=True):
            normalized_alias = _normalized_words(alias)
            needle = f" {normalized_alias} "
            if needle in remaining:
                matched = True
                consumed_words.update(normalized_alias.split())
                remaining = remaining.replace(needle, " ")
        if matched:
            _append_unique(categories, rule.categories)
            _append_unique(semantic_terms, rule.semantic_terms)
            diversify = diversify or rule.diversify

    for aliases, catalog_colors in _COLOR_ALIASES:
        matched = False
        for alias in sorted(aliases, key=len, reverse=True):
            normalized_alias = _normalized_words(alias)
            needle = f" {normalized_alias} "
            if needle in remaining:
                matched = True
                consumed_words.update(normalized_alias.split())
                remaining = remaining.replace(needle, " ")
        if matched:
            _append_unique(colors, catalog_colors)

    words = tuple(word for word in normalized.split() if len(word) >= 2)
    query_terms = tuple(word for word in words if word not in _STOP_WORDS)
    brand_terms = tuple(
        word
        for word in query_terms
        if word not in consumed_words and word not in _NON_BRAND_TERMS
    )

    explicit_brand = bool(re.search(r"\b(?:brand|by|from)\b", normalized))
    if brand_terms:
        # A query made entirely of otherwise-unclassified words may be a
        # multi-word brand ("Fear of God", "CMMN SWDN"). Preserve the original
        # compact phrase so the database can recognize it exactly.
        unclassified_words = {
            word for word in words if word not in _STOP_WORDS
        }
        if not categories and not colors and set(brand_terms) == unclassified_words:
            brand_query = normalized
        else:
            brand_query = " ".join(brand_terms)
    else:
        brand_query = None

    # Requiring a brand is safe for explicit "by/from/brand" syntax and for a
    # single otherwise-unclassified token such as "Kowtow". Mixed natural
    # language only receives a brand boost, avoiding false empty result sets.
    require_brand = bool(
        brand_query
        and (
            explicit_brand
            or (
                not categories
                and not colors
                and len(brand_terms) == 1
                and len(query_terms) == 1
            )
        )
    )

    if semantic_terms:
        semantic_query = (
            f"{query}. Related fashion categories: {', '.join(semantic_terms)}."
        )
    else:
        semantic_query = query

    return SearchIntent(
        query=query,
        semantic_query=semantic_query,
        categories=tuple(categories),
        colors=tuple(colors),
        query_terms=query_terms,
        brand_query=brand_query,
        brand_terms=brand_terms,
        require_brand=require_brand,
        diversify_categories=diversify and len(categories) > 1,
    )


def _hybrid_rpc_is_missing(exc: Exception) -> bool:
    message = str(exc).casefold()
    return (
        "pgrst202" in message
        or "could not find the function" in message
        or ("schema cache" in message and SEARCH_RPC.casefold() in message)
    )


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
        """Rank products with semantic, category, color, and brand evidence.

        gender must already be RPC-normalized (outfit_builder._rpc_gender):
        'masculine'/'feminine' filter to that label + genderless, None means
        no filter.
        """
        intent = parse_search_intent(query)
        embedding: Optional[List[float]]
        embedding_error: Optional[Exception] = None
        try:
            embedding = self.embed_query(intent.semantic_query)
        except Exception as exc:
            # The hybrid RPC can still answer structured and lexical searches
            # when the embedding provider has a temporary outage.
            embedding = None
            embedding_error = exc
            logger.warning(
                "Query embedding failed; attempting lexical search error=%s", exc
            )

        params = intent.hybrid_rpc_params(embedding, limit)
        if gender:
            params["p_gender"] = gender

        try:
            result = self.supabase.rpc(SEARCH_RPC, params).execute()
        except Exception as exc:
            if not _hybrid_rpc_is_missing(exc):
                raise
            if embedding is None:
                raise RuntimeError(
                    "Search embedding failed and the hybrid search RPC is unavailable"
                ) from embedding_error

            # Safe staged-deployment fallback: code can ship before migration
            # 010 reaches Supabase without breaking the Search tab.
            logger.warning(
                "Hybrid search RPC unavailable; falling back to %s", LEGACY_SEARCH_RPC
            )
            legacy_params: Dict[str, Any] = {
                "query_embedding": embedding,
                "match_count": limit,
            }
            if gender:
                legacy_params["p_gender"] = gender
            result = self.supabase.rpc(LEGACY_SEARCH_RPC, legacy_params).execute()

        return [self._serialize(row) for row in (result.data or [])]

    @staticmethod
    def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
        """Match the product shape used by outfit endpoints and iOS search."""
        price = row.get("product_price_amount")
        similarity = row.get("similarity")
        search_score = row.get("search_score")
        item: Dict[str, Any] = {
            "product_id": row.get("product_id"),
            "product_title": row.get("product_title"),
            "product_brand": row.get("product_brand"),
            "product_price_amount": float(price) if price is not None else None,
            "product_img_link": row.get("product_img_link"),
            "product_link": row.get("product_link"),
            "similarity": round(float(similarity), 4) if similarity is not None else None,
        }
        if search_score is not None:
            item["search_score"] = round(float(search_score), 4)
        slot = row.get("product_slot")
        category = row.get("product_category") or slot
        if slot:
            item["slot"] = slot
        if category:
            item["category"] = category
        color = row.get("product_color")
        if color:
            item["product_color"] = color
        description = row.get("product_description")
        if description:
            item["description"] = description
        return item
