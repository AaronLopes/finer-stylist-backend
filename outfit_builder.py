#!/usr/bin/env python3
"""
Outfit Builder for Finer

Unified outfit building service that supports:
1. Quiz-based outfit generation (Web + iOS)
2. Chat-based natural language queries (iOS)

This module orchestrates the outfit building process by:
- Translating quiz answers or chat queries into structured parameters
- Calling ff_build_outfit_v2 for each slot with context
- Maintaining outfit cohesion (color, texture, formality balance)

Usage:
    from outfit_builder import OutfitBuilder

    builder = OutfitBuilder()
    outfit = builder.build_from_quiz(quiz_answers)
    outfit = builder.build_from_chat(query, user_profile)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set

from dotenv import load_dotenv
from openai import OpenAI
import supabase

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Product table priority. "omega" = omega-primary/chi-fallback (default).
# "chi" = chi-primary/omega-fallback. Set via env to switch without deploy.
_primary = os.getenv("PRIMARY_RPC", "omega").lower()
if _primary == "chi":
    RPC_ORDER = [("ff_build_outfit_chi", "chi"), ("ff_build_outfit_v3", "omega")]
else:
    RPC_ORDER = [("ff_build_outfit_v3", "omega"), ("ff_build_outfit_chi", "chi")]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Product table priority: %s-primary", RPC_ORDER[0][1])


class ColorStrategy(str, Enum):
    BALANCED = "balanced"
    BOLD = "bold"
    NEUTRAL = "neutral"
    MONOCHROME = "monochrome"


@dataclass
class OutfitParams:
    """Unified parameters for outfit building - used by both quiz and chat modes."""

    # Core filters
    gender: str  # masculine, feminine, genderless
    occasion: str  # date, work, casual, going-out, gym
    budget_min: float = 0
    budget_max: float = 10000

    # Tag-based matching
    style_tags: List[str] = field(default_factory=list)
    occasion_tags: List[str] = field(default_factory=list)
    season_tags: List[str] = field(default_factory=list)
    avoid_tags: List[str] = field(default_factory=list)
    category_includes: Dict[str, List[str]] = field(default_factory=dict)
    category_excludes: Dict[str, List[str]] = field(default_factory=dict)

    # Stylist modifiers
    hero_boost: float = 1.0  # 1.5 for "serve-looks"
    color_strategy: ColorStrategy = ColorStrategy.BALANCED

    # Affiliate boosting
    boost_affiliates: bool = True  # Default to boosting affiliates
    affiliate_boost_weight: float = 0.15  # How much to boost (0-1)


@dataclass
class SlotConfig:
    """Configuration for a single outfit slot."""

    slot: str  # top, bottom, footwear, outerwear, accessory
    formality: int  # 1-5
    is_hero: bool = False
    include_categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None


@dataclass
class OutfitFormula:
    """Defines how to build an outfit for a specific occasion."""

    base_formality: int
    slots: List[SlotConfig]
    balance_rule: str  # "match", "contrast", "elevate"


# Occasion → Outfit Formula mappings
OCCASION_FORMULAS: Dict[str, OutfitFormula] = {
    "work": OutfitFormula(
        base_formality=4,
        slots=[
            SlotConfig("top", formality=4, is_hero=False),
            SlotConfig("bottom", formality=4, is_hero=True),
            SlotConfig("footwear", formality=4, is_hero=False),
        ],
        balance_rule="match",
    ),
    "date": OutfitFormula(
        base_formality=3,
        slots=[
            SlotConfig("top", formality=4, is_hero=True),
            SlotConfig("bottom", formality=2, is_hero=False),
            SlotConfig("footwear", formality=3, is_hero=False),
        ],
        balance_rule="contrast",
    ),
    "going-out": OutfitFormula(
        base_formality=4,
        slots=[
            SlotConfig("top", formality=4, is_hero=True),
            SlotConfig("bottom", formality=4, is_hero=False),
            SlotConfig("footwear", formality=4, is_hero=False),
        ],
        balance_rule="elevate",
    ),
    "gym": OutfitFormula(
        base_formality=1,
        slots=[
            SlotConfig("top", formality=1, is_hero=False),
            SlotConfig("bottom", formality=1, is_hero=False),
            SlotConfig("footwear", formality=1, is_hero=False),
        ],
        balance_rule="match",
    ),
    "casual": OutfitFormula(
        base_formality=2,
        slots=[
            SlotConfig("top", formality=2, is_hero=False),
            SlotConfig("bottom", formality=2, is_hero=False),
            SlotConfig("footwear", formality=2, is_hero=False),
        ],
        balance_rule="match",
    ),
}

VALID_CATEGORIES_BY_SLOT: Dict[str, List[str]] = {
    "top": [
        "t_shirt",
        "shirt",
        "blouse",
        "tank_top",
        "sweater",
        "sweatshirt_hoodie",
        "polo",
        "bodysuit",
        "dress",
        "jumpsuit_romper",
        "top_other",
    ],
    "bottom": [
        "pants",
        "jeans",
        "shorts",
        "skirt_skort",
        "leggings",
        "underwear",
        "bottom_other",
    ],
    "footwear": [
        "sneaker",
        "boot",
        "sandal",
        "heel",
        "loafer_flat",
        "footwear_other",
    ],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "vest", "outerwear_other"],
    "accessory": [
        "bag",
        "jewelry",
        "hat",
        "belt",
        "scarf",
        "eyewear",
        "watch",
        "accessory_other",
    ],
}

DEFAULT_EXCLUDE_CATEGORIES_BY_SLOT: Dict[str, List[str]] = {
    "top": ["dress", "jumpsuit_romper"],
    "bottom": ["underwear"],
}

OCCASION_CATEGORY_TIERS: Dict[str, Dict[str, List[List[str]]]] = {
    "work": {
        "top": [
            ["shirt", "blouse", "sweater", "polo"],
            ["shirt", "blouse", "sweater", "polo", "t_shirt"],
        ],
        "bottom": [["pants"], ["pants", "skirt_skort", "jeans"]],
        "footwear": [
            ["loafer_flat", "boot", "heel"],
            ["loafer_flat", "boot", "heel", "sneaker"],
        ],
    },
    "date": {
        "top": [
            ["blouse", "shirt", "sweater", "bodysuit"],
            ["blouse", "shirt", "sweater", "bodysuit", "tank_top", "t_shirt"],
        ],
        "bottom": [
            ["jeans", "skirt_skort", "pants"],
            ["jeans", "skirt_skort", "pants", "shorts"],
        ],
        "footwear": [
            ["heel", "boot", "loafer_flat", "sandal"],
            ["heel", "boot", "loafer_flat", "sandal", "sneaker"],
        ],
    },
    "going-out": {
        "top": [
            ["blouse", "bodysuit", "tank_top", "shirt"],
            ["blouse", "bodysuit", "tank_top", "shirt", "sweater"],
        ],
        "bottom": [
            ["pants", "skirt_skort", "jeans"],
            ["pants", "skirt_skort", "jeans", "shorts"],
        ],
        "footwear": [
            ["heel", "boot", "loafer_flat"],
            ["heel", "boot", "loafer_flat", "sandal"],
        ],
    },
    "gym": {
        "top": [
            ["t_shirt", "tank_top", "sweatshirt_hoodie"],
            ["t_shirt", "tank_top", "sweatshirt_hoodie", "top_other"],
        ],
        "bottom": [
            ["leggings", "shorts", "pants"],
            ["leggings", "shorts", "pants", "bottom_other"],
        ],
        "footwear": [["sneaker"], ["sneaker", "footwear_other"]],
    },
    "casual": {
        "top": [
            ["t_shirt", "shirt", "sweater", "sweatshirt_hoodie"],
            [
                "t_shirt",
                "shirt",
                "sweater",
                "sweatshirt_hoodie",
                "tank_top",
                "polo",
                "blouse",
            ],
        ],
        "bottom": [
            ["jeans", "pants", "shorts", "leggings"],
            ["jeans", "pants", "shorts", "leggings", "skirt_skort"],
        ],
        "footwear": [
            ["sneaker", "boot", "sandal", "loafer_flat"],
            ["sneaker", "boot", "sandal", "loafer_flat", "footwear_other"],
        ],
    },
}

CHAT_CATEGORY_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "top": {
        "t_shirt": [r"\btees?\b", r"\bt-?shirts?\b"],
        "shirt": [r"\bshirts?\b", r"\bbutton[-\s]?downs?\b"],
        "blouse": [r"\bblouses?\b"],
        "tank_top": [r"\btanks?\b", r"\btank tops?\b"],
        "sweater": [r"\bsweaters?\b", r"\bknits?\b"],
        "sweatshirt_hoodie": [r"\bhoodies?\b", r"\bsweatshirts?\b"],
        "polo": [r"\bpolos?\b"],
        "bodysuit": [r"\bbodysuits?\b"],
    },
    "bottom": {
        "pants": [r"\bpants?\b", r"\btrousers?\b", r"\bslacks?\b", r"\bchinos?\b"],
        "jeans": [r"\bjeans?\b", r"\bdenim\b"],
        "shorts": [r"\bshorts?\b"],
        "skirt_skort": [r"\bskirts?\b", r"\bskorts?\b"],
        "leggings": [r"\bleggings?\b"],
    },
    "footwear": {
        "sneaker": [r"\bsneakers?\b", r"\btennis shoes?\b"],
        "boot": [r"\bboots?\b"],
        "sandal": [r"\bsandals?\b"],
        "heel": [r"\bheels?\b"],
        "loafer_flat": [r"\bloafers?\b", r"\bflats?\b"],
    },
}

SLOT_CANDIDATE_POOL_SIZE = 8

UNDERWEAR_RE = re.compile(
    r"\b(underwear|boxer\s*briefs?|boxers?|briefs?|panties|bras?|bralettes?|"
    r"lingerie|thongs?|knickers?)\b",
    re.I,
)


def _normalize_gender(value: Optional[str]) -> str:
    gender = (value or "").strip().lower()
    if gender in {"masculine", "feminine"}:
        return gender
    if gender in {"genderless", "unisex", "non-binary", "nonbinary", ""}:
        return "genderless"
    return "genderless"


def _rpc_gender(value: Optional[str]) -> Optional[str]:
    gender = _normalize_gender(value)
    return None if gender == "genderless" else gender


def _dedupe_lower(values: Optional[List[str]]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values or []:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _valid_categories_for_slot(slot: str, categories: Optional[List[str]]) -> List[str]:
    valid = set(VALID_CATEGORIES_BY_SLOT.get(slot, []))
    return [cat for cat in _dedupe_lower(categories) if cat in valid]


def _sanitize_category_map(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    sanitized: Dict[str, List[str]] = {}
    for slot, categories in value.items():
        slot_key = str(slot or "").strip().lower()
        if slot_key not in VALID_CATEGORIES_BY_SLOT:
            continue
        if isinstance(categories, str):
            raw_categories = [categories]
        elif isinstance(categories, list):
            raw_categories = categories
        else:
            continue
        valid_categories = _valid_categories_for_slot(slot_key, raw_categories)
        if valid_categories:
            sanitized[slot_key] = valid_categories
    return sanitized


def _merge_category_maps(*maps: Any) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for category_map in maps:
        for slot, categories in _sanitize_category_map(category_map).items():
            merged[slot] = _valid_categories_for_slot(
                slot, [*merged.get(slot, []), *categories]
            )
    return {slot: categories for slot, categories in merged.items() if categories}

# Setting → Tag modifiers
# NOTE: Avoid "casual" - 72% of products have it, making it meaningless noise
SETTING_TAG_MAP: Dict[str, List[str]] = {
    "city": ["urban", "polished", "structured", "smart_casual", "streetwear"],
    "country": ["relaxed", "earth_tones", "bohemian"],
    "suburbs": ["relaxed", "everyday", "classic"],
    "beach": ["summer", "relaxed", "bright", "linen"],
    "night-out": ["dark", "going_out", "fitted", "dressy"],
}

# Weather → Season/Material tags
WEATHER_TAG_MAP: Dict[str, Dict[str, List[str]]] = {
    "hot": {
        "season_tags": ["summer", "all_season"],
        "material_tags": ["linen", "cotton"],
        "avoid_tags": ["wool", "fleece", "winter", "long_sleeve"],
    },
    "moderate": {
        "season_tags": ["spring", "fall", "all_season"],
        "material_tags": ["cotton"],
        "avoid_tags": [],
    },
    "cold": {
        "season_tags": ["winter", "fall"],
        "material_tags": ["wool", "fleece"],
        "avoid_tags": ["sleeveless", "linen", "summer"],
    },
}

# Style → Core tags
# NOTE: Avoid "casual" - 72% of products have it, making it meaningless noise
STYLE_TAG_MAP: Dict[str, List[str]] = {
    "minimalist": ["minimalist", "solid", "neutral", "classic"],
    "bohemian": ["bohemian", "floral", "earth_tones", "relaxed"],
    "fitted": ["fitted", "slim", "structured"],
    "smart-casual": ["smart_casual", "classic", "polished"],
    "classic": ["classic", "neutral", "solid"],
    "streetwear": ["streetwear", "fitted", "dark"],
}

# Occasion-specific tag overrides (these REPLACE style tags for specific occasions)
# NOTE: Avoid "casual" - 72% of products have it, making it meaningless noise
OCCASION_STYLE_OVERRIDE: Dict[str, List[str]] = {
    "gym": ["gym", "activewear", "athletic", "workout"],
    "beach": ["swimwear", "summer", "linen", "bright"],
}

# Goals → Strategy modifiers
GOALS_STRATEGY: Dict[str, Dict[str, Any]] = {
    "serve-looks": {
        "hero_boost": 1.5,
        "color_strategy": ColorStrategy.BOLD,
        "boost_affiliates": True,
        "affiliate_boost_weight": 0.08,
    },
    "inspired": {
        "hero_boost": 1.0,
        "color_strategy": ColorStrategy.BALANCED,
        "boost_affiliates": True,
        "affiliate_boost_weight": 0.08,
    },
    "discover-brands": {
        "hero_boost": 1.0,
        "color_strategy": ColorStrategy.BALANCED,
        "boost_affiliates": True,
        "affiliate_boost_weight": 0.15,  # Moderate affiliate boost for brand discovery
    },
    "capsule": {
        "hero_boost": 0.8,
        "color_strategy": ColorStrategy.NEUTRAL,
        "boost_affiliates": True,
        "affiliate_boost_weight": 0.05,
    },
}

# Budget → Price ranges
# NOTE: These are per-item ranges, not total outfit
# Ranges overlap to ensure we find products at budget boundaries
BUDGET_RANGES: Dict[str, Dict[str, float]] = {
    "$": {"min": 0, "max": 75},
    "$$": {"min": 0, "max": 150},  # Overlaps with $ to catch more
    "$$$": {"min": 0, "max": 300},  # No min - include affordable activewear etc
    "$$$$": {"min": 0, "max": 10000},  # Luxury - no limits
}

# Calendar event keyword → occasion. Substring-matched against event title.
# Order matters only when keywords overlap; keep most-specific first.
CALENDAR_OCCASION_MAP: Dict[str, str] = {
    # Work-ish
    "standup": "work",
    "stand-up": "work",
    "1:1": "work",
    "interview": "work",
    "presentation": "work",
    "conference": "work",
    "review": "work",
    "sprint": "work",
    "office": "work",
    "meeting": "work",
    # Date-ish
    "anniversary": "date",
    "date night": "date",
    "dinner with": "date",
    "dinner": "date",
    "drinks with": "date",
    # Going-out
    "wedding": "going-out",
    "gala": "going-out",
    "concert": "going-out",
    "club": "going-out",
    "party": "going-out",
    "show": "going-out",
    # Gym
    "workout": "gym",
    "yoga": "gym",
    "pilates": "gym",
    "training": "gym",
    "gym": "gym",
    "run": "gym",
}


def _temp_to_weather_bucket(temp_f: float) -> str:
    """Map a Fahrenheit temperature to a WEATHER_TAG_MAP bucket."""
    if temp_f < 50:
        return "cold"
    if temp_f >= 75:
        return "hot"
    return "moderate"


def _event_to_occasion(event: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Map a calendar event dict {title, category?, all_day?} to an OCCASION_FORMULAS key.
    Returns None when no confident match — caller falls through to profile/default.
    """
    if not event:
        return None
    category = (event.get("category") or "").lower()
    if category in OCCASION_FORMULAS:
        return category
    title = (event.get("title") or "").lower()
    if not title:
        return None
    for keyword, occasion in CALENDAR_OCCASION_MAP.items():
        if keyword in title:
            return occasion
    return None


class OutfitBuilder:
    """
    Main outfit building service.

    Supports two input modes:
    1. Quiz mode: Structured answers from quiz flow
    2. Chat mode: Natural language queries
    """

    _location_cache: Dict[str, str] = {}

    def __init__(self):
        self.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OutfitBuilder initialized")

    # =========================================================================
    # LOCATION CLASSIFICATION (LLM + cache)
    # =========================================================================

    def _classify_location(self, city_label: str) -> str:
        """
        Classify a city_label (e.g. "Brooklyn, NY") into a SETTING_TAG_MAP key
        using gpt-4o-mini. Results are cached in-memory per unique city_label.
        Returns one of: city, suburbs, country, beach.  Falls back to "city".
        """
        key = city_label.strip().lower()
        if key in self._location_cache:
            return self._location_cache[key]

        valid_settings = ["city", "suburbs", "country", "beach"]
        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify locations for a fashion styling app. "
                            "Given a location label, respond with exactly one word: "
                            "city, suburbs, country, or beach. Nothing else."
                        ),
                    },
                    {"role": "user", "content": city_label},
                ],
                max_tokens=5,
                temperature=0,
            )
            setting = resp.choices[0].message.content.strip().lower()
            if setting not in valid_settings:
                setting = "city"
        except Exception as e:
            logger.warning("Location classification failed for '%s': %s", city_label, e)
            setting = "city"

        self._location_cache[key] = setting
        logger.info("Classified location '%s' → %s", city_label, setting)
        return setting

    # =========================================================================
    # CHI/OMEGA FALLBACK
    # =========================================================================

    @staticmethod
    def _category_filters(slot_config: SlotConfig) -> Dict[str, Optional[List[str]]]:
        include = _valid_categories_for_slot(
            slot_config.slot, slot_config.include_categories
        )
        hard_excludes = DEFAULT_EXCLUDE_CATEGORIES_BY_SLOT.get(slot_config.slot, [])
        exclude = _valid_categories_for_slot(
            slot_config.slot, [*hard_excludes, *(slot_config.exclude_categories or [])]
        )
        return {"include": include or None, "exclude": exclude}

    @staticmethod
    def _slot_category_attempts(
        params: OutfitParams, slot_config: SlotConfig
    ) -> List[SlotConfig]:
        slot = slot_config.slot
        slot_excludes = _valid_categories_for_slot(
            slot,
            [
                *(slot_config.exclude_categories or []),
                *params.category_excludes.get(slot, []),
            ],
        )

        explicit_include = _valid_categories_for_slot(
            slot, slot_config.include_categories
        )
        chat_include = _valid_categories_for_slot(
            slot, params.category_includes.get(slot, [])
        )
        if explicit_include:
            include_tiers = [explicit_include]
        elif chat_include:
            include_tiers = [chat_include]
        else:
            include_tiers = OCCASION_CATEGORY_TIERS.get(
                params.occasion, OCCASION_CATEGORY_TIERS["casual"]
            ).get(slot, [])

        attempts: List[SlotConfig] = []
        seen: Set[tuple] = set()
        for include in include_tiers:
            valid_include = _valid_categories_for_slot(slot, include)
            if not valid_include:
                continue
            key = tuple(valid_include)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                SlotConfig(
                    slot=slot,
                    formality=slot_config.formality,
                    is_hero=slot_config.is_hero,
                    include_categories=valid_include,
                    exclude_categories=slot_excludes,
                )
            )

        attempts.append(
            SlotConfig(
                slot=slot,
                formality=slot_config.formality,
                is_hero=slot_config.is_hero,
                include_categories=None,
                exclude_categories=slot_excludes,
            )
        )
        return attempts

    @staticmethod
    def _rpc_params_for_name(
        rpc_name: str, rpc_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if rpc_name == "ff_build_outfit_v3":
            return rpc_params
        return {
            k: v
            for k, v in rpc_params.items()
            if k not in {"p_include_categories", "p_exclude_categories"}
        }

    @staticmethod
    def _candidate_category(item: Dict[str, Any]) -> Optional[str]:
        category = (item.get("product_category") or item.get("category") or "").lower()
        if category:
            return category
        text_parts = [item.get("product_title") or "", item.get("product_description") or ""]
        tags = item.get("product_tags") or []
        if isinstance(tags, list):
            text_parts.extend(str(t) for t in tags)
        else:
            text_parts.append(str(tags))
        if UNDERWEAR_RE.search(" ".join(text_parts)):
            return "underwear"
        return None

    def _candidate_allowed_for_slot(
        self, item: Dict[str, Any], slot_config: SlotConfig
    ) -> bool:
        filters = self._category_filters(slot_config)
        category = self._candidate_category(item)
        include = [c.lower() for c in filters["include"] or []]
        exclude = [c.lower() for c in filters["exclude"] or []]
        if include and category not in include:
            return False
        if category in exclude:
            logger.info(
                "  ↓ rejected slot=%s category=%s title=%s",
                slot_config.slot,
                category,
                (item.get("product_title") or "Unknown")[:50],
            )
            return False
        return True

    def _query_slot_with_fallback(
        self,
        rpc_params: Dict[str, Any],
        slot_name: str,
        slot_config: Optional[SlotConfig] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Try the primary product table first. If it returns empty, fall back
        to the secondary. Order controlled by PRIMARY_RPC env var.
        Returns the best item dict with a _source annotation, or None.
        """
        slot_config = slot_config or SlotConfig(slot_name, formality=3)
        category_filters = self._category_filters(slot_config)
        query_params = dict(rpc_params)
        if category_filters["include"] or category_filters["exclude"]:
            query_params["p_n"] = max(
                int(query_params.get("p_n") or 1), SLOT_CANDIDATE_POOL_SIZE
            )
        for rpc_name, label in RPC_ORDER:
            try:
                result = self.supabase.rpc(
                    rpc_name, self._rpc_params_for_name(rpc_name, query_params)
                ).execute()
                if result.data and len(result.data) > 0:
                    for item in result.data:
                        if not self._candidate_allowed_for_slot(item, slot_config):
                            continue
                        item["_source"] = label
                        logger.info(
                            "  ✓ [%s] slot=%s found %d candidates",
                            label, slot_name, len(result.data),
                        )
                        return item
                logger.info("  ↓ [%s] slot=%s empty, trying next", label, slot_name)
            except Exception as e:
                logger.warning(
                    "  ⚠ [%s] slot=%s RPC error: %s", label, slot_name, e
                )
        return None

    def _query_slot_candidates_with_fallback(
        self,
        rpc_params: Dict[str, Any],
        slot_name: str,
        desired_n: int = 5,
        slot_config: Optional[SlotConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Like _query_slot_with_fallback but returns multiple candidates.
        Tries primary first; if fewer than desired_n, tops up from secondary.
        Dedupes by product_id.
        """
        slot_config = slot_config or SlotConfig(slot_name, formality=3)
        seen_ids: set = set()
        all_candidates: List[Dict[str, Any]] = []

        for rpc_name, label in RPC_ORDER:
            if len(all_candidates) >= desired_n:
                break
            try:
                params = {**rpc_params, "p_n": desired_n}
                result = self.supabase.rpc(
                    rpc_name, self._rpc_params_for_name(rpc_name, params)
                ).execute()
                for item in result.data or []:
                    if not self._candidate_allowed_for_slot(item, slot_config):
                        continue
                    pid = item.get("product_id")
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        item["_source"] = label
                        all_candidates.append(item)
            except Exception as e:
                logger.warning(
                    "  ⚠ [%s] slot=%s candidates RPC error: %s", label, slot_name, e
                )

        logger.info(
            "  slot=%s total candidates=%d", slot_name, len(all_candidates)
        )
        return all_candidates

    # =========================================================================
    # QUIZ MODE
    # =========================================================================

    def build_from_quiz(self, quiz_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an outfit from quiz answers.

        Args:
            quiz_answers: {
                "gender": "feminine",
                "occasion": "date",
                "weather": ["moderate"],
                "setting": "city",
                "goals": ["serve-looks"],
                "style": "minimalist",
                "budget": "$$"
            }

        Returns:
            Complete outfit with items for each slot
        """
        logger.info("Building outfit from quiz: %s", quiz_answers)

        # Convert quiz answers to OutfitParams
        params = self._quiz_to_params(quiz_answers)

        # Get outfit formula for occasion
        occasion = quiz_answers.get("occasion", "casual")
        formula = OCCASION_FORMULAS.get(occasion, OCCASION_FORMULAS["casual"])

        # Build outfit slot by slot
        return self._build_outfit(params, formula)

    # =========================================================================
    # CANONICAL PARAMS BUILDER (used by quiz, chat, and today)
    # =========================================================================

    def build_params(
        self,
        *,
        profile: Optional[Dict[str, Any]] = None,
        chat_extract: Optional[Dict[str, Any]] = None,
        today_context: Optional[Dict[str, Any]] = None,
    ) -> OutfitParams:
        """
        Single canonical translator from any input shape to OutfitParams.

        Layers (each overrides only the fields it provides):
            1. profile          — quiz-style structured answers (base layer)
            2. chat_extract     — gpt-4o-mini extraction from a NL query
            3. today_context    — live weather + calendar from device

        Live signals (Today) win over chat hints, which win over stored profile.
        """

        # Defaults
        gender = "genderless"
        occasion = "casual"
        style_tags: List[str] = []
        season_tags: List[str] = []
        avoid_tags: List[str] = []
        category_includes: Dict[str, List[str]] = {}
        category_excludes: Dict[str, List[str]] = {}
        budget_min = 0.0
        budget_max = 10000.0
        hero_boost = 1.0
        color_strategy = ColorStrategy.BALANCED
        boost_affiliates = True
        affiliate_boost_weight = 0.15

        # ----- Layer 1: profile (quiz logic) -----
        if profile:
            if profile.get("gender"):
                gender = _normalize_gender(profile["gender"])
                logger.info(
                    "build_params layer=profile set gender=%s", gender
                )
            else:
                logger.info(
                    "build_params layer=profile gender=MISSING — defaulting to 'genderless' (no gender filter)"
                )
            if profile.get("occasion"):
                occasion = profile["occasion"]

            if occasion in OCCASION_STYLE_OVERRIDE:
                style_tags = OCCASION_STYLE_OVERRIDE[occasion].copy()
            else:
                style = profile.get("style")
                if style and style in STYLE_TAG_MAP:
                    style_tags.extend(STYLE_TAG_MAP[style])
                setting = profile.get("setting")
                if setting and setting in SETTING_TAG_MAP:
                    style_tags.extend(SETTING_TAG_MAP[setting])

            weather_list = profile.get("weather") or []
            if isinstance(weather_list, str):
                weather_list = [weather_list]
            for weather in weather_list:
                if weather in WEATHER_TAG_MAP:
                    cfg = WEATHER_TAG_MAP[weather]
                    season_tags.extend(cfg.get("season_tags", []))
                    avoid_tags.extend(cfg.get("avoid_tags", []))

            budget_key = profile.get("budget")
            if budget_key and budget_key in BUDGET_RANGES:
                br = BUDGET_RANGES[budget_key]
                budget_min = br["min"]
                budget_max = br["max"]

            goals = profile.get("goals") or []
            if isinstance(goals, str):
                goals = [goals]
            for goal in goals:
                if goal in GOALS_STRATEGY:
                    strategy = GOALS_STRATEGY[goal]
                    hero_boost = max(hero_boost, strategy.get("hero_boost", 1.0))
                    if strategy.get("color_strategy"):
                        color_strategy = strategy["color_strategy"]
                    if strategy.get("affiliate_boost_weight", 0) > affiliate_boost_weight:
                        affiliate_boost_weight = strategy["affiliate_boost_weight"]
                    if "boost_affiliates" in strategy:
                        boost_affiliates = strategy["boost_affiliates"]

        # ----- Layer 2: chat overlay -----
        if chat_extract:
            if chat_extract.get("gender"):
                logger.info(
                    "build_params layer=chat_extract OVERRIDE gender %s -> %s",
                    gender,
                    chat_extract["gender"],
                )
                gender = _normalize_gender(chat_extract["gender"])

            if chat_extract.get("occasion"):
                occasion = chat_extract["occasion"]
                # Re-apply override if the new occasion has one (gym/beach)
                if occasion in OCCASION_STYLE_OVERRIDE:
                    style_tags = OCCASION_STYLE_OVERRIDE[occasion].copy()

            # Append (don't replace) embedding-matched style tags so chat hints
            # narrow rather than overwrite the profile's curated tag set.
            descriptors = chat_extract.get("style_descriptors") or []
            if descriptors:
                matched = self._match_tags_by_embedding(descriptors)
                style_tags.extend(matched)

            # Setting hint maps the same way as profile's setting field
            setting_hint = chat_extract.get("setting_hint")
            if setting_hint and setting_hint in SETTING_TAG_MAP:
                style_tags.extend(SETTING_TAG_MAP[setting_hint])

            # Weather hint only if profile didn't already set season tags
            weather_hint = chat_extract.get("weather_hint")
            if weather_hint and weather_hint in WEATHER_TAG_MAP and not season_tags:
                cfg = WEATHER_TAG_MAP[weather_hint]
                season_tags.extend(cfg.get("season_tags", []))
                avoid_tags.extend(cfg.get("avoid_tags", []))

            if chat_extract.get("budget_max"):
                budget_max = float(chat_extract["budget_max"])
                budget_min = 0.0

            category_includes = _sanitize_category_map(
                chat_extract.get("category_includes")
            )
            category_excludes = _sanitize_category_map(
                chat_extract.get("category_excludes")
            )

        # ----- Layer 3: today overlay (live signals win) -----
        if today_context is not None:
            # Weather: live temp wins, fallback to "moderate"
            temp_f = today_context.get("temp_f")
            if temp_f is not None:
                bucket = _temp_to_weather_bucket(float(temp_f))
            else:
                bucket = "moderate"
            cfg = WEATHER_TAG_MAP[bucket]
            season_tags = list(cfg.get("season_tags", []))
            avoid_tags = list(cfg.get("avoid_tags", []))

            # Occasion: calendar event wins, fallback to "casual"
            inferred_event_occasion = _event_to_occasion(
                today_context.get("primary_event")
            )
            if inferred_event_occasion:
                occasion = inferred_event_occasion
                if occasion in OCCASION_STYLE_OVERRIDE:
                    style_tags = OCCASION_STYLE_OVERRIDE[occasion].copy()

            # Location: LLM classification wins, fallback to "city"
            city_label = today_context.get("city_label")
            if city_label:
                setting = self._classify_location(city_label)
            else:
                setting = "city"
            if setting in SETTING_TAG_MAP:
                style_tags.extend(SETTING_TAG_MAP[setting])

        # Occasion tags: include both hyphenated and underscored forms
        occasion_key = occasion.replace("-", "_")
        occasion_tags = list(set([occasion, occasion_key]))

        return OutfitParams(
            gender=gender,
            occasion=occasion,
            budget_min=budget_min,
            budget_max=budget_max,
            style_tags=list(dict.fromkeys(style_tags)),
            occasion_tags=occasion_tags,
            season_tags=list(dict.fromkeys(season_tags)),
            avoid_tags=list(dict.fromkeys(avoid_tags)),
            category_includes=category_includes,
            category_excludes=category_excludes,
            hero_boost=hero_boost,
            color_strategy=color_strategy,
            boost_affiliates=boost_affiliates,
            affiliate_boost_weight=affiliate_boost_weight,
        )

    def _quiz_to_params(self, quiz: Dict[str, Any]) -> OutfitParams:
        """Backward-compatible alias. New code should call build_params(profile=...)."""
        return self.build_params(profile=quiz)

    # =========================================================================
    # CHAT MODE
    # =========================================================================

    def build_from_chat(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an outfit from a natural language query.

        Args:
            query: "something flowy and romantic for a vineyard date, under $200"
            user_profile: Optional saved user preferences (gender, etc.)
            context: Optional follow-up context {intent, original_query, last_outfit_summary}

        Returns:
            Complete outfit with items for each slot
        """
        logger.info("Building outfit from chat: %s", query)

        params = self.parse_chat_query(query, user_profile, context)

        # Use the resolved occasion from build_params (LLM extraction → keyword
        # fallback → profile → "casual"). Avoids the prior dual-source bug
        # where _infer_occasion_from_query overrode the LLM result.
        formula = OCCASION_FORMULAS.get(params.occasion, OCCASION_FORMULAS["casual"])
        return self._build_outfit(params, formula)

    def parse_chat_query(
        self,
        query: str,
        user_profile: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OutfitParams:
        """
        Parse a natural language query (with optional follow-up context) into
        OutfitParams via gpt-4o-mini extraction layered onto the user's profile.

        Public — also called by /api/chat/parse for previewing intent.
        """
        user_profile = user_profile or {}
        context = context or {}

        intent = context.get("intent") or "new_query"
        original_query = context.get("original_query") or ""
        last_outfit_summary = context.get("last_outfit_summary") or ""

        # If this is a refinement, surface the prior turn so the parser knows
        # which fields the user is changing vs preserving.
        if intent == "refine_existing" and (original_query or last_outfit_summary):
            user_content = (
                f"Previous request: {original_query or '(none)'}\n"
                f"Previous outfit: {last_outfit_summary or '(none)'}\n"
                f"Refinement: {query}"
            )
        else:
            user_content = query

        system_prompt = """You are a fashion query parser. Extract structured information from user queries about outfits.

Return a JSON object with these fields:
- occasion: one of [work, date, going-out, gym, casual] or null
- style_descriptors: list of style words from the query (e.g., ["flowy", "romantic", "edgy"])
- budget_max: number or null (extract from phrases like "under $200")
- weather_hint: one of [hot, moderate, cold] or null
- gender: one of [masculine, feminine, genderless] or null (only if explicitly mentioned)
- setting_hint: one of [city, country, suburbs, beach, night-out] or null

Only include fields you can confidently extract. Be concise."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            extracted = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error("Failed to parse query: %s", e)
            extracted = {}

        # Snapshot exactly what GPT-4o-mini returned, before any fallback edits,
        # so the breakdown log can show LLM output vs. fallback separately.
        gpt_extracted = dict(extracted)

        # Keyword inference is a *fallback* when the LLM didn't return an
        # occasion — never an override.
        if not extracted.get("occasion"):
            inferred = self._infer_occasion_from_query(query)
            if inferred and inferred != "casual":
                extracted["occasion"] = inferred

        category_intent = self._extract_category_intent(query)
        extracted["category_includes"] = _merge_category_maps(
            extracted.get("category_includes"), category_intent["category_includes"]
        )
        extracted["category_excludes"] = _merge_category_maps(
            extracted.get("category_excludes"), category_intent["category_excludes"]
        )

        if gpt_extracted.get("occasion"):
            occasion_source = "gpt-4o-mini"
        elif extracted.get("occasion"):
            occasion_source = "keyword fallback"
        else:
            occasion_source = "default -> casual"

        params = self.build_params(profile=user_profile, chat_extract=extracted)

        # ─── Visibility: how this chat query decomposed into the RPC request ───
        # Printed to the backend terminal on every chat build so you can see the
        # full in-between (raw GPT extraction + fallbacks + final scored params).
        logger.info(
            "\n"
            "════════════════ CHAT QUERY BREAKDOWN ════════════════\n"
            "  query             : %r\n"
            "  intent            : %s\n"
            "  user_profile in   : %s\n"
            "  ── GPT-4o-mini extracted from query ──\n"
            "  occasion          : %s\n"
            "  style_descriptors : %s\n"
            "  budget_max        : %s\n"
            "  weather_hint      : %s\n"
            "  gender            : %s\n"
            "  setting_hint      : %s\n"
            "  category_includes : %s\n"
            "  category_excludes : %s\n"
            "  occasion source   : %s\n"
            "  ── final params → Supabase RPC (%s) ──\n"
            "  gender            : %s\n"
            "  occasion          : %s\n"
            "  budget            : $%.0f – $%.0f\n"
            "  style_tags        : %s\n"
            "  occasion_tags     : %s\n"
            "  season_tags       : %s\n"
            "  avoid_tags        : %s\n"
            "  color_strategy    : %s\n"
            "═══════════════════════════════════════════════════════",
            query,
            intent,
            user_profile or {},
            gpt_extracted.get("occasion"),
            gpt_extracted.get("style_descriptors"),
            gpt_extracted.get("budget_max"),
            gpt_extracted.get("weather_hint"),
            gpt_extracted.get("gender"),
            gpt_extracted.get("setting_hint"),
            extracted.get("category_includes"),
            extracted.get("category_excludes"),
            occasion_source,
            RPC_ORDER[0][0],
            params.gender,
            params.occasion,
            params.budget_min,
            params.budget_max,
            params.style_tags,
            params.occasion_tags,
            params.season_tags,
            params.avoid_tags,
            params.color_strategy.value,
        )

        return params

    def _match_tags_by_embedding(self, descriptors: List[str]) -> List[str]:
        """
        Match style descriptors to product tags using embedding similarity.

        One embedding per descriptor (batched in a single OpenAI call) plus one
        match_tags RPC per descriptor. Returns up to ~4 tags per descriptor,
        deduped — keeps the resulting tag set tight so the SQL scorer's
        proportion-of-tags-found metric isn't diluted into noise.
        """
        descriptors = [d for d in (descriptors or []) if d and d.strip()][:5]
        if not descriptors:
            return []

        try:
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=descriptors,
            )
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return []

        matched_tags: List[str] = []
        for descriptor, emb_obj in zip(descriptors, response.data):
            try:
                result = self.supabase.rpc(
                    "match_tags",
                    {
                        "query_embedding": emb_obj.embedding,
                        "match_count": 4,
                        "match_threshold": 0.3,
                    },
                ).execute()
                for row in result.data or []:
                    tag = row.get("tag")
                    if tag and tag not in matched_tags:
                        matched_tags.append(tag)
            except Exception as e:
                logger.warning(
                    "Tag matching failed for descriptor '%s': %s", descriptor, e
                )
                continue

        logger.info("Matched tags for %s: %s", descriptors, matched_tags)
        return matched_tags

    # =========================================================================
    # TODAY MODE
    # =========================================================================

    def build_for_today(
        self,
        profile: Optional[Dict[str, Any]] = None,
        today_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build an outfit for the Today tab using live weather + calendar context
        layered on the user's stored profile.

        Args:
            profile: StyleProfile dict — base preferences
            today_context: {
                "lat": float | None,
                "lon": float | None,
                "temp_f": float | None,            # fallback: moderate
                "condition": str | None,
                "daypart": str | None,
                "city_label": str | None,          # e.g. "Brooklyn, NY" → setting inference
                "primary_event": {                 # fallback: casual occasion
                    "title": str,
                    "category": str | None,
                    "all_day": bool | None,
                } | None,
            }

        Returns:
            Complete outfit dict (same shape as build_from_quiz / build_from_chat).
        """
        logger.info(
            "Building outfit for today: profile_keys=%s context_keys=%s",
            list((profile or {}).keys()),
            list((today_context or {}).keys()),
        )
        params = self.build_params(profile=profile, today_context=today_context)

        # Masculine gym remap: neither chi nor omega has meaningful masculine
        # gym inventory. Remap to casual with athletic-casual style tags.
        if params.gender == "masculine" and params.occasion == "gym":
            logger.info("build_for_today: remapping masculine gym → casual")
            params = OutfitParams(
                gender=params.gender,
                occasion="casual",
                budget_min=params.budget_min,
                budget_max=params.budget_max,
                style_tags=["streetwear", "fitted", "athletic"],
                occasion_tags=["casual"],
                season_tags=params.season_tags,
                avoid_tags=params.avoid_tags,
                category_includes=params.category_includes,
                category_excludes=params.category_excludes,
                hero_boost=params.hero_boost,
                color_strategy=params.color_strategy,
                boost_affiliates=params.boost_affiliates,
                affiliate_boost_weight=params.affiliate_boost_weight,
            )

        formula = OCCASION_FORMULAS.get(params.occasion, OCCASION_FORMULAS["casual"])
        return self._build_outfit(params, formula)

    # =========================================================================
    # CHAT HELPERS
    # =========================================================================

    def _infer_occasion_from_query(self, query: str) -> str:
        """Simple keyword-based occasion inference."""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["work", "office", "meeting", "professional"]
        ):
            return "work"
        if any(word in query_lower for word in ["date", "romantic", "dinner"]):
            return "date"
        if any(
            word in query_lower for word in ["party", "club", "night out", "going out"]
        ):
            return "going-out"
        if any(
            word in query_lower for word in ["gym", "workout", "exercise", "athletic"]
        ):
            return "gym"

        return "casual"

    @staticmethod
    def _extract_category_intent(query: str) -> Dict[str, Dict[str, List[str]]]:
        """Infer simple slot/category directives from stylist chat text."""
        query_lower = (query or "").lower()
        includes: Dict[str, List[str]] = {}
        excludes: Dict[str, List[str]] = {}

        for slot, category_aliases in CHAT_CATEGORY_ALIASES.items():
            for category, patterns in category_aliases.items():
                mentioned = False
                negated = False
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        mentioned = True
                    negation_pattern = (
                        r"(?:\bno\b|\bwithout\b|\bavoid\b|\bskip\b|\bnot\b|"
                        r"\bdon'?t want\b|\bdo not want\b|\bnever\b)"
                        rf"\s+(?:\w+\s+){{0,3}}{pattern}"
                    )
                    if re.search(negation_pattern, query_lower):
                        negated = True
                if negated:
                    excludes.setdefault(slot, []).append(category)
                elif mentioned:
                    includes.setdefault(slot, []).append(category)

        return {
            "category_includes": _sanitize_category_map(includes),
            "category_excludes": _sanitize_category_map(excludes),
        }

    # =========================================================================
    # CORE OUTFIT BUILDING
    # =========================================================================

    def _build_outfit(
        self, params: OutfitParams, formula: OutfitFormula
    ) -> Dict[str, Any]:
        """
        Build a complete outfit by querying each slot with context.

        Maintains outfit cohesion by passing:
        - Already selected colors (avoid exact repeats)
        - Already selected textures (encourage diversity)
        """
        outfit = {
            "items": {},
            "params_used": {
                "gender": params.gender,
                "occasion": params.occasion,
                "style_tags": params.style_tags,
                "color_strategy": params.color_strategy.value,
            },
            "total_price": 0.0,
        }

        selected_colors: List[str] = []
        selected_textures: List[str] = []

        for slot_config in formula.slots:
            logger.info(
                "Building slot: %s (formality=%d, hero=%s) gender=%s occasion=%s",
                slot_config.slot,
                slot_config.formality,
                slot_config.is_hero,
                params.gender,
                params.occasion,
            )
            item: Optional[Dict[str, Any]] = None
            for attempt_index, attempt_config in enumerate(
                self._slot_category_attempts(params, slot_config), start=1
            ):
                category_filters = self._category_filters(attempt_config)
                logger.info(
                    "  category attempt %d slot=%s include=%s exclude=%s",
                    attempt_index,
                    slot_config.slot,
                    category_filters["include"],
                    category_filters["exclude"],
                )

                rpc_params = {
                    "p_slot": slot_config.slot,
                    "p_gender": _rpc_gender(params.gender),
                    "p_min_price": params.budget_min,
                    "p_max_price": params.budget_max,
                    "p_style_tags": params.style_tags,
                    "p_occasion_tags": params.occasion_tags,
                    "p_season_tags": params.season_tags,
                    "p_avoid_tags": params.avoid_tags,
                    "p_target_formality": slot_config.formality,
                    "p_hero_slot": slot_config.is_hero,
                    "p_existing_colors": selected_colors,
                    "p_existing_textures": selected_textures,
                    "p_color_strategy": params.color_strategy.value,
                    "p_boost_affiliates": params.boost_affiliates,
                    "p_affiliate_boost_weight": params.affiliate_boost_weight,
                    "p_include_categories": category_filters["include"],
                    "p_exclude_categories": category_filters["exclude"],
                    "p_n": 1,
                }

                item = self._query_slot_with_fallback(
                    rpc_params, slot_config.slot, slot_config=attempt_config
                )
                if item:
                    break

            if item:
                outfit["items"][slot_config.slot] = item
                if item.get("product_color"):
                    selected_colors.append(item["product_color"])
                if item.get("product_texture"):
                    selected_textures.append(item["product_texture"])
                if item.get("product_price_amount"):
                    outfit["total_price"] += float(item["product_price_amount"])
                logger.info(
                    "  ✓ Selected: %s...",
                    item.get("product_title", "Unknown")[:50],
                )
            else:
                logger.warning("  ⚠ No products found for slot: %s", slot_config.slot)
                outfit["items"][slot_config.slot] = None

        outfit["total_price"] = round(outfit["total_price"], 2)
        logger.info("Outfit complete. Total: $%s", outfit["total_price"])

        return outfit

    # =========================================================================
    # SWAP FUNCTIONALITY
    # =========================================================================

    def swap_item(
        self,
        current_outfit: Dict[str, Any],
        slot_to_swap: str,
        exclude_product_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Swap a single item in an existing outfit while maintaining cohesion.

        Args:
            current_outfit: The current outfit dict
            slot_to_swap: Which slot to replace (e.g., "top")
            exclude_product_ids: Product IDs to exclude (e.g., current item)

        Returns:
            New item for the slot, or None if no alternatives found
        """
        params_used = current_outfit.get("params_used", {})
        items = current_outfit.get("items", {})

        # Get colors/textures from OTHER items (not the one being swapped)
        existing_colors = []
        existing_textures = []
        for slot, item in items.items():
            if slot != slot_to_swap and item:
                if item.get("product_color"):
                    existing_colors.append(item["product_color"])
                if item.get("product_texture"):
                    existing_textures.append(item["product_texture"])

        # Determine formality from occasion
        occasion = params_used.get("occasion", "casual")
        formula = OCCASION_FORMULAS.get(occasion, OCCASION_FORMULAS["casual"])

        slot_config = None
        for sc in formula.slots:
            if sc.slot == slot_to_swap:
                slot_config = sc
                break

        if not slot_config:
            slot_config = SlotConfig(slot_to_swap, formality=3, is_hero=False)

        occasion_key = occasion.replace("-", "_")
        occasion_tags = list(set([occasion, occasion_key]))
        category_filters = self._category_filters(slot_config)

        rpc_params = {
            "p_slot": slot_to_swap,
            "p_gender": _rpc_gender(params_used.get("gender", "genderless")),
            "p_style_tags": params_used.get("style_tags", []),
            "p_occasion_tags": occasion_tags,
            "p_target_formality": slot_config.formality,
            "p_hero_slot": slot_config.is_hero,
            "p_existing_colors": existing_colors,
            "p_existing_textures": existing_textures,
            "p_color_strategy": params_used.get("color_strategy", "balanced"),
            "p_boost_affiliates": params_used.get("boost_affiliates", True),
            "p_affiliate_boost_weight": params_used.get(
                "affiliate_boost_weight", 0.15
            ),
            "p_include_categories": category_filters["include"],
            "p_exclude_categories": category_filters["exclude"],
        }

        all_candidates = self._query_slot_candidates_with_fallback(
            rpc_params, slot_to_swap, desired_n=5, slot_config=slot_config,
        )

        exclude_ids = set(exclude_product_ids or [])
        candidates = [
            item for item in all_candidates
            if item.get("product_id") not in exclude_ids
        ]

        if candidates:
            return candidates[0]
        elif all_candidates:
            return all_candidates[0]
        else:
            return None


# Convenience function for API usage
def create_outfit_builder() -> OutfitBuilder:
    """Factory function to create OutfitBuilder instance."""
    return OutfitBuilder()
