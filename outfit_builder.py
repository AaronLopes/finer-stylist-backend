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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI
import supabase

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorStrategy(str, Enum):
    BALANCED = "balanced"
    BOLD = "bold"
    NEUTRAL = "neutral"
    MONOCHROME = "monochrome"


@dataclass
class OutfitParams:
    """Unified parameters for outfit building - used by both quiz and chat modes."""

    # Core filters
    gender: str  # masculine, feminine, unisex, non-binary
    occasion: str  # date, work, casual, going-out, gym
    budget_min: float = 0
    budget_max: float = 10000

    # Tag-based matching
    style_tags: List[str] = field(default_factory=list)
    occasion_tags: List[str] = field(default_factory=list)
    season_tags: List[str] = field(default_factory=list)
    avoid_tags: List[str] = field(default_factory=list)

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


class OutfitBuilder:
    """
    Main outfit building service.

    Supports two input modes:
    1. Quiz mode: Structured answers from quiz flow
    2. Chat mode: Natural language queries
    """

    def __init__(self):
        self.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OutfitBuilder initialized")

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

    def _quiz_to_params(self, quiz: Dict[str, Any]) -> OutfitParams:
        """Convert quiz answers to OutfitParams."""

        occasion = quiz.get("occasion", "casual")

        # Check if this occasion has a style override (gym, beach, etc.)
        # If so, use occasion-specific tags instead of generic style/setting tags
        if occasion in OCCASION_STYLE_OVERRIDE:
            # For gym/beach: use specific activewear/swimwear tags
            style_tags = OCCASION_STYLE_OVERRIDE[occasion].copy()
            logger.info("Using occasion override tags for %s: %s", occasion, style_tags)
        else:
            # Normal flow: collect style tags from quiz
            style_tags = []
            style = quiz.get("style")
            if style and style in STYLE_TAG_MAP:
                style_tags.extend(STYLE_TAG_MAP[style])

            # Add setting tags
            setting = quiz.get("setting")
            if setting and setting in SETTING_TAG_MAP:
                style_tags.extend(SETTING_TAG_MAP[setting])

        # Collect season/weather tags
        season_tags = []
        avoid_tags = []
        weather_list = quiz.get("weather", ["moderate"])
        if isinstance(weather_list, str):
            weather_list = [weather_list]

        for weather in weather_list:
            if weather in WEATHER_TAG_MAP:
                weather_config = WEATHER_TAG_MAP[weather]
                season_tags.extend(weather_config.get("season_tags", []))
                avoid_tags.extend(weather_config.get("avoid_tags", []))

        # Occasion tags
        occasion_tags = [occasion.replace("-", "_")]

        # Budget
        budget_key = quiz.get("budget", "$$")
        budget_range = BUDGET_RANGES.get(budget_key, BUDGET_RANGES["$$"])

        # Goals → strategy
        goals = quiz.get("goals", [])
        if isinstance(goals, str):
            goals = [goals]

        hero_boost = 1.0
        color_strategy = ColorStrategy.BALANCED
        boost_affiliates = True
        affiliate_boost_weight = 0.15  # Default boost

        for goal in goals:
            if goal in GOALS_STRATEGY:
                strategy = GOALS_STRATEGY[goal]
                hero_boost = max(hero_boost, strategy.get("hero_boost", 1.0))
                if strategy.get("color_strategy"):
                    color_strategy = strategy["color_strategy"]
                # Use highest affiliate boost if multiple goals
                if strategy.get("affiliate_boost_weight", 0) > affiliate_boost_weight:
                    affiliate_boost_weight = strategy["affiliate_boost_weight"]
                if "boost_affiliates" in strategy:
                    boost_affiliates = strategy["boost_affiliates"]

        return OutfitParams(
            gender=quiz.get("gender", "unisex"),
            occasion=occasion,
            budget_min=budget_range["min"],
            budget_max=budget_range["max"],
            style_tags=list(set(style_tags)),
            occasion_tags=occasion_tags,
            season_tags=list(set(season_tags)),
            avoid_tags=list(set(avoid_tags)),
            hero_boost=hero_boost,
            color_strategy=color_strategy,
            boost_affiliates=boost_affiliates,
            affiliate_boost_weight=affiliate_boost_weight,
        )

    # =========================================================================
    # CHAT MODE
    # =========================================================================

    def build_from_chat(
        self, query: str, user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build an outfit from a natural language query.

        Args:
            query: "something flowy and romantic for a vineyard date, under $200"
            user_profile: Optional saved user preferences (gender, etc.)

        Returns:
            Complete outfit with items for each slot
        """
        logger.info("Building outfit from chat: %s", query)

        # Parse query into structured params
        params = self.parse_chat_query(query, user_profile)

        # Infer occasion from query or default
        occasion = self._infer_occasion_from_query(query)
        formula = OCCASION_FORMULAS.get(occasion, OCCASION_FORMULAS["casual"])

        # Build outfit
        return self._build_outfit(params, formula)

    def parse_chat_query(
        self, query: str, user_profile: Optional[Dict[str, Any]] = None
    ) -> OutfitParams:
        """
        Parse natural language query into OutfitParams using GPT-4o-mini.

        This is a public method that can be called directly for the /api/chat/parse endpoint.
        """
        user_profile = user_profile or {}

        # Use GPT-4o-mini to extract structured intent
        system_prompt = """You are a fashion query parser. Extract structured information from user queries about outfits.

Return a JSON object with these fields:
- occasion: one of [work, date, going-out, gym, casual] or null
- style_descriptors: list of style words from the query (e.g., ["flowy", "romantic", "edgy"])
- budget_max: number or null (extract from phrases like "under $200")
- weather_hint: one of [hot, moderate, cold] or null
- gender: one of [masculine, feminine, unisex] or null (only if explicitly mentioned)
- setting_hint: one of [city, country, suburbs, beach, night-out] or null

Only include fields you can confidently extract. Be concise."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
            )

            extracted = json.loads(response.choices[0].message.content)
            logger.info("Extracted from query: %s", extracted)

        except Exception as e:
            logger.error("Failed to parse query: %s", e)
            extracted = {}

        # Map style descriptors to tags via embedding similarity
        style_tags = []
        style_descriptors = extracted.get("style_descriptors", [])
        if style_descriptors:
            style_tags = self._match_tags_by_embedding(style_descriptors)

        # Add setting tags if hint provided
        setting_hint = extracted.get("setting_hint")
        if setting_hint and setting_hint in SETTING_TAG_MAP:
            style_tags.extend(SETTING_TAG_MAP[setting_hint])

        # Weather → season tags
        season_tags = []
        avoid_tags = []
        weather_hint = extracted.get("weather_hint")
        if weather_hint and weather_hint in WEATHER_TAG_MAP:
            weather_config = WEATHER_TAG_MAP[weather_hint]
            season_tags = weather_config.get("season_tags", [])
            avoid_tags = weather_config.get("avoid_tags", [])

        # Budget
        budget_max = extracted.get("budget_max", 300) or 300

        # Gender from profile or extraction
        gender = user_profile.get("gender") or extracted.get("gender") or "unisex"

        # Occasion
        occasion = extracted.get("occasion") or "casual"

        return OutfitParams(
            gender=gender,
            occasion=occasion,
            budget_min=0,
            budget_max=budget_max,
            style_tags=list(set(style_tags)),
            occasion_tags=[occasion.replace("-", "_")],
            season_tags=season_tags,
            avoid_tags=avoid_tags,
            hero_boost=1.0,
            color_strategy=ColorStrategy.BALANCED,
        )

    def _match_tags_by_embedding(self, descriptors: List[str]) -> List[str]:
        """
        Match style descriptors to product tags using embedding similarity.

        Args:
            descriptors: ["flowy", "romantic", "edgy"]

        Returns:
            Matching tags from vocabulary: ["bohemian", "relaxed", "fitted"]
        """
        if not descriptors:
            return []

        try:
            # Generate embedding for combined descriptors
            combined_text = " ".join(descriptors)
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text,
            )
            query_embedding = response.data[0].embedding

            # Query Supabase for matching tags
            result = self.supabase.rpc(
                "match_tags",
                {
                    "query_embedding": query_embedding,
                    "match_count": 8,
                    "match_threshold": 0.3,
                },
            ).execute()

            matched_tags = [row["tag"] for row in result.data]
            logger.info("Matched tags for '%s': %s", combined_text, matched_tags)
            return matched_tags

        except Exception as e:
            logger.error("Tag matching failed: %s", e)
            return []

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
                "Building slot: %s (formality=%d, hero=%s)",
                slot_config.slot,
                slot_config.formality,
                slot_config.is_hero,
            )

            # Query ff_build_outfit_v3 for this slot (with affiliate boosting)
            try:
                result = self.supabase.rpc(
                    "ff_build_outfit_v3",
                    {
                        "p_slot": slot_config.slot,
                        "p_gender": params.gender,
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
                        "p_n": 1,  # Get top candidate
                    },
                ).execute()

                if result.data and len(result.data) > 0:
                    item = result.data[0]
                    outfit["items"][slot_config.slot] = item

                    # Track selected attributes for next slot
                    if item.get("product_color"):
                        selected_colors.append(item["product_color"])
                    if item.get("product_texture"):
                        selected_textures.append(item["product_texture"])

                    # Accumulate price
                    if item.get("product_price_amount"):
                        outfit["total_price"] += float(item["product_price_amount"])

                    logger.info(
                        "  ✓ Selected: %s...",
                        item.get("product_title", "Unknown")[:50],
                    )
                else:
                    logger.warning(
                        "  ⚠ No products found for slot: %s", slot_config.slot
                    )
                    outfit["items"][slot_config.slot] = None

            except Exception as e:
                logger.error("  ✗ Error querying slot %s: %s", slot_config.slot, e)
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

        # Query for alternatives
        try:
            result = self.supabase.rpc(
                "ff_build_outfit_v3",
                {
                    "p_slot": slot_to_swap,
                    "p_gender": params_used.get("gender", "unisex"),
                    "p_style_tags": params_used.get("style_tags", []),
                    "p_occasion_tags": [occasion.replace("-", "_")],
                    "p_target_formality": slot_config.formality,
                    "p_hero_slot": slot_config.is_hero,
                    "p_existing_colors": existing_colors,
                    "p_existing_textures": existing_textures,
                    "p_color_strategy": params_used.get("color_strategy", "balanced"),
                    "p_boost_affiliates": params_used.get("boost_affiliates", True),
                    "p_affiliate_boost_weight": params_used.get(
                        "affiliate_boost_weight", 0.15
                    ),
                    "p_n": 5,  # Get multiple to filter
                },
            ).execute()

            # Filter out excluded products
            exclude_ids = set(exclude_product_ids or [])
            candidates = [
                item
                for item in result.data
                if item.get("product_id") not in exclude_ids
            ]

            if candidates:
                return candidates[0]
            elif result.data:
                return result.data[0]  # Fallback to any result
            else:
                return None

        except Exception as e:
            logger.error("Swap failed: %s", e)
            return None


# Convenience function for API usage
def create_outfit_builder() -> OutfitBuilder:
    """Factory function to create OutfitBuilder instance."""
    return OutfitBuilder()
