"""Unit tests for OutfitBuilder.build_params — the canonical params translator.

These tests exercise pure logic: profile → params, chat overlay, today overlay.
The supabase + openai clients are constructed but never called for the
profile-only and today-overlay paths. The chat-overlay test that uses
style_descriptors monkeypatches _match_tags_by_embedding to skip the network.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy env so client constructors don't blow up at import time
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from outfit_builder import (  # noqa: E402
    ColorStrategy,
    OutfitBuilder,
    OutfitParams,
    _event_to_occasion,
    _temp_to_weather_bucket,
)


@pytest.fixture(scope="module")
def builder():
    return OutfitBuilder()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def test_temp_bucket_boundaries():
    assert _temp_to_weather_bucket(40) == "cold"
    assert _temp_to_weather_bucket(49.9) == "cold"
    assert _temp_to_weather_bucket(50) == "moderate"
    assert _temp_to_weather_bucket(74.9) == "moderate"
    assert _temp_to_weather_bucket(75) == "hot"
    assert _temp_to_weather_bucket(95) == "hot"


def test_event_to_occasion_keyword_match():
    assert _event_to_occasion({"title": "Standup with team"}) == "work"
    assert _event_to_occasion({"title": "Dinner with Anna"}) == "date"
    assert _event_to_occasion({"title": "Yoga class"}) == "gym"
    assert _event_to_occasion({"title": "Wedding reception"}) == "going-out"
    assert _event_to_occasion({"title": "Coffee"}) is None
    assert _event_to_occasion({"title": ""}) is None
    assert _event_to_occasion(None) is None


def test_event_to_occasion_category_wins_over_title():
    assert (
        _event_to_occasion({"title": "Random thing", "category": "work"}) == "work"
    )


# -----------------------------------------------------------------------------
# Profile layer (quiz parity)
# -----------------------------------------------------------------------------


def test_profile_only_minimalist_date_balanced(builder):
    params = builder.build_params(
        profile={
            "gender": "feminine",
            "occasion": "date",
            "weather": ["moderate"],
            "setting": "city",
            "goals": ["serve-looks"],
            "style": "minimalist",
            "budget": "$$",
        }
    )
    assert isinstance(params, OutfitParams)
    assert params.gender == "feminine"
    assert params.occasion == "date"
    assert params.occasion_tags == ["date"]
    assert params.budget_min == 0
    assert params.budget_max == 150
    assert params.color_strategy == ColorStrategy.BOLD
    assert params.hero_boost == 1.5
    # minimalist + city tags both present
    assert "minimalist" in params.style_tags
    assert "urban" in params.style_tags
    # moderate weather tags
    assert any(t in params.season_tags for t in ("spring", "fall", "all_season"))


def test_profile_only_gym_uses_override(builder):
    params = builder.build_params(
        profile={
            "gender": "masculine",
            "occasion": "gym",
            "weather": ["moderate"],
            "setting": "city",
            "style": "minimalist",
            "budget": "$$",
        }
    )
    assert params.occasion == "gym"
    # gym override replaces style/setting tags
    assert set(params.style_tags) == {"gym", "activewear", "athletic", "workout"}
    # The keyword-overridden tag list ignores 'minimalist'/'urban'
    assert "minimalist" not in params.style_tags


def test_profile_only_cold_weather_brings_avoid_tags(builder):
    params = builder.build_params(
        profile={
            "gender": "feminine",
            "occasion": "casual",
            "weather": ["cold"],
            "style": "classic",
            "budget": "$$",
        }
    )
    assert "winter" in params.season_tags or "fall" in params.season_tags
    assert any(t in params.avoid_tags for t in ("sleeveless", "linen", "summer"))


def test_profile_only_dash_occasion_normalized(builder):
    params = builder.build_params(
        profile={
            "gender": "unisex",
            "occasion": "going-out",
            "style": "minimalist",
            "budget": "$$",
        }
    )
    # dashes become underscores in the SQL-bound occasion_tags array
    assert params.occasion_tags == ["going_out"]


def test_profile_empty_yields_safe_defaults(builder):
    params = builder.build_params(profile={})
    assert params.gender == "unisex"
    assert params.occasion == "casual"
    assert params.color_strategy == ColorStrategy.BALANCED
    assert params.budget_min == 0
    assert params.budget_max == 10000
    assert params.style_tags == []
    assert params.season_tags == []
    assert params.avoid_tags == []


# -----------------------------------------------------------------------------
# Chat overlay
# -----------------------------------------------------------------------------


def test_chat_overlay_overrides_occasion(builder):
    params = builder.build_params(
        profile={
            "gender": "feminine",
            "occasion": "casual",
            "style": "classic",
            "budget": "$$",
        },
        chat_extract={"occasion": "date"},
    )
    assert params.occasion == "date"
    # Profile budget preserved when chat doesn't extract one
    assert params.budget_max == 150


def test_chat_overlay_budget_overrides_profile(builder):
    params = builder.build_params(
        profile={"budget": "$$$$"},  # 10000 max
        chat_extract={"budget_max": 200},
    )
    assert params.budget_max == 200


def test_chat_overlay_appends_descriptor_tags(builder, monkeypatch):
    monkeypatch.setattr(
        OutfitBuilder,
        "_match_tags_by_embedding",
        lambda self, descriptors: ["bohemian", "relaxed"],
    )
    params = builder.build_params(
        profile={"style": "minimalist"},
        chat_extract={"style_descriptors": ["flowy", "romantic"]},
    )
    # Profile's minimalist tags are preserved AND chat descriptors append
    assert "minimalist" in params.style_tags
    assert "bohemian" in params.style_tags
    assert "relaxed" in params.style_tags


def test_chat_overlay_weather_hint_only_when_profile_silent(builder):
    # Profile already has weather → chat hint ignored
    params = builder.build_params(
        profile={"weather": ["moderate"]},
        chat_extract={"weather_hint": "cold"},
    )
    # Should still be moderate-derived, not cold
    assert "winter" not in params.season_tags

    # Profile silent → chat hint takes effect
    params2 = builder.build_params(
        profile={},
        chat_extract={"weather_hint": "cold"},
    )
    assert any(t in params2.season_tags for t in ("winter", "fall"))


def test_chat_overlay_new_occasion_triggers_override_tags(builder):
    # Profile is minimalist/casual; chat says "gym" → tags get replaced with override
    params = builder.build_params(
        profile={"style": "minimalist", "setting": "city", "occasion": "casual"},
        chat_extract={"occasion": "gym"},
    )
    assert params.occasion == "gym"
    assert "gym" in params.style_tags
    assert "activewear" in params.style_tags
    assert "minimalist" not in params.style_tags


# -----------------------------------------------------------------------------
# Today overlay (live signals win)
# -----------------------------------------------------------------------------


def test_today_overlay_temp_replaces_weather(builder):
    # Profile says "hot" but live device sees 40F
    params = builder.build_params(
        profile={"weather": ["hot"], "occasion": "casual"},
        today_context={"temp_f": 40},
    )
    assert any(t in params.season_tags for t in ("winter", "fall"))
    assert any(
        t in params.avoid_tags for t in ("sleeveless", "linen", "summer")
    )


def test_today_overlay_event_overrides_profile_occasion(builder):
    params = builder.build_params(
        profile={"occasion": "casual", "style": "minimalist"},
        today_context={
            "temp_f": 35,
            "primary_event": {"title": "Standup with team"},
        },
    )
    assert params.occasion == "work"
    assert params.occasion_tags == ["work"]


def test_today_overlay_no_event_keeps_profile_occasion(builder):
    params = builder.build_params(
        profile={"occasion": "date"},
        today_context={"temp_f": 70},  # no primary_event
    )
    assert params.occasion == "date"


def test_today_overlay_gym_event_applies_override_tags(builder):
    params = builder.build_params(
        profile={"style": "classic", "occasion": "casual"},
        today_context={
            "temp_f": 65,
            "primary_event": {"title": "Yoga 6pm"},
        },
    )
    assert params.occasion == "gym"
    assert "gym" in params.style_tags
    assert "activewear" in params.style_tags


def test_today_overlay_with_no_temp_or_event_is_a_noop(builder):
    profile_only = builder.build_params(
        profile={"occasion": "date", "weather": ["moderate"], "style": "classic"}
    )
    layered = builder.build_params(
        profile={"occasion": "date", "weather": ["moderate"], "style": "classic"},
        today_context={},
    )
    assert profile_only.occasion == layered.occasion
    assert set(profile_only.style_tags) == set(layered.style_tags)
    assert set(profile_only.season_tags) == set(layered.season_tags)


# -----------------------------------------------------------------------------
# Layering — profile + chat + today simultaneously
# -----------------------------------------------------------------------------


def test_full_stack_today_wins_over_chat_wins_over_profile(builder, monkeypatch):
    monkeypatch.setattr(
        OutfitBuilder,
        "_match_tags_by_embedding",
        lambda self, descriptors: ["bohemian"],
    )
    params = builder.build_params(
        profile={
            "gender": "feminine",
            "occasion": "casual",
            "weather": ["hot"],
            "style": "minimalist",
            "budget": "$$",
        },
        chat_extract={
            "occasion": "going-out",
            "style_descriptors": ["edgy"],
            "budget_max": 250,
        },
        today_context={
            "temp_f": 38,  # cold — should override profile's "hot"
            "primary_event": {"title": "Gym session"},  # should override chat's going-out
        },
    )
    # Today wins
    assert params.occasion == "gym"
    # Live cold weather wins
    assert any(t in params.season_tags for t in ("winter", "fall"))
    # Today's gym occasion forced override tags
    assert "gym" in params.style_tags
    # Chat budget still applies (today doesn't touch budget)
    assert params.budget_max == 250
