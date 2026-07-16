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
    WEEKDAY_OCCASION_POOL,
    WEEKEND_OCCASION_POOL,
    _event_to_occasion,
    _rotating_fallback_occasion,
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
    assert "date" in params.occasion_tags
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


def test_profile_only_dash_occasion_includes_both_forms(builder):
    params = builder.build_params(
        profile={
            "gender": "unisex",
            "occasion": "going-out",
            "style": "minimalist",
            "budget": "$$",
        }
    )
    assert params.gender == "genderless"
    # Both hyphenated and underscored forms for tag matching across chi + omega
    assert "going_out" in params.occasion_tags
    assert "going-out" in params.occasion_tags


def test_profile_empty_yields_safe_defaults(builder):
    params = builder.build_params(profile={})
    assert params.gender == "genderless"
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
    assert "work" in params.occasion_tags


def test_today_overlay_no_event_rotates_occasion(builder):
    # No calendar event → day-aware rotation, NOT the stored quiz occasion.
    # 2026-07-15 is a Wednesday; "date" only joins the weekend pool.
    params = builder.build_params(
        profile={"occasion": "date"},
        today_context={"temp_f": 70, "local_date": "2026-07-15"},
    )
    assert params.occasion in WEEKDAY_OCCASION_POOL


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


def test_today_overlay_empty_context_applies_fallback_defaults(builder):
    """Empty today_context triggers fallbacks: moderate weather, city setting."""
    params = builder.build_params(
        profile={"occasion": "date", "style": "classic"},
        today_context={},
    )
    # No local_date → rotation uses server date; just assert it's a valid pool member
    assert params.occasion in set(
        WEEKDAY_OCCASION_POOL + WEEKEND_OCCASION_POOL + ["date"]
    )
    # Moderate weather fallback
    assert any(t in params.season_tags for t in ("spring", "fall", "all_season"))
    # City setting fallback (urban, polished, etc.)
    assert "urban" in params.style_tags


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


# -----------------------------------------------------------------------------
# Location classification
# -----------------------------------------------------------------------------


def test_classify_location_uses_cache(builder, monkeypatch):
    call_count = {"n": 0}
    original_create = builder.openai.chat.completions.create

    def mock_create(**kwargs):
        call_count["n"] += 1

        class _Choice:
            class _Msg:
                content = "city"
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    monkeypatch.setattr(builder.openai.chat.completions, "create", mock_create)
    OutfitBuilder._location_cache.clear()

    result1 = builder._classify_location("Brooklyn, NY")
    result2 = builder._classify_location("Brooklyn, NY")
    assert result1 == "city"
    assert result2 == "city"
    assert call_count["n"] == 1  # cached on second call


def test_classify_location_fallback_on_error(builder, monkeypatch):
    monkeypatch.setattr(
        builder.openai.chat.completions,
        "create",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    OutfitBuilder._location_cache.clear()
    assert builder._classify_location("Unknown Place") == "city"


def test_today_context_city_label_adds_setting_tags(builder, monkeypatch):
    monkeypatch.setattr(
        OutfitBuilder, "_classify_location", lambda self, label: "beach"
    )
    params = builder.build_params(
        profile={"style": "classic"},
        today_context={"city_label": "Malibu, CA", "temp_f": 85},
    )
    assert "summer" in params.style_tags
    assert "relaxed" in params.style_tags


def test_today_context_no_city_label_defaults_to_city(builder):
    params = builder.build_params(
        profile={"style": "classic"},
        today_context={"temp_f": 70},
    )
    assert "urban" in params.style_tags


# -----------------------------------------------------------------------------
# Today fallback defaults
# -----------------------------------------------------------------------------


def test_today_no_temp_defaults_to_moderate(builder):
    params = builder.build_params(
        profile={},
        today_context={},
    )
    assert any(t in params.season_tags for t in ("spring", "fall", "all_season"))
    assert params.avoid_tags == []  # moderate has no avoid tags


# -----------------------------------------------------------------------------
# No-event occasion rotation
# -----------------------------------------------------------------------------


def test_rotation_is_deterministic():
    a = _rotating_fallback_occasion("date", "feminine", "2026-07-15", "user-1")
    b = _rotating_fallback_occasion("date", "feminine", "2026-07-15", "user-1")
    assert a == b


def test_rotation_consecutive_days_differ():
    # 2026-07-13/14 are Mon/Tue — same weekday pool, so rotation must alternate.
    mon = _rotating_fallback_occasion("work", "feminine", "2026-07-13", "user-1")
    tue = _rotating_fallback_occasion("work", "feminine", "2026-07-14", "user-1")
    assert mon != tue
    assert {mon, tue} <= set(WEEKDAY_OCCASION_POOL)


def test_rotation_weekend_uses_weekend_pool():
    # 2026-07-18/19 are Sat/Sun; profile "date" joins the weekend pool.
    for d in ("2026-07-18", "2026-07-19"):
        occ = _rotating_fallback_occasion("date", "feminine", d, "user-1")
        assert occ in set(WEEKEND_OCCASION_POOL + ["date"])


def test_rotation_gym_profile_joins_pool_except_masculine():
    dates = [f"2026-07-{day:02d}" for day in range(13, 27)]  # two full weeks
    feminine = {
        _rotating_fallback_occasion("gym", "feminine", d, "user-1") for d in dates
    }
    masculine = {
        _rotating_fallback_occasion("gym", "masculine", d, "user-1") for d in dates
    }
    assert "gym" in feminine  # gym cycles in for feminine users
    assert "gym" not in masculine  # never steered into thin masculine gym inventory


def test_rotation_bad_local_date_does_not_raise():
    occ = _rotating_fallback_occasion("casual", "feminine", "not-a-date", "user-1")
    assert occ in set(WEEKDAY_OCCASION_POOL + WEEKEND_OCCASION_POOL)
    occ = _rotating_fallback_occasion(None, "genderless", None, None)
    assert occ in set(WEEKDAY_OCCASION_POOL + WEEKEND_OCCASION_POOL)


def test_today_profile_fallback_event_is_ignored(builder):
    # Legacy iOS builds synthesized a fake event from the quiz occasion —
    # the backend must strip it and rotate instead. Weekday date, so a
    # forced "date" occasion would be a bug.
    params = builder.build_params(
        profile={"occasion": "date"},
        today_context={
            "temp_f": 70,
            "local_date": "2026-07-15",  # Wednesday
            "primary_event": {"title": "Date Night", "category": "date"},
            "sources": {"event": "profile_fallback"},
        },
    )
    assert params.occasion in WEEKDAY_OCCASION_POOL


def test_today_live_event_still_wins(builder):
    params = builder.build_params(
        profile={"occasion": "date"},
        today_context={
            "temp_f": 70,
            "local_date": "2026-07-18",  # Saturday — rotation would avoid "work"
            "primary_event": {"title": "Standup with team"},
            "sources": {"event": "live"},
        },
    )
    assert params.occasion == "work"


# -----------------------------------------------------------------------------
# Masculine gym remap
# -----------------------------------------------------------------------------


def test_build_for_today_masculine_gym_remaps_to_casual(builder, monkeypatch):
    monkeypatch.setattr(
        OutfitBuilder,
        "_build_outfit",
        lambda self, params, formula, variance_seed=None: {
            "params": params,
            "formula": formula,
        },
    )
    result = builder.build_for_today(
        profile={"gender": "masculine", "occasion": "gym", "budget": "$$"},
        today_context={"temp_f": 70, "primary_event": {"title": "Gym"}},
    )
    params = result["params"]
    assert params.occasion == "casual"
    assert params.gender == "masculine"
    assert "streetwear" in params.style_tags
    assert "athletic" in params.style_tags


def test_build_for_today_feminine_gym_not_remapped(builder, monkeypatch):
    monkeypatch.setattr(
        OutfitBuilder,
        "_build_outfit",
        lambda self, params, formula, variance_seed=None: {
            "params": params,
            "formula": formula,
        },
    )
    result = builder.build_for_today(
        profile={"gender": "feminine", "occasion": "gym", "budget": "$$"},
        today_context={"temp_f": 70, "primary_event": {"title": "Gym"}},
    )
    params = result["params"]
    assert params.occasion == "gym"


# -----------------------------------------------------------------------------
# Occasion tags: both forms
# -----------------------------------------------------------------------------


def test_occasion_tags_non_hyphenated_dedupes(builder):
    params = builder.build_params(profile={"occasion": "work"})
    assert params.occasion_tags == ["work"]


def test_occasion_tags_going_out_has_both_forms(builder):
    params = builder.build_params(profile={"occasion": "going-out"})
    assert set(params.occasion_tags) == {"going-out", "going_out"}
