"""Tests for category-guided outfit construction."""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from outfit_builder import OutfitBuilder, OutfitFormula, OutfitParams, SlotConfig  # noqa: E402


@pytest.fixture
def builder():
    return OutfitBuilder()


def _item(product_id="p1", category="shirt", slot="top"):
    return {
        "product_id": product_id,
        "product_title": f"{category} test item",
        "product_category": category,
        "product_slot": slot,
        "product_price_amount": 42,
        "product_color": "black",
        "product_texture": "woven",
    }


def _empty_rpc(calls):
    def rpc(name, params):
        calls.append((name, dict(params)))
        mock = MagicMock()
        mock.execute.return_value = MagicMock(data=[])
        return mock

    return rpc


@pytest.mark.parametrize(
    ("occasion", "slot", "expected_include"),
    [
        ("work", "bottom", ["pants"]),
        ("date", "top", ["blouse", "shirt", "sweater", "bodysuit"]),
        ("going-out", "top", ["blouse", "bodysuit", "tank_top", "shirt"]),
        ("gym", "footwear", ["sneaker"]),
        ("casual", "top", ["t_shirt", "shirt", "sweater", "sweatshirt_hoodie"]),
    ],
)
def test_occasion_first_category_tier(builder, monkeypatch, occasion, slot, expected_include):
    calls = []
    monkeypatch.setattr(builder.supabase, "rpc", _empty_rpc(calls))

    builder._build_outfit(
        OutfitParams(gender="feminine", occasion=occasion),
        OutfitFormula(
            base_formality=2,
            slots=[SlotConfig(slot, formality=2)],
            balance_rule="match",
        ),
    )

    omega_call = next(params for name, params in calls if name == "ff_build_outfit_v3")
    assert omega_call["p_include_categories"] == expected_include


def test_category_fallback_relaxes_to_next_occasion_tier(builder, monkeypatch):
    calls = []

    def rpc(name, params):
        calls.append((name, dict(params)))
        mock = MagicMock()
        data = []
        if (
            name == "ff_build_outfit_v3"
            and params.get("p_include_categories") == ["pants", "skirt_skort", "jeans"]
        ):
            data = [_item("skirt_1", "skirt_skort", "bottom")]
        mock.execute.return_value = MagicMock(data=data)
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)

    outfit = builder._build_outfit(
        OutfitParams(gender="feminine", occasion="work"),
        OutfitFormula(
            base_formality=4,
            slots=[SlotConfig("bottom", formality=4)],
            balance_rule="match",
        ),
    )

    omega_includes = [
        params.get("p_include_categories")
        for name, params in calls
        if name == "ff_build_outfit_v3"
    ]
    assert omega_includes[:2] == [["pants"], ["pants", "skirt_skort", "jeans"]]
    assert outfit["items"]["bottom"]["product_id"] == "skirt_1"


def test_category_fallback_reaches_broad_slot(builder, monkeypatch):
    calls = []

    def rpc(name, params):
        calls.append((name, dict(params)))
        mock = MagicMock()
        data = []
        if name == "ff_build_outfit_v3" and params.get("p_include_categories") is None:
            data = [_item("mystery_top", None, "top")]
        mock.execute.return_value = MagicMock(data=data)
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)

    outfit = builder._build_outfit(
        OutfitParams(gender="feminine", occasion="casual"),
        OutfitFormula(
            base_formality=2,
            slots=[SlotConfig("top", formality=2)],
            balance_rule="match",
        ),
    )

    omega_includes = [
        params.get("p_include_categories")
        for name, params in calls
        if name == "ff_build_outfit_v3"
    ]
    assert omega_includes[-1] is None
    assert outfit["items"]["top"]["product_id"] == "mystery_top"


@pytest.mark.parametrize("category", ["dress", "jumpsuit_romper"])
def test_full_body_top_categories_are_not_default_top_candidates(builder, category):
    assert (
        builder._candidate_allowed_for_slot(
            _item("full_body", category, "top"),
            SlotConfig("top", formality=2),
        )
        is False
    )


def _mock_chat_response(payload):
    class _Choice:
        class _Msg:
            content = json.dumps(payload)

        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


@pytest.mark.parametrize(
    ("query", "expected_includes", "expected_excludes"),
    [
        ("make the bottom a skirt", {"bottom": ["skirt_skort"]}, {}),
        ("no heels please", {}, {"footwear": ["heel"]}),
        ("wear sneakers with this", {"footwear": ["sneaker"]}, {}),
    ],
)
def test_chat_category_intent(builder, monkeypatch, query, expected_includes, expected_excludes):
    monkeypatch.setattr(
        builder.openai.chat.completions,
        "create",
        lambda **kwargs: _mock_chat_response({}),
    )

    params = builder.parse_chat_query(query, user_profile={})

    assert params.category_includes == expected_includes
    assert params.category_excludes == expected_excludes


def test_chat_include_overrides_occasion_category_preference(builder):
    params = OutfitParams(
        gender="feminine",
        occasion="work",
        category_includes={"bottom": ["skirt_skort"]},
    )

    attempts = builder._slot_category_attempts(
        params, SlotConfig("bottom", formality=4)
    )

    assert attempts[0].include_categories == ["skirt_skort"]
