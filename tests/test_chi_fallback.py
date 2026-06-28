"""Unit tests for the omega-primary / chi-fallback slot querying.

PRIMARY_RPC defaults to "omega" (outfit_builder.RPC_ORDER), so omega
(ff_build_outfit_v3) is tried first and chi (ff_build_outfit_chi) only
backfills slots omega leaves empty.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from outfit_builder import OutfitBuilder, OutfitFormula, OutfitParams, SlotConfig  # noqa: E402

SAMPLE_ITEM = {
    "product_id": "abc123",
    "product_title": "Test Top",
    "product_price_amount": 49.99,
    "product_color": "black",
    "product_texture": "woven",
}

RPC_PARAMS = {
    "p_slot": "top",
    "p_gender": "feminine",
    "p_style_tags": ["minimalist"],
    "p_n": 1,
}


def _make_rpc_mock(omega_data=None, chi_data=None, omega_error=False):
    """Create a supabase.rpc mock that dispatches by RPC name."""

    def rpc_side_effect(name, params):
        mock_exec = MagicMock()
        if name == "ff_build_outfit_v3":
            if omega_error:
                raise RuntimeError("omega connection error")
            mock_exec.execute.return_value = MagicMock(data=omega_data)
        elif name == "ff_build_outfit_chi":
            mock_exec.execute.return_value = MagicMock(data=chi_data)
        else:
            mock_exec.execute.return_value = MagicMock(data=None)
        return mock_exec

    return rpc_side_effect


@pytest.fixture
def builder():
    return OutfitBuilder()


def test_omega_returns_data_chi_not_called(builder, monkeypatch):
    omega_item = {**SAMPLE_ITEM, "product_id": "omega_1"}
    calls = []

    def rpc(name, params):
        calls.append(name)
        mock = MagicMock()
        if name == "ff_build_outfit_v3":
            mock.execute.return_value = MagicMock(data=[omega_item])
        else:
            mock.execute.return_value = MagicMock(data=[])
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["product_id"] == "omega_1"
    assert item["_source"] == "omega"
    assert "ff_build_outfit_chi" not in calls


def test_omega_empty_falls_to_chi(builder, monkeypatch):
    chi_item = {**SAMPLE_ITEM, "product_id": "chi_1"}
    monkeypatch.setattr(
        builder.supabase, "rpc", _make_rpc_mock(omega_data=[], chi_data=[chi_item])
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["product_id"] == "chi_1"
    assert item["_source"] == "chi"


def test_omega_error_falls_to_chi(builder, monkeypatch):
    chi_item = {**SAMPLE_ITEM, "product_id": "chi_1"}
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(omega_error=True, chi_data=[chi_item]),
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["_source"] == "chi"


def test_both_empty_returns_none(builder, monkeypatch):
    monkeypatch.setattr(
        builder.supabase, "rpc", _make_rpc_mock(omega_data=[], chi_data=[])
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is None


def test_candidates_merges_omega_and_chi(builder, monkeypatch):
    omega_items = [
        {**SAMPLE_ITEM, "product_id": "omega_1"},
        {**SAMPLE_ITEM, "product_id": "omega_2"},
    ]
    chi_items = [
        {**SAMPLE_ITEM, "product_id": "omega_1"},  # duplicate
        {**SAMPLE_ITEM, "product_id": "chi_1"},
        {**SAMPLE_ITEM, "product_id": "chi_2"},
    ]
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(omega_data=omega_items, chi_data=chi_items),
    )

    candidates = builder._query_slot_candidates_with_fallback(
        {**RPC_PARAMS, "p_n": 5}, "top", desired_n=5
    )
    ids = [c["product_id"] for c in candidates]
    assert ids == ["omega_1", "omega_2", "chi_1", "chi_2"]  # deduped, omega first
    assert candidates[0]["_source"] == "omega"
    assert candidates[2]["_source"] == "chi"


def test_bottom_build_passes_underwear_exclusion_to_omega(builder, monkeypatch):
    calls = []

    def rpc(name, params):
        calls.append((name, params))
        mock = MagicMock()
        mock.execute.return_value = MagicMock(data=[])
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)
    params = OutfitParams(gender="feminine", occasion="casual")
    formula = OutfitFormula(
        base_formality=2,
        slots=[SlotConfig("bottom", formality=2)],
        balance_rule="match",
    )

    builder._build_outfit(params, formula)

    omega_call = next(params for name, params in calls if name == "ff_build_outfit_v3")
    assert omega_call["p_exclude_categories"] == ["underwear"]


def test_genderless_build_sends_null_gender_filter(builder, monkeypatch):
    calls = []

    def rpc(name, params):
        calls.append((name, params))
        mock = MagicMock()
        mock.execute.return_value = MagicMock(data=[])
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)
    params = OutfitParams(gender="genderless", occasion="casual")
    formula = OutfitFormula(
        base_formality=2,
        slots=[SlotConfig("top", formality=2)],
        balance_rule="match",
    )

    builder._build_outfit(params, formula)

    omega_call = next(params for name, params in calls if name == "ff_build_outfit_v3")
    assert omega_call["p_gender"] is None


def test_underwear_candidate_is_skipped_for_normal_bottom(builder, monkeypatch):
    underwear = {
        **SAMPLE_ITEM,
        "product_id": "u1",
        "product_title": "Cotton Boxer Briefs",
        "product_category": "underwear",
    }
    pants = {
        **SAMPLE_ITEM,
        "product_id": "p1",
        "product_title": "Wide Leg Pants",
        "product_category": "pants",
    }
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(omega_data=[underwear, pants], chi_data=[]),
    )

    item = builder._query_slot_with_fallback(
        {"p_slot": "bottom", "p_gender": "feminine"},
        "bottom",
        slot_config=SlotConfig("bottom", formality=2),
    )

    assert item is not None
    assert item["product_id"] == "p1"


def test_underwear_title_is_skipped_even_without_category(builder, monkeypatch):
    underwear = {
        **SAMPLE_ITEM,
        "product_id": "u1",
        "product_title": "Cotton Boxer Briefs",
    }
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(omega_data=[underwear], chi_data=[]),
    )

    item = builder._query_slot_with_fallback(
        {"p_slot": "bottom", "p_gender": "feminine"},
        "bottom",
        slot_config=SlotConfig("bottom", formality=2),
    )

    assert item is None


def test_bottom_swap_passes_underwear_exclusion(builder, monkeypatch):
    captured = {}

    def fake_candidates(rpc_params, slot_name, desired_n=5, slot_config=None):
        captured["rpc_params"] = rpc_params
        captured["slot_name"] = slot_name
        captured["slot_config"] = slot_config
        return []

    monkeypatch.setattr(
        builder, "_query_slot_candidates_with_fallback", fake_candidates
    )

    builder.swap_item(
        {
            "params_used": {
                "gender": "feminine",
                "occasion": "casual",
                "style_tags": ["minimalist"],
            },
            "items": {},
        },
        "bottom",
    )

    assert captured["slot_name"] == "bottom"
    assert captured["rpc_params"]["p_exclude_categories"] == ["underwear"]
    assert captured["slot_config"].slot == "bottom"
