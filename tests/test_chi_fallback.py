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

from outfit_builder import OutfitBuilder  # noqa: E402

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
