"""Unit tests for the chi-primary / omega-fallback slot querying."""

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


def _make_rpc_mock(chi_data=None, omega_data=None, chi_error=False):
    """Create a supabase.rpc mock that dispatches by RPC name."""

    def rpc_side_effect(name, params):
        mock_exec = MagicMock()
        if name == "ff_build_outfit_chi":
            if chi_error:
                raise RuntimeError("chi connection error")
            mock_exec.execute.return_value = MagicMock(data=chi_data)
        elif name == "ff_build_outfit_v3":
            mock_exec.execute.return_value = MagicMock(data=omega_data)
        else:
            mock_exec.execute.return_value = MagicMock(data=None)
        return mock_exec

    return rpc_side_effect


@pytest.fixture
def builder():
    return OutfitBuilder()


def test_chi_returns_data_omega_not_called(builder, monkeypatch):
    chi_item = {**SAMPLE_ITEM, "product_id": "chi_1"}
    calls = []

    def rpc(name, params):
        calls.append(name)
        mock = MagicMock()
        if name == "ff_build_outfit_chi":
            mock.execute.return_value = MagicMock(data=[chi_item])
        else:
            mock.execute.return_value = MagicMock(data=[])
        return mock

    monkeypatch.setattr(builder.supabase, "rpc", rpc)

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["product_id"] == "chi_1"
    assert item["_source"] == "chi"
    assert "ff_build_outfit_v3" not in calls


def test_chi_empty_falls_to_omega(builder, monkeypatch):
    omega_item = {**SAMPLE_ITEM, "product_id": "omega_1"}
    monkeypatch.setattr(
        builder.supabase, "rpc", _make_rpc_mock(chi_data=[], omega_data=[omega_item])
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["product_id"] == "omega_1"
    assert item["_source"] == "omega"


def test_chi_error_falls_to_omega(builder, monkeypatch):
    omega_item = {**SAMPLE_ITEM, "product_id": "omega_1"}
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(chi_error=True, omega_data=[omega_item]),
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is not None
    assert item["_source"] == "omega"


def test_both_empty_returns_none(builder, monkeypatch):
    monkeypatch.setattr(
        builder.supabase, "rpc", _make_rpc_mock(chi_data=[], omega_data=[])
    )

    item = builder._query_slot_with_fallback(RPC_PARAMS, "top")
    assert item is None


def test_candidates_merges_chi_and_omega(builder, monkeypatch):
    chi_items = [
        {**SAMPLE_ITEM, "product_id": "chi_1"},
        {**SAMPLE_ITEM, "product_id": "chi_2"},
    ]
    omega_items = [
        {**SAMPLE_ITEM, "product_id": "chi_1"},  # duplicate
        {**SAMPLE_ITEM, "product_id": "omega_1"},
        {**SAMPLE_ITEM, "product_id": "omega_2"},
    ]
    monkeypatch.setattr(
        builder.supabase,
        "rpc",
        _make_rpc_mock(chi_data=chi_items, omega_data=omega_items),
    )

    candidates = builder._query_slot_candidates_with_fallback(
        {**RPC_PARAMS, "p_n": 5}, "top", desired_n=5
    )
    ids = [c["product_id"] for c in candidates]
    assert ids == ["chi_1", "chi_2", "omega_1", "omega_2"]  # deduped, chi first
    assert candidates[0]["_source"] == "chi"
    assert candidates[2]["_source"] == "omega"
