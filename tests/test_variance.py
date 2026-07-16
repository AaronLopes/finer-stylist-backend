"""Unit tests for seeded item-selection variance in Today builds.

Stubs builder.supabase.rpc so no network is touched. Variance is a pick
policy inside _query_slot_with_fallback / _build_outfit: with a seeded RNG
the slot picks from the top-N allowed candidates; without one it keeps the
original deterministic top-1 behavior.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from outfit_builder import (  # noqa: E402
    OCCASION_FORMULAS,
    OutfitBuilder,
    TODAY_VARIANCE_POOL_SIZE,
)


def _candidates(slot: str, n: int = TODAY_VARIANCE_POOL_SIZE):
    return [
        {
            "product_id": f"{slot}-{i}",
            "product_title": f"{slot} item {i}",
            "product_color": "black",
            "product_texture": "cotton",
            "product_price_amount": 50.0 + i,
        }
        for i in range(n)
    ]


class _StubExecute:
    def __init__(self, data):
        self.data = data

    def execute(self):
        return self


@pytest.fixture()
def builder(monkeypatch):
    b = OutfitBuilder()

    def stub_rpc(rpc_name, rpc_params):
        return _StubExecute(_candidates(rpc_params.get("p_slot") or "top"))

    monkeypatch.setattr(b.supabase, "rpc", stub_rpc)
    return b


def test_no_rng_returns_top_candidate(builder):
    item = builder._query_slot_with_fallback({"p_slot": "top", "p_n": 1}, "top")
    assert item["product_id"] == "top-0"


def test_same_seed_same_pick(builder):
    first = builder._query_slot_with_fallback(
        {"p_slot": "top", "p_n": TODAY_VARIANCE_POOL_SIZE},
        "top",
        rng=random.Random(42),
    )
    second = builder._query_slot_with_fallback(
        {"p_slot": "top", "p_n": TODAY_VARIANCE_POOL_SIZE},
        "top",
        rng=random.Random(42),
    )
    assert first["product_id"] == second["product_id"]


def test_different_seeds_can_differ(builder):
    picks = {
        builder._query_slot_with_fallback(
            {"p_slot": "top", "p_n": TODAY_VARIANCE_POOL_SIZE},
            "top",
            rng=random.Random(seed),
        )["product_id"]
        for seed in range(20)
    }
    assert len(picks) > 1


def test_build_outfit_same_seed_is_idempotent(builder):
    params = builder.build_params(profile={"occasion": "casual", "budget": "$$"})
    formula = OCCASION_FORMULAS["casual"]

    first = builder._build_outfit(params, formula, variance_seed=1234)
    second = builder._build_outfit(params, formula, variance_seed=1234)

    ids = lambda outfit: {  # noqa: E731
        slot: item and item["product_id"] for slot, item in outfit["items"].items()
    }
    assert ids(first) == ids(second)


def test_build_outfit_without_seed_keeps_top1(builder):
    params = builder.build_params(profile={"occasion": "casual", "budget": "$$"})
    formula = OCCASION_FORMULAS["casual"]

    outfit = builder._build_outfit(params, formula)
    for slot, item in outfit["items"].items():
        if item is not None:
            assert item["product_id"] == f"{slot}-0"
