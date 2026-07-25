"""Unit tests for GET /products/search — route validation and envelope shape.

The embedding model and Supabase RPC are never touched: a fake service is
injected into api_server's module-level singleton. These tests cover the
contract the iOS Search tab depends on (finer-ios docs/backend-blanks.md).
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy env so client constructors don't blow up at import time
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import api_server  # noqa: E402


FAKE_RESULT = {
    "product_id": "awin_114434_1",
    "product_title": "Linen Shirt: White Tops - Size Medium",
    "product_brand": "Haus",
    "product_price_amount": 24.99,
    "product_img_link": "https://img.example/1.jpg",
    "product_link": "https://shop.example/1",
    "slot": "top",
    "category": "top",
    "similarity": 0.6123,
}


class FakeSearchService:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [FAKE_RESULT]
        self.error = error
        self.calls = []

    def search(self, query, limit=20, gender=None):
        self.calls.append((query, limit, gender))
        if self.error:
            raise self.error
        return self.results


@pytest.fixture
def client():
    api_server.app.config["TESTING"] = True
    with api_server.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def fake_service():
    service = FakeSearchService()
    api_server._product_search_service = service
    yield service
    api_server._product_search_service = None


def test_missing_q_returns_400(client, fake_service):
    response = client.get("/products/search")
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert body["results"] == []
    assert "q" in body["error"]
    assert fake_service.calls == []


def test_blank_q_returns_400(client, fake_service):
    response = client.get("/products/search?q=%20%20")
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_invalid_limit_returns_400(client, fake_service):
    response = client.get("/products/search?q=dress&limit=abc")
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert "limit" in body["error"].lower()


def test_success_envelope(client, fake_service):
    response = client.get("/products/search?q=linen+shirt")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["results"] == [FAKE_RESULT]
    assert fake_service.calls == [("linen shirt", 20, None)]


def test_limit_is_capped_at_50(client, fake_service):
    response = client.get("/products/search?q=dress&limit=500")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 50, None)]


def test_limit_floor_is_1(client, fake_service):
    response = client.get("/products/search?q=dress&limit=-3")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 1, None)]


def test_gender_masculine_passes_through(client, fake_service):
    response = client.get("/products/search?q=dress&gender=masculine")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 20, "masculine")]


def test_gender_feminine_passes_through(client, fake_service):
    response = client.get("/products/search?q=dress&gender=Feminine")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 20, "feminine")]


def test_gender_nonbinary_means_no_filter(client, fake_service):
    for value in ("unisex", "non-binary", "genderless", "anything-else"):
        response = client.get(f"/products/search?q=dress&gender={value}")
        assert response.status_code == 200
    assert fake_service.calls == [("dress", 20, None)] * 4


def test_service_failure_returns_500_envelope(client):
    service = FakeSearchService(error=RuntimeError("rpc exploded"))
    api_server._product_search_service = service
    try:
        response = client.get("/products/search?q=dress")
        assert response.status_code == 500
        body = response.get_json()
        assert body["success"] is False
        assert body["results"] == []
        assert "rpc exploded" in body["error"]
    finally:
        api_server._product_search_service = None


def test_serializer_omits_missing_optional_fields():
    from product_search_service import ProductSearchService

    row = {
        "product_id": "p1",
        "product_title": "Jacket",
        "product_price_amount": "49.50",
        "product_img_link": "https://img.example/p1.jpg",
        "product_link": "https://shop.example/p1",
        "product_slot": None,
        "product_description": None,
        "similarity": 0.71234,
    }
    item = ProductSearchService._serialize(row)
    assert item["product_price_amount"] == 49.50
    assert item["similarity"] == 0.7123
    assert "slot" not in item
    assert "category" not in item
    assert "description" not in item


def test_serializer_includes_slot_and_description_when_present():
    from product_search_service import ProductSearchService

    row = {
        "product_id": "p2",
        "product_title": "Dress",
        "product_price_amount": 80,
        "product_img_link": "https://img.example/p2.jpg",
        "product_link": "https://shop.example/p2",
        "product_slot": "dress",
        "product_description": "A flowy midi dress.",
        "similarity": 0.6,
    }
    item = ProductSearchService._serialize(row)
    assert item["slot"] == "dress"
    assert item["category"] == "dress"
    assert item["description"] == "A flowy midi dress."


def test_pants_intent_expands_to_multiple_bottom_categories():
    from product_search_service import parse_search_intent

    intent = parse_search_intent("pants")

    assert intent.categories == ("pants", "jeans", "leggings")
    assert intent.diversify_categories is True
    assert "trousers" in intent.semantic_query
    assert "jeans" in intent.semantic_query
    assert intent.brand_query is None


def test_specific_category_and_color_are_exact_facets():
    from product_search_service import parse_search_intent

    intent = parse_search_intent("black jeans")

    assert intent.categories == ("jeans",)
    assert intent.colors == ("black",)
    assert intent.diversify_categories is False
    assert intent.brand_query is None


def test_bare_brand_is_required_without_a_static_brand_registry():
    from product_search_service import parse_search_intent

    intent = parse_search_intent("Kowtow")

    assert intent.brand_query == "kowtow"
    assert intent.brand_terms == ("kowtow",)
    assert intent.require_brand is True


def test_mixed_brand_category_and_color_keep_all_three_intents():
    from product_search_service import parse_search_intent

    intent = parse_search_intent("black Kowtow jeans")

    assert intent.categories == ("jeans",)
    assert intent.colors == ("black",)
    assert intent.brand_query == "kowtow"
    assert intent.brand_terms == ("kowtow",)
    assert intent.require_brand is False


def test_specific_phrase_does_not_leak_into_second_category():
    from product_search_service import parse_search_intent

    intent = parse_search_intent("navy dress pants")

    assert intent.categories == ("pants",)
    assert "dress" not in intent.categories
    assert intent.colors == ("blue",)


class _FakeRPCCall:
    def __init__(self, outcome):
        self.outcome = outcome

    def execute(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(data=self.outcome)


class _RecordingSupabase:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def rpc(self, name, params):
        self.calls.append((name, params))
        return _FakeRPCCall(self.outcomes.pop(0))


def _service_with(supabase_client, embedding=None, embedding_error=None):
    from product_search_service import ProductSearchService

    service = object.__new__(ProductSearchService)
    service.supabase = supabase_client
    if embedding_error:
        service.embed_query = lambda _query: (_ for _ in ()).throw(embedding_error)
    else:
        service.embed_query = lambda _query: embedding or [0.1, 0.2, 0.3]
    return service


def test_service_passes_structured_intent_to_hybrid_rpc(monkeypatch):
    import product_search_service

    database = _RecordingSupabase([[{
        "product_id": "p1",
        "product_title": "Classic Jeans",
        "product_brand": "Kowtowclothing",
        "product_price_amount": 115,
        "product_img_link": "https://img.example/jeans.jpg",
        "product_link": "https://shop.example/jeans",
        "product_slot": "bottom",
        "product_category": "jeans",
        "product_color": "black",
        "similarity": 0.51,
        "search_score": 1.2,
    }]])
    service = _service_with(database)
    monkeypatch.setattr(product_search_service, "SEARCH_RPC", "match_products_hybrid")

    results = service.search("black Kowtow jeans", limit=12, gender="feminine")

    assert results[0]["category"] == "jeans"
    assert results[0]["product_color"] == "black"
    assert results[0]["search_score"] == 1.2
    rpc_name, params = database.calls[0]
    assert rpc_name == "match_products_hybrid"
    assert params["match_count"] == 12
    assert params["p_categories"] == ["jeans"]
    assert params["p_colors"] == ["black"]
    assert params["p_brand_query"] == "kowtow"
    assert params["p_brand_terms"] == ["kowtow"]
    assert params["p_gender"] == "feminine"


def test_missing_hybrid_rpc_falls_back_to_legacy(monkeypatch):
    import product_search_service

    database = _RecordingSupabase(
        [
            RuntimeError("PGRST202: Could not find the function"),
            [dict(FAKE_RESULT)],
        ]
    )
    service = _service_with(database)
    monkeypatch.setattr(product_search_service, "SEARCH_RPC", "match_products_hybrid")
    monkeypatch.setattr(
        product_search_service, "LEGACY_SEARCH_RPC", "match_products_semantic"
    )

    results = service.search("pants", limit=7)

    assert results[0]["product_id"] == FAKE_RESULT["product_id"]
    assert results[0]["similarity"] == FAKE_RESULT["similarity"]
    assert database.calls[0][0] == "match_products_hybrid"
    assert database.calls[1] == (
        "match_products_semantic",
        {"query_embedding": [0.1, 0.2, 0.3], "match_count": 7},
    )


def test_embedding_outage_still_calls_hybrid_lexical_search(monkeypatch):
    import product_search_service

    database = _RecordingSupabase([[dict(FAKE_RESULT)]])
    service = _service_with(database, embedding_error=RuntimeError("OpenAI timeout"))
    monkeypatch.setattr(product_search_service, "SEARCH_RPC", "match_products_hybrid")

    results = service.search("red", limit=5)

    assert results[0]["product_id"] == FAKE_RESULT["product_id"]
    assert database.calls[0][1]["query_embedding"] is None
    assert database.calls[0][1]["p_colors"] == ["red"]
