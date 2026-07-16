"""Unit tests for GET /products/search — route validation and envelope shape.

The embedding model and Supabase RPC are never touched: a fake service is
injected into api_server's module-level singleton. These tests cover the
contract the iOS Search tab depends on (finer-ios docs/backend-blanks.md).
"""

import os
import sys

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

    def search(self, query, limit=20):
        self.calls.append((query, limit))
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
    assert fake_service.calls == [("linen shirt", 20)]


def test_limit_is_capped_at_50(client, fake_service):
    response = client.get("/products/search?q=dress&limit=500")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 50)]


def test_limit_floor_is_1(client, fake_service):
    response = client.get("/products/search?q=dress&limit=-3")
    assert response.status_code == 200
    assert fake_service.calls == [("dress", 1)]


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
