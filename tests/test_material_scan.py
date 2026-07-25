"""Tests for the Gemini-backed Material Scanner."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import api_server  # noqa: E402
from material_scan_service import (  # noqa: E402
    MaterialScanError,
    MaterialScanService,
)


SCAN_RESULT = {
    "garment_type": "shirt",
    "material_family": "natural_cellulosic",
    "likely_materials": [
        {"name": "cotton", "confidence_score": 0.78},
        {"name": "linen", "confidence_score": 0.44},
    ],
    "fabric_construction": "woven",
    "confidence": "medium",
    "confidence_score": 0.74,
    "summary": "This appears to be a cotton-rich woven fabric.",
    "evidence_source": "visual_and_metadata",
    "needs_more_input": True,
    "suggested_next_input": "Add a close-up of the fabric.",
}


class FakeMaterialScanService:
    model = "gemini-3.6-flash"

    def __init__(self, result=None, error=None):
        self.result = result or SCAN_RESULT
        self.error = error
        self.calls = []

    def analyze(self, image, **kwargs):
        self.calls.append((image, kwargs))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    api_server.app.config["TESTING"] = True
    with api_server.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def fake_service():
    service = FakeMaterialScanService()
    api_server._material_scan_service = service
    yield service
    api_server._material_scan_service = None


def _image(filename="garment.jpg", mime_type="image/jpeg"):
    return io.BytesIO(b"not-a-real-image"), filename, mime_type


def test_scan_requires_multipart(client, fake_service):
    response = client.post("/materials/scan", json={"image": "abc"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert "multipart/form-data" in body["error"]
    assert fake_service.calls == []


def test_scan_requires_image(client, fake_service):
    response = client.post(
        "/materials/scan",
        data={"brand": "Finer"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "image" in response.get_json()["error"]
    assert fake_service.calls == []


def test_scan_rejects_unsupported_image_type(client, fake_service):
    response = client.post(
        "/materials/scan",
        data={"image": _image("garment.gif", "image/gif")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["error"]
    assert fake_service.calls == []


def test_scan_passes_images_and_metadata_to_service(client, fake_service):
    response = client.post(
        "/materials/scan",
        data={
            "image": _image(),
            "texture_image": _image("texture.webp", "image/webp"),
            "brand": "  Everlane ",
            "garment_type": "shirt",
            "product_name": "Relaxed Linen Shirt",
            "style_code": "ABC-123",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body == {
        "success": True,
        "scan": SCAN_RESULT,
        "model": "gemini-3.6-flash",
        "error": None,
    }
    assert len(fake_service.calls) == 1
    image, kwargs = fake_service.calls[0]
    assert image == (b"not-a-real-image", "image/jpeg")
    assert kwargs["texture_image"] == (b"not-a-real-image", "image/webp")
    assert kwargs["brand"] == "Everlane"
    assert kwargs["style_code"] == "ABC-123"


def test_scan_maps_gemini_failure_to_502(client):
    api_server._material_scan_service = FakeMaterialScanService(
        error=MaterialScanError("upstream failed")
    )
    try:
        response = client.post(
            "/materials/scan",
            data={"image": _image()},
            content_type="multipart/form-data",
        )
    finally:
        api_server._material_scan_service = None

    assert response.status_code == 502
    body = response.get_json()
    assert body["success"] is False
    assert body["scan"] is None
    assert "temporarily unavailable" in body["error"]


class _GeminiResponse:
    def __init__(self, body, *, ok=True, status_code=200):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _GeminiSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_service_uses_gemini_36_minimal_thinking_and_structured_output():
    response_body = {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(SCAN_RESULT)}],
            }
        ],
        "usage": {
            "total_input_tokens": 800,
            "total_output_tokens": 120,
            "total_thought_tokens": 0,
        },
    }
    session = _GeminiSession(_GeminiResponse(response_body))
    service = MaterialScanService(api_key="gemini-key", session=session)

    result = service.analyze(
        (b"garment", "image/jpeg"),
        brand="Everlane",
        garment_type="shirt",
    )

    assert result["material_family"] == "natural_cellulosic"
    assert result["confidence"] == "medium"
    assert len(session.calls) == 1
    _url, call = session.calls[0]
    assert call["headers"]["x-goog-api-key"] == "gemini-key"
    payload = call["json"]
    assert payload["model"] == "gemini-3.6-flash"
    assert payload["store"] is False
    assert payload["generation_config"] == {"thinking_level": "minimal"}
    assert payload["response_format"]["mime_type"] == "application/json"
    assert payload["input"][1]["resolution"] == "medium"
    assert "percentages" in payload["input"][0]["text"]


def test_service_uses_high_resolution_only_for_texture_closeup():
    response_body = {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": json.dumps(SCAN_RESULT)}],
            }
        ],
    }
    session = _GeminiSession(_GeminiResponse(response_body))
    service = MaterialScanService(api_key="gemini-key", session=session)

    service.analyze(
        (b"garment", "image/jpeg"),
        texture_image=(b"texture", "image/jpeg"),
    )

    image_parts = [
        item for item in session.calls[0][1]["json"]["input"] if item["type"] == "image"
    ]
    assert [item["resolution"] for item in image_parts] == ["medium", "high"]


def test_service_forces_unknown_or_low_confidence_to_request_more_input():
    normalized = MaterialScanService._normalize_scan(
        {
            **SCAN_RESULT,
            "material_family": "something_invalid",
            "confidence_score": 0.2,
            "confidence": "high",
            "needs_more_input": False,
        }
    )

    assert normalized["material_family"] == "unknown"
    assert normalized["confidence"] == "low"
    assert normalized["needs_more_input"] is True


def test_service_raises_safe_error_for_gemini_http_failure():
    session = _GeminiSession(
        _GeminiResponse({"error": {"message": "bad request"}}, ok=False, status_code=400)
    )
    service = MaterialScanService(api_key="gemini-key", session=session)

    with pytest.raises(MaterialScanError, match="HTTP 400"):
        service.analyze((b"garment", "image/jpeg"))
