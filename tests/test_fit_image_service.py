"""Unit tests for FitImageService list behavior."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from fit_image_service import FitImageService  # noqa: E402


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _Resp(self._data)


class _Supabase:
    def __init__(self):
        self._tables = {
            "user_fits": [
                {
                    "id": "fit_1",
                    "user_id": "user_1",
                    "title": "Work look",
                    "source": "stylist_chat",
                    "tags": ["work"],
                    "items": [],
                    "created_at": "2026-06-28T00:00:00",
                }
            ],
            "user_fit_images": [
                {"fit_id": "fit_1", "image_path": "user_1/fit_1/1.png", "position": 0},
                {"fit_id": "fit_1", "image_path": "user_1/fit_1/2.png", "position": 1},
            ],
        }

    def table(self, name):
        return _Query(self._tables[name])


def test_list_fits_skips_failed_signed_urls(monkeypatch):
    service = FitImageService.__new__(FitImageService)
    service.supabase = _Supabase()

    def fake_signed_url(path):
        if path.endswith("2.png"):
            raise TimeoutError("signed URL timeout")
        return f"https://signed.example/{path}"

    monkeypatch.setattr(service, "_create_signed_url", fake_signed_url)

    fits = service.list_fits("user_1")

    assert len(fits) == 1
    assert fits[0]["image_urls"] == ["https://signed.example/user_1/fit_1/1.png"]


def test_persona_key_normalizes_and_orders():
    key = FitImageService.persona_key(
        {"gender": " Masculine ", "style": "Classic", "occasion": "WORK", "setting": "City"}
    )
    assert key == "masculine|classic|work|city"


def test_persona_key_fills_missing_axes_with_any():
    # Missing / empty axes must not shift positions — order is always
    # gender|style|occasion|setting so the key matches the iOS client.
    key = FitImageService.persona_key({"gender": "feminine", "occasion": "date"})
    assert key == "feminine|any|date|any"


def test_persona_items_flattens_and_drops_empty_slots():
    service = FitImageService.__new__(FitImageService)
    items = {
        "top": {
            "product_id": 42,
            "product_title": "Linen Shirt",
            "product_brand": "Afends",
            "product_price_amount": 80,
            "product_img_link": "https://img/top.jpg",
            "product_link": "https://shop/top",
        },
        "bottom": None,  # unfilled slot — should be skipped
        "footwear": {
            "product_id": "shoe-1",
            "product_title": "Loafers",
            "product_img_link": "https://img/shoe.jpg",
        },
    }
    result = service._persona_items(items)

    assert [i["slot"] for i in result] == ["top", "shoes"]  # footwear -> shoes
    assert result[0]["id"] == "42"
    assert result[0]["imageUrl"] == "https://img/top.jpg"
    assert result[0]["brand"] == "Afends"
    assert result[1]["brand"] == ""  # no product_brand -> empty fallback
    assert result[1]["price"] == 0
