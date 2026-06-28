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
