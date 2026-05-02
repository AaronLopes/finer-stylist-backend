"""
Manual integration test for POST /outfit/build/today.

Hits a running backend (local Flask or Render). Not part of pytest because
it requires the live database, products, and OPENAI_API_KEY — but it
exercises the cache, the fallback layers, and the persistence side effects
end-to-end.

Usage:
    python3 tests/test_today_endpoint.py [http://localhost:8000]
    python3 tests/test_today_endpoint.py https://finer-stylist-backend.onrender.com
"""

import json
import sys
import time
import uuid

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TEST_USER_ID = f"test-user-{uuid.uuid4().hex[:8]}"


def post(path: str, body: dict) -> dict:
    print(f"\n--> POST {BASE_URL}{path}")
    print(f"    user_id={body.get('user_id', '(none)')}")
    if "today_context" in body:
        print(f"    context={list(body['today_context'].keys())}")
    r = requests.post(f"{BASE_URL}{path}", json=body, timeout=30)
    print(f"    HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        print(f"    NON-JSON RESPONSE: {r.text[:300]}")
        sys.exit(1)
    return data


def assert_eq(actual, expected, label: str) -> None:
    status = "PASS" if actual == expected else "FAIL"
    print(f"    [{status}] {label}: expected {expected!r}, got {actual!r}")
    assert actual == expected, label


def assert_truthy(value, label: str) -> None:
    status = "PASS" if value else "FAIL"
    print(f"    [{status}] {label}: {value!r}")
    assert value, label


def case_full_context_caches() -> None:
    print("\n=== case 1: full context, calendar overrides occasion to 'work' ===")
    body = {
        "user_id": TEST_USER_ID,
        "profile": {
            "gender": "feminine",
            "occasion": "casual",
            "weather": ["moderate"],
            "setting": "city",
            "goals": ["serve-looks"],
            "style": "minimalist",
            "budget": "$$",
        },
        "today_context": {
            "lat": 40.7128,
            "lon": -74.0060,
            "temp_f": 38,
            "condition": "cloudy",
            "daypart": "morning",
            "primary_event": {"title": "Team standup with engineering"},
            "local_date": time.strftime("%Y-%m-%d"),
        },
    }

    first = post("/outfit/build/today", body)
    assert_eq(first.get("success"), True, "first call success")
    assert_eq(first.get("cached"), False, "first call NOT cached")
    assert_eq(first.get("occasion"), "work", "calendar event maps to occasion=work")
    assert_truthy(first.get("fit_id"), "fit_id present after generate")
    items = first.get("items") or {}
    has_items = any(v is not None for v in items.values())
    assert_truthy(has_items, "items dict populated")
    fit_id = first["fit_id"]

    print("\n--- second call same context (should hit cache) ---")
    second = post("/outfit/build/today", body)
    assert_eq(second.get("cached"), True, "second call IS cached")
    assert_eq(second.get("fit_id"), fit_id, "cache returns same fit_id")


def case_different_occasion_skips_cache() -> None:
    print("\n=== case 2: different calendar event = different occasion = new fit ===")
    body = {
        "user_id": TEST_USER_ID,
        "profile": {"gender": "feminine", "style": "minimalist", "budget": "$$"},
        "today_context": {
            "temp_f": 38,
            "primary_event": {"title": "Dinner with Anna"},
            "local_date": time.strftime("%Y-%m-%d"),
        },
    }
    response = post("/outfit/build/today", body)
    assert_eq(response.get("success"), True, "success")
    assert_eq(response.get("occasion"), "date", "dinner event maps to occasion=date")
    assert_eq(response.get("cached"), False, "different occasion = cache miss")


def case_profile_fallback_only() -> None:
    print("\n=== case 3: no live signals — profile fallback drives everything ===")
    body = {
        # No user_id — exercises the no-cache code path too
        "profile": {
            "gender": "masculine",
            "occasion": "gym",
            "weather": ["moderate"],
            "style": "fitted",
            "budget": "$$",
        },
        "today_context": {"local_date": time.strftime("%Y-%m-%d")},
    }
    response = post("/outfit/build/today", body)
    assert_eq(response.get("success"), True, "fallback path succeeds")
    assert_eq(response.get("occasion"), "gym", "occasion taken from profile")
    assert_eq(response.get("cached"), False, "no user_id = no cache")
    assert_eq(response.get("fit_id"), None, "no user_id = no persisted fit")


def case_empty_payload_does_not_500() -> None:
    print("\n=== case 4: empty profile + empty context (defaults only) ===")
    response = post("/outfit/build/today", {"profile": {}, "today_context": {}})
    # Either generates a casual default outfit OR returns a clean 404.
    # The only failure mode is 500.
    assert response.get("success") is not None or response.get("error") is not None, \
        "even empty input must return a structured response"
    print(f"    [INFO] empty-input outcome: success={response.get('success')!r}")


if __name__ == "__main__":
    print(f"Testing {BASE_URL}")
    print(f"Using throwaway user_id={TEST_USER_ID}")
    print(f"NOTE: cases 1+2 will create rows in user_fits + finer_daily_fits.")
    print(f"      Clean up with: DELETE FROM finer_daily_fits WHERE user_id='{TEST_USER_ID}';")
    print(f"                     DELETE FROM user_fits WHERE user_id='{TEST_USER_ID}';")

    case_full_context_caches()
    case_different_occasion_skips_cache()
    case_profile_fallback_only()
    case_empty_payload_does_not_500()

    print("\nAll cases finished.")
