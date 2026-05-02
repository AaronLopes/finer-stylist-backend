#!/usr/bin/env python3
"""
Finer Outfit Builder API Server

Flask server providing unified outfit building endpoints for:
- Web quiz flow
- iOS quiz flow
- iOS chat-based search

Endpoints:
    POST /api/outfit/build     - Build outfit from quiz or chat
    POST /api/outfit/swap      - Swap single item in outfit
    POST /api/chat/parse       - Parse chat query (preview what we understood)
    GET  /api/health           - Health check

Usage:
    gunicorn api_server:app --bind 0.0.0.0:8000
    # or for development:
    python api_server.py

Environment Variables Required:
    - SUPABASE_URL
    - SUPABASE_KEY
    - OPENAI_API_KEY
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from chat_service import get_chat_composer, get_chat_resolver
from fit_image_service import FitImageService
from outfit_builder import OutfitBuilder, create_outfit_builder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# CORS for web and mobile clients
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize outfit builder (lazy loading)
_outfit_builder: Optional[OutfitBuilder] = None
_fit_image_service: Optional[FitImageService] = None


def get_outfit_builder() -> OutfitBuilder:
    global _outfit_builder
    if _outfit_builder is None:
        _outfit_builder = create_outfit_builder()
    return _outfit_builder


def get_fit_image_service() -> FitImageService:
    global _fit_image_service
    if _fit_image_service is None:
        _fit_image_service = FitImageService()
    return _fit_image_service


def _short_user_id(user_id: Optional[str]) -> str:
    if not user_id:
        return "missing"
    if len(user_id) <= 8:
        return user_id
    return f"{user_id[:4]}...{user_id[-4:]}"


def _is_missing_profile_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _is_sparse_profile(profile: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(profile, dict) or not profile:
        return True

    core_keys = ["gender", "occasion", "style", "budget", "weather", "setting", "goals"]
    populated_count = 0

    for key in core_keys:
        if not _is_missing_profile_value(profile.get(key)):
            populated_count += 1

    return populated_count < 2


def _merge_profile_with_fallback(
    user_profile: Optional[Dict[str, Any]], fallback_profile: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(user_profile or {})
    fallback = fallback_profile or {}

    for key, value in fallback.items():
        if key not in merged or _is_missing_profile_value(merged.get(key)):
            merged[key] = value

    return merged


def _fetch_latest_quiz_profile(user_id: str) -> Dict[str, Any]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.warning(
            "/outfit/build chat profile hydration skipped: missing Supabase env vars"
        )
        return {}

    try:
        from supabase import create_client

        client = create_client(supabase_url, supabase_key)
        response = (
            client.table("user_profile")
            .select("quiz_answers,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )

        rows = response.data or []
        if not rows:
            logger.info(
                "/outfit/build chat profile hydration no quiz profile found user=%s",
                _short_user_id(user_id),
            )
            return {}

        quiz_answers = rows[0].get("quiz_answers")
        if not isinstance(quiz_answers, dict):
            logger.warning(
                "/outfit/build chat profile hydration invalid quiz_answers type user=%s type=%s",
                _short_user_id(user_id),
                type(quiz_answers).__name__,
            )
            return {}

        return quiz_answers
    except Exception as exc:
        logger.exception(
            "/outfit/build chat profile hydration failed user=%s error=%s",
            _short_user_id(user_id),
            exc,
        )
        return {}


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_quiz_answers(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate quiz answers payload."""
    required = ["gender", "occasion", "style", "budget"]
    for field in required:
        if field not in data:
            return False, f"Missing required field: {field}"
    return True, ""


def validate_chat_query(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate chat query payload."""
    if "query" not in data or not data["query"]:
        return False, "Missing required field: query"
    return True, ""


def validate_swap_request(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate swap request payload."""
    if "current_outfit" not in data:
        return False, "Missing required field: current_outfit"
    if "slot_to_swap" not in data:
        return False, "Missing required field: slot_to_swap"
    return True, ""


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with environment and connection validation."""

    checks = {
        "version": "2.0.0",
        "env": {},
        "connections": {},
    }
    all_ok = True

    # Check environment variables exist
    env_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            checks["env"][var] = "set"
        else:
            checks["env"][var] = "missing"
            all_ok = False

    # Test Supabase connection
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        try:
            import supabase

            client = supabase.create_client(
                os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
            )
            # Simple query to test connection
            client.table("finer_products_omega").select("product_id").limit(1).execute()
            checks["connections"]["supabase"] = "ok"
        except Exception as e:
            checks["connections"]["supabase"] = f"error: {str(e)}"
            all_ok = False

    # Test OpenAI connection
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Minimal API call to verify key
            client.models.list()
            checks["connections"]["openai"] = "ok"
        except Exception as e:
            checks["connections"]["openai"] = f"error: {str(e)}"
            all_ok = False

    checks["status"] = "healthy" if all_ok else "unhealthy"

    return jsonify(checks), 200 if all_ok else 503


@app.route("/outfit/build", methods=["POST"])
def build_outfit():
    """
    Build a complete outfit from quiz answers or chat query.

    Supports two modes:
    - quiz: Structured answers from quiz flow
    - chat: Natural language query

    Returns outfit with items for each slot (top, bottom, footwear, etc.)
    """
    data = request.get_json()

    if not data:
        logger.warning("/outfit/build missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    mode = data.get("mode")
    logger.info(
        "/outfit/build request received mode=%s keys=%s", mode, list(data.keys())
    )

    if mode not in ["quiz", "chat"]:
        logger.warning("/outfit/build invalid mode=%s", mode)
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Invalid mode: {mode}. Use 'quiz' or 'chat'",
                }
            ),
            400,
        )

    builder = get_outfit_builder()

    try:
        if mode == "quiz":
            quiz_answers = data.get("quiz_answers")
            if not quiz_answers:
                logger.warning("/outfit/build quiz mode missing quiz_answers")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "quiz_answers required for quiz mode",
                        }
                    ),
                    400,
                )

            valid, error = validate_quiz_answers(quiz_answers)
            if not valid:
                logger.warning("/outfit/build quiz validation failed: %s", error)
                return jsonify({"success": False, "error": error}), 400

            outfit = builder.build_from_quiz(quiz_answers)

        else:  # chat mode
            chat_query = data.get("chat_query")
            if not chat_query:
                logger.warning("/outfit/build chat mode missing chat_query")
                return (
                    jsonify(
                        {"success": False, "error": "chat_query required for chat mode"}
                    ),
                    400,
                )

            valid, error = validate_chat_query(chat_query)
            if not valid:
                logger.warning("/outfit/build chat validation failed: %s", error)
                return jsonify({"success": False, "error": error}), 400

            query_text = chat_query.get("query") or ""
            user_profile = chat_query.get("user_profile") or {}
            chat_user_id = chat_query.get("user_id")

            if _is_sparse_profile(user_profile):
                if chat_user_id:
                    fallback_profile = _fetch_latest_quiz_profile(chat_user_id)
                    if fallback_profile:
                        user_profile = _merge_profile_with_fallback(
                            user_profile, fallback_profile
                        )
                        logger.info(
                            "/outfit/build chat profile hydrated user=%s fallback_keys=%s merged_keys=%s",
                            _short_user_id(chat_user_id),
                            list(fallback_profile.keys()),
                            list(user_profile.keys()),
                        )
                    else:
                        logger.info(
                            "/outfit/build chat profile sparse user=%s no fallback data",
                            _short_user_id(chat_user_id),
                        )
                else:
                    logger.info(
                        "/outfit/build chat profile sparse without user_id, cannot hydrate"
                    )

            logger.info(
                "/outfit/build chat query_len=%s profile_keys=%s",
                len(query_text),
                list(user_profile.keys()) if isinstance(user_profile, dict) else [],
            )

            chat_context = chat_query.get("context") or None
            outfit = builder.build_from_chat(
                query=query_text,
                user_profile=user_profile,
                context=chat_context,
            )

            has_real_items = any(
                v is not None for v in (outfit.get("items") or {}).values()
            )
            if not has_real_items:
                logger.warning(
                    "/outfit/build chat empty result; attempting quiz fallback user=%s",
                    _short_user_id(chat_user_id),
                )

                quiz_fallback_payload = {
                    "gender": (user_profile or {}).get("gender"),
                    "occasion": (user_profile or {}).get("occasion"),
                    "style": (user_profile or {}).get("style"),
                    "budget": (user_profile or {}).get("budget"),
                    "weather": (user_profile or {}).get("weather"),
                    "setting": (user_profile or {}).get("setting"),
                    "goals": (user_profile or {}).get("goals"),
                }

                valid_quiz_fallback, quiz_fallback_error = validate_quiz_answers(
                    quiz_fallback_payload
                )

                if valid_quiz_fallback:
                    fallback_outfit = builder.build_from_quiz(quiz_fallback_payload)
                    if fallback_outfit.get("items"):
                        outfit = fallback_outfit
                        logger.info(
                            "/outfit/build chat fallback success user=%s item_slots=%s",
                            _short_user_id(chat_user_id),
                            list((outfit.get("items") or {}).keys()),
                        )
                    else:
                        logger.warning(
                            "/outfit/build chat fallback returned empty items user=%s",
                            _short_user_id(chat_user_id),
                        )
                else:
                    logger.warning(
                        "/outfit/build chat fallback skipped user=%s reason=%s",
                        _short_user_id(chat_user_id),
                        quiz_fallback_error,
                    )

            logger.info(
                "/outfit/build chat success item_slots=%s total_price=%s",
                list((outfit.get("items") or {}).keys()),
                outfit.get("total_price", 0.0),
            )

        return jsonify(
            {
                "success": True,
                "items": outfit.get("items", {}),
                "total_price": outfit.get("total_price", 0.0),
                "params_used": outfit.get("params_used", {}),
            }
        )

    except Exception as e:
        logger.exception("Outfit build failed mode=%s error=%s", mode, e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/outfit/swap", methods=["POST"])
def swap_outfit_item():
    """
    Swap a single item in an existing outfit.

    Maintains outfit cohesion by considering colors and textures
    of other items in the outfit.
    """
    data = request.get_json()

    if not data:
        logger.warning("/outfit/swap missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    valid, error = validate_swap_request(data)
    if not valid:
        return jsonify({"success": False, "error": error}), 400

    builder = get_outfit_builder()

    try:
        new_item = builder.swap_item(
            current_outfit=data.get("current_outfit"),
            slot_to_swap=data.get("slot_to_swap"),
            exclude_product_ids=data.get("exclude_product_ids", []),
        )

        if new_item:
            return jsonify(
                {
                    "success": True,
                    "new_item": new_item,
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "message": "No alternative items found",
                    "new_item": None,
                }
            )

    except Exception as e:
        logger.error("Swap failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat/parse", methods=["POST"])
def parse_chat_query():
    """
    Parse a chat query and return what we understood.

    Useful for iOS to show users what the AI understood before
    generating the outfit. Builds trust and allows refinement.
    """
    data = request.get_json()

    if not data:
        logger.warning("/chat/parse missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    logger.info(
        "/chat/parse request query_len=%s profile_keys=%s",
        len((data.get("query") or "")),
        (
            list((data.get("user_profile") or {}).keys())
            if isinstance(data.get("user_profile"), dict)
            else []
        ),
    )

    query = data.get("query")
    if not query:
        return (
            jsonify({"success": False, "error": "Missing required field: query"}),
            400,
        )

    builder = get_outfit_builder()

    try:
        params = builder.parse_chat_query(
            query=query,
            user_profile=data.get("user_profile"),
        )

        return jsonify(
            {
                "success": True,
                "understood": {
                    "occasion": params.occasion,
                    "gender": params.gender,
                    "budget_range": {
                        "min": params.budget_min,
                        "max": params.budget_max,
                    },
                    "color_strategy": params.color_strategy.value,
                },
                "matched_tags": params.style_tags
                + params.occasion_tags
                + params.season_tags,
            }
        )

    except Exception as e:
        logger.error("Parse failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# CONVENIENCE ENDPOINTS
# =============================================================================


@app.route("/outfit/build/quiz", methods=["POST"])
def build_outfit_from_quiz():
    """
    Convenience endpoint for quiz mode.
    Equivalent to POST /api/outfit/build with mode="quiz"
    """
    data = request.get_json()

    if not data:
        logger.warning("/outfit/build/quiz missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    # Wrap in the unified format
    wrapped = {
        "mode": "quiz",
        "quiz_answers": data,
    }

    # Temporarily replace request json
    with app.test_request_context(json=wrapped):
        return build_outfit()


@app.route("/fits", methods=["POST"])
def save_fit():
    """
    Persist a fit's metadata without generating images. Used so chat-built
    fits land in the user's Fits tab immediately on creation.

    Expected payload:
    {
      "user_id": "uuid",
      "title": "optional",
      "tags": ["date"],
      "source": "stylist_chat" | "today" | ...,
      "items": [{ id, slot, name, brand?, price?, imageUrl?, link? }, ...]
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    user_id = data.get("user_id")
    items = data.get("items")
    tags = data.get("tags") or []

    if not user_id:
        return (
            jsonify({"success": False, "error": "Missing required field: user_id"}),
            400,
        )
    if not isinstance(items, list) or not items:
        return (
            jsonify({"success": False, "error": "Missing required field: items"}),
            400,
        )

    service = get_fit_image_service()
    try:
        fit = service.save_fit(
            user_id=user_id,
            title=data.get("title"),
            tags=tags,
            source=data.get("source") or "stylist_chat",
            items=items,
        )
        logger.info(
            "/fits saved user=%s fit_id=%s source=%s item_count=%s",
            _short_user_id(user_id),
            fit.get("id"),
            data.get("source") or "stylist_chat",
            len(items),
        )
        return jsonify({"success": True, "fit": fit})
    except Exception as e:
        logger.exception(
            "Save fit failed user=%s error=%s", _short_user_id(user_id), e
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/fits/generate-images", methods=["POST"])
def generate_fit_images():
    """
    Generate fit images, upload to Supabase Storage, and persist fit metadata.

    Expected payload:
    {
      "user_id": "uuid",
      "title": "optional title",
      "tags": ["night-out"],
      "source": "stylist_chat",
      "items": [{...mapped fit item with imageUrl + slot...}],
      "profile": {...optional style profile hints...},
      "count": 3
    }
    """
    data = request.get_json()
    if not data:
        logger.warning("/fits/generate-images missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    user_id = data.get("user_id")
    items = data.get("items")
    tags = data.get("tags") or []
    profile = data.get("profile") or {}

    logger.info(
        "/fits/generate-images request user=%s item_count=%s tag_count=%s source=%s count=%s",
        _short_user_id(user_id),
        len(items) if isinstance(items, list) else "invalid",
        len(tags) if isinstance(tags, list) else "invalid",
        data.get("source") or "stylist_chat",
        data.get("count") or 3,
    )

    if not user_id:
        logger.warning("/fits/generate-images validation failed: missing user_id")
        return (
            jsonify({"success": False, "error": "Missing required field: user_id"}),
            400,
        )

    if not isinstance(items, list) or len(items) == 0:
        logger.warning(
            "/fits/generate-images validation failed: invalid items payload_type=%s",
            type(items).__name__,
        )
        return (
            jsonify({"success": False, "error": "Missing required field: items"}),
            400,
        )

    missing_image_slots: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slot = item.get("slot")
        if slot in ["top", "bottom", "shoes"] and not item.get("imageUrl"):
            missing_image_slots.append(str(slot))

    if missing_image_slots:
        logger.warning(
            "/fits/generate-images items missing imageUrl for slots=%s",
            missing_image_slots,
        )

    service = get_fit_image_service()
    source = data.get("source") or "stylist_chat"

    try:
        result = service.create_fit_with_images(
            user_id=user_id,
            title=data.get("title"),
            tags=tags,
            source=source,
            items=items,
            profile=profile,
            count=int(data.get("count") or 3),
        )

        logger.info(
            "/fits/generate-images success user=%s fit_id=%s generated_count=%s profile_keys=%s",
            _short_user_id(user_id),
            result.get("fit", {}).get("id"),
            len(result.get("image_urls") or []),
            list(profile.keys()) if isinstance(profile, dict) else [],
        )

        # If this fit originated from the Today tab, write the composed image
        # URLs back to its finer_daily_fits row so the next /outfit/build/today
        # call this day returns them inline instead of generating again.
        if source == "today":
            new_fit_id = result.get("fit", {}).get("id")
            urls = result.get("image_urls") or []
            if new_fit_id and urls:
                try:
                    service.supabase.table("finer_daily_fits").update(
                        {"image_urls": urls}
                    ).eq("user_id", user_id).eq("fit_id", new_fit_id).execute()
                    logger.info(
                        "/fits/generate-images cached %s images on finer_daily_fits fit_id=%s",
                        len(urls),
                        new_fit_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "/fits/generate-images cache write to finer_daily_fits failed: %s",
                        exc,
                    )

        return jsonify(
            {
                "success": True,
                "fit": result["fit"],
                "image_urls": result["image_urls"],
            }
        )
    except Exception as e:
        logger.exception(
            "Fit image generation failed user=%s error=%s",
            _short_user_id(user_id),
            e,
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/fits", methods=["GET"])
def list_user_fits():
    """
    List persisted fits for a user, including signed image URLs.

    Query params:
      - user_id (required)
    """
    user_id = request.args.get("user_id")
    if not user_id:
        logger.warning("/fits missing user_id query param")
        return (
            jsonify(
                {"success": False, "error": "Missing required query param: user_id"}
            ),
            400,
        )

    service = get_fit_image_service()

    try:
        fits = service.list_fits(user_id)
        logger.info(
            "/fits success user=%s fit_count=%s", _short_user_id(user_id), len(fits)
        )
        return jsonify({"success": True, "fits": fits})
    except Exception as e:
        logger.exception(
            "Fit listing failed user=%s error=%s", _short_user_id(user_id), e
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat/resolve-intent", methods=["POST"])
def resolve_chat_intent():
    """
    Resolve an ambiguous user message into a standalone outfit query.
    Called by the mobile app when client-side confidence is low.

    Expects JSON:
        - message: the user's raw text
        - last_query: the previous outfit query
        - last_outfit_summary: (optional) item names from the last outfit
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    message = data.get("message", "")
    last_query = data.get("last_query", "")

    if not message:
        return (
            jsonify({"success": False, "error": "Missing required field: message"}),
            400,
        )

    logger.info(
        "/chat/resolve-intent msg_len=%s last_query_len=%s",
        len(message),
        len(last_query),
    )

    resolver = get_chat_resolver()
    result = resolver.resolve(
        user_message=message,
        last_query=last_query,
        last_outfit_summary=data.get("last_outfit_summary", ""),
    )

    return jsonify({"success": True, **result})


@app.route("/outfit/build/chat", methods=["POST"])
def build_outfit_from_chat():
    """
    Convenience endpoint for chat mode.
    Equivalent to POST /api/outfit/build with mode="chat"
    Enriches response with natural stylist reply via ChatResponseComposer.
    """
    data = request.get_json()

    if not data:
        logger.warning("/outfit/build/chat missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    query = data.get("query") or ""
    intent = (data.get("context") or {}).get("intent", "new_query")

    logger.info(
        "/outfit/build/chat request query_len=%s has_user_id=%s intent=%s profile_keys=%s",
        len(query),
        bool(data.get("user_id")),
        intent,
        (
            list((data.get("user_profile") or {}).keys())
            if isinstance(data.get("user_profile"), dict)
            else []
        ),
    )

    # Wrap in the unified format
    wrapped = {
        "mode": "chat",
        "chat_query": data,
    }

    # Get the base outfit response
    with app.test_request_context(json=wrapped):
        base_response = build_outfit()

    # Parse the base response to enrich with stylist reply
    try:
        base_data = base_response.get_json()

        if base_data.get("success") and base_data.get("items"):
            items = base_data["items"]
            item_names = [
                v.get("product_title", "item")
                for v in items.values()
                if v is not None and isinstance(v, dict)
            ]
            outfit_summary = ", ".join(item_names[:4])

            composer = get_chat_composer()
            composed = composer.compose_reply(
                query=query,
                outfit_summary=outfit_summary,
                intent=intent,
                user_profile=data.get("user_profile"),
            )

            base_data["reply_text"] = composed["reply_text"]
            base_data["follow_up_chips"] = composed["follow_up_chips"]
            base_data["cta_hint"] = composed["cta_hint"]

            return jsonify(base_data)

    except Exception as exc:
        logger.warning("/outfit/build/chat composer enrichment failed: %s", exc)

    return base_response


@app.route("/outfit/build/today", methods=["POST"])
def build_outfit_today():
    """
    Build (or fetch cached) outfit for the Today tab.

    Cache key: (user_id, today's date in user's locale, inferred occasion).
    A second request the same day with the same occasion returns the cached
    fit. A different inferred occasion (lunch meeting → dinner date)
    generates a new fit.

    Expected payload:
    {
      "user_id": "uuid" | null,            # required for cache; without it, no cache
      "profile": { ...StyleProfile },      # required for sane fallback
      "today_context": {
        "lat": float | null,
        "lon": float | null,
        "temp_f": float | null,
        "condition": "sunny" | "rainy" | ... | null,
        "daypart": "morning" | "afternoon" | "evening" | null,
        "primary_event": {                 # null if calendar denied / empty
          "title": str,
          "category": str | null,
          "all_day": bool | null
        } | null,
        "local_date": "YYYY-MM-DD" | null  # iOS-supplied for cache key
      }
    }
    """
    from datetime import date as _date

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    user_id = data.get("user_id")
    profile = data.get("profile") or {}
    today_context = data.get("today_context") or {}
    local_date_str = today_context.get("local_date")

    builder = get_outfit_builder()

    # Resolve params first so we know the inferred occasion (cache key component)
    params = builder.build_params(profile=profile, today_context=today_context)
    occasion = params.occasion
    cache_date = local_date_str or _date.today().isoformat()

    logger.info(
        "/outfit/build/today user=%s date=%s occasion=%s context_keys=%s",
        _short_user_id(user_id),
        cache_date,
        occasion,
        list(today_context.keys()),
    )

    fit_service = get_fit_image_service()
    supabase_client = fit_service.supabase

    # ----- Cache lookup -----
    if user_id:
        try:
            cached = (
                supabase_client.table("finer_daily_fits")
                .select("fit_id, image_urls, generated_at")
                .eq("user_id", user_id)
                .eq("date", cache_date)
                .eq("occasion", occasion)
                .limit(1)
                .execute()
            )
            if cached.data:
                fit_id = cached.data[0]["fit_id"]
                cached_image_urls = cached.data[0].get("image_urls") or []
                fit_row = (
                    supabase_client.table("user_fits")
                    .select("*")
                    .eq("id", fit_id)
                    .limit(1)
                    .execute()
                )
                if fit_row.data:
                    cached_fit = fit_row.data[0]
                    logger.info(
                        "/outfit/build/today cache hit user=%s fit_id=%s images=%s",
                        _short_user_id(user_id),
                        fit_id,
                        len(cached_image_urls),
                    )
                    return jsonify(
                        {
                            "success": True,
                            "cached": True,
                            "fit_id": fit_id,
                            "items": cached_fit.get("items"),
                            "title": cached_fit.get("title"),
                            "occasion": occasion,
                            "image_urls": cached_image_urls,
                        }
                    )
        except Exception as exc:
            logger.warning("/outfit/build/today cache lookup failed: %s", exc)

    # ----- Cache miss → generate -----
    try:
        outfit = builder.build_for_today(profile=profile, today_context=today_context)
    except Exception as exc:
        logger.exception(
            "/outfit/build/today build failed user=%s error=%s",
            _short_user_id(user_id),
            exc,
        )
        return jsonify({"success": False, "error": str(exc)}), 500

    items_dict = outfit.get("items") or {}
    has_items = any(v is not None for v in items_dict.values())
    if not has_items:
        return (
            jsonify({"success": False, "error": "No products found for today's context"}),
            404,
        )

    # Persist as a regular fit so it appears in the user's Fits tab.
    saved_fit = None
    fit_items_payload = []
    for slot, product in items_dict.items():
        if not product:
            continue
        fit_items_payload.append(
            {
                "id": str(product.get("product_id") or ""),
                "slot": "shoes" if slot == "footwear" else slot,
                "name": product.get("product_title") or "Untitled",
                "brand": "FinerFit",
                "price": float(product.get("product_price_amount") or 0),
                "imageUrl": product.get("product_img_link"),
                "link": product.get("product_link"),
            }
        )

    if user_id and fit_items_payload:
        try:
            saved_fit = fit_service.save_fit(
                user_id=user_id,
                title=f"Today's {occasion.replace('-', ' ').title()} Look",
                tags=[occasion],
                source="today",
                items=fit_items_payload,
            )
        except Exception as exc:
            logger.warning("/outfit/build/today fit persistence failed: %s", exc)

        # Cache row only if persistence succeeded (we need the fit_id)
        if saved_fit and saved_fit.get("id"):
            try:
                supabase_client.table("finer_daily_fits").upsert(
                    {
                        "user_id": user_id,
                        "date": cache_date,
                        "occasion": occasion,
                        "fit_id": saved_fit["id"],
                        "context_snapshot": today_context,
                    },
                    on_conflict="user_id,date,occasion",
                ).execute()
            except Exception as exc:
                logger.warning("/outfit/build/today cache write failed: %s", exc)

    response_payload = {
        "success": True,
        "cached": False,
        "fit_id": saved_fit.get("id") if saved_fit else None,
        "items": items_dict,
        "total_price": outfit.get("total_price", 0.0),
        "occasion": occasion,
        # Empty array on first build — iOS interprets this as "trigger image
        # generation, then call us back via /fits/generate-images". We'll
        # write the resulting URLs into finer_daily_fits.image_urls so the
        # next /outfit/build/today call this day returns them inline.
        "image_urls": [],
    }
    logger.info(
        "/outfit/build/today success user=%s fit_id=%s slots=%s",
        _short_user_id(user_id),
        response_payload["fit_id"],
        list(items_dict.keys()),
    )
    return jsonify(response_payload)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )
