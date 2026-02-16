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

            outfit = builder.build_from_chat(
                query=query_text,
                user_profile=user_profile,
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

    try:
        result = service.create_fit_with_images(
            user_id=user_id,
            title=data.get("title"),
            tags=tags,
            source=data.get("source") or "stylist_chat",
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
