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
    GET  /products/search      - Semantic product search (iOS Search tab)
    GET  /products/<id>        - Single product detail + sustainability score
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables before importing service modules, since some of
# them (e.g. fit_image_service) read env vars at module-import time.
load_dotenv()

from chat_service import get_chat_composer, get_chat_resolver
from fit_image_service import FitImageService
from outfit_builder import OutfitBuilder, _rpc_gender, _stable_hash, create_outfit_builder
from product_search_service import ProductSearchService
from push_service import get_push_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# CORS for web and mobile clients
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize outfit builder (lazy loading)
_outfit_builder: Optional[OutfitBuilder] = None
_fit_image_service: Optional[FitImageService] = None
_product_search_service: Optional[ProductSearchService] = None

MAX_SEARCH_LIMIT = 50


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


def get_product_search_service() -> ProductSearchService:
    global _product_search_service
    if _product_search_service is None:
        _product_search_service = ProductSearchService()
    return _product_search_service


_supabase_client = None


def get_supabase_client():
    """Process-wide service-role Supabase client (RLS-bypassing). Used by the
    device-token registry and broadcast endpoints."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required.")
        _supabase_client = create_client(url, key)
    return _supabase_client


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


@app.route("/persona/resolve", methods=["POST"])
def resolve_persona():
    """Get-or-create a shared style persona image for the post-quiz reveal.

    Keyed on a coarse combination (gender|style|occasion|setting) so images are
    reused across users: cache hits return instantly, misses build an outfit +
    render one full-body model image, then cache it. No auth required — this
    runs during onboarding before the user has an account.

    Expected payload: the quiz answers ({gender, occasion, weather, setting,
    goals, style, budget}). Response:
    {
      "success": true,
      "cached": true|false,
      "persona_key": "masculine|classic|work|city",
      "title": "The Classic",
      "subtitle": null,
      "image_url": "https://.../storage/v1/object/public/personas/....png",
      "items": [{ id, slot, name, price, imageUrl, link }, ...]
    }
    """
    data = request.get_json()
    if not data:
        logger.warning("/persona/resolve missing JSON payload")
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    valid, error = validate_quiz_answers(data)
    if not valid:
        logger.warning("/persona/resolve validation failed: %s", error)
        return jsonify({"success": False, "error": error}), 400

    service = get_fit_image_service()
    key = service.persona_key(data)

    # Fast path: cache hit — no outfit build, no Gemini call.
    try:
        cached = service.get_persona(key)
    except Exception as e:
        logger.exception("Persona lookup failed key=%s error=%s", key, e)
        cached = None

    if cached:
        logger.info("/persona/resolve HIT key=%s", key)
        return jsonify({"success": True, "cached": True, **cached})

    logger.info("/persona/resolve MISS key=%s — building + rendering", key)
    try:
        builder = get_outfit_builder()
        outfit = builder.build_from_quiz(data)
        items = outfit.get("items", {}) or {}
        if not any(v is not None for v in items.values()):
            return (
                jsonify({"success": False, "error": "Could not build an outfit"}),
                502,
            )
        persona = service.create_persona(key, data, items)
        logger.info("/persona/resolve CREATED key=%s", key)
        return jsonify({"success": True, "cached": False, **persona})
    except Exception as e:
        logger.exception("Persona create failed key=%s error=%s", key, e)
        return jsonify({"success": False, "error": str(e)}), 500


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
    existing_fit_id = data.get("fit_id")
    # Optional Precision Fit likeness reference (Pro). When present, the model is
    # rendered with the user's own face instead of a generic model.
    selfie_url = data.get("selfie_url") or None
    # Resolved occasion for the image prompt. Today fits pass this explicitly
    # because their occasion (calendar/rotation) can differ from the stored
    # profile occasion.
    occasion = data.get("occasion") or (
        profile.get("occasion") if isinstance(profile, dict) else None
    )

    try:
        if existing_fit_id:
            # Update path — used so Today's already-persisted fit doesn't
            # spawn a duplicate row when iOS auto-triggers image gen.
            result = service.update_fit_with_images(
                fit_id=existing_fit_id,
                items=items,
                profile=profile,
                count=int(data.get("count") or 3),
                selfie_url=selfie_url,
                occasion=occasion,
            )
        else:
            result = service.create_fit_with_images(
                user_id=user_id,
                title=data.get("title"),
                tags=tags,
                source=source,
                items=items,
                profile=profile,
                count=int(data.get("count") or 3),
                selfie_url=selfie_url,
                occasion=occasion,
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


@app.route("/products/search", methods=["GET"])
def search_products():
    """
    Semantic product search over the catalog.

    Backs the iOS Search tab (ProductSearching protocol — see
    finer-ios docs/backend-blanks.md). Embeds the query with fashionSigLIP
    and ranks by pgvector cosine similarity via the match_products_semantic
    RPC.

    Query params:
      - q (required): natural-language search text
      - limit (optional, default 20, capped at 50)
      - gender (optional): profile gender; masculine/feminine restrict
        results to that label + genderless, anything else means no filter
        (same rule as the outfit RPCs, migrations/009_gender_filtering.sql)
    """
    query = (request.args.get("q") or "").strip()
    if not query:
        logger.warning("/products/search missing q param")
        return (
            jsonify(
                {
                    "success": False,
                    "results": [],
                    "error": "Missing required query param: q",
                }
            ),
            400,
        )

    limit_raw = request.args.get("limit", "20")
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "success": False,
                    "results": [],
                    "error": f"Invalid limit: {limit_raw}",
                }
            ),
            400,
        )
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    gender = _rpc_gender(request.args.get("gender"))

    logger.info(
        "/products/search q_len=%s limit=%s gender=%s", len(query), limit, gender
    )

    try:
        service = get_product_search_service()
        results = service.search(query, limit=limit, gender=gender)
        logger.info("/products/search success result_count=%s", len(results))
        return jsonify({"success": True, "results": results, "error": None})
    except Exception as e:
        logger.exception("/products/search failed q_len=%s error=%s", len(query), e)
        return jsonify({"success": False, "results": [], "error": str(e)}), 500


@app.route("/products/<product_id>", methods=["GET"])
def get_product_detail(product_id: str):
    """
    Single-product detail lookup, including the sustainability scoring
    columns (s-mvp-v2). Backs the iOS ProductDetailView sheet — scores only
    render there, so the client fetches this lazily per product instead of
    the outfit/search payloads carrying score data.

    Response `product` matches the outfit item serialization, plus:
      - material: raw product_material text (nullable)
      - sustainability:
          score              0-100 float, null when withheld
          grade              A-E letter, null when withheld
          data_completeness  0-1 confidence in the inputs
          score_version      e.g. "s-mvp-v2"
          breakdown          full score_breakdown jsonb — the substantiation
                             the UI must surface next to any score (fibers,
                             certs, transport, sources). For withheld scores
                             it carries only {reason, fiber_source,
                             score_version}.
    """
    pid = (product_id or "").strip()
    if not pid:
        return jsonify({"success": False, "product": None, "error": "Missing product id"}), 400

    try:
        client = get_supabase_client()
        result = (
            client.table("finer_products_omega")
            .select(
                "product_id,product_title,product_brand,product_img_link,"
                "product_link,product_price_amount,product_slot,"
                "product_description,product_material,product_s_score,"
                "s_score_grade,score_data_completeness,score_version,"
                "score_breakdown"
            )
            .eq("product_id", pid)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return jsonify({"success": False, "product": None, "error": "Product not found"}), 404

        row = rows[0]
        price = row.get("product_price_amount")
        s_score = row.get("product_s_score")
        product = {
            "product_id": row.get("product_id"),
            "product_title": row.get("product_title"),
            "product_brand": row.get("product_brand"),
            "product_price_amount": float(price) if price is not None else None,
            "product_img_link": row.get("product_img_link"),
            "product_link": row.get("product_link"),
            "slot": row.get("product_slot"),
            "description": row.get("product_description"),
            "material": row.get("product_material"),
            "sustainability": {
                "score": float(s_score) if s_score is not None else None,
                "grade": row.get("s_score_grade"),
                "data_completeness": row.get("score_data_completeness"),
                "score_version": row.get("score_version"),
                "breakdown": row.get("score_breakdown"),
            },
        }
        logger.info(
            "/products/<id> success product_id=%s scored=%s",
            pid,
            s_score is not None,
        )
        return jsonify({"success": True, "product": product, "error": None})
    except Exception as e:
        logger.exception("/products/<id> failed product_id=%s error=%s", pid, e)
        return jsonify({"success": False, "product": None, "error": str(e)}), 500


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


def _persisted_items_to_api_dict(persisted_items: Any) -> Dict[str, Dict[str, Any]]:
    """Convert the persisted user_fits.items list back to the {slot: product_dict}
    shape that /outfit/build/today returns on cache miss. Without this, cache
    HIT and cache MISS responses had different shapes — iOS decoded the cache
    hit as empty and rendered the Today empty state.
    """
    if not isinstance(persisted_items, list):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for item in persisted_items:
        if not isinstance(item, dict):
            continue
        slot = item.get("slot") or "unknown"
        # Persisted format normalizes "footwear" to "shoes"; reverse for the
        # API's slot-keyed dict that the rest of the response uses.
        api_slot = "footwear" if slot == "shoes" else slot
        result[api_slot] = {
            "product_id": item.get("id"),
            "product_title": item.get("name"),
            "product_brand": item.get("brand"),
            "product_price_amount": item.get("price"),
            "product_img_link": item.get("imageUrl"),
            "product_link": item.get("link"),
        }
    return result


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
        "temp_f": float | null,             # fallback: "moderate" weather
        "condition": "sunny" | "rainy" | ... | null,
        "daypart": "morning" | "afternoon" | "evening" | null,
        "city_label": str | null,           # e.g. "Brooklyn, NY" → setting inference; fallback: "city"
        "primary_event": {                  # null → day-aware occasion rotation
          "title": str,
          "category": str | null,
          "all_day": bool | null
        } | null,
        "local_date": "YYYY-MM-DD" | null   # iOS-supplied for cache key + rotation
      }
    }

    With no calendar event, the occasion is no longer the stored quiz answer:
    it rotates per (user, day) between day-appropriate occasions (see
    outfit_builder._rotating_fallback_occasion). Item selection on cache miss
    picks from the top candidates per slot with a per-(user, day) seeded RNG;
    same-day stability comes from the finer_daily_fits cache (the slot RPCs
    are themselves randomized).
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
    params = builder.build_params(
        profile=profile, today_context=today_context, user_id=user_id
    )
    occasion = params.occasion
    cache_date = local_date_str or _date.today().isoformat()
    variance_seed = _stable_hash(f"{user_id or 'anon'}:{cache_date}")

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
                            # Convert persisted list back to {slot: product_dict}
                            # so iOS gets the same shape as on cache miss.
                            "items": _persisted_items_to_api_dict(cached_fit.get("items")),
                            "title": cached_fit.get("title"),
                            "occasion": occasion,
                            "image_urls": cached_image_urls,
                        }
                    )
        except Exception as exc:
            logger.warning("/outfit/build/today cache lookup failed: %s", exc)

    # ----- Cache miss → generate -----
    try:
        outfit = builder.build_for_today(
            profile=profile,
            today_context=today_context,
            user_id=user_id,
            variance_seed=variance_seed,
        )
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
                "brand": product.get("product_brand") or "FinerFit",
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
# PUSH NOTIFICATIONS
# =============================================================================


@app.route("/devices/register", methods=["POST"])
def register_device():
    """
    Register (upsert) an APNs device token so it can receive pushes.

    Called by the iOS app after the user grants notification permission and on
    subsequent launches (tokens can rotate). user_id is optional — guest
    devices still receive broadcasts.

    Expected payload:
    {
      "device_token": "hex-encoded APNs token",   # required
      "user_id": "uuid",                           # optional (null for guests)
      "environment": "production" | "sandbox",     # optional, default production
      "app_version": "1.2.0",                      # optional
      "platform": "ios"                            # optional, default ios
    }
    """
    data = request.get_json() or {}

    token = (data.get("device_token") or "").strip()
    if not token:
        return (
            jsonify({"success": False, "error": "Missing required field: device_token"}),
            400,
        )

    environment = data.get("environment") or "production"
    if environment not in ("production", "sandbox"):
        environment = "production"

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "device_token": token,
        "user_id": data.get("user_id"),
        "platform": data.get("platform") or "ios",
        "environment": environment,
        "app_version": data.get("app_version"),
        "is_active": True,
        "updated_at": now,
        "last_registered_at": now,
    }

    try:
        get_supabase_client().table("device_tokens").upsert(
            row, on_conflict="device_token"
        ).execute()
    except Exception as exc:
        logger.exception("Device register failed token=%s err=%s", token[:8], exc)
        return jsonify({"success": False, "error": str(exc)}), 500

    logger.info(
        "/devices/register token=%s user=%s env=%s",
        token[:8],
        _short_user_id(data.get("user_id")),
        environment,
    )
    return jsonify({"success": True})


def _require_admin() -> bool:
    """Fail-closed admin gate for broadcast. Returns True if authorized."""
    expected = os.getenv("ADMIN_PUSH_TOKEN")
    if not expected:
        return False
    return request.headers.get("X-Admin-Token") == expected


@app.route("/admin/push/broadcast", methods=["POST"])
def push_broadcast():
    """
    Send an announcement push to registered devices. Admin-only.

    Auth: header `X-Admin-Token: <ADMIN_PUSH_TOKEN>`.

    Expected payload:
    {
      "title": "The new update is here",       # required
      "body": "We refreshed the catalog ✨",    # required
      "subtitle": "optional",
      "user_ids": ["uuid", ...],   # optional: target these users' devices only
      "device_tokens": ["...."],   # optional: target these exact tokens only
      "data": { "route": "discover" }  # optional custom payload keys
    }

    With neither user_ids nor device_tokens, sends to ALL active devices.
    """
    if not _require_admin():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if not title or not body:
        return (
            jsonify({"success": False, "error": "Missing required field: title/body"}),
            400,
        )

    supabase = get_supabase_client()
    try:
        query = (
            supabase.table("device_tokens")
            .select("device_token,environment")
            .eq("is_active", True)
        )
        if data.get("device_tokens"):
            query = query.in_("device_token", data["device_tokens"])
        elif data.get("user_ids"):
            query = query.in_("user_id", data["user_ids"])
        rows = query.execute().data or []
    except Exception as exc:
        logger.exception("Broadcast token fetch failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

    if not rows:
        return jsonify({"success": True, "sent": 0, "failed": 0, "deactivated": 0})

    results = get_push_service().send_many(
        rows,
        title=title,
        body=body,
        subtitle=data.get("subtitle"),
        data=data.get("data"),
    )

    sent = sum(1 for r in results if r.ok)
    dead = [r.device_token for r in results if r.should_deactivate]

    # Prune tokens APNs told us are permanently invalid.
    if dead:
        try:
            now = datetime.now(timezone.utc).isoformat()
            supabase.table("device_tokens").update(
                {"is_active": False, "updated_at": now}
            ).in_("device_token", dead).execute()
        except Exception as exc:
            logger.warning("Failed to deactivate dead tokens: %s", exc)

    logger.info(
        "/admin/push/broadcast targeted=%s sent=%s failed=%s deactivated=%s",
        len(rows),
        sent,
        len(results) - sent,
        len(dead),
    )
    return jsonify(
        {
            "success": True,
            "targeted": len(rows),
            "sent": sent,
            "failed": len(results) - sent,
            "deactivated": len(dead),
        }
    )


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
