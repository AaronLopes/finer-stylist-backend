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


def get_outfit_builder() -> OutfitBuilder:
    global _outfit_builder
    if _outfit_builder is None:
        _outfit_builder = create_outfit_builder()
    return _outfit_builder


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
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    mode = data.get("mode")

    if mode not in ["quiz", "chat"]:
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
                return jsonify({"success": False, "error": error}), 400

            outfit = builder.build_from_quiz(quiz_answers)

        else:  # chat mode
            chat_query = data.get("chat_query")
            if not chat_query:
                return (
                    jsonify(
                        {"success": False, "error": "chat_query required for chat mode"}
                    ),
                    400,
                )

            valid, error = validate_chat_query(chat_query)
            if not valid:
                return jsonify({"success": False, "error": error}), 400

            outfit = builder.build_from_chat(
                query=chat_query.get("query"),
                user_profile=chat_query.get("user_profile"),
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
        logger.error("Outfit build failed: %s", e)
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
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

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
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    # Wrap in the unified format
    wrapped = {
        "mode": "quiz",
        "quiz_answers": data,
    }

    # Temporarily replace request json
    with app.test_request_context(json=wrapped):
        return build_outfit()


@app.route("/outfit/build/chat", methods=["POST"])
def build_outfit_from_chat():
    """
    Convenience endpoint for chat mode.
    Equivalent to POST /api/outfit/build with mode="chat"
    """
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "error": "No JSON payload provided"}), 400

    # Wrap in the unified format
    wrapped = {
        "mode": "chat",
        "chat_query": data,
    }

    # Temporarily replace request json
    with app.test_request_context(json=wrapped):
        return build_outfit()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )
