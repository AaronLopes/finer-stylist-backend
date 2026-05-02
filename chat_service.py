"""
Chat Response Composer for Finer Stylist

Generates natural, conversational stylist replies using a lightweight LLM call.
Falls back to template responses on error/timeout.

Usage:
    from chat_service import ChatResponseComposer

    composer = ChatResponseComposer()
    result = composer.compose_reply(
        query="NYC winter streetwear look",
        outfit_summary="Ontong Black, Venturi Alveomesh, Sorry I'm Late Hoodie",
        intent="new_query",
        user_profile={"gender": "masculine"},
    )
    # result = {"reply_text": "...", "follow_up_chips": [...], "cta_hint": "..."}
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
COMPOSER_TIMEOUT_S = float(os.getenv("COMPOSER_TIMEOUT_S", "3.0"))

SYSTEM_PROMPT = """You are a sharp, friendly personal stylist in a mobile chat app. You just built an outfit for the user.

Rules:
- Reference 1-2 item names from the outfit_summary, never invent items not listed
- Keep response under 50 words
- Match the user's energy (casual query → casual tone, formal → polished)
- End with ONE short follow-up question or suggestion
- If the outfit has items, be enthusiastic but not over the top
- If no items were found, empathize and suggest a clearer request
- Never discuss pricing unless the user mentioned budget
- Do NOT use emojis

You must output valid JSON only:
{"reply_text": "your reply here", "follow_up_chips": ["chip1", "chip2", "chip3"], "cta_hint": "generate_images|refine|new_query"}

follow_up_chips: 2-4 short actionable suggestions the user can tap (e.g. "Again", "More casual", "Cheaper", "Different shoes")
cta_hint: what the user should probably do next"""

INTENT_CONTEXT = {
    "new_query": "This is a fresh outfit request.",
    "regenerate": "The user asked for a new version of the same look.",
    "budget_refine": "The user wanted to adjust the budget/price.",
    "style_refine": "The user wanted to adjust the style direction.",
    "slot_swap": "The user wanted to swap a specific item.",
}

DEFAULT_CHIPS = ["Again", "More casual", "Dressier", "Cheaper"]


class ChatResponseComposer:
    """Generates natural stylist chat replies via a small LLM call."""

    def __init__(self):
        self._client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        if not self._client:
            logger.warning(
                "ChatResponseComposer: no OPENAI_API_KEY, will use templates only"
            )

    def compose_reply(
        self,
        query: str,
        outfit_summary: str,
        intent: str = "new_query",
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a natural stylist reply.

        Returns dict with keys: reply_text, follow_up_chips, cta_hint
        Falls back to template on any failure.
        """
        if not self._client:
            return self._template_reply(query, outfit_summary, intent)

        user_msg = self._build_user_message(query, outfit_summary, intent, user_profile)

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=120,
                temperature=0.8,
                timeout=COMPOSER_TIMEOUT_S,
            )

            raw = response.choices[0].message.content or "{}"
            elapsed = time.monotonic() - t0
            logger.info(
                "ChatResponseComposer completed in %.2fs model=%s tokens_out=%s",
                elapsed,
                CHAT_MODEL,
                response.usage.completion_tokens if response.usage else "?",
            )

            parsed = json.loads(raw)
            return {
                "reply_text": parsed.get("reply_text", "").strip()
                or self._template_text(query, outfit_summary, intent),
                "follow_up_chips": parsed.get("follow_up_chips", DEFAULT_CHIPS)[:4],
                "cta_hint": parsed.get("cta_hint", "generate_images"),
            }

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "ChatResponseComposer failed after %.2fs: %s — using template fallback",
                elapsed,
                exc,
            )
            return self._template_reply(query, outfit_summary, intent)

    def _build_user_message(
        self,
        query: str,
        outfit_summary: str,
        intent: str,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        parts = [
            f'User query: "{query[:400]}"',
            f"Outfit summary: {outfit_summary[:300]}",
            f"Context: {INTENT_CONTEXT.get(intent, 'New request.')}",
        ]
        if user_profile:
            gender = user_profile.get("gender", "")
            occasion = user_profile.get("occasion", "")
            if gender or occasion:
                parts.append(f"User: gender={gender}, occasion={occasion}")
        return "\n".join(parts)

    def _template_reply(
        self, query: str, outfit_summary: str, intent: str
    ) -> Dict[str, Any]:
        return {
            "reply_text": self._template_text(query, outfit_summary, intent),
            "follow_up_chips": DEFAULT_CHIPS,
            "cta_hint": "generate_images" if outfit_summary else "new_query",
        }

    @staticmethod
    def _template_text(query: str, outfit_summary: str, intent: str) -> str:
        if not outfit_summary:
            return (
                "I could not build a look just now. "
                'Try a more specific request like "smart-casual work fit under $200".'
            )
        pieces = outfit_summary.split(", ")[:2]
        summary = " and ".join(pieces)
        if intent == "regenerate":
            return f"Fresh take — here's a new look with {summary}. Want me to keep going or refine?"
        if intent in ("budget_refine", "style_refine"):
            return f"Refined it for you — {summary}. How's this feel?"
        return f'Built a look for "{query}" — {summary}. Want me to refine this?'


RESOLVER_SYSTEM_PROMPT = """You are a query intent resolver for a fashion stylist chat app.

The user sent a short message in the context of an ongoing outfit conversation.
You are given the user's current message and the last outfit query they made.

Determine what the user wants and rewrite their message into a standalone outfit query.

Output valid JSON only:
{"intent": "regenerate|refine_budget|refine_style|slot_swap|new_query|clarify", "resolved_query": "standalone outfit query to send to the builder", "confidence": 0.0-1.0}

- "regenerate": user wants a new version of the same look
- "refine_budget": user wants price adjustment
- "refine_style": user wants style/vibe adjustment
- "slot_swap": user wants to change a specific item slot
- "new_query": this is a completely new outfit request
- "clarify": you cannot determine what the user wants (set resolved_query to empty string)

If intent is "clarify", set confidence to 0.0."""

RESOLVER_TIMEOUT_S = float(os.getenv("RESOLVER_TIMEOUT_S", "2.5"))


class ChatIntentResolver:
    """Resolves ambiguous user messages into standalone outfit queries via a small LLM call."""

    def __init__(self):
        self._client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def resolve(
        self,
        user_message: str,
        last_query: str,
        last_outfit_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Resolve an ambiguous user message into intent + standalone query.

        Returns dict: intent, resolved_query, confidence
        """
        if not self._client:
            return {
                "intent": "new_query",
                "resolved_query": user_message,
                "confidence": 0.5,
            }

        user_msg = (
            f'User message: "{user_message[:300]}"\n'
            f'Last outfit query: "{last_query[:300]}"\n'
            f"Last outfit items: {last_outfit_summary[:200]}"
        )

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=60,
                temperature=0.2,
                timeout=RESOLVER_TIMEOUT_S,
            )

            raw = response.choices[0].message.content or "{}"
            elapsed = time.monotonic() - t0
            logger.info(
                "ChatIntentResolver completed in %.2fs model=%s",
                elapsed,
                CHAT_MODEL,
            )

            parsed = json.loads(raw)
            return {
                "intent": parsed.get("intent", "new_query"),
                "resolved_query": parsed.get("resolved_query", user_message),
                "confidence": float(parsed.get("confidence", 0.5)),
            }

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "ChatIntentResolver failed after %.2fs: %s — falling back to raw query",
                elapsed,
                exc,
            )
            return {
                "intent": "new_query",
                "resolved_query": user_message,
                "confidence": 0.5,
            }


# Singletons
_composer: Optional[ChatResponseComposer] = None
_resolver: Optional[ChatIntentResolver] = None


def get_chat_composer() -> ChatResponseComposer:
    global _composer
    if _composer is None:
        _composer = ChatResponseComposer()
    return _composer


def get_chat_resolver() -> ChatIntentResolver:
    global _resolver
    if _resolver is None:
        _resolver = ChatIntentResolver()
    return _resolver
