"""Gemini-backed material analysis for the Finer Material Scanner.

The first pass intentionally identifies coarse material families and likely
fibers. It does not invent blend percentages from an RGB photograph.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

MATERIAL_FAMILIES = {
    "natural_cellulosic",
    "animal_protein",
    "regenerated_cellulosic",
    "synthetic",
    "leather_or_suede",
    "blended",
    "unknown",
}
EVIDENCE_SOURCES = {"visual_estimate", "visual_and_metadata", "label_text"}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "garment_type": {"type": "string"},
        "material_family": {
            "type": "string",
            "enum": sorted(MATERIAL_FAMILIES),
        },
        "likely_materials": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence_score": {"type": "number"},
                },
                "required": ["name", "confidence_score"],
            },
        },
        "fabric_construction": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "confidence_score": {"type": "number"},
        "summary": {"type": "string"},
        "evidence_source": {
            "type": "string",
            "enum": sorted(EVIDENCE_SOURCES),
        },
        "needs_more_input": {"type": "boolean"},
        "suggested_next_input": {"type": "string"},
    },
    "required": [
        "garment_type",
        "material_family",
        "likely_materials",
        "fabric_construction",
        "confidence",
        "confidence_score",
        "summary",
        "evidence_source",
        "needs_more_input",
        "suggested_next_input",
    ],
}

SYSTEM_PROMPT = """You are Finer's Material Scanner.

Analyze the garment and identify its BASIC material family and up to three
likely fibers. Keep fiber identity separate from fabric construction: cotton,
linen, wool, silk, viscose, polyester, nylon, and similar terms are fibers;
denim, jersey, satin, fleece, twill, knit, and woven describe construction.

Rules:
- Never invent exact fiber percentages from visual appearance.
- Treat brand alone as weak context. It must not override visible evidence.
- Product name and style code are useful context, but only care-label text is
  direct evidence of composition.
- Use "unknown" and request more input when cotton/linen/viscose/polyester or
  wool/acrylic blends cannot be distinguished reliably.
- Reserve high confidence for distinctive evidence or readable label text.
- Prefer the next input in this order: fabric close-up, garment type, product
  name/style code, then a care-label photo or recognized label text.
- Keep the summary under two short sentences and avoid sustainability or
  quality claims.
"""


class MaterialScanError(RuntimeError):
    """Raised when Gemini cannot produce a usable material scan."""


class MaterialScanService:
    """Analyze garment photos with Gemini 3.6 Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout_seconds: int = 45,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Material Scanner.")
        self.model = model or os.getenv("MATERIAL_SCANNER_MODEL", DEFAULT_MODEL)
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def analyze(
        self,
        image: Tuple[bytes, str],
        *,
        texture_image: Optional[Tuple[bytes, str]] = None,
        brand: Optional[str] = None,
        garment_type: Optional[str] = None,
        product_name: Optional[str] = None,
        style_code: Optional[str] = None,
        label_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._context_text(
            brand=brand,
            garment_type=garment_type,
            product_name=product_name,
            style_code=style_code,
            label_text=label_text,
            has_texture_image=texture_image is not None,
        )
        inputs: List[Dict[str, Any]] = [
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{context}"},
            self._image_part(*image, resolution="medium"),
        ]
        if texture_image is not None:
            inputs.append(
                {
                    "type": "text",
                    "text": "The next image is a close-up of the same garment's fabric.",
                }
            )
            inputs.append(self._image_part(*texture_image, resolution="high"))

        payload = {
            "model": self.model,
            "input": inputs,
            "store": False,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": RESPONSE_SCHEMA,
            },
            "generation_config": {"thinking_level": "minimal"},
        }

        started = time.monotonic()
        try:
            response = self.session.post(
                INTERACTIONS_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise MaterialScanError("Gemini Material Scanner request failed.") from exc

        elapsed_ms = round((time.monotonic() - started) * 1000)
        if not response.ok:
            logger.warning(
                "Material Scanner Gemini failure model=%s status=%s elapsed_ms=%s body=%s",
                self.model,
                response.status_code,
                elapsed_ms,
                response.text[:300],
            )
            raise MaterialScanError(
                f"Gemini Material Scanner returned HTTP {response.status_code}."
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise MaterialScanError(
                "Gemini Material Scanner returned invalid JSON."
            ) from exc

        if result.get("status") not in (None, "completed"):
            raise MaterialScanError(
                f"Gemini Material Scanner ended with status {result.get('status')}."
            )

        scan = self._parse_scan(self._extract_output_text(result))
        usage = result.get("usage") or {}
        logger.info(
            "Material Scanner completed model=%s elapsed_ms=%s input_tokens=%s "
            "output_tokens=%s thought_tokens=%s confidence=%s needs_more_input=%s",
            self.model,
            elapsed_ms,
            usage.get("total_input_tokens"),
            usage.get("total_output_tokens"),
            usage.get("total_thought_tokens"),
            scan["confidence"],
            scan["needs_more_input"],
        )
        return scan

    @staticmethod
    def _image_part(
        image_bytes: bytes, mime_type: str, *, resolution: str
    ) -> Dict[str, str]:
        return {
            "type": "image",
            "data": base64.b64encode(image_bytes).decode("ascii"),
            "mime_type": mime_type,
            "resolution": resolution,
        }

    @staticmethod
    def _context_text(
        *,
        brand: Optional[str],
        garment_type: Optional[str],
        product_name: Optional[str],
        style_code: Optional[str],
        label_text: Optional[str],
        has_texture_image: bool,
    ) -> str:
        values = {
            "Brand": brand,
            "User-provided garment type": garment_type,
            "Product name": product_name,
            "Style code": style_code,
            "Recognized care-label text": label_text,
        }
        lines = ["Optional context supplied by the user:"]
        populated = False
        for label, value in values.items():
            if value:
                lines.append(f"- {label}: {value}")
                populated = True
        if not populated:
            lines.append("- None")
        lines.append(f"- Fabric close-up supplied: {'yes' if has_texture_image else 'no'}")
        return "\n".join(lines)

    @staticmethod
    def _extract_output_text(result: Dict[str, Any]) -> str:
        direct = result.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct

        for step in reversed(result.get("steps") or []):
            if step.get("type") != "model_output":
                continue
            texts = [
                content.get("text", "")
                for content in step.get("content") or []
                if content.get("type") == "text"
            ]
            combined = "".join(texts).strip()
            if combined:
                return combined

        raise MaterialScanError("Gemini Material Scanner returned no text output.")

    @classmethod
    def _parse_scan(cls, output_text: str) -> Dict[str, Any]:
        cleaned = output_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
        try:
            raw = json.loads(cleaned)
        except (TypeError, ValueError) as exc:
            raise MaterialScanError(
                "Gemini Material Scanner returned malformed structured output."
            ) from exc
        if not isinstance(raw, dict):
            raise MaterialScanError(
                "Gemini Material Scanner returned an invalid response shape."
            )
        return cls._normalize_scan(raw)

    @staticmethod
    def _normalize_scan(raw: Dict[str, Any]) -> Dict[str, Any]:
        family = str(raw.get("material_family") or "unknown").strip().lower()
        if family not in MATERIAL_FAMILIES:
            family = "unknown"

        evidence_source = str(
            raw.get("evidence_source") or "visual_estimate"
        ).strip().lower()
        if evidence_source not in EVIDENCE_SOURCES:
            evidence_source = "visual_estimate"

        try:
            confidence_score = float(raw.get("confidence_score", 0))
        except (TypeError, ValueError):
            confidence_score = 0.0
        confidence_score = round(max(0.0, min(1.0, confidence_score)), 3)
        if confidence_score >= 0.82:
            confidence = "high"
        elif confidence_score >= 0.58:
            confidence = "medium"
        else:
            confidence = "low"

        likely_materials = []
        for item in raw.get("likely_materials") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                item_confidence = float(item.get("confidence_score", 0))
            except (TypeError, ValueError):
                item_confidence = 0.0
            likely_materials.append(
                {
                    "name": name,
                    "confidence_score": round(
                        max(0.0, min(1.0, item_confidence)), 3
                    ),
                }
            )
            if len(likely_materials) == 3:
                break

        needs_more_input = bool(raw.get("needs_more_input"))
        if confidence == "low" or family == "unknown":
            needs_more_input = True

        return {
            "garment_type": str(raw.get("garment_type") or "unknown").strip()[:80],
            "material_family": family,
            "likely_materials": likely_materials,
            "fabric_construction": str(
                raw.get("fabric_construction") or "unknown"
            ).strip()[:80],
            "confidence": confidence,
            "confidence_score": confidence_score,
            "summary": str(raw.get("summary") or "").strip()[:320],
            "evidence_source": evidence_source,
            "needs_more_input": needs_more_input,
            "suggested_next_input": str(
                raw.get("suggested_next_input") or ""
            ).strip()[:160],
        }
