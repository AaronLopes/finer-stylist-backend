#!/usr/bin/env python3
"""Fit image generation + persistence service."""

import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import supabase

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FIT_IMAGES_BUCKET = os.getenv("FIT_IMAGES_BUCKET", "fit-images-private")
SIGNED_URL_TTL_SECONDS = int(os.getenv("FIT_SIGNED_URL_TTL_SECONDS", "604800"))


class FitImageService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required")

        self.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

    def create_fit_with_images(
        self,
        user_id: str,
        title: Optional[str],
        tags: List[str],
        source: str,
        items: List[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
        count: int = 3,
    ) -> Dict[str, Any]:
        slots = self._extract_slots(items)
        generated_images = self._generate_images(
            top_image=slots.get("top") or "",
            bottom_image=slots.get("bottom") or "",
            shoes_image=slots.get("shoes") or "",
            profile=profile or {},
            count=count,
        )

        fit_payload = {
            "user_id": user_id,
            "title": title,
            "source": source,
            "tags": tags,
            "items": items,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        fit_insert = self.supabase.table("user_fits").insert(fit_payload).execute()
        fit_data = fit_insert.data[0]
        fit_id = fit_data["id"]

        image_rows: List[Dict[str, Any]] = []
        image_urls: List[str] = []

        for idx, image_bytes in enumerate(generated_images):
            image_path = f"{user_id}/{fit_id}/{idx + 1}.png"
            self._upload_to_storage(image_path, image_bytes)
            signed_url = self._create_signed_url(image_path)

            image_rows.append(
                {
                    "fit_id": fit_id,
                    "user_id": user_id,
                    "image_path": image_path,
                    "position": idx,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
            image_urls.append(signed_url)

        if image_rows:
            self.supabase.table("user_fit_images").insert(image_rows).execute()

        return {
            "fit": {
                "id": fit_id,
                "user_id": user_id,
                "title": title,
                "source": source,
                "tags": tags,
                "items": items,
                "created_at": fit_data.get("created_at"),
            },
            "image_urls": image_urls,
        }

    def list_fits(self, user_id: str) -> List[Dict[str, Any]]:
        fits_resp = (
            self.supabase.table("user_fits")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        fits = fits_resp.data or []
        if not fits:
            return []

        fit_ids = [fit["id"] for fit in fits]
        images_resp = (
            self.supabase.table("user_fit_images")
            .select("fit_id,image_path,position")
            .in_("fit_id", fit_ids)
            .order("position")
            .execute()
        )

        images_by_fit: Dict[str, List[Dict[str, Any]]] = {}
        for row in images_resp.data or []:
            fit_id = row["fit_id"]
            images_by_fit.setdefault(fit_id, []).append(row)

        normalized: List[Dict[str, Any]] = []
        for fit in fits:
            fit_images = images_by_fit.get(fit["id"], [])
            image_urls = [self._create_signed_url(img["image_path"]) for img in fit_images]
            normalized.append(
                {
                    "id": fit["id"],
                    "user_id": fit["user_id"],
                    "title": fit.get("title"),
                    "source": fit.get("source", "stylist_chat"),
                    "tags": fit.get("tags") or [],
                    "items": fit.get("items") or [],
                    "created_at": fit.get("created_at"),
                    "image_urls": image_urls,
                }
            )

        return normalized

    def _extract_slots(self, items: List[Dict[str, Any]]) -> Dict[str, str]:
        slots: Dict[str, str] = {}
        for item in items:
            slot = (item.get("slot") or "").lower()
            image_url = item.get("imageUrl") or item.get("image_url")
            if not slot or not image_url:
                continue
            if slot == "footwear":
                slot = "shoes"
            slots[slot] = image_url
        return slots

    def _generate_images(
        self,
        top_image: str,
        bottom_image: str,
        shoes_image: str,
        profile: Dict[str, Any],
        count: int,
    ) -> List[bytes]:
        available = [url for url in [top_image, bottom_image, shoes_image] if url]
        if not available:
            raise ValueError("Need at least one image URL to generate fit images")

        image_parts = []
        for url in [top_image, bottom_image, shoes_image]:
            if not url:
                continue
            inline = self._convert_image_to_inline_data(url)
            if inline:
                image_parts.append({"inlineData": inline})

        if not image_parts:
            raise ValueError("Could not load any item images for generation")

        generated: List[bytes] = []
        gender = profile.get("gender") or ""
        style = profile.get("style") or ""
        occasion = profile.get("occasion") or ""

        for variation in range(max(1, count)):
            prompt = self._build_prompt(gender, style, occasion, variation + 1)
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}, *image_parts],
                    }
                ]
            }
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=90,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Gemini request failed ({response.status_code}): {response.text[:250]}"
                )

            result = response.json()
            image_bytes = self._extract_image_bytes(result)
            if image_bytes:
                generated.append(image_bytes)

        if not generated:
            raise RuntimeError("Gemini did not return any images")

        return generated

    def _convert_image_to_inline_data(self, image_url: str) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(image_url, timeout=20)
            if not response.ok:
                logger.warning("Image fetch failed (%s): %s", response.status_code, image_url)
                return None
            mime_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
            return {
                "mimeType": mime_type,
                "data": base64.b64encode(response.content).decode("utf-8"),
            }
        except Exception as exc:
            logger.warning("Image conversion failed for %s: %s", image_url, exc)
            return None

    def _extract_image_bytes(self, gemini_result: Dict[str, Any]) -> Optional[bytes]:
        candidates = gemini_result.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        return None

    def _upload_to_storage(self, image_path: str, image_bytes: bytes) -> None:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{FIT_IMAGES_BUCKET}/{image_path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "x-upsert": "true",
            "Content-Type": "image/png",
        }
        response = requests.post(upload_url, headers=headers, data=image_bytes, timeout=30)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Storage upload failed ({response.status_code}): {response.text[:250]}"
            )

    def _create_signed_url(self, image_path: str) -> str:
        sign_url = f"{SUPABASE_URL}/storage/v1/object/sign/{FIT_IMAGES_BUCKET}/{image_path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json",
        }
        response = requests.post(
            sign_url,
            headers=headers,
            json={"expiresIn": SIGNED_URL_TTL_SECONDS},
            timeout=15,
        )
        if not response.ok:
            raise RuntimeError(
                f"Could not create signed URL ({response.status_code}): {response.text[:250]}"
            )

        signed_path = response.json().get("signedURL", "")
        if not signed_path:
            raise RuntimeError("Signed URL response missing signedURL")

        if signed_path.startswith("http://") or signed_path.startswith("https://"):
            return signed_path

        return f"{SUPABASE_URL}/storage/v1{signed_path}"

    def _build_prompt(self, gender: str, style: str, occasion: str, variation: int) -> str:
        return (
            "Create a photorealistic full-body fashion model image featuring the outfit items provided in the reference images. "
            "Preserve garment details, silhouettes, and colors accurately. "
            f"Gender expression: {gender or 'unspecified'}. "
            f"Style: {style or 'modern'}. "
            f"Occasion: {occasion or 'everyday'}. "
            f"Variation {variation}: use a different model pose and background while keeping the same outfit pieces. "
            "Output only one image."
        )
