#!/usr/bin/env python3
"""Fit image generation + persistence service."""

import base64
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Public bucket for shared style personas (see migrations/006_personas.sql).
PERSONAS_BUCKET = os.getenv("PERSONAS_BUCKET", "personas")
# Private per-user bucket (selfies + personalized persona renders). Folders are
# keyed by client id; RLS is defined in the iOS repo's docs/sql/selfie-profile.sql.
SELFIES_BUCKET = os.getenv("SELFIES_BUCKET", "user-selfies")

# Curated display names per style aesthetic for the persona card.
_PERSONA_TITLES = {
    "minimalist": "The Minimalist",
    "classic": "The Classic",
    "fitted": "The Trendsetter",
    "smart-casual": "The Effortless",
    "bohemian": "The Free Spirit",
}


class FitImageService:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required")

        self.supabase = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

    def update_fit_with_images(
        self,
        fit_id: str,
        items: List[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
        count: int = 3,
        selfie_url: Optional[str] = None,
        occasion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate Gemini images for an EXISTING fit row and attach them.

        Used so a Today fit (already persisted via save_fit when /outfit/build/today
        ran) gets its outfit pictures appended to the same row instead of
        spawning a duplicate. The fit_id must already exist in user_fits.
        occasion overrides the profile occasion in the prompt — Today fits'
        resolved occasion (calendar/rotation) can differ from the stored one.
        """
        slots = self._extract_slots(items)
        generated_images = self._generate_images(
            top_image=slots.get("top") or "",
            bottom_image=slots.get("bottom") or "",
            shoes_image=slots.get("shoes") or "",
            profile=profile or {},
            count=count,
            selfie_url=selfie_url,
            occasion=occasion,
        )

        # Touch the fit's updated_at so the Fits list re-orders correctly.
        existing = (
            self.supabase.table("user_fits")
            .select("*")
            .eq("id", fit_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            raise RuntimeError(f"fit_id {fit_id} does not exist in user_fits")
        fit_row = existing.data[0]
        user_id = fit_row["user_id"]

        self.supabase.table("user_fits").update(
            {"updated_at": datetime.utcnow().isoformat()}
        ).eq("id", fit_id).execute()

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
                "title": fit_row.get("title"),
                "source": fit_row.get("source"),
                "tags": fit_row.get("tags") or [],
                "items": fit_row.get("items") or [],
                "created_at": fit_row.get("created_at"),
            },
            "image_urls": image_urls,
        }

    def save_fit(
        self,
        user_id: str,
        title: Optional[str],
        tags: List[str],
        source: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist a fit's metadata without generating images.

        Used so chat-built fits appear in the user's Fits tab immediately on
        creation; image generation can happen later (lazy / on-demand).
        """
        now = datetime.utcnow().isoformat()
        fit_payload = {
            "user_id": user_id,
            "title": title,
            "source": source,
            "tags": tags,
            "items": items,
            "created_at": now,
            "updated_at": now,
        }
        result = self.supabase.table("user_fits").insert(fit_payload).execute()
        fit_data = result.data[0]
        return {
            "id": fit_data["id"],
            "user_id": user_id,
            "title": title,
            "source": source,
            "tags": tags,
            "items": items,
            "created_at": fit_data.get("created_at"),
        }

    def create_fit_with_images(
        self,
        user_id: str,
        title: Optional[str],
        tags: List[str],
        source: str,
        items: List[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
        count: int = 3,
        selfie_url: Optional[str] = None,
        occasion: Optional[str] = None,
    ) -> Dict[str, Any]:
        slots = self._extract_slots(items)
        generated_images = self._generate_images(
            top_image=slots.get("top") or "",
            bottom_image=slots.get("bottom") or "",
            shoes_image=slots.get("shoes") or "",
            profile=profile or {},
            count=count,
            selfie_url=selfie_url,
            occasion=occasion,
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

    # ------------------------------------------------------------------
    # Style personas (shared, combination-keyed cache)
    # ------------------------------------------------------------------

    @staticmethod
    def persona_key(profile: Dict[str, Any]) -> str:
        """Coarse cache key from the visually dominant quiz axes.

        gender|style|occasion|setting, lowercased. Missing axes fall back to
        "any" so the key is always well-formed. MUST stay in sync with the
        client-side key in StyleProfile.personaKey (iOS).
        """

        def norm(value: Any) -> str:
            text = str(value or "").strip().lower()
            return text or "any"

        return "|".join(
            [
                norm(profile.get("gender")),
                norm(profile.get("style")),
                norm(profile.get("occasion")),
                norm(profile.get("setting")),
            ]
        )

    def get_persona(self, persona_key: str) -> Optional[Dict[str, Any]]:
        """Return the cached persona for a key, or None on a miss."""
        resp = (
            self.supabase.table("personas")
            .select("*")
            .eq("persona_key", persona_key)
            .limit(1)
            .execute()
        )
        if resp.data:
            return self._persona_response(resp.data[0])
        return None

    def create_persona(
        self,
        persona_key: str,
        profile: Dict[str, Any],
        items: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Render one model image for a fresh combination and cache it.

        `items` is the outfit builder's slot->product dict. We render a single
        generic (no-selfie) full-body model wearing the pieces, upload it to the
        public personas bucket, upsert the row, and return the response shape.
        Upsert (not insert) so concurrent cold-starts on the same key converge
        instead of racing to a duplicate-key error.
        """
        items_list = self._persona_items(items)
        slots = self._extract_slots(items_list)

        generated = self._generate_images(
            top_image=slots.get("top") or "",
            bottom_image=slots.get("bottom") or "",
            shoes_image=slots.get("shoes") or "",
            profile=profile or {},
            count=1,
        )
        image_bytes = generated[0]

        image_path = f"{persona_key.replace('|', '_')}.png"
        self._upload_to_storage(image_path, image_bytes, bucket=PERSONAS_BUCKET)
        image_url = self._public_url(image_path, PERSONAS_BUCKET)

        style = (profile.get("style") or "").strip().lower()
        row = {
            "persona_key": persona_key,
            "gender": (profile.get("gender") or "").strip().lower() or None,
            "style": style or None,
            "occasion": (profile.get("occasion") or "").strip().lower() or None,
            "setting": (profile.get("setting") or "").strip().lower() or None,
            "title": _PERSONA_TITLES.get(style, "Your Style"),
            "subtitle": None,
            "image_path": image_path,
            "image_url": image_url,
            "items": items_list,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.supabase.table("personas").upsert(
            row, on_conflict="persona_key"
        ).execute()

        return self._persona_response(row)

    def personalize_persona(
        self,
        persona: Dict[str, Any],
        profile: Dict[str, Any],
        client_id: str,
        selfie_inline: Optional[Dict[str, str]] = None,
        selfie_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-render an already-resolved persona with the user's face.

        `persona` is the /persona/resolve response shape — its items are reused
        verbatim so the personalized render wears the SAME outfit as the generic
        card. The image goes to the private user-selfies bucket next to the
        selfie it derives from ({client_id}/persona-<key>.png, upserted), never
        the shared public personas cache. Raises ValueError when no usable
        selfie is available — a generic image must never be returned as "you".
        """
        face_inline = None
        if selfie_inline and selfie_inline.get("data"):
            face_inline = selfie_inline
        elif selfie_url:
            face_inline = self._convert_image_to_inline_data(selfie_url)
        if not face_inline:
            raise ValueError("Could not load selfie")

        items_list = persona.get("items") or []
        slots = self._extract_slots(items_list)

        generated = self._generate_images(
            top_image=slots.get("top") or "",
            bottom_image=slots.get("bottom") or "",
            shoes_image=slots.get("shoes") or "",
            profile=profile or {},
            count=1,
            selfie_inline=face_inline,
        )

        key = persona.get("persona_key") or self.persona_key(profile)
        image_path = f"{client_id}/persona-{key.replace('|', '_')}.png"
        self._upload_to_storage(image_path, generated[0], bucket=SELFIES_BUCKET)
        image_url = self._create_signed_url(image_path, bucket=SELFIES_BUCKET)

        return {
            "persona_key": key,
            "title": persona.get("title"),
            "subtitle": persona.get("subtitle"),
            "image_url": image_url,
            "image_path": image_path,
            "items": items_list,
        }

    def _persona_items(self, items: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten the builder's slot->product dict into the fit-item shape
        (slot + imageUrl + display fields) used for rendering and shop-the-look."""
        result: List[Dict[str, Any]] = []
        for slot, product in (items or {}).items():
            if not product:
                continue
            result.append(
                {
                    "id": str(product.get("product_id") or slot),
                    "slot": "shoes" if slot == "footwear" else slot,
                    "name": product.get("product_title") or "Item",
                    "brand": product.get("product_brand") or "",
                    "price": product.get("product_price_amount") or 0,
                    "imageUrl": product.get("product_img_link"),
                    "link": product.get("product_link"),
                }
            )
        return result

    def _persona_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "persona_key": row.get("persona_key"),
            "title": row.get("title"),
            "subtitle": row.get("subtitle"),
            "image_url": row.get("image_url"),
            "items": row.get("items") or [],
        }

    def _public_url(self, image_path: str, bucket: str) -> str:
        return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{image_path}"

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

        # One batch sign call for every image across every fit — signing
        # per-image serially (one storage round-trip each) made /fits take
        # 10-15s at ~40 fits, which stalled the single-worker queue and
        # timed out the iOS Today request behind it.
        all_paths = [
            img["image_path"]
            for fit_images in images_by_fit.values()
            for img in fit_images
        ]
        signed_by_path = self._create_signed_urls_batch(all_paths)

        normalized: List[Dict[str, Any]] = []
        for fit in fits:
            fit_images = images_by_fit.get(fit["id"], [])
            image_urls: List[str] = []
            for img in fit_images:
                signed_url = signed_by_path.get(img["image_path"])
                if signed_url:
                    image_urls.append(signed_url)
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

    def _create_signed_urls_batch(self, image_paths: List[str]) -> Dict[str, str]:
        """Sign many storage paths in one request.

        Uses the bulk variant of the storage sign endpoint (same route as
        `_create_signed_url`, but with a `paths` array). Returns
        {path: full_signed_url}; paths that fail to sign are omitted.
        """
        if not image_paths:
            return {}

        sign_url = f"{SUPABASE_URL}/storage/v1/object/sign/{FIT_IMAGES_BUCKET}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                sign_url,
                headers=headers,
                json={"expiresIn": SIGNED_URL_TTL_SECONDS, "paths": image_paths},
                timeout=15,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Could not batch-create signed URLs ({response.status_code}): {response.text[:250]}"
                )
            rows = response.json()
        except Exception as exc:
            logger.warning(
                "Batch signed URL request failed (%s paths) error=%s",
                len(image_paths),
                exc,
            )
            return {}

        signed_by_path: Dict[str, str] = {}
        for row in rows or []:
            path = row.get("path")
            signed_path = row.get("signedURL") or ""
            if not path or not signed_path:
                continue
            if signed_path.startswith(("http://", "https://")):
                signed_by_path[path] = signed_path
            else:
                signed_by_path[path] = f"{SUPABASE_URL}/storage/v1{signed_path}"
        return signed_by_path

    def _create_signed_url_for_listing(self, image_path: str) -> Optional[str]:
        try:
            return self._create_signed_url(image_path)
        except Exception as exc:
            logger.warning(
                "Skipping fit image signed URL path=%s error=%s",
                image_path,
                exc,
            )
            return None

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
        selfie_url: Optional[str] = None,
        selfie_inline: Optional[Dict[str, str]] = None,
        occasion: Optional[str] = None,
    ) -> List[bytes]:
        available = [url for url in [top_image, bottom_image, shoes_image] if url]
        if not available:
            raise ValueError("Need at least one image URL to generate fit images")

        # Precision Fit: when a selfie is supplied, load it as the FIRST image
        # part so the prompt can reference it as the person's face/likeness.
        # selfie_inline ({mimeType, data}) wins over selfie_url — the guest
        # persona flow sends bytes directly since nothing is uploaded yet. If
        # the URL fails to load we silently fall back to a generic model.
        face_part = None
        if selfie_inline and selfie_inline.get("data"):
            face_part = {"inlineData": selfie_inline}
        elif selfie_url:
            face_inline = self._convert_image_to_inline_data(selfie_url)
            if face_inline:
                face_part = {"inlineData": face_inline}
            else:
                logger.warning("Selfie load failed; rendering generic model")
        has_face = face_part is not None

        garment_parts = []
        for url in [top_image, bottom_image, shoes_image]:
            if not url:
                continue
            inline = self._convert_image_to_inline_data(url)
            if inline:
                garment_parts.append({"inlineData": inline})

        if not garment_parts:
            raise ValueError("Could not load any item images for generation")

        image_parts = ([face_part] if has_face else []) + garment_parts

        gender = profile.get("gender") or ""
        style = profile.get("style") or ""
        occasion = occasion or profile.get("occasion") or ""

        def _call_gemini(variation: int) -> Optional[bytes]:
            prompt = self._build_prompt(gender, style, occasion, variation, has_face)
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}, *image_parts],
                    }
                ]
            }
            t0 = time.monotonic()
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=120,
            )
            elapsed = time.monotonic() - t0
            if not resp.ok:
                logger.warning(
                    "Gemini variation %s failed (%s) after %.1fs: %s",
                    variation,
                    resp.status_code,
                    elapsed,
                    resp.text[:200],
                )
                return None
            logger.info("Gemini variation %s completed in %.1fs", variation, elapsed)
            return self._extract_image_bytes(resp.json())

        t_start = time.monotonic()
        generated: List[bytes] = []
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = {
                pool.submit(_call_gemini, v + 1): v for v in range(max(1, count))
            }
            for future in as_completed(futures):
                try:
                    img = future.result()
                    if img:
                        generated.append(img)
                except Exception as exc:
                    logger.warning("Gemini worker error: %s", exc)

        logger.info(
            "Image generation total: %d/%d images in %.1fs",
            len(generated),
            count,
            time.monotonic() - t_start,
        )

        if not generated:
            raise RuntimeError("Gemini did not return any images")

        return generated

    def _convert_image_to_inline_data(self, image_url: str) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(image_url, timeout=20)
            if not response.ok:
                logger.warning(
                    "Image fetch failed (%s): %s", response.status_code, image_url
                )
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

    def _upload_to_storage(
        self, image_path: str, image_bytes: bytes, bucket: Optional[str] = None
    ) -> None:
        bucket = bucket or FIT_IMAGES_BUCKET
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{image_path}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "apikey": SUPABASE_KEY,
            "x-upsert": "true",
            "Content-Type": "image/png",
        }
        response = requests.post(
            upload_url, headers=headers, data=image_bytes, timeout=30
        )
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Storage upload failed ({response.status_code}): {response.text[:250]}"
            )

    def _create_signed_url(self, image_path: str, bucket: Optional[str] = None) -> str:
        bucket = bucket or FIT_IMAGES_BUCKET
        sign_url = f"{SUPABASE_URL}/storage/v1/object/sign/{bucket}/{image_path}"
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

    def _build_prompt(
        self,
        gender: str,
        style: str,
        occasion: str,
        variation: int,
        has_face: bool = False,
    ) -> str:
        if has_face:
            intro = (
                "The FIRST reference image is a selfie of a real person. Create a photorealistic "
                "full-body fashion image of THIS SAME PERSON — their face, skin tone, and likeness "
                "must closely and faithfully match the selfie. Dress them in the outfit items shown "
                "in the REMAINING reference images. "
            )
        else:
            intro = (
                "Create a photorealistic full-body fashion model image featuring the outfit items "
                "provided in the reference images. "
            )
        return (
            intro
            + "Preserve garment details, silhouettes, and colors accurately. "
            f"Gender expression: {gender or 'unspecified'}. "
            f"Style: {style or 'modern'}. "
            f"Occasion: {occasion or 'everyday'}. "
            f"Variation {variation}: use a different pose and background while keeping the same outfit pieces"
            + (" and the same person's face" if has_face else "")
            + ". Output only one image."
        )
