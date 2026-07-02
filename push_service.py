#!/usr/bin/env python3
"""
APNs push notification sender (token-based / .p8 auth).

Uses Apple's modern provider-token authentication (a single ES256-signed JWT
built from a .p8 AuthKey) instead of legacy per-App-ID SSL certificates. One
key works for every app under the team and for both the sandbox and production
APNs hosts; it never expires.

Talks HTTP/2 to APNs via httpx. A single httpx.Client is reused across requests
(connection pooling), and the provider JWT is cached and refreshed on an
interval — Apple rejects tokens older than 1 hour and rate-limits refreshes
faster than every ~20 minutes, so we rotate at 40 minutes.

Config (environment variables):
    APNS_KEY_ID     - 10-char Key ID from the APNs AuthKey (Keys section)
    APNS_TEAM_ID    - 10-char Apple Developer Team ID
    APNS_BUNDLE_ID  - app bundle id / APNs topic (default: app.finer.fit)
    APNS_KEY_PATH   - path to the AuthKey_XXXX.p8 file
                      (on Render: a Secret File, e.g. /etc/secrets/apns.p8)
    APNS_KEY_P8     - alternative to APNS_KEY_PATH: the .p8 contents inline
                      (use when a secret-file mount isn't available)

Either APNS_KEY_PATH or APNS_KEY_P8 must be set.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import jwt

logger = logging.getLogger(__name__)

# APNs hosts. Which one a device token belongs to is decided by the build that
# minted it (see device_tokens.environment), not by anything server-side.
_APNS_HOSTS = {
    "production": "https://api.push.apple.com",
    "sandbox": "https://api.sandbox.push.apple.com",
}

# Refresh the provider JWT well inside Apple's 1-hour ceiling.
_TOKEN_TTL_SECONDS = 40 * 60

# APNs reasons that mean "this token is permanently dead" — prune on sight.
_DEAD_TOKEN_REASONS = {"Unregistered", "BadDeviceToken", "DeviceTokenNotForTopic"}


@dataclass
class PushResult:
    """Outcome of a single send, per device token."""

    device_token: str
    ok: bool
    status: int
    reason: Optional[str] = None
    # True when the token should be marked inactive (uninstalled / invalid).
    should_deactivate: bool = False


class APNsClient:
    """Thread-safe APNs sender. Instantiate once (see get_push_service())."""

    def __init__(
        self,
        key_id: str,
        team_id: str,
        bundle_id: str,
        private_key: str,
    ) -> None:
        self._key_id = key_id
        self._team_id = team_id
        self._bundle_id = bundle_id
        self._private_key = private_key

        # httpx.Client with http2=True is safe to share across threads; it
        # keeps a pooled HTTP/2 connection per host and multiplexes over it.
        self._http = httpx.Client(http2=True, timeout=10.0)

        self._jwt_lock = threading.Lock()
        self._jwt: Optional[str] = None
        self._jwt_issued_at: float = 0.0

    # -- provider token -----------------------------------------------------

    def _provider_token(self) -> str:
        """Return a cached JWT, minting a fresh one past the TTL."""
        with self._jwt_lock:
            age = time.monotonic() - self._jwt_issued_at
            if self._jwt is None or age >= _TOKEN_TTL_SECONDS:
                self._jwt = jwt.encode(
                    {"iss": self._team_id, "iat": int(time.time())},
                    self._private_key,
                    algorithm="ES256",
                    headers={"kid": self._key_id},
                )
                self._jwt_issued_at = time.monotonic()
                logger.info("APNs provider token refreshed (kid=%s)", self._key_id)
            return self._jwt

    # -- sending ------------------------------------------------------------

    def send(
        self,
        device_token: str,
        *,
        title: str,
        body: str,
        environment: str = "production",
        subtitle: Optional[str] = None,
        badge: Optional[int] = None,
        sound: Optional[str] = "default",
        data: Optional[Dict[str, Any]] = None,
        collapse_id: Optional[str] = None,
    ) -> PushResult:
        """Send one alert push. Returns a PushResult (never raises for APNs
        rejections — network errors surface as ok=False, status=0)."""
        host = _APNS_HOSTS.get(environment, _APNS_HOSTS["production"])
        url = f"{host}/3/device/{device_token}"

        alert: Dict[str, Any] = {"title": title, "body": body}
        if subtitle:
            alert["subtitle"] = subtitle

        aps: Dict[str, Any] = {"alert": alert}
        if sound is not None:
            aps["sound"] = sound
        if badge is not None:
            aps["badge"] = badge

        payload: Dict[str, Any] = {"aps": aps}
        if data:
            # Custom keys ride alongside "aps" at the top level.
            payload.update(data)

        headers = {
            "authorization": f"bearer {self._provider_token()}",
            "apns-topic": self._bundle_id,
            "apns-push-type": "alert",
            "apns-priority": "10",
            "apns-expiration": "0",  # deliver-now-or-drop; no store-and-forward
        }
        if collapse_id:
            headers["apns-collapse-id"] = collapse_id

        try:
            resp = self._http.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("APNs network error token=%s err=%s", device_token[:8], exc)
            return PushResult(device_token, ok=False, status=0, reason=str(exc))

        if resp.status_code == 200:
            return PushResult(device_token, ok=True, status=200)

        reason = None
        try:
            reason = resp.json().get("reason")
        except Exception:
            pass

        return PushResult(
            device_token,
            ok=False,
            status=resp.status_code,
            reason=reason,
            should_deactivate=reason in _DEAD_TOKEN_REASONS,
        )

    def send_many(
        self,
        tokens: List[Dict[str, str]],
        *,
        title: str,
        body: str,
        subtitle: Optional[str] = None,
        badge: Optional[int] = None,
        sound: Optional[str] = "default",
        data: Optional[Dict[str, Any]] = None,
        collapse_id: Optional[str] = None,
        max_workers: int = 16,
    ) -> List[PushResult]:
        """Fan a single message out to many tokens concurrently.

        `tokens` is a list of {"device_token": ..., "environment": ...} dicts
        (the row shape from the device_tokens table)."""
        if not tokens:
            return []

        def _one(row: Dict[str, str]) -> PushResult:
            return self.send(
                row["device_token"],
                title=title,
                body=body,
                environment=row.get("environment", "production"),
                subtitle=subtitle,
                badge=badge,
                sound=sound,
                data=data,
                collapse_id=collapse_id,
            )

        workers = min(max_workers, len(tokens))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, tokens))


# ---------------------------------------------------------------------------
# Lazy singleton (mirrors the other services in this codebase)
# ---------------------------------------------------------------------------

_push_service: Optional[APNsClient] = None
_push_lock = threading.Lock()


def _load_private_key() -> str:
    import glob
    import os

    # 1. Inline contents (APNS_KEY_P8), if provided.
    inline = os.getenv("APNS_KEY_P8")
    if inline:
        return inline

    # 2. Explicit path (APNS_KEY_PATH), if provided.
    path = os.getenv("APNS_KEY_PATH")

    # 3. Auto-discover the Render Secret File: a single .p8 under /etc/secrets.
    #    Lets the mounted key "just work" with no env var to point at it.
    if not path:
        matches = sorted(glob.glob("/etc/secrets/*.p8"))
        if len(matches) == 1:
            path = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                "Multiple .p8 files in /etc/secrets; set APNS_KEY_PATH to pick "
                f"one: {matches}"
            )

    if path:
        with open(path, "r") as fh:
            return fh.read()

    raise RuntimeError(
        "APNs key not configured: add the .p8 as a Render Secret File (mounts "
        "at /etc/secrets/), or set APNS_KEY_PATH / APNS_KEY_P8."
    )


def get_push_service() -> APNsClient:
    """Return the process-wide APNsClient, building it on first use."""
    global _push_service
    if _push_service is None:
        with _push_lock:
            if _push_service is None:
                import os

                key_id = os.getenv("APNS_KEY_ID")
                team_id = os.getenv("APNS_TEAM_ID")
                if not key_id or not team_id:
                    raise RuntimeError(
                        "APNs not configured: APNS_KEY_ID and APNS_TEAM_ID are required."
                    )
                _push_service = APNsClient(
                    key_id=key_id,
                    team_id=team_id,
                    bundle_id=os.getenv("APNS_BUNDLE_ID", "app.finer.fit"),
                    private_key=_load_private_key(),
                )
    return _push_service
