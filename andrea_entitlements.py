"""Trusted account entitlement lookup; never accepts client Pro claims."""
from datetime import datetime, timezone


class EntitlementLookupError(Exception):
    pass


class SupabaseEntitlements:
    """Read the existing billing-owned table using the authenticated Supabase UID.

    Subscription writers must maintain is_pro and expires_at. A missing row is
    free, not a grant based on email or a historical Stripe checkout. Keep ratings
    disabled until the Stripe/StoreKit sync and table write policies are verified.
    """
    def __init__(self, client_provider, fallback=None, now=None):
        self.client_provider = client_provider
        self.fallback = fallback
        self.now = now or (lambda: datetime.now(timezone.utc))

    def available(self):
        return True

    def is_pro(self, user_id):
        try:
            rows = (self.client_provider().table('entitlements')
                    .select('is_pro,expires_at').eq('user_id', user_id).execute().data)
            if not isinstance(rows, list):
                raise ValueError('Invalid entitlement response')
            for row in rows:
                if row.get('is_pro') is not True:
                    continue
                expiry = row.get('expires_at')
                if expiry is None:
                    return True
                expires_at = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
                if expires_at.tzinfo is None:
                    raise ValueError('Entitlement expiration must have a timezone')
                if expires_at > self.now():
                    return True
        except Exception as exc:
            raise EntitlementLookupError('Subscription lookup failed') from exc
        # Optional compatibility with native purchases that have not been
        # mirrored to Supabase. Both inputs are server-verified sources.
        if self.fallback and self.fallback.available():
            return self.fallback.is_pro(user_id)
        return False
