"""Server-owned rollout, separate from authentication and paid entitlements."""
import os
from uuid import UUID

# Private preview approved by the account owner. This is a rollout identifier,
# not a credential or Pro grant; routes still verify JWTs, entitlements and quota.
# Clear ANDREA_RATINGS_PREVIEW_USER_IDS to disable the preview without a deploy.
DEFAULT_RATING_PREVIEW_USER_IDS = frozenset({
    '1b8d78cd-5996-46a0-9532-a0f563691ef2',
})


def ratings_rollout_allows(user_id):
    try:
        user_id = str(UUID(str(user_id)))
    except (ValueError, TypeError, AttributeError):
        return False
    if os.getenv('ANDREA_RATINGS_ENABLED', 'false').lower() == 'true':
        return True
    configured = os.getenv('ANDREA_RATINGS_PREVIEW_USER_IDS')
    if configured is None:
        allowed = DEFAULT_RATING_PREVIEW_USER_IDS
    else:
        try:
            allowed = {str(UUID(value.strip())) for value in configured.split(',') if value.strip()}
        except (ValueError, TypeError, AttributeError):
            # Misconfigured rollout must fail closed, never enable all accounts.
            return False
    return user_id in allowed
