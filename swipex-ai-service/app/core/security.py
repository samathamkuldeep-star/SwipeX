"""Security dependencies including constant-time internal API Key verification."""

import hmac

from fastapi import Header

from app.core.config import settings
from app.core.exceptions import UnauthorizedError


async def verify_api_key(
    x_internal_api_key: str | None = Header(None, alias="X-Internal-Api-Key")
) -> None:
    """Dependency enforcing header X-Internal-Api-Key matching configured INTERNAL_API_KEY.

    Skips check ONLY if ENVIRONMENT == 'dev' and no INTERNAL_API_KEY is configured.
    """
    configured_key = settings.INTERNAL_API_KEY

    # Skip auth in dev mode if no key is configured
    if settings.ENVIRONMENT == "dev" and not configured_key:
        return

    if not x_internal_api_key or not configured_key:
        raise UnauthorizedError("Missing or unconfigured internal API key.")

    # Constant-time comparison to prevent timing side-channel attacks
    is_valid = hmac.compare_digest(
        x_internal_api_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    )

    if not is_valid:
        raise UnauthorizedError("Invalid internal API key.")
