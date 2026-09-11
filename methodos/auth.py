"""Optional API-key authentication for the hosted service.

When the environment variable `PGRAPH_API_KEY` is set, every request to a
protected route must include the key as either:
- `Authorization: Bearer <key>` header (preferred), or
- `X-API-Key: <key>` header (alternative)

When `PGRAPH_API_KEY` is unset (the default), authentication is a no-op
and every request is accepted. This keeps the default installation
dev-friendly while letting operators lock the service down by exporting
one variable.

The auth dependency is implemented as a FastAPI `Depends` callable so it
composes with the existing endpoints without per-route boilerplate.
Rotation: change the env var, then restart the service. There is no
multi-key or per-tenant support in v0.1.

Security notes:
- Use a high-entropy key (>= 32 random bytes, base64- or hex-encoded).
- Always run behind TLS termination; the key is sent in cleartext over
  the wire otherwise.
- Audit the `methodos_auth_failures_total` counter (via `/metrics`) to
  detect brute-force attempts.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

from fastapi import Header, HTTPException, Request, status


class APIKeyAuth:
    """Bearer-token check; no-op when `PGRAPH_API_KEY` is unset.

    Compare is constant-time to mitigate timing attacks. The class is
    instantiated once at app startup (via `create_app`) so the configured
    key is captured once rather than re-read on every request.
    """

    def __init__(self, *, configured_key: str | None) -> None:
        # Treat empty strings the same as unset so operators don't get
        # locked out by `PGRAPH_API_KEY=`.
        self.configured_key = configured_key if configured_key else None

    @property
    def enabled(self) -> bool:
        """Whether auth is enforced (i.e. a key has been configured)."""
        return self.configured_key is not None

    def check(
        self,
        *,
        authorization: str | None,
        x_api_key: str | None,
    ) -> None:
        """Verify the request; raise `HTTPException(401)` on failure.

        No-op when auth is disabled. Both header values are accepted;
        the first match wins.
        """
        if not self.enabled:
            return

        candidate = extract_candidate(authorization, x_api_key)
        if candidate is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(candidate, self.configured_key):  # type: ignore[type-var]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )


def extract_candidate(authorization: str | None, x_api_key: str | None) -> str | None:
    """Return the API key candidate from headers, or None if absent."""
    if x_api_key:
        return x_api_key
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    return None


def auth_dependency(
    auth: APIKeyAuth,
) -> Callable[..., Awaitable[None]]:
    """Build a FastAPI dependency that delegates to `auth.check`.

    Returns a callable suitable for `Depends(...)`. Captures `auth` via
    closure so request handlers don't need to thread it through.
    """

    async def require_api_key(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        # Best-effort client IP for logging; behind a proxy, the operator
        # should configure `ProxyHeadersMiddleware` so `request.client.host`
        # reflects the real client.
        client_ip = request.client.host if request.client else "unknown"
        auth.check(authorization=authorization, x_api_key=x_api_key)
        # Stash metadata on request.state so downstream middleware and
        # dependencies (e.g. the rate limiter, structured log records)
        # can read the IP and API key without re-parsing headers.
        request.state.methodos_client_ip = client_ip
        request.state.methodos_api_key = extract_candidate(authorization, x_api_key)

    return require_api_key


def api_key_from_env(env: dict[str, str] | None = None) -> str | None:
    """Read `PGRAPH_API_KEY` from the given env mapping (defaults to os.environ)."""
    source = env if env is not None else os.environ
    return source.get("PGRAPH_API_KEY") or None


__all__ = ["APIKeyAuth", "api_key_from_env", "auth_dependency", "extract_candidate"]
