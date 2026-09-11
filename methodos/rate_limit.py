"""Simple in-process rate limiting for the hosted service.

Production deployments may scale horizontally; this implementation is
intentionally **per-process** and acts as a backstop rather than a
distributed limiter. Behind a load balancer, each replica enforces its
own quota - the aggregate limit is `N x per-replica-limit`. Operators
needing a cluster-wide limit should front the service with a shared
store (e.g., Redis + `slowapi`).

Algorithm: **token bucket** with refill. Each (subject, bucket) pair
gets `capacity` tokens, refilled at `refill_per_second`. A request
consumes one token; exhaustion yields `HTTP 429 Too Many Requests`.

Subjects are derived from the request - by default, the client IP.
When API-key auth is enabled, the API key is used instead so a single
misbehaving key cannot starve other tenants sharing an IP.

Tuning via env vars:
- `PGRAPH_RATE_LIMIT_CAPACITY`: max tokens per bucket (default: 60)
- `PGRAPH_RATE_LIMIT_REFILL_PER_SECOND`: refill rate (default: 1.0)
- `PGRAPH_RATE_LIMIT_DISABLED=1`: bypass entirely (default: enabled)

The bucket store is bounded; entries idle for `idle_ttl_seconds`
(default: 600) are evicted by an opportunistic sweep on each request.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request, status


@dataclass(slots=True)
class Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """Thread-safe in-memory token bucket limiter."""

    def __init__(
        self,
        *,
        capacity: float,
        refill_per_second: float,
        idle_ttl_seconds: float = 600.0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if refill_per_second <= 0:
            raise ValueError(f"refill_per_second must be positive, got {refill_per_second}")
        if idle_ttl_seconds <= 0:
            raise ValueError(f"idle_ttl_seconds must be positive, got {idle_ttl_seconds}")
        self._capacity = capacity
        self._refill = refill_per_second
        self._idle_ttl = idle_ttl_seconds
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def refill_per_second(self) -> float:
        return self._refill

    def check(self, subject: str, *, cost: float = 1.0) -> None:
        """Consume `cost` tokens for `subject`; raise 429 on exhaustion.

        Tokens refill continuously since `last_refill` at the configured
        rate up to `capacity`. Buckets idle longer than `idle_ttl_seconds`
        are evicted on each call.
        """
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")
        if cost > self._capacity:
            # Single request larger than the bucket can ever hold.
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="request exceeds rate-limit capacity",
                headers={"Retry-After": str(int(self._capacity / self._refill))},
            )

        now = time.monotonic()
        with self._lock:
            self.sweep_idle(now)
            bucket = self._buckets.get(subject)
            if bucket is None:
                bucket = Bucket(tokens=self._capacity, last_refill=now)
                self._buckets[subject] = bucket
            else:
                elapsed = now - bucket.last_refill
                bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill)
                bucket.last_refill = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return

            # Compute Retry-After based on time to accumulate `cost` tokens.
            deficit = cost - bucket.tokens
            retry_after = max(1, int(deficit / self._refill) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

    def sweep_idle(self, now: float) -> None:
        """Drop buckets that haven't been touched for `idle_ttl_seconds`.

        Called under the lock; the bound is small in practice (one
        eviction per call) and bounded by the number of subjects.
        """
        threshold = now - self._idle_ttl
        stale = [k for k, b in self._buckets.items() if b.last_refill < threshold]
        for key in stale:
            del self._buckets[key]

    def reset(self) -> None:
        """Clear all buckets. Test helper."""
        with self._lock:
            self._buckets.clear()


def limiter_from_env(env: dict[str, str] | None = None) -> TokenBucketLimiter | None:
    """Build a `TokenBucketLimiter` from env vars.

    Returns `None` when `PGRAPH_RATE_LIMIT_DISABLED=1`, signaling "no
    limiter" to the service factory.
    """
    source = env if env is not None else os.environ
    if source.get("PGRAPH_RATE_LIMIT_DISABLED") == "1":
        return None
    capacity = float(source.get("PGRAPH_RATE_LIMIT_CAPACITY", "60"))
    refill = float(source.get("PGRAPH_RATE_LIMIT_REFILL_PER_SECOND", "1.0"))
    return TokenBucketLimiter(capacity=capacity, refill_per_second=refill)


def subject_for_request(
    request: Request,
    *,
    api_key: str | None,
) -> str:
    """Pick the bucket subject for this request.

    Prefers the API key when present so per-tenant quotas are respected
    behind shared NAT. Falls back to the client IP (with an `ip:` prefix
    to keep the namespaces disjoint).
    """
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def rate_limit_dependency(
    limiter: TokenBucketLimiter | None,
    *,
    api_key_provider: Callable[[Request], str | None] | None = None,
) -> Callable[..., Awaitable[None]]:
    """Build a FastAPI dependency that delegates to `limiter.check`.

    Pass `api_key_provider` to make the limiter key-aware; default is
    raw client IP. When `limiter is None`, the returned dependency is a
    no-op.
    """

    async def enforce(request: Request) -> None:
        if limiter is None:
            return
        api_key = api_key_provider(request) if api_key_provider is not None else None
        subject = subject_for_request(request, api_key=api_key)
        limiter.check(subject)

    return enforce


__all__ = [
    "Bucket",
    "TokenBucketLimiter",
    "limiter_from_env",
    "rate_limit_dependency",
    "subject_for_request",
]
