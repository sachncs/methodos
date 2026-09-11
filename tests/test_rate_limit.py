"""Tests for `methodos.rate_limit`."""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from methodos.rate_limit import (
    Bucket,
    TokenBucketLimiter,
    limiter_from_env,
    rate_limit_dependency,
    subject_for_request,
)


class TestTokenBucketLimiter:
    def test_initial_request_passes(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0)
        limiter.check("ip:1.1.1.1")
        limiter.check("ip:1.1.1.1")

    def test_third_request_blocked(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0)
        limiter.check("ip:1.1.1.1")
        limiter.check("ip:1.1.1.1")
        with pytest.raises(HTTPException) as exc:
            limiter.check("ip:1.1.1.1")
        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers

    def test_refill_unblocks(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_per_second=100.0)
        limiter.check("ip:1.1.1.1")
        limiter.check("ip:1.1.1.1")
        time.sleep(0.05)  # refill ~5 tokens
        limiter.check("ip:1.1.1.1")

    def test_cost_larger_than_capacity_blocked(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0)
        with pytest.raises(HTTPException) as exc:
            limiter.check("ip:1.1.1.1", cost=5)
        assert exc.value.status_code == 429

    def test_negative_cost_rejected(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0)
        with pytest.raises(ValueError):
            limiter.check("ip:1.1.1.1", cost=-1)

    def test_separate_subjects_independent(self) -> None:
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=0.01)
        limiter.check("ip:1.1.1.1")
        # Different subject still has full bucket
        limiter.check("ip:2.2.2.2")

    def test_idle_eviction(self) -> None:
        limiter = TokenBucketLimiter(capacity=10, refill_per_second=1.0, idle_ttl_seconds=0.01)
        limiter.check("ip:1.1.1.1")
        time.sleep(0.05)
        # Next call should evict the idle bucket; bucket is rebuilt at full
        limiter.check("ip:1.1.1.1")
        # With full bucket, we should have at least 9 tokens left (consumed 1)
        # Check via another immediate call (should pass since 9 tokens remain)
        limiter.check("ip:1.1.1.1")

    def test_reset_clears_buckets(self) -> None:
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=0.01)
        limiter.check("ip:1.1.1.1")
        limiter.reset()
        # After reset, fresh bucket at full capacity
        limiter.check("ip:1.1.1.1")

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=0, refill_per_second=1.0)

    def test_invalid_refill(self) -> None:
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=1, refill_per_second=0)

    def test_invalid_idle_ttl(self) -> None:
        with pytest.raises(ValueError):
            TokenBucketLimiter(capacity=1, refill_per_second=1.0, idle_ttl_seconds=0)


class TestLimiterFromEnv:
    def test_default_builds_enabled_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PGRAPH_RATE_LIMIT_DISABLED", raising=False)
        monkeypatch.delenv("PGRAPH_RATE_LIMIT_CAPACITY", raising=False)
        monkeypatch.delenv("PGRAPH_RATE_LIMIT_REFILL_PER_SECOND", raising=False)
        limiter = limiter_from_env()
        assert limiter is not None
        assert limiter.capacity == 60.0

    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGRAPH_RATE_LIMIT_DISABLED", "1")
        assert limiter_from_env() is None

    def test_custom_capacity_and_refill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGRAPH_RATE_LIMIT_CAPACITY", "120")
        monkeypatch.setenv("PGRAPH_RATE_LIMIT_REFILL_PER_SECOND", "2.5")
        limiter = limiter_from_env()
        assert limiter is not None
        assert limiter.capacity == 120.0
        assert limiter.refill_per_second == 2.5


class TestSubjectForRequest:
    def test_api_key_takes_precedence(self) -> None:
        from starlette.requests import Request

        scope: dict = {"type": "http", "client": ("1.2.3.4", 5000)}
        req = Request(scope)
        assert subject_for_request(req, api_key="key1") == "key:key1"

    def test_ip_when_no_api_key(self) -> None:
        from starlette.requests import Request

        scope: dict = {"type": "http", "client": ("1.2.3.4", 5000)}
        req = Request(scope)
        assert subject_for_request(req, api_key=None) == "ip:1.2.3.4"

    def test_unknown_ip_when_no_client(self) -> None:
        from starlette.requests import Request

        scope: dict = {"type": "http"}
        req = Request(scope)
        assert subject_for_request(req, api_key=None) == "ip:unknown"


class TestRateLimitDependency:
    def test_returns_callable(self) -> None:
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=1.0)
        dep = rate_limit_dependency(limiter)
        assert callable(dep)

    def test_none_limiter_returns_noop(self) -> None:
        dep = rate_limit_dependency(None)
        assert callable(dep)


class TestBucketDataclass:
    def test_construct(self) -> None:
        b = Bucket(tokens=1.5, last_refill=time.monotonic())
        assert b.tokens == 1.5
        assert isinstance(b.last_refill, float)
