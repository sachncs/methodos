"""Tests for the bounded HTTP load probe."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from scripts.http_load_smoke import percentile, run_check


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([0.1, 0.2, 0.3, 0.4], 0.95) == 0.4
    with pytest.raises(ValueError, match="not be empty"):
        percentile([], 0.95)


def test_run_check_reports_successes(capsys: pytest.CaptureFixture[str]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200)

    result = asyncio.run(
        run_check(
            url="https://example.test/health",
            requests=8,
            concurrency=3,
            timeout=1,
            transport=httpx.MockTransport(handler),
        )
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "succeeded=8" in output
    assert "failed=0" in output
    assert "p99_seconds=" in output


def test_run_check_returns_failure_for_non_200(capsys: pytest.CaptureFixture[str]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, text="unavailable")

    result = asyncio.run(
        run_check(
            url="https://example.test/health",
            requests=2,
            concurrency=1,
            timeout=1,
            transport=httpx.MockTransport(handler),
        )
    )

    assert result == 1
    captured = capsys.readouterr()
    assert "failed=2" in captured.out
    assert "HTTP 503" in captured.err


@pytest.mark.parametrize(
    ("requests", "concurrency", "timeout"),
    [(0, 1, 1), (100001, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_run_check_rejects_invalid_limits(requests: int, concurrency: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            run_check(
                url="https://example.test/health",
                requests=requests,
                concurrency=concurrency,
                timeout=timeout,
            )
        )
