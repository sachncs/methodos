"""Tests for `methodos.llm`."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

import methodos.llm as llm_module
from methodos.llm import (
    CircuitBreaker,
    CircuitOpenError,
    FailoverLLMClient,
    LiteLLMClient,
    LLMClient,
    LLMError,
    is_retryable_provider_error,
)


class SimpleSchema(BaseModel):
    """Sample Pydantic schema for structured-output tests."""

    answer: str
    confidence: float


class StubClient:
    """Small protocol-compatible provider double."""

    def __init__(self, result: str | None = None, error: LLMError | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        del system, user, json_schema, temperature
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def test_failover_client_requires_a_provider() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FailoverLLMClient([])


async def test_failover_client_uses_next_provider_after_llm_failure() -> None:
    first = StubClient(error=LLMError("primary unavailable"))
    second = StubClient(result="recovered")
    client = FailoverLLMClient([first, second])

    assert await client.complete(system="s", user="u") == "recovered"
    assert first.calls == 1
    assert second.calls == 1


async def test_failover_client_reports_all_provider_failures() -> None:
    first = StubClient(error=LLMError("primary unavailable"))
    second = StubClient(error=LLMError("backup unavailable"))
    client = FailoverLLMClient([first, second])

    with pytest.raises(LLMError, match="all 2 LLM providers failed") as excinfo:
        await client.complete(system="s", user="u")
    assert "primary unavailable" in "\n".join(excinfo.value.__notes__ or [])
    assert "backup unavailable" in "\n".join(excinfo.value.__notes__ or [])


async def test_failover_client_does_not_swallow_programming_errors() -> None:
    class BrokenClient:
        async def complete(self, **kwargs: Any) -> str:
            del kwargs
            raise ValueError("bug")

    backup = StubClient(result="unused")
    client = FailoverLLMClient([BrokenClient(), backup])
    with pytest.raises(ValueError, match="bug"):
        await client.complete(system="s", user="u")
    assert backup.calls == 0


def test_circuit_breaker_opens_and_recovers() -> None:
    """Repeated provider failures shed work, then permit a recovery probe."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=5.0)
    breaker.record_failure()
    assert breaker.allow_request()
    breaker.record_failure()
    assert breaker.is_open
    assert not breaker.allow_request()
    breaker.opened_at = time.monotonic() - 6.0
    assert breaker.allow_request()
    breaker.record_success()
    assert not breaker.is_open
    assert breaker.failure_count == 0


def test_circuit_breaker_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="recovery_seconds"):
        CircuitBreaker(recovery_seconds=0)


def test_llm_client_is_runtime_checkable_protocol() -> None:
    """LLMClient is `runtime_checkable` so isinstance checks work at runtime."""

    # Confirm @runtime_checkable was applied; `isinstance` against a Protocol
    # without that decorator returns False even for matching shapes.
    class ProtocolClient:
        async def complete(
            self,
            *,
            system: str,
            user: str,
            json_schema: type[BaseModel] | None = None,
            temperature: float = 0.0,
        ) -> str:
            return "ok"

    assert isinstance(ProtocolClient(), LLMClient)


def test_litellm_client_construct_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        LiteLLMClient(model="")


def test_litellm_client_construct_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        LiteLLMClient(model="gpt-4o-mini", timeout_seconds=0)
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        LiteLLMClient(model="gpt-4o-mini", timeout_seconds=-1)


def test_litellm_client_construct_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        LiteLLMClient(model="gpt-4o-mini", max_retries=-1)


def test_litellm_client_rejects_invalid_budget_settings() -> None:
    with pytest.raises(ValueError, match="retry_backoff_seconds"):
        LiteLLMClient(model="gpt-4o-mini", retry_backoff_seconds=-1)
    with pytest.raises(ValueError, match="max_output_tokens"):
        LiteLLMClient(model="gpt-4o-mini", max_output_tokens=0)
    with pytest.raises(ValueError, match="max_retry_backoff_seconds"):
        LiteLLMClient(model="gpt-4o-mini", max_retry_backoff_seconds=0.1)


def test_provider_error_policy_distinguishes_transient_and_permanent_errors() -> None:
    class UnauthorizedError(Exception):
        status_code = 401

    class RateLimitError(Exception):
        status_code = 429

    assert not is_retryable_provider_error(UnauthorizedError())
    assert is_retryable_provider_error(RateLimitError())
    assert is_retryable_provider_error(TimeoutError())


def test_litellm_client_model_property() -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    assert client.model == "gpt-4o-mini"


class LitellmModuleDouble:
    """Module-shaped double for monkeypatching `litellm` in tests."""

    def __init__(self, recorded_acompletion: Any) -> None:
        self.acompletion = staticmethod(recorded_acompletion)


@pytest.fixture
def patched_litellm(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Patch `litellm.acompletion` to record calls and return canned responses."""
    state: dict[str, Any] = {
        "calls": [],
        "responses": [{"choices": [{"message": {"content": "ok"}}]}],
        "exceptions": [],
    }

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        state["calls"].append(kwargs)
        if state["exceptions"]:
            exc = state["exceptions"].pop(0)
            raise exc
        if state["responses"]:
            response: dict[str, Any] = state["responses"].pop(0)
            return response
        return {"choices": [{"message": {"content": "default"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))
    yield state


async def test_litellm_complete_returns_content(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    out = await client.complete(system="s", user="u")
    assert out == "ok"
    assert len(patched_litellm["calls"]) == 1
    call = patched_litellm["calls"][0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


async def test_litellm_complete_passes_api_key_and_base(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(
        model="anthropic/claude-sonnet-4-5",
        api_key="k",
        api_base="https://example.test/v1",
    )
    await client.complete(system="s", user="u")
    call = patched_litellm["calls"][0]
    assert call["api_key"] == "k"
    assert call["api_base"] == "https://example.test/v1"
    assert call["timeout"] == 60.0


async def test_litellm_complete_includes_structured_output(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    await client.complete(system="s", user="u", json_schema=SimpleSchema)
    call = patched_litellm["calls"][0]
    assert "response_format" in call
    assert call["response_format"]["type"] == "json_schema"
    schema = call["response_format"]["json_schema"]
    assert schema["name"] == "SimpleSchema"
    assert "properties" in schema["schema"]


async def test_litellm_complete_retries_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failures then success → returns the success without raising."""
    call_count = {"n": 0}

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError(f"transient {call_count['n']}")
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=3)
    out = await client.complete(system="s", user="u")
    assert out == "recovered"
    assert call_count["n"] == 3


async def test_litellm_complete_raises_llm_error_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All retries fail → raises LLMError wrapping the last underlying error."""

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("always fails")

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=2)
    with pytest.raises(LLMError) as excinfo:
        await client.complete(system="s", user="u")
    assert "failed after 3 attempts" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "always fails" in str(excinfo.value.__cause__)


async def test_litellm_complete_sheds_calls_when_circuit_is_open(
    patched_litellm: dict[str, Any],
) -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60.0)
    breaker.record_failure()
    client = LiteLLMClient(model="gpt-4o-mini", circuit_breaker=breaker)
    with pytest.raises(CircuitOpenError):
        await client.complete(system="s", user="u")
    assert patched_litellm["calls"] == []


async def test_litellm_complete_does_not_retry_on_programmatic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMError raised inside the call path (non-retryable) propagates immediately."""
    call_count = {"n": 0}

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        # Return a content payload that is NOT a string → triggers LLMError.
        return {"choices": [{"message": {"content": 42}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=5)
    with pytest.raises(LLMError, match="expected str content"):
        await client.complete(system="s", user="u")
    assert call_count["n"] == 1


async def test_litellm_complete_does_not_retry_permanent_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnauthorizedError(Exception):
        status_code = 401

    call_count = {"n": 0}

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        raise UnauthorizedError("bad key")

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))
    client = LiteLLMClient(model="gpt-4o-mini", max_retries=5)
    with pytest.raises(LLMError, match="non-retryable"):
        await client.complete(system="s", user="u")
    assert call_count["n"] == 1


async def test_litellm_complete_uses_configured_timeout(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini", timeout_seconds=12.5)
    await client.complete(system="s", user="u")
    assert patched_litellm["calls"][0]["timeout"] == 12.5


async def test_litellm_complete_enforces_timeout_at_client_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider that ignores its timeout cannot hold the request forever."""

    async def hanging_acompletion(**kwargs: Any) -> dict[str, Any]:
        await asyncio.Event().wait()
        return {"choices": [{"message": {"content": "never"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(hanging_acompletion))
    client = LiteLLMClient(model="gpt-4o-mini", timeout_seconds=0.01, max_retries=0)
    with pytest.raises(LLMError, match="failed after 1 attempts"):
        await client.complete(system="s", user="u")


async def test_litellm_complete_propagates_temperature(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    await client.complete(system="s", user="u", temperature=0.7)
    assert patched_litellm["calls"][0]["temperature"] == 0.7


async def test_litellm_error_is_runtime_error() -> None:
    """LLMError inherits RuntimeError so callers can catch broad."""
    assert issubclass(LLMError, RuntimeError)
