"""Tests for `methodos.llm`."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import litellm
import pytest
from pydantic import BaseModel

import methodos.llm as llm_module
from methodos.llm import (
    AsyncCircuitBreaker,
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    FailoverLLMClient,
    LiteLLMClient,
    LLMClient,
    LLMError,
    ScopedTokenBudget,
    SQLiteCircuitBreaker,
    SQLiteTokenBudget,
    TokenBudget,
    TokenPricing,
    is_retryable_provider_error,
)


class SimpleSchema(BaseModel):
    """Sample Pydantic schema for structured-output tests."""

    answer: str
    confidence: float


def test_token_budget_validates_and_accounts() -> None:
    with pytest.raises(ValueError, match="max_total_tokens"):
        TokenBudget(max_total_tokens=0, max_input_tokens=10)
    budget = TokenBudget(max_total_tokens=20, max_input_tokens=10)
    assert budget.reserve(system="1234", user="1234", output_tokens=5) == 2
    assert budget.consumed_tokens == 7
    with pytest.raises(BudgetExceededError, match="cumulative"):
        budget.reserve(system="1234", user="1234", output_tokens=14)


def test_token_budget_rejects_large_input() -> None:
    budget = TokenBudget(max_total_tokens=100, max_input_tokens=2)
    with pytest.raises(BudgetExceededError, match="input"):
        budget.reserve(system="123456789", user="u", output_tokens=1)


def test_token_budget_enforces_pricing_ceiling() -> None:
    pricing = TokenPricing(
        max_total_cost_usd=0.001,
        input_cost_per_1k_usd=0.5,
        output_cost_per_1k_usd=1.0,
    )
    assert pricing.estimate(input_tokens=2, output_tokens=5) == pytest.approx(0.006)
    budget = TokenBudget(max_total_tokens=100, max_input_tokens=100, pricing=pricing)
    with pytest.raises(BudgetExceededError, match="cost"):
        budget.reserve(system="1234", user="1234", output_tokens=5)


def test_token_pricing_validates_values() -> None:
    with pytest.raises(ValueError, match="max_total_cost_usd"):
        TokenPricing(max_total_cost_usd=0, input_cost_per_1k_usd=1, output_cost_per_1k_usd=1)
    with pytest.raises(ValueError, match="non-negative"):
        TokenPricing(max_total_cost_usd=1, input_cost_per_1k_usd=-1, output_cost_per_1k_usd=1)


def test_sqlite_token_budget_persists_and_scopes_keys(tmp_path: Path) -> None:
    database = tmp_path / "token-budget.db"
    first = SQLiteTokenBudget(
        database_path=database,
        key="tenant-a",
        max_total_tokens=10,
        max_input_tokens=10,
    )
    assert first.reserve(system="1234", user="1234", output_tokens=5) == 2

    reopened = SQLiteTokenBudget(
        database_path=database,
        key="tenant-a",
        max_total_tokens=10,
        max_input_tokens=10,
    )
    with pytest.raises(BudgetExceededError, match="cumulative"):
        reopened.reserve(system="1234", user="1234", output_tokens=5)

    other_tenant = SQLiteTokenBudget(
        database_path=database,
        key="tenant-b",
        max_total_tokens=10,
        max_input_tokens=10,
    )
    assert other_tenant.reserve(system="1234", user="1234", output_tokens=5) == 2


def test_sqlite_token_budget_persists_pricing_ceiling(tmp_path: Path) -> None:
    database = tmp_path / "priced-token-budget.db"
    pricing = TokenPricing(
        max_total_cost_usd=0.01,
        input_cost_per_1k_usd=1.0,
        output_cost_per_1k_usd=1.0,
    )
    first = SQLiteTokenBudget(
        database_path=database,
        key="tenant-a",
        max_total_tokens=100,
        max_input_tokens=100,
        pricing=pricing,
    )
    first.reserve(system="1234", user="1234", output_tokens=5)
    reopened = SQLiteTokenBudget(
        database_path=database,
        key="tenant-a",
        max_total_tokens=100,
        max_input_tokens=100,
        pricing=pricing,
    )
    with pytest.raises(BudgetExceededError, match="cost"):
        reopened.reserve(system="1234", user="1234", output_tokens=5)


def test_sqlite_token_budget_validates_key_and_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="key must not be empty"):
        SQLiteTokenBudget(
            database_path=tmp_path / "invalid-key.db",
            key="",
            max_total_tokens=10,
            max_input_tokens=10,
        )
    budget = SQLiteTokenBudget(
        database_path=tmp_path / "invalid-request.db",
        key="tenant-a",
        max_total_tokens=10,
        max_input_tokens=2,
    )
    with pytest.raises(BudgetExceededError, match="input"):
        budget.reserve(system="123456789", user="u", output_tokens=1)
    with pytest.raises(ValueError, match="output_tokens"):
        budget.reserve(system="1", user="1", output_tokens=0)


def test_retry_after_accepts_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("rate limited")
    error.headers = {"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}  # type: ignore[attr-defined]
    monkeypatch.setattr(time, "time", lambda: 0.0)

    assert llm_module._retry_after_seconds(error) == 10.0


def test_retry_after_rejects_malformed_http_date() -> None:
    error = RuntimeError("rate limited")
    error.headers = {"Retry-After": "not-a-date"}  # type: ignore[attr-defined]

    assert llm_module._retry_after_seconds(error) is None


def test_sqlite_token_budget_concurrent_reservations_are_atomic(tmp_path: Path) -> None:
    database = tmp_path / "concurrent-token-budget.db"
    budgets = [
        SQLiteTokenBudget(
            database_path=database,
            key="tenant-a",
            max_total_tokens=10,
            max_input_tokens=10,
        )
        for _ in range(8)
    ]

    async def reserve_all() -> list[object]:
        return await asyncio.gather(
            *(
                asyncio.to_thread(
                    budget.reserve,
                    system="1234",
                    user="1234",
                    output_tokens=5,
                )
                for budget in budgets
            ),
            return_exceptions=True,
        )

    outcomes = asyncio.run(reserve_all())
    assert sum(not isinstance(outcome, BudgetExceededError) for outcome in outcomes) == 1


def test_sqlite_token_budget_supports_async_reservation(tmp_path: Path) -> None:
    budget = SQLiteTokenBudget(
        database_path=tmp_path / "async-token-budget.db",
        key="tenant-a",
        max_total_tokens=10,
        max_input_tokens=10,
    )
    assert asyncio.run(budget.reserve_async(system="1234", user="1234", output_tokens=5)) == 2


def test_scoped_token_budget_isolates_graph_scopes() -> None:
    created: dict[str, TokenBudget] = {}

    def factory(scope: str) -> TokenBudget:
        created[scope] = TokenBudget(max_total_tokens=10, max_input_tokens=10)
        return created[scope]

    budget = ScopedTokenBudget(factory)
    assert (
        asyncio.run(
            budget.reserve_scoped_async(
                scope="tenant-a:graph-a", system="1234", user="1234", output_tokens=5
            )
        )
        == 2
    )
    assert (
        asyncio.run(
            budget.reserve_scoped_async(
                scope="tenant-a:graph-b", system="1234", user="1234", output_tokens=5
            )
        )
        == 2
    )
    assert set(created) == {"tenant-a:graph-a", "tenant-a:graph-b"}
    with pytest.raises(ValueError, match="scope must not be empty"):
        asyncio.run(
            budget.reserve_scoped_async(scope=" ", system="1234", user="1234", output_tokens=1)
        )


def test_scoped_token_budget_supports_durable_async_budgets(tmp_path: Path) -> None:
    database = tmp_path / "scoped-token-budget.db"
    budget = ScopedTokenBudget(
        lambda scope: SQLiteTokenBudget(
            database_path=database,
            key=scope,
            max_total_tokens=10,
            max_input_tokens=10,
        )
    )
    assert (
        asyncio.run(
            budget.reserve_scoped_async(
                scope="tenant-a:graph-a", system="1234", user="1234", output_tokens=5
            )
        )
        == 2
    )


async def test_scoped_budget_and_llm_client_close_owned_resources() -> None:
    class ClosableBudget:
        def __init__(self) -> None:
            self.closed = False

        async def reserve_async(self, *, system: str, user: str, output_tokens: int) -> int:
            del system, user, output_tokens
            return 1

        async def close(self) -> None:
            self.closed = True

    owned = ClosableBudget()
    scoped = ScopedTokenBudget(lambda _scope: owned)
    await scoped.reserve_scoped_async(scope="tenant:graph", system="s", user="u", output_tokens=1)
    client = LiteLLMClient(model="test-model", budget=scoped)
    await client.close()
    assert owned.closed
    assert not scoped._budgets


async def test_litellm_client_requires_scope_for_scoped_budget() -> None:
    client = LiteLLMClient(
        model="test-model",
        budget=ScopedTokenBudget(
            lambda _scope: TokenBudget(max_total_tokens=10, max_input_tokens=10)
        ),
    )
    with pytest.raises(BudgetExceededError, match="explicit budget_scope"):
        await client.complete(system="s", user="u")


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
        budget_scope: str | None = None,
    ) -> str:
        del system, user, json_schema, temperature, budget_scope
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


async def test_failover_client_preserves_budget_scope() -> None:
    class ScopedProvider:
        def __init__(self) -> None:
            self.scope: str | None = None

        async def complete(
            self,
            *,
            system: str,
            user: str,
            json_schema: type[BaseModel] | None = None,
            temperature: float = 0.0,
            budget_scope: str | None = None,
        ) -> str:
            del system, user, json_schema, temperature
            self.scope = budget_scope
            return "scoped"

    provider = ScopedProvider()
    client = FailoverLLMClient([provider])
    assert await client.complete(system="s", user="u", budget_scope="tenant:graph") == "scoped"
    assert provider.scope == "tenant:graph"


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


def test_sqlite_circuit_breaker_serializes_half_open_probes(tmp_path: Path) -> None:
    breaker = SQLiteCircuitBreaker(
        database_path=tmp_path / "circuit.db",
        key="provider-a",
        failure_threshold=1,
        recovery_seconds=30.0,
    )

    async def exercise() -> None:
        assert await breaker.allow_request_async()
        await breaker.record_failure_async()
        assert not await breaker.allow_request_async()
        with sqlite3.connect(tmp_path / "circuit.db") as connection:
            connection.execute(
                "UPDATE circuit_breakers SET opened_at = ?, probe_until = NULL WHERE key = ?",
                (0.0, "provider-a"),
            )
        outcomes = await asyncio.gather(
            breaker.allow_request_async(), breaker.allow_request_async()
        )
        assert sum(outcomes) == 1
        await breaker.record_success_async()
        assert await breaker.allow_request_async()

    asyncio.run(exercise())


def test_async_circuit_breaker_is_runtime_checkable() -> None:
    class Breaker:
        async def allow_request_async(self) -> bool:
            return True

        async def record_success_async(self) -> None:
            return None

        async def record_failure_async(self) -> None:
            return None

    assert isinstance(Breaker(), AsyncCircuitBreaker)


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
            budget_scope: str | None = None,
        ) -> str:
            del budget_scope
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

    def litellm_error(name: str) -> Exception:
        error_class = cast(type[Exception], getattr(litellm, name))
        return error_class("x", "test", "m")

    assert is_retryable_provider_error(litellm_error("APIConnectionError"))
    assert is_retryable_provider_error(litellm_error("RateLimitError"))
    assert not is_retryable_provider_error(litellm_error("AuthenticationError"))
    assert not is_retryable_provider_error(litellm_error("BadRequestError"))


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


async def test_litellm_complete_uses_async_circuit_breaker(
    patched_litellm: dict[str, Any],
) -> None:
    class AsyncBreaker:
        def __init__(self) -> None:
            self.successes = 0

        async def allow_request_async(self) -> bool:
            return True

        async def record_success_async(self) -> None:
            self.successes += 1

        async def record_failure_async(self) -> None:
            raise AssertionError("failure path should not run")

    breaker = AsyncBreaker()
    client = LiteLLMClient(model="gpt-4o-mini", circuit_breaker=breaker)
    assert await client.complete(system="s", user="u") == "ok"
    assert breaker.successes == 1


async def test_litellm_complete_records_async_circuit_failure(
    patched_litellm: dict[str, Any],
) -> None:
    class AsyncBreaker:
        def __init__(self) -> None:
            self.failures = 0

        async def allow_request_async(self) -> bool:
            return True

        async def record_success_async(self) -> None:
            raise AssertionError("success path should not run")

        async def record_failure_async(self) -> None:
            self.failures += 1

    patched_litellm["exceptions"] = [RuntimeError("provider down")]
    breaker = AsyncBreaker()
    client = LiteLLMClient(
        model="gpt-4o-mini",
        max_retries=0,
        circuit_breaker=breaker,
    )
    with pytest.raises(LLMError):
        await client.complete(system="s", user="u")
    assert breaker.failures == 1


async def test_litellm_complete_uses_async_token_budget(
    patched_litellm: dict[str, Any],
    tmp_path: Path,
) -> None:
    budget = SQLiteTokenBudget(
        database_path=tmp_path / "client-token-budget.db",
        key="tenant-a",
        max_total_tokens=10,
        max_input_tokens=10,
    )
    client = LiteLLMClient(model="gpt-4o-mini", max_output_tokens=5, budget=budget)
    assert await client.complete(system="s", user="u") == "ok"


async def test_litellm_complete_uses_scoped_token_budget(
    patched_litellm: dict[str, Any],
) -> None:
    budget = ScopedTokenBudget(lambda _scope: TokenBudget(max_total_tokens=10, max_input_tokens=10))
    client = LiteLLMClient(model="gpt-4o-mini", max_output_tokens=5, budget=budget)
    assert await client.complete(system="s", user="u", budget_scope="tenant-a:graph-a") == "ok"


async def test_litellm_complete_enforces_shared_budget(
    patched_litellm: dict[str, Any],
) -> None:
    budget = TokenBudget(max_total_tokens=10, max_input_tokens=10)
    client = LiteLLMClient(model="gpt-4o-mini", max_output_tokens=5, budget=budget)
    await client.complete(system="s", user="u")
    with pytest.raises(BudgetExceededError, match="cumulative"):
        await client.complete(system="s", user="u")
    assert len(patched_litellm["calls"]) == 1


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


async def test_litellm_complete_honors_bounded_retry_after_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    call_count = {"n": 0}

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def recorded_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            error = RuntimeError("rate limited")
            error.status_code = 429  # type: ignore[attr-defined]
            error.headers = {"Retry-After": "20"}  # type: ignore[attr-defined]
            raise error
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(recorded_acompletion))
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = LiteLLMClient(
        model="gpt-4o-mini",
        max_retries=1,
        retry_backoff_seconds=0.25,
        max_retry_backoff_seconds=3.0,
    )

    assert await client.complete(system="s", user="u") == "recovered"
    assert delays == [3.0]


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
    with pytest.raises(LLMError, match="exceeded its total timeout"):
        await client.complete(system="s", user="u")


async def test_litellm_total_timeout_includes_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries cannot extend the configured total provider deadline."""
    calls = 0

    async def failing_acompletion(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("transient provider failure")

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(failing_acompletion))
    client = LiteLLMClient(
        model="gpt-4o-mini",
        timeout_seconds=0.02,
        max_retries=5,
        retry_backoff_seconds=1.0,
    )

    with pytest.raises(LLMError, match="exceeded its total timeout"):
        await client.complete(system="s", user="u")
    assert calls == 1


async def test_litellm_complete_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request cancellation must stop an in-flight provider call immediately."""
    started = asyncio.Event()

    async def cancellable_acompletion(**kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"choices": [{"message": {"content": "never"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(cancellable_acompletion))
    client = LiteLLMClient(model="gpt-4o-mini", timeout_seconds=60.0, max_retries=0)
    task = asyncio.create_task(client.complete(system="s", user="u"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_litellm_complete_propagates_temperature(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    await client.complete(system="s", user="u", temperature=0.7)
    assert patched_litellm["calls"][0]["temperature"] == 0.7


async def test_litellm_error_is_runtime_error() -> None:
    """LLMError inherits RuntimeError so callers can catch broad."""
    assert issubclass(LLMError, RuntimeError)
