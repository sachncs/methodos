"""LLM client interface and a litellm-backed implementation.

This module defines:
- `LLMClient`: a `typing.Protocol` describing the contract any LLM backend
  must satisfy. Concrete implementations satisfy it via duck typing.
- `LiteLLMClient`: the default production implementation, backed by
  `litellm`. Supports multi-provider routing (OpenAI, Anthropic, Google,
  Grok, local models), structured outputs via `json_schema`, and retries
  with exponential backoff.

Engineering notes:
- All I/O is `async def`. litellm exposes `acompletion`.
- Public configuration attributes are documented; implementation-only state
  follows the one-leading-underscore convention.
- No lazy imports: `litellm` is a hard dependency at the top of the file.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import litellm
from opentelemetry import trace
from prometheus_client import Counter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
llm_tracer = trace.get_tracer("methodos.llm")
LLM_ATTEMPTS = Counter(
    "methodos_llm_attempts_total",
    "LLM provider attempts by model and outcome.",
    ["model", "outcome"],
)


@runtime_checkable
class LLMClient(Protocol):
    """Contract any LLM backend must satisfy."""

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant text for `system` + `user`."""
        ...


class LLMError(RuntimeError):
    """Raised when an LLM backend fails after exhausting retries."""


class CircuitOpenError(LLMError):
    """Raised while a provider circuit is open and calls are being shed."""


class BudgetExceededError(LLMError):
    """Raised before a request would exceed its configured token budget."""


class TokenBudget:
    """Shared, in-process token budget that can be injected per tenant.

    The budget uses a conservative four-characters-per-token estimate before
    dispatch. A caller can share one instance across clients and requests;
    successful or failed reservations both count because the provider may
    charge for failed attempts.
    """

    def __init__(self, *, max_total_tokens: int, max_input_tokens: int) -> None:
        """Initialize positive cumulative and per-request input limits."""
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens
        self.consumed_tokens = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate tokens conservatively from UTF-8-independent text length."""
        return max(1, math.ceil(len(text) / 4))

    def reserve(self, *, system: str, user: str, output_tokens: int) -> int:
        """Reserve one request and return its estimated input-token count."""
        input_tokens = self.estimate_tokens(system) + self.estimate_tokens(user)
        if input_tokens > self.max_input_tokens:
            raise BudgetExceededError(
                f"LLM input budget exceeded: {input_tokens} > {self.max_input_tokens}"
            )
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        requested = input_tokens + output_tokens
        if self.consumed_tokens + requested > self.max_total_tokens:
            raise BudgetExceededError(
                "LLM cumulative token budget exceeded: "
                f"{self.consumed_tokens + requested} > {self.max_total_tokens}"
            )
        self.consumed_tokens += requested
        return input_tokens


class SQLiteTokenBudget(TokenBudget):
    """SQLite-backed cumulative token budget shared across local workers.

    The reservation is committed in an ``IMMEDIATE`` SQLite transaction, so
    separate processes cannot reserve the same remaining budget concurrently.
    The class keeps the same synchronous ``reserve`` contract as
    :class:`TokenBudget` and can be injected into ``LiteLLMClient``.
    """

    def __init__(
        self,
        *,
        database_path: Path,
        key: str,
        max_total_tokens: int,
        max_input_tokens: int,
    ) -> None:
        """Initialize a durable budget identified by `key`."""
        super().__init__(
            max_total_tokens=max_total_tokens,
            max_input_tokens=max_input_tokens,
        )
        if not key:
            raise ValueError("key must not be empty")
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a contention-safe SQLite connection."""
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _ensure_schema(self) -> None:
        """Create the durable budget table if needed."""
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS token_budgets ("
                "key TEXT PRIMARY KEY, consumed_tokens INTEGER NOT NULL)"
            )

    def reserve(self, *, system: str, user: str, output_tokens: int) -> int:
        """Atomically reserve tokens or raise `BudgetExceededError`."""
        input_tokens = self.estimate_tokens(system) + self.estimate_tokens(user)
        if input_tokens > self.max_input_tokens:
            raise BudgetExceededError(
                f"LLM input budget exceeded: {input_tokens} > {self.max_input_tokens}"
            )
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        requested = input_tokens + output_tokens
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT consumed_tokens FROM token_budgets WHERE key = ?",
                    (self.key,),
                ).fetchone()
                consumed = 0 if row is None else int(row[0])
                total = consumed + requested
                if total > self.max_total_tokens:
                    raise BudgetExceededError(
                        "LLM cumulative token budget exceeded: "
                        f"{total} > {self.max_total_tokens}"
                    )
                connection.execute(
                    "INSERT INTO token_budgets (key, consumed_tokens) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET consumed_tokens = excluded.consumed_tokens",
                    (self.key, total),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.consumed_tokens = total
        return input_tokens


def is_retryable_provider_error(exc: Exception) -> bool:
    """Return whether a provider failure is safe to retry."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429 or status_code >= 500:
            return True
        if 400 <= status_code < 500:
            return False
    error_name = type(exc).__name__.lower()
    permanent_markers = ("authentication", "authorization", "badrequest", "invalidrequest")
    if any(marker in error_name for marker in permanent_markers):
        return False
    return isinstance(exc, (ConnectionError, OSError, TimeoutError, RuntimeError))


class CircuitBreaker:
    """Small process-local circuit breaker for one upstream provider.

    The breaker is deliberately dependency-free. Deployments with multiple
    workers should place the same policy behind a shared gateway or store;
    this local guard still prevents one worker from retrying a failing
    provider indefinitely.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        """Initialize the breaker with bounded failure and recovery values."""
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failure_count = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """Return whether calls should currently be shed."""
        if self.opened_at is None:
            return False
        return time.monotonic() - self.opened_at < self.recovery_seconds

    def allow_request(self) -> bool:
        """Allow a normal or one half-open probe request."""
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.recovery_seconds:
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        """Close the circuit and clear consecutive failures."""
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        """Count a failed upstream call and open at the threshold."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()


class LiteLLMClient:
    """Multi-provider LLM client backed by litellm.

    Args:
        model: model identifier understood by litellm (e.g. `"gpt-4o-mini"`,
            `"anthropic/claude-sonnet-4-5"`, `"gemini/gemini-1.5-pro"`).
        api_key: optional API key override; otherwise read from the
            provider's standard environment variable by litellm.
        api_base: optional custom API base URL.
        timeout_seconds: per-request timeout.
        max_retries: number of retries on transient failures (total
            attempts = `max_retries + 1`).
        retry_backoff_seconds: initial exponential delay between retries.
        max_retry_backoff_seconds: upper bound for exponential backoff.
        max_output_tokens: upper bound for generated response tokens.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        max_retry_backoff_seconds: float = 8.0,
        max_output_tokens: int = 2048,
        circuit_breaker: CircuitBreaker | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        """Initialize the LiteLLM-backed client.

        Args:
            model: litellm model identifier.
            api_key: optional API key override.
            api_base: optional custom API base URL.
            timeout_seconds: per-request timeout.
            max_retries: retries on transient failures; total attempts = max_retries + 1.
            retry_backoff_seconds: initial exponential delay between retries.
            max_retry_backoff_seconds: upper bound for exponential backoff.
            max_output_tokens: upper bound for generated response tokens.
            circuit_breaker: optional upstream failure breaker; a new local
                breaker is created when omitted.
            budget: optional shared token budget, suitable for tenant-scoped
                accounting when injected by the application.
        """
        if not model:
            raise ValueError("model must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if retry_backoff_seconds < 0:
            raise ValueError(
                f"retry_backoff_seconds must be non-negative, got {retry_backoff_seconds}"
            )
        if max_retry_backoff_seconds < retry_backoff_seconds:
            raise ValueError(
                "max_retry_backoff_seconds must be >= retry_backoff_seconds"
            )
        if max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {max_output_tokens}")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_retry_backoff_seconds = max_retry_backoff_seconds
        self.max_output_tokens = max_output_tokens
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.budget = budget

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat-completion request and return the assistant text.

        Retries up to `max_retries` times on transient failures (network
        errors, rate limits). After exhaustion, raises the last underlying
        exception as `LLMError`.

        When `json_schema` is provided, the response is requested with
        litellm's `response_format` set to a JSON Schema describing the
        target Pydantic model.

        Args:
            system: System prompt text.
            user: User prompt text.
            json_schema: Optional Pydantic model for structured output.
            temperature: Sampling temperature.

        Returns:
            Assistant text (or JSON string if `json_schema` is set).
        """
        if self.budget is not None:
            try:
                self.budget.reserve(
                    system=system,
                    user=user,
                    output_tokens=self.max_output_tokens,
                )
            except BudgetExceededError:
                LLM_ATTEMPTS.labels(self.model, "budget_rejected").inc()
                raise
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
            "max_tokens": self.max_output_tokens,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.__name__,
                    "schema": json_schema.model_json_schema(),
                },
            }

        last_exc: Exception | None = None
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            if not self.circuit_breaker.allow_request():
                LLM_ATTEMPTS.labels(self.model, "circuit_open").inc()
                raise CircuitOpenError(
                    f"LLM provider circuit is open for model {self.model}"
                )
            LLM_ATTEMPTS.labels(self.model, "started").inc()
            try:
                with llm_tracer.start_as_current_span("methodos.llm.attempt") as span:
                    span.set_attribute("methodos.llm.model", self.model)
                    span.set_attribute("methodos.llm.attempt", attempt + 1)
                    try:
                        async with asyncio.timeout(self.timeout):
                            response = await litellm.acompletion(**kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        raise
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise LLMError(
                        f"expected str content from {self.model}, got {type(content).__name__}"
                    )
                self.circuit_breaker.record_success()
                LLM_ATTEMPTS.labels(self.model, "succeeded").inc()
                return content
            except LLMError:
                LLM_ATTEMPTS.labels(self.model, "failed").inc()
                raise
            except Exception as exc:
                LLM_ATTEMPTS.labels(self.model, "failed").inc()
                last_exc = exc
                logger.warning(
                    "litellm attempt %d/%d for model %s failed: %s",
                    attempt + 1,
                    total_attempts,
                    self.model,
                    exc,
                )
                if not is_retryable_provider_error(exc):
                    raise LLMError(
                        f"non-retryable LLM call failure for {self.model}"
                    ) from exc
                self.circuit_breaker.record_failure()
                if attempt < total_attempts - 1 and self.retry_backoff_seconds > 0:
                    delay = min(
                        self.max_retry_backoff_seconds,
                        self.retry_backoff_seconds * (2**attempt),
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        logger.error(
            "litellm exhausted %d attempts for model %s",
            total_attempts,
            self.model,
        )
        raise LLMError(
            f"LLM call to {self.model} failed after {total_attempts} attempts"
        ) from last_exc


class FailoverLLMClient:
    """Ordered failover strategy over multiple :class:`LLMClient` providers.

    Each provider owns its retry and circuit-breaker policy. This wrapper only
    moves to the next provider after a provider reports an ``LLMError``. It
    deliberately does not catch ``asyncio.CancelledError`` or arbitrary
    exceptions, so request cancellation and programming errors remain visible
    to the caller.
    """

    def __init__(self, providers: Sequence[LLMClient]) -> None:
        """Initialize with at least one provider in priority order."""
        if not providers:
            raise ValueError("providers must contain at least one client")
        self.providers = tuple(providers)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the first successful response, failing only after all providers."""
        failures: list[LLMError] = []
        for index, provider in enumerate(self.providers):
            with llm_tracer.start_as_current_span("methodos.llm.failover") as span:
                span.set_attribute("methodos.llm.provider_index", index)
                span.set_attribute(
                    "methodos.llm.provider_type", type(provider).__name__
                )
                try:
                    return await provider.complete(
                        system=system,
                        user=user,
                        json_schema=json_schema,
                        temperature=temperature,
                    )
                except LLMError as exc:
                    span.record_exception(exc)
                    failures.append(exc)
                    logger.warning(
                        "LLM provider %d/%d failed; trying the next provider",
                        index + 1,
                        len(self.providers),
                    )
        error = LLMError(
            f"all {len(self.providers)} LLM providers failed"
        )
        for failure in failures:
            error.add_note(str(failure))
        raise error from failures[-1]


__all__ = [
    "LLM_ATTEMPTS",
    "BudgetExceededError",
    "CircuitBreaker",
    "CircuitOpenError",
    "FailoverLLMClient",
    "LLMClient",
    "LLMError",
    "LiteLLMClient",
    "SQLiteTokenBudget",
    "TokenBudget",
    "is_retryable_provider_error",
]
