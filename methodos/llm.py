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
import time
from typing import Any, Protocol, runtime_checkable

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
        max_output_tokens: int = 2048,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the LiteLLM-backed client.

        Args:
            model: litellm model identifier.
            api_key: optional API key override.
            api_base: optional custom API base URL.
            timeout_seconds: per-request timeout.
            max_retries: retries on transient failures; total attempts = max_retries + 1.
            retry_backoff_seconds: initial exponential delay between retries.
            max_output_tokens: upper bound for generated response tokens.
            circuit_breaker: optional upstream failure breaker; a new local
                breaker is created when omitted.
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
        if max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {max_output_tokens}")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_output_tokens = max_output_tokens
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

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
                raise CircuitOpenError(
                    f"LLM provider circuit is open for model {self.model}"
                )
            try:
                response = await litellm.acompletion(**kwargs)
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise LLMError(
                        f"expected str content from {self.model}, got {type(content).__name__}"
                    )
                self.circuit_breaker.record_success()
                return content
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                self.circuit_breaker.record_failure()
                logger.warning(
                    "litellm attempt %d/%d for model %s failed: %s",
                    attempt + 1,
                    total_attempts,
                    self.model,
                    exc,
                )
                if attempt < total_attempts - 1 and self.retry_backoff_seconds > 0:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_exc is not None
        logger.error(
            "litellm exhausted %d attempts for model %s",
            total_attempts,
            self.model,
        )
        raise LLMError(
            f"LLM call to {self.model} failed after {total_attempts} attempts"
        ) from last_exc


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "LLMClient",
    "LLMError",
    "LiteLLMClient",
]
