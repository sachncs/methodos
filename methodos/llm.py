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
- The Protocol uses ellipsis method bodies — runtime duck typing plus
  static `typing.Protocol` conformance via mypy.
- `LiteLLMClient` does NOT use lazy imports: `litellm` is a hard
  dependency, so it sits in the top-of-file import block.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Contract any LLM backend must satisfy.

    Methods:
        complete: send a chat-completion request and return the assistant
            message text (and optionally parse as `json_schema`).
    """

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str: ...


class LLMError(RuntimeError):
    """Raised when an LLM backend fails after exhausting retries."""


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
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def model(self) -> str:
        """Configured model identifier (read-only)."""
        return self._model

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
        errors, rate limits). After exhaustion, raises the last
        underlying exception as `LLMError`.

        When `json_schema` is provided, the response is requested with
        litellm's `response_format` set to a JSON Schema describing the
        target Pydantic model. The returned text is the model's JSON
        response (not parsed — parsing is the caller's responsibility).
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.__name__,
                    "schema": json_schema.model_json_schema(),
                },
            }

        last_exc: Exception | None = None
        total_attempts = self._max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = await litellm.acompletion(**kwargs)
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise LLMError(
                        f"expected str content from {self._model}, "
                        f"got {type(content).__name__}"
                    )
                return content
            except LLMError:
                # Programmatic content-shape errors should not be retried.
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "litellm attempt %d/%d for model %s failed: %s",
                    attempt + 1,
                    total_attempts,
                    self._model,
                    exc,
                )
        assert last_exc is not None
        logger.error(
            "litellm exhausted %d attempts for model %s",
            total_attempts,
            self._model,
        )
        raise LLMError(
            f"LLM call to {self._model} failed after {total_attempts} attempts"
        ) from last_exc


__all__ = ["LLMClient", "LLMError", "LiteLLMClient"]
