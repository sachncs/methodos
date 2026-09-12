"""Tests for `methodos.llm`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel

import methodos.llm as llm_module
from methodos.llm import LiteLLMClient, LLMClient, LLMError


class SimpleSchema(BaseModel):
    """Sample Pydantic schema for structured-output tests."""

    answer: str
    confidence: float


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


def test_litellm_client_model_property() -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    assert client.model == "gpt-4o-mini"


class LitellmModuleDouble:
    """Module-shaped double for monkeypatching `litellm` in tests."""

    def __init__(self, fake_acompletion: Any) -> None:
        self.acompletion = staticmethod(fake_acompletion)


@pytest.fixture
def patched_litellm(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Patch `litellm.acompletion` to record calls and return canned responses."""
    state: dict[str, Any] = {
        "calls": [],
        "responses": [{"choices": [{"message": {"content": "ok"}}]}],
        "exceptions": [],
    }

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        state["calls"].append(kwargs)
        if state["exceptions"]:
            exc = state["exceptions"].pop(0)
            raise exc
        if state["responses"]:
            response: dict[str, Any] = state["responses"].pop(0)
            return response
        return {"choices": [{"message": {"content": "default"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(fake_acompletion))
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

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError(f"transient {call_count['n']}")
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(fake_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=3)
    out = await client.complete(system="s", user="u")
    assert out == "recovered"
    assert call_count["n"] == 3


async def test_litellm_complete_raises_llm_error_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All retries fail → raises LLMError wrapping the last underlying error."""

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("always fails")

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(fake_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=2)
    with pytest.raises(LLMError) as excinfo:
        await client.complete(system="s", user="u")
    assert "failed after 3 attempts" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "always fails" in str(excinfo.value.__cause__)


async def test_litellm_complete_does_not_retry_on_programmatic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMError raised inside the call path (non-retryable) propagates immediately."""
    call_count = {"n": 0}

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        # Return a content payload that is NOT a string → triggers LLMError.
        return {"choices": [{"message": {"content": 42}}]}

    monkeypatch.setattr(llm_module, "litellm", LitellmModuleDouble(fake_acompletion))

    client = LiteLLMClient(model="gpt-4o-mini", max_retries=5)
    with pytest.raises(LLMError, match="expected str content"):
        await client.complete(system="s", user="u")
    assert call_count["n"] == 1


async def test_litellm_complete_uses_configured_timeout(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini", timeout_seconds=12.5)
    await client.complete(system="s", user="u")
    assert patched_litellm["calls"][0]["timeout"] == 12.5


async def test_litellm_complete_propagates_temperature(
    patched_litellm: dict[str, Any],
) -> None:
    client = LiteLLMClient(model="gpt-4o-mini")
    await client.complete(system="s", user="u", temperature=0.7)
    assert patched_litellm["calls"][0]["temperature"] == 0.7


async def test_litellm_error_is_runtime_error() -> None:
    """LLMError inherits RuntimeError so callers can catch broad."""
    assert issubclass(LLMError, RuntimeError)
