"""Tests for `methodos.observability` (logging + metrics)."""

from __future__ import annotations

import io
import json
import logging

import pytest

from methodos.observability import (
    EVOLUTION_ROUNDS_TOTAL,
    GUIDANCE_CACHE_TOTAL,
    INFO,
    LEVELS,
    LLM_ATTEMPTS_TOTAL,
    REQUEST_DURATION_SECONDS,
    REQUESTS_TOTAL,
    RESERVED_LOG_KEYS,
    JSONFormatter,
    configure_logging,
)


@pytest.fixture(autouse=True)
def reset_root_logger() -> None:
    """Snapshot root handlers before each test and restore after."""
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved:
        root.addHandler(handler)
    root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_default_text_format(self) -> None:
        stream = io.StringIO()
        configure_logging(stream=stream)
        logger = logging.getLogger("methodos.test.text")
        logger.info("hello %s", "world")
        output = stream.getvalue()
        assert "methodos.test.text" in output
        assert "INFO" in output
        assert "hello world" in output

    def test_json_format(self) -> None:
        stream = io.StringIO()
        configure_logging(stream=stream, json_format=True)
        logger = logging.getLogger("methodos.test.json")
        logger.info("structured", extra={"graph_id": "g1", "round": 3})
        line = stream.getvalue().strip()
        payload = json.loads(line)
        assert payload["message"] == "structured"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "methodos.test.json"
        assert payload["graph_id"] == "g1"
        assert payload["round"] == 3
        assert payload["timestamp"].endswith("Z")

    def test_level_case_insensitive(self) -> None:
        configure_logging(level="debug", stream=io.StringIO())
        assert logging.getLogger().level == logging.DEBUG

    def test_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError):
            configure_logging(level="LOUD")

    def test_replaces_existing_handlers(self) -> None:
        root = logging.getLogger()
        root.addHandler(logging.NullHandler())
        root.addHandler(logging.NullHandler())
        configure_logging(stream=io.StringIO())
        assert len(root.handlers) == 1

    def test_extra_with_non_serializable_value(self) -> None:
        stream = io.StringIO()
        configure_logging(stream=stream, json_format=True)
        logger = logging.getLogger("methodos.test.bad")

        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        logger.info("bad extra", extra={"obj": Weird()})
        line = stream.getvalue().strip()
        payload = json.loads(line)
        assert payload["obj"] == "<weird>"


class TestJSONFormatter:
    def test_includes_exception_info(self) -> None:
        stream = io.StringIO()
        configure_logging(stream=stream, json_format=True)
        logger = logging.getLogger("methodos.test.exc")
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("oops")
        payload = json.loads(stream.getvalue().strip())
        assert payload["exception"]["type"] == "RuntimeError"
        assert payload["exception"]["message"] == "boom"
        assert "Traceback" in payload["traceback"]

    def test_reserved_keys_skipped(self) -> None:
        # LogRecord internals like 'args', 'pathname' must not leak into
        # the JSON payload. `message` is the only reserved key we
        # intentionally expose.
        stream = io.StringIO()
        configure_logging(stream=stream, json_format=True)
        logger = logging.getLogger("methodos.test.reserved")
        logger.info("msg")
        payload = json.loads(stream.getvalue().strip())
        for key in RESERVED_LOG_KEYS:
            if key == "message":
                continue
            assert key not in payload, f"reserved key {key!r} leaked into payload"


class TestLoggingConstants:
    def test_levels_contains_expected(self) -> None:
        assert "DEBUG" in LEVELS
        assert "INFO" in LEVELS
        assert "WARNING" in LEVELS
        assert "ERROR" in LEVELS
        assert "CRITICAL" in LEVELS

    def test_reserved_keys_is_frozenset(self) -> None:
        assert isinstance(RESERVED_LOG_KEYS, frozenset)


class TestFormatterStandalone:
    """`JSONFormatter` is usable without `configure_logging`."""

    def test_format_record(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="x.py",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        payload = json.loads(output)
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "x"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetricsInstruments:
    def test_increment_request_counter(self) -> None:
        # The autouse `reset_prometheus_counters` fixture in conftest
        # gives us a clean slate, so a single .inc() must equal 1.
        REQUESTS_TOTAL.labels(method="GET", path="/x", status="200").inc()
        assert (
            REQUESTS_TOTAL.labels(method="GET", path="/x", status="200")._value.get()  # type: ignore[attr-defined]
            == 1.0
        )

    def test_observe_request_duration(self) -> None:
        # Smoke test: recording samples does not raise.
        REQUEST_DURATION_SECONDS.labels(method="GET", path="/x").observe(0.01)
        REQUEST_DURATION_SECONDS.labels(method="POST", path="/y").observe(2.5)

    def test_guidance_cache_counter(self) -> None:
        GUIDANCE_CACHE_TOTAL.labels(result="hit").inc()
        GUIDANCE_CACHE_TOTAL.labels(result="miss").inc(2)
        assert GUIDANCE_CACHE_TOTAL.labels(result="hit")._value.get() == 1.0  # type: ignore[attr-defined]
        assert GUIDANCE_CACHE_TOTAL.labels(result="miss")._value.get() == 2.0  # type: ignore[attr-defined]

    def test_llm_attempts_counter(self) -> None:
        LLM_ATTEMPTS_TOTAL.labels(outcome="success", model="gpt-4o-mini").inc()
        LLM_ATTEMPTS_TOTAL.labels(outcome="retry", model="gpt-4o-mini").inc()
        LLM_ATTEMPTS_TOTAL.labels(outcome="exhausted", model="claude").inc()
        assert (
            LLM_ATTEMPTS_TOTAL.labels(
                outcome="success", model="gpt-4o-mini"
            )._value.get()  # type: ignore[attr-defined]
            == 1.0
        )

    def test_evolution_rounds_counter(self) -> None:
        EVOLUTION_ROUNDS_TOTAL.labels(outcome="accepted").inc()
        EVOLUTION_ROUNDS_TOTAL.labels(outcome="rejected").inc()
        assert (
            EVOLUTION_ROUNDS_TOTAL.labels(outcome="accepted")._value.get() == 1.0  # type: ignore[attr-defined]
        )

    def test_info_gauge(self) -> None:
        INFO.labels(version="test", component="service").set(1)
        assert INFO.labels(version="test", component="service")._value.get() == 1.0
