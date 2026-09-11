"""Observability primitives for the `methodos` service.

This module groups the two observability concerns:

- **Logging** (`configure_logging`, `JSONFormatter`, `LEVELS`,
  `RESERVED_LOG_KEYS`): human-readable or structured JSON log lines,
  routed through stdlib `logging` so existing
  `logger = logging.getLogger(__name__)` call sites keep working.

- **Metrics** (`REQUESTS_TOTAL`, `REQUEST_DURATION_SECONDS`,
  `GUIDANCE_CACHE_TOTAL`, `LLM_ATTEMPTS_TOTAL`, `EVOLUTION_ROUNDS_TOTAL`,
  `INFO`): Prometheus instruments on the default
  `prometheus_client` registry, exposed at `GET /metrics`.

Logging
-------

`configure_logging(level="INFO", json_format=False)` configures the
root logger once. In JSON mode, each log line is a single-line JSON
object with `timestamp` (UTC ISO-8601 with `Z` suffix), `level`,
`logger`, `message`, and any `extra={...}` fields passed at the call
site. Non-serializable values are rendered via `repr`.

Metrics
-------

Each instrument is a module-level singleton; importing this module is
sufficient to register them on the default registry. The `INFO` gauge
records the build version and component (call `INFO.labels(version=...,
component=...).set(1)` once at startup).

Usage from the service:

    from prometheus_client import generate_latest
    from methodos.observability import CONTENT_TYPE_LATEST

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


# Keys reserved by the logging library; never copy them into the JSON
# payload from `extra=` or the LogRecord's own attributes.
RESERVED_LOG_KEYS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


LEVELS: Final[dict[str, int]] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class JSONFormatter(logging.Formatter):
    """Render `LogRecord`s as single-line JSON objects.

    Custom fields may be added by passing them through `extra=` at the log
    call site (e.g. `logger.info("saved", extra={"graph_id": "g1"})`).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in RESERVED_LOG_KEYS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
            }
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    *,
    level: str = "INFO",
    json_format: bool = False,
    stream: Any = None,
) -> None:
    """Configure the root logger for `methodos`.

    Args:
        level: log level name (case-insensitive). One of DEBUG/INFO/
            WARNING/ERROR/CRITICAL.
        json_format: when true, emit structured JSON log lines; otherwise
            use a human-readable text format.
        stream: an IO stream for log output; defaults to `sys.stderr`.
            Pass a `StringIO` in tests to capture output.
    """
    normalized = level.upper()
    if normalized not in LEVELS:
        raise ValueError(f"unknown log level: {level!r}; expected one of {sorted(LEVELS)}")

    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace any pre-existing handlers (e.g., uvicorn's defaults) so the
    # chosen format is what operators see.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(LEVELS[normalized])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


REQUESTS_TOTAL: Counter = Counter(
    "methodos_requests_total",
    "Total HTTP requests handled by the methodos service.",
    labelnames=("method", "path", "status"),
)

REQUEST_DURATION_SECONDS: Histogram = Histogram(
    "methodos_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

GUIDANCE_CACHE_TOTAL: Counter = Counter(
    "methodos_guidance_cache_total",
    "Guidance cache lookups by outcome (hit/miss).",
    labelnames=("result",),
)

LLM_ATTEMPTS_TOTAL: Counter = Counter(
    "methodos_llm_attempts_total",
    "LLM completion attempts by outcome (success/retry/exhausted).",
    labelnames=("outcome", "model"),
)

EVOLUTION_ROUNDS_TOTAL: Counter = Counter(
    "methodos_evolution_rounds_total",
    "Evolution rounds by outcome.",
    labelnames=("outcome",),
)

INFO: Gauge = Gauge(
    "methodos_info",
    "Build info; value is always 1, labels carry metadata.",
    labelnames=("version", "component"),
)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "EVOLUTION_ROUNDS_TOTAL",
    "GUIDANCE_CACHE_TOTAL",
    "INFO",
    "LEVELS",
    "LLM_ATTEMPTS_TOTAL",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION_SECONDS",
    "RESERVED_LOG_KEYS",
    "JSONFormatter",
    "configure_logging",
    "generate_latest",
]
