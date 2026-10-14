"""FastAPI app factory for the `methodos` hosted service.

Engineering notes:
- `create_app(repo, llm)` is the public factory; everything else lives
  inside it (routers, lifespan, dependencies). This is the standard
  FastAPI pattern for testable apps.
- HTTP request/response models live in this module as Pydantic DTOs,
  distinct from the domain models in `methodos.schema`.
- No module-level globals except `app = create_app()` for the uvicorn
  entry point.
- Lifespan wires the configured `repo` and `llm` into `app.state`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import Message

from methodos.graph import match_node, neighborhood
from methodos.guidance import generate_guidance
from methodos.jobs import (
    EvolutionJobMetricsStore,
    EvolutionJobWorker,
    JobIdempotencyConflictError,
    JobQuotaExceededError,
)
from methodos.llm import (
    LiteLLMClient,
    LLMClient,
    PostgresTokenBudget,
    ScopedTokenBudget,
    SQLiteTokenBudget,
    TokenPricing,
)
from methodos.repo import (
    AtomicRepository,
    FilesystemRepository,
    GraphAlreadyExistsError,
    GraphRevisionConflictError,
    PostgresRepository,
    RateLimitRepository,
    Repository,
    SnapshotRepository,
    SQLiteRepository,
    VersionedRepository,
    build_repository,
)
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger("methodos.service")
request_tracer = trace.get_tracer("methodos.service")
repository_tracer = trace.get_tracer("methodos.repository")
GRAPH_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
TENANT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"


HTTP_REQUESTS = Counter(
    "methodos_http_requests_total",
    "Total HTTP requests handled by methodos.",
    ["method", "status"],
)
HTTP_LATENCY = Histogram(
    "methodos_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method"],
)
AUTH_FAILURES = Counter(
    "methodos_auth_failures_total",
    "Authentication and authorization failures by stable reason.",
    ["reason"],
)
EVOLUTION_JOBS = Gauge(
    "methodos_evolution_jobs",
    "Evolution jobs by queue state.",
    ["state"],
)
EVOLUTION_WORKER = Gauge(
    "methodos_evolution_worker",
    "Evolution worker liveness signals.",
    ["state"],
)


# ----------------------------------------------------------------------------
# HTTP DTOs (separate from domain models to keep the wire format flexible)
# ----------------------------------------------------------------------------


class GraphCreateRequest(BaseModel):
    """Request body for `POST /v1/graphs`."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128, pattern=GRAPH_ID_PATTERN)
    from_graph_id: str | None = Field(default=None, max_length=128, pattern=GRAPH_ID_PATTERN)


class GuidanceRequest(BaseModel):
    """Request body for `POST /v1/graphs/{graph_id}/guidance`."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)
    trajectory: list[tuple[str, str]] = Field(default_factory=list)
    window: int = Field(default=3, ge=0, le=20)
    guidance_hops: int = Field(default=2, ge=0, le=10)


class GuidanceResponse(BaseModel):
    """Response body for the guidance endpoint."""

    model_config = ConfigDict(extra="forbid")
    guidance: str


class EvolveRequest(BaseModel):
    """Request body for `POST /v1/graphs/{graph_id}/evolve`.

    Evolution is queued through an injected host-owned EvolutionRunner. The
    default app has no runner and fails closed; production hosts must provide
    a durable job implementation.
    """

    model_config = ConfigDict(extra="forbid")
    k_rounds: int = Field(default=10, ge=1, le=100)


class EvolveResponse(BaseModel):
    """Response body for an accepted or unavailable evolution request."""

    model_config = ConfigDict(extra="forbid")
    detail: str
    job_id: str | None = None
    status: str | None = None


class EvolutionJobStatus(BaseModel):
    """Public status returned while an evolution job is being processed."""

    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=128, pattern=GRAPH_ID_PATTERN)
    status: str = Field(min_length=1, max_length=32)
    detail: str | None = None


@runtime_checkable
class EvolutionRunner(Protocol):
    """Host-owned job launcher for asynchronous evolution."""

    async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
        """Queue evolution and return a stable job identifier."""
        ...

    async def status(self, job_id: str) -> EvolutionJobStatus | None:
        """Return a job status or `None` when the job is unknown."""
        ...


@runtime_checkable
class IdempotentEvolutionRunner(EvolutionRunner, Protocol):
    """Evolution runner capability for safe client retries."""

    async def start_idempotent(
        self,
        *,
        graph_id: str,
        k_rounds: int,
        request_id: str,
        idempotency_key: str,
    ) -> str:
        """Return the same job id for repeated keys with the same request."""
        ...


@runtime_checkable
class TenantScopedEvolutionRunner(EvolutionRunner, Protocol):
    """Runner capability that durably binds jobs and status to a tenant."""

    async def start_for_tenant(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        k_rounds: int,
        request_id: str,
    ) -> str:
        """Queue a job owned by `tenant_id`."""
        ...

    async def start_idempotent_for_tenant(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        k_rounds: int,
        request_id: str,
        idempotency_key: str,
    ) -> str:
        """Queue or deduplicate a tenant-owned job."""
        ...

    async def status_for_tenant(
        self,
        *,
        tenant_id: str,
        job_id: str,
    ) -> EvolutionJobStatus | None:
        """Return status only when the job belongs to `tenant_id`."""
        ...


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    model_config = ConfigDict(extra="forbid")
    status: str


class ReadinessResponse(BaseModel):
    """Response body for `GET /ready`."""

    model_config = ConfigDict(extra="forbid")
    status: str


class ErrorResponse(BaseModel):
    """Stable error envelope returned by every API failure."""

    model_config = ConfigDict(extra="forbid")
    detail: str | list[dict[str, object]]
    code: str
    request_id: str


def validate_graph_id(graph_id: str) -> str:
    """Validate a graph identifier before it reaches a repository."""
    if re.fullmatch(GRAPH_ID_PATTERN, graph_id) is None:
        raise HTTPException(status_code=422, detail="invalid graph_id")
    return graph_id


def validate_tenant_id(tenant_id: str) -> str:
    """Validate a tenant identifier before it enters the request context."""
    if re.fullmatch(TENANT_ID_PATTERN, tenant_id) is None:
        raise ValueError("tenant ids must match the tenant identifier pattern")
    return tenant_id


def validate_idempotency_key(key: str | None) -> str | None:
    """Validate an optional idempotency key before passing it to a runner."""
    if key is None:
        return None
    if re.fullmatch(IDEMPOTENCY_KEY_PATTERN, key) is None:
        raise HTTPException(status_code=422, detail="invalid Idempotency-Key")
    return key


def validate_cors_origins(origins: list[str], *, authenticated: bool) -> list[str]:
    """Validate explicit browser origins before enabling CORS.

    Wildcard origins are rejected for authenticated services because they make
    browser-origin policy indistinguishable from public access.
    """
    validated: list[str] = []
    for origin in origins:
        parsed = urlsplit(origin)
        if origin == "*":
            if authenticated:
                raise ValueError("wildcard CORS origins are not allowed with authentication")
            raise ValueError("invalid CORS origin: '*' requires an explicit origin")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"invalid CORS origin: {origin!r}")
        validated.append(origin)
    return validated


VALID_ROLES = frozenset({"reader", "writer", "operator"})


def parse_api_key_roles(raw: str) -> dict[str, frozenset[str]]:
    """Parse `key=role|role` entries without exposing key material."""
    roles_by_key: dict[str, frozenset[str]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, raw_roles = item.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "METHODOS_API_KEY_ROLES must be comma-separated key=role|role entries"
            ) from exc
        key = key.strip()
        roles = frozenset(role.strip().lower() for role in raw_roles.split("|") if role.strip())
        if not key or not roles or not roles <= VALID_ROLES:
            raise ValueError("METHODOS_API_KEY_ROLES contains an invalid key or role")
        if key in roles_by_key:
            raise ValueError("duplicate API key in METHODOS_API_KEY_ROLES")
        roles_by_key[key] = roles
    return roles_by_key


def validate_api_key_roles(
    roles_by_key: dict[str, frozenset[str]], configured_keys: tuple[str, ...]
) -> None:
    """Reject role assignments that cannot affect an authenticated key.

    A typo in a production role mapping must fail startup instead of creating
    a false impression that a key has been restricted.
    """
    unknown_keys = sorted(set(roles_by_key) - set(configured_keys))
    if unknown_keys:
        raise ValueError(
            "METHODOS_API_KEY_ROLES contains keys that are not configured: "
            + ", ".join(repr(key) for key in unknown_keys)
        )


def _optional_positive_int(name: str) -> int | None:
    """Parse an optional positive integer environment setting."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_nonnegative_float(name: str) -> float | None:
    """Parse an optional non-negative decimal environment setting."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _positive_float(name: str, default: float) -> float:
    """Parse a finite positive decimal environment setting."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _build_scoped_budget(repo: Repository) -> ScopedTokenBudget | None:
    """Build durable graph-scoped budgets from explicit deployment settings."""
    total = _optional_positive_int("METHODOS_TOKEN_BUDGET_TOTAL")
    maximum_input = _optional_positive_int("METHODOS_TOKEN_BUDGET_INPUT")
    maximum_cost = _optional_nonnegative_float("METHODOS_TOKEN_BUDGET_COST_USD")
    input_price = _optional_nonnegative_float("METHODOS_TOKEN_PRICE_INPUT_PER_1K_USD")
    output_price = _optional_nonnegative_float("METHODOS_TOKEN_PRICE_OUTPUT_PER_1K_USD")
    if total is None and maximum_input is None:
        if any(value is not None for value in (maximum_cost, input_price, output_price)):
            raise ValueError(
                "METHODOS_TOKEN_BUDGET_TOTAL and METHODOS_TOKEN_BUDGET_INPUT must be set "
                "when pricing is configured"
            )
        return None
    if total is None or maximum_input is None:
        raise ValueError(
            "METHODOS_TOKEN_BUDGET_TOTAL and METHODOS_TOKEN_BUDGET_INPUT must be set together"
        )
    pricing: TokenPricing | None = None
    if any(value is not None for value in (maximum_cost, input_price, output_price)):
        if maximum_cost is None or input_price is None or output_price is None:
            raise ValueError(
                "METHODOS_TOKEN_BUDGET_COST_USD, METHODOS_TOKEN_PRICE_INPUT_PER_1K_USD, "
                "and METHODOS_TOKEN_PRICE_OUTPUT_PER_1K_USD must be set together"
            )
        pricing = TokenPricing(
            max_total_cost_usd=maximum_cost,
            input_cost_per_1k_usd=input_price,
            output_cost_per_1k_usd=output_price,
        )

    factory: Callable[[str], SQLiteTokenBudget | PostgresTokenBudget]
    if isinstance(repo, SQLiteRepository):

        def sqlite_factory(scope: str) -> SQLiteTokenBudget:
            return SQLiteTokenBudget(
                database_path=repo.db_path,
                key=f"llm:{scope}",
                max_total_tokens=total,
                max_input_tokens=maximum_input,
                pricing=pricing,
            )

        factory = sqlite_factory

    elif isinstance(repo, PostgresRepository):

        def postgres_factory(scope: str) -> PostgresTokenBudget:
            return PostgresTokenBudget(
                dsn=repo.dsn,
                key=f"llm:{scope}",
                max_total_tokens=total,
                max_input_tokens=maximum_input,
                pricing=pricing,
            )

        factory = postgres_factory

    else:
        raise ValueError(
            "durable token budgets require PGRAPH_BACKEND=sqlite or postgres; "
            "filesystem does not provide shared quota enforcement"
        )
    return ScopedTokenBudget(factory)


def validate_runtime_mode(
    mode: str,
    *,
    repo: Repository,
    authenticated: bool,
    credentials: tuple[str, ...] = (),
    role_keys: frozenset[str] = frozenset(),
) -> str:
    """Fail closed for deployment modes that are intended for shared hosts.

    Filesystem persistence is process-local and API authentication is disabled
    by default for developer ergonomics. Neither default is safe for a shared
    staging or production service, so those modes require explicit durable
    persistence and credentials at startup.
    """
    normalized = mode.strip().lower() or "development"
    if normalized not in {"development", "test", "staging", "production"}:
        raise ValueError("METHODOS_ENV must be one of development, test, staging, or production")
    if normalized in {"staging", "production"}:
        if not authenticated:
            raise ValueError(
                f"METHODOS_ENV={normalized} requires METHODOS_API_KEYS or METHODOS_TENANT_KEYS"
            )
        if any(len(credential) < 16 for credential in credentials):
            raise ValueError(
                f"METHODOS_ENV={normalized} requires API and tenant keys of at least 16 characters"
            )
        if set(role_keys) != set(credentials):
            raise ValueError(
                f"METHODOS_ENV={normalized} requires explicit roles for every configured API key"
            )
        if isinstance(repo, FilesystemRepository):
            raise ValueError(
                f"METHODOS_ENV={normalized} requires SQLite or Postgres persistence; "
                "filesystem is process-local"
            )
    return normalized


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------


def create_app(
    *,
    repo: Repository | None = None,
    llm: LLMClient | None = None,
    evolution_runner: EvolutionRunner | None = None,
    evolution_worker: EvolutionJobWorker | None = None,
) -> FastAPI:
    """Build a FastAPI app wired to the given (or default-built) backend.

    Args:
        repo: Repository implementation. Defaults to `build_repository()`
            which selects based on environment variables.
        llm: LLMClient implementation. Defaults to LiteLLMClient with
            model `gpt-4o-mini` (override via `OPENAI_API_KEY` env var).
        evolution_runner: optional host-owned evolution job launcher.
        evolution_worker: optional supervised worker whose liveness is exported
            through `/metrics`.
    """
    backend_repo: Repository = repo if repo is not None else build_repository()
    scoped_budget = _build_scoped_budget(backend_repo) if llm is None else None
    backend_llm: LLMClient = (
        llm
        if llm is not None
        else LiteLLMClient(
            model="gpt-4o-mini",
            budget=scoped_budget,
        )
    )
    configured_api_keys = tuple(
        key.strip() for key in os.environ.get("METHODOS_API_KEYS", "").split(",") if key.strip()
    )
    if not configured_api_keys:
        legacy_api_key = os.environ.get("METHODOS_API_KEY", "").strip()
        configured_api_keys = (legacy_api_key,) if legacy_api_key else ()
    configured_tenant_keys = os.environ.get("METHODOS_TENANT_KEYS", "").strip()
    tenant_keys: dict[str, str] = {}
    if configured_tenant_keys:
        for item in configured_tenant_keys.split(","):
            try:
                tenant_id, tenant_key = item.split(":", maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    "METHODOS_TENANT_KEYS must be comma-separated tenant:key pairs"
                ) from exc
            tenant_id = validate_tenant_id(tenant_id.strip())
            tenant_key = tenant_key.strip()
            if not tenant_key:
                raise ValueError("METHODOS_TENANT_KEYS cannot contain empty keys")
            if tenant_id in tenant_keys:
                raise ValueError(f"duplicate tenant id: {tenant_id}")
            tenant_keys[tenant_id] = tenant_key
    configured_tenant_id = validate_tenant_id(
        os.environ.get("METHODOS_TENANT_ID", "default").strip() or "default"
    )
    api_key_roles = parse_api_key_roles(os.environ.get("METHODOS_API_KEY_ROLES", ""))
    validate_api_key_roles(
        api_key_roles,
        tuple(tenant_keys.values()) if tenant_keys else configured_api_keys,
    )
    validate_runtime_mode(
        os.environ.get("METHODOS_ENV", "development"),
        repo=backend_repo,
        authenticated=bool(configured_api_keys or tenant_keys),
        credentials=tuple(tenant_keys.values()) if tenant_keys else configured_api_keys,
        role_keys=frozenset(api_key_roles),
    )
    configured_rate_limit = int(os.environ.get("METHODOS_RATE_LIMIT_PER_MINUTE", "0"))
    if configured_rate_limit < 0:
        raise ValueError("METHODOS_RATE_LIMIT_PER_MINUTE must be non-negative")
    max_body_bytes = int(os.environ.get("METHODOS_MAX_BODY_BYTES", str(1024 * 1024)))
    if max_body_bytes <= 0:
        raise ValueError("METHODOS_MAX_BODY_BYTES must be positive")
    body_read_timeout = _positive_float("METHODOS_BODY_READ_TIMEOUT_SECONDS", 10.0)
    request_windows: defaultdict[str, deque[float]] = defaultdict(deque)
    shared_rate_limiter = backend_repo if isinstance(backend_repo, RateLimitRepository) else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.repo = backend_repo
        app.state.llm = backend_llm
        initialize = getattr(backend_repo, "initialize", None)
        close = getattr(backend_repo, "close", None)
        close_llm = getattr(backend_llm, "close", None)
        if initialize is not None:
            await initialize()
        logger.info("methodos service started")
        try:
            yield
        finally:
            if close_llm is not None:
                await close_llm()
            if close is not None:
                await close()
            logger.info("methodos service stopped")

    app = FastAPI(title="methodos", version="0.1.0", lifespan=lifespan)

    def error_payload(request: Request, detail: object, code: str) -> dict[str, object]:
        """Build the documented error envelope without leaking internals."""
        safe_detail: str | list[dict[str, object]]
        if isinstance(detail, str):
            safe_detail = detail
        elif isinstance(detail, list) and all(isinstance(item, dict) for item in detail):
            encoded_detail = jsonable_encoder(detail)
            safe_detail = (
                encoded_detail
                if isinstance(encoded_detail, list)
                and all(isinstance(item, dict) for item in encoded_detail)
                else "request failed"
            )
        else:
            safe_detail = "request failed"
        return {
            "detail": safe_detail,
            "code": code,
            "request_id": getattr(request.state, "request_id", "unknown"),
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Return HTTP errors using the stable application envelope."""
        code = {
            401: "unauthorized",
            403: "forbidden",
            400: "invalid_request",
            404: "not_found",
            409: "conflict",
            422: "invalid_request",
            428: "precondition_required",
            429: "rate_limited",
            413: "request_too_large",
            501: "not_implemented",
            503: "unavailable",
        }.get(exc.status_code, "request_failed")
        if exc.status_code in {401, 403}:
            AUTH_FAILURES.labels(code).inc()
        headers = dict(exc.headers or {})
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(request, exc.detail, code),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return Pydantic/FastAPI validation failures in the same envelope."""
        return JSONResponse(
            status_code=422,
            content=error_payload(request, exc.errors(), "invalid_request"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Hide implementation details while preserving a correlated log."""
        logger.exception("unhandled request failure", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_payload(request, "internal server error", "internal_error"),
        )

    def harden_response(response: Response, request_id: str) -> Response:
        """Attach correlation and browser-security headers to every response."""
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["cache-control"] = "no-store"
        return response

    cors_origins = [
        origin.strip()
        for origin in os.environ.get("METHODOS_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    cors_origins = validate_cors_origins(
        cors_origins,
        authenticated=bool(configured_api_keys or tenant_keys),
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=[
                "content-type",
                "idempotency-key",
                "x-api-key",
                "x-request-id",
                "x-tenant-id",
            ],
        )

    @app.middleware("http")
    async def request_hardening(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Attach request correlation and baseline browser security headers."""
        supplied_request_id = request.headers.get("x-request-id")
        request_id = (
            supplied_request_id
            if supplied_request_id and re.fullmatch(REQUEST_ID_PATTERN, supplied_request_id)
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        started_at = time.perf_counter()
        original_receive = request._receive
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = -1
            if declared_length < 0 or declared_length > max_body_bytes:
                oversized_response: Response = JSONResponse(
                    status_code=413,
                    content=error_payload(
                        request, "request body is too large", "request_too_large"
                    ),
                )
                HTTP_REQUESTS.labels(request.method, "413").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                return harden_response(oversized_response, request_id)
        body = bytearray()
        while True:
            try:
                message = await asyncio.wait_for(original_receive(), body_read_timeout)
            except TimeoutError:
                timeout_response: Response = JSONResponse(
                    status_code=408,
                    content=error_payload(
                        request,
                        "request body read timed out",
                        "request_timeout",
                    ),
                )
                HTTP_REQUESTS.labels(request.method, "408").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                return harden_response(timeout_response, request_id)
            if message["type"] == "http.disconnect":
                request._receive = original_receive
                disconnected_response: Response = JSONResponse(
                    status_code=499,
                    content=error_payload(
                        request,
                        "client disconnected before the request was received",
                        "client_disconnected",
                    ),
                )
                HTTP_REQUESTS.labels(request.method, "499").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                return harden_response(disconnected_response, request_id)
            chunk = message.get("body", b"")
            if isinstance(chunk, bytes):
                if len(body) + len(chunk) > max_body_bytes:
                    oversized_response = JSONResponse(
                        status_code=413,
                        content=error_payload(
                            request, "request body is too large", "request_too_large"
                        ),
                    )
                    HTTP_REQUESTS.labels(request.method, "413").inc()
                    HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                    return harden_response(oversized_response, request_id)
                body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def replay_body() -> Message:
            """Replay the bounded body once to downstream request parsing."""
            request._receive = original_receive
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        request._receive = replay_body
        public_path = request.url.path in {
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        }
        if configured_rate_limit and not public_path:
            now = time.monotonic()
            client_host = request.client.host if request.client else "unknown"
            rate_key = (
                hashlib.sha256(request.headers.get("x-api-key", "").encode()).hexdigest()
                if tenant_keys
                else client_host
            )
            if shared_rate_limiter is not None:
                allowed, retry_after = await shared_rate_limiter.consume_rate_limit(
                    hashlib.sha256((rate_key or client_host).encode()).hexdigest(),
                    configured_rate_limit,
                )
            else:
                window = request_windows[rate_key or client_host]
                while window and now - window[0] >= 60:
                    window.popleft()
                allowed = len(window) < configured_rate_limit
                retry_after = 60
                if allowed:
                    window.append(now)
            if not allowed:
                limited_response: Response = JSONResponse(
                    status_code=429,
                    content=error_payload(request, "rate limit exceeded", "rate_limited"),
                    headers={"retry-after": str(retry_after)},
                )
                HTTP_REQUESTS.labels(request.method, "429").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                return harden_response(limited_response, request_id)
        if (configured_api_keys or tenant_keys) and not public_path:
            try:
                tenant_id, _ = tenant_auth(request)
            except HTTPException as exc:
                code = "forbidden" if exc.status_code == 403 else "unauthorized"
                AUTH_FAILURES.labels(code).inc()
                auth_response = JSONResponse(
                    status_code=exc.status_code,
                    content=error_payload(request, exc.detail, code),
                    headers={"www-authenticate": "ApiKey"} if exc.status_code == 401 else {},
                )
                HTTP_REQUESTS.labels(request.method, str(exc.status_code)).inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                return harden_response(auth_response, request_id)
            request.state.tenant_id = tenant_id
        with request_tracer.start_as_current_span("methodos.http.request") as span:
            span.set_attribute("http.request.method", request.method)
            span.set_attribute("url.path", request.url.path)
            span.set_attribute("methodos.request_id", request_id)
        response = await call_next(request)
        HTTP_REQUESTS.labels(request.method, str(response.status_code)).inc()
        duration = time.perf_counter() - started_at
        HTTP_LATENCY.labels(request.method).observe(duration)
        structured_logger.info(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=round(duration, 6),
        )
        return harden_response(response, request_id)

    def get_repo() -> Repository:
        repo: Repository = app.state.repo
        return repo

    def get_llm() -> LLMClient:
        llm: LLMClient = app.state.llm
        return llm

    def request_tenant(request: Request) -> str:
        """Return the authenticated tenant assigned by request middleware."""
        tenant_id = getattr(request.state, "tenant_id", configured_tenant_id)
        return validate_tenant_id(tenant_id)

    def require_role(request: Request, role: str) -> None:
        """Require a role assigned by authentication middleware."""
        roles = getattr(request.state, "roles", VALID_ROLES)
        role_grants = {
            "reader": {"reader", "writer", "operator"},
            "writer": {"writer", "operator"},
            "operator": {"operator"},
        }
        if not roles.intersection(role_grants[role]):
            raise HTTPException(status_code=403, detail=f"{role} role is required")

    def storage_graph_id(tenant_id: str, graph_id: str) -> str:
        """Map a public graph id into an isolated tenant namespace."""
        if not tenant_keys:
            return graph_id
        digest = hashlib.sha256(f"{tenant_id}:{graph_id}".encode()).hexdigest()
        return digest

    async def load_scoped_graph(repo: Repository, tenant_id: str, graph_id: str) -> ProceduralGraph:
        """Load a graph from the authenticated tenant namespace."""
        with repository_tracer.start_as_current_span("methodos.repository.load_graph") as span:
            span.set_attribute("methodos.repository.backend", type(repo).__name__)
            stored = await repo.load_graph(storage_graph_id(tenant_id, graph_id))
        return stored.model_copy(update={"id": graph_id})

    async def create_scoped_graph(repo: Repository, tenant_id: str, graph: ProceduralGraph) -> None:
        """Persist a graph under the authenticated tenant namespace."""
        stored = graph.model_copy(update={"id": storage_graph_id(tenant_id, graph.id)})
        with repository_tracer.start_as_current_span("methodos.repository.create_graph") as span:
            span.set_attribute("methodos.repository.backend", type(repo).__name__)
            if isinstance(repo, AtomicRepository):
                await repo.create_graph(stored)
            else:
                await repo.save_graph(stored)

    def tenant_auth(request: Request) -> tuple[str, str]:
        """Authenticate a request and return `(tenant_id, api_key)`."""
        supplied_key = request.headers.get("x-api-key", "")
        requested_tenant = request.headers.get("x-tenant-id")
        if tenant_keys:
            for tenant_id, tenant_key in tenant_keys.items():
                if secrets.compare_digest(supplied_key, tenant_key):
                    if requested_tenant and requested_tenant != tenant_id:
                        raise HTTPException(status_code=403, detail="tenant mismatch")
                    request.state.roles = api_key_roles.get(supplied_key, frozenset({"writer"}))
                    return tenant_id, supplied_key
            raise HTTPException(status_code=401, detail="missing or invalid API key")
        if configured_api_keys:
            matched_key = next(
                (
                    configured_key
                    for configured_key in configured_api_keys
                    if secrets.compare_digest(supplied_key, configured_key)
                ),
                None,
            )
            if matched_key is None:
                raise HTTPException(status_code=401, detail="missing or invalid API key")
            request.state.roles = api_key_roles.get(matched_key, frozenset({"writer"}))
        if requested_tenant and requested_tenant != configured_tenant_id:
            raise HTTPException(status_code=403, detail="tenant mismatch")
        return configured_tenant_id, supplied_key

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready(repo: Repository = Depends(get_repo)) -> ReadinessResponse:
        """Report whether the configured repository can accept traffic."""
        try:
            await repo.load_graph("__methodos_readiness_probe__")
        except FileNotFoundError:
            return ReadinessResponse(status="ready")
        except Exception as exc:
            logger.exception("repository readiness probe failed: %s", exc)
            raise HTTPException(status_code=503, detail="repository unavailable") from exc
        return ReadinessResponse(status="ready")

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Expose process and client-library metrics for scraping."""
        job_store = getattr(evolution_runner, "store", None)
        if isinstance(job_store, EvolutionJobMetricsStore):
            try:
                counts = await job_store.counts()
                EVOLUTION_JOBS.labels("queued").set(counts.queued)
                EVOLUTION_JOBS.labels("running").set(counts.running)
                EVOLUTION_JOBS.labels("failed").set(counts.failed)
            except Exception:
                logger.exception("evolution job metrics probe failed")
        if evolution_worker is not None:
            health = evolution_worker.health()
            EVOLUTION_WORKER.labels("running").set(int(health.running))
            EVOLUTION_WORKER.labels("consecutive_errors").set(health.consecutive_errors)
            EVOLUTION_WORKER.labels("last_heartbeat").set(health.last_heartbeat or 0)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/graphs/{graph_id}", response_model=ProceduralGraph)
    async def get_graph(
        graph_id: str,
        request: Request,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        validate_graph_id(graph_id)
        require_role(request, "reader")
        try:
            return await load_scoped_graph(repo, request_tenant(request), graph_id)
        except FileNotFoundError:
            logger.info("graph not found: %s", graph_id)
            raise HTTPException(status_code=404, detail="graph not found") from None

    @app.post("/v1/graphs", response_model=ProceduralGraph, status_code=201)
    async def create_graph(
        request: GraphCreateRequest,
        http_request: Request,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        require_role(http_request, "writer")
        tenant_id = request_tenant(http_request)
        if request.from_graph_id:
            try:
                base = await load_scoped_graph(repo, tenant_id, request.from_graph_id)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="source graph not found") from None
            new_graph = base.model_copy(update={"id": request.id})
        else:
            new_graph = ProceduralGraph(id=request.id)
        if isinstance(repo, AtomicRepository):
            try:
                await create_scoped_graph(repo, tenant_id, new_graph)
            except GraphAlreadyExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        else:
            try:
                await load_scoped_graph(repo, tenant_id, request.id)
            except FileNotFoundError:
                await create_scoped_graph(repo, tenant_id, new_graph)
            else:
                raise HTTPException(
                    status_code=409,
                    detail=f"graph {request.id!r} already exists",
                )
        return new_graph

    @app.post(
        "/v1/graphs/{graph_id}/guidance",
        response_model=GuidanceResponse,
    )
    async def get_guidance(
        graph_id: str,
        request: GuidanceRequest,
        http_request: Request,
        repo: Repository = Depends(get_repo),
        llm: LLMClient = Depends(get_llm),
    ) -> GuidanceResponse:
        validate_graph_id(graph_id)
        require_role(http_request, "reader")
        try:
            graph = await load_scoped_graph(repo, request_tenant(http_request), graph_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="graph not found") from None

        # Match the most recent action to a node; fall back to full graph
        # on miss (paper §3.2).
        last_action = request.trajectory[-1][0] if request.trajectory else "Start"
        node_id = match_node(last_action, graph.nodes)
        sub = (
            neighborhood(graph, node_id, h=request.guidance_hops) if node_id is not None else graph
        )

        steps = request.trajectory
        guidance = await generate_guidance(
            llm=llm,
            graph=sub,
            query=request.query,
            trajectory=steps,
            window=request.window,
            budget_scope=f"{request_tenant(http_request)}:{graph_id}",
        )
        return GuidanceResponse(guidance=guidance)

    @app.put("/v1/graphs/{graph_id}", response_model=ProceduralGraph)
    async def update_graph(
        graph_id: str,
        graph: ProceduralGraph,
        http_request: Request,
        if_match: str | None = Header(default=None, alias="If-Match"),
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        """Compare-and-swap a graph using its integer `If-Match` revision."""
        validate_graph_id(graph_id)
        require_role(http_request, "writer")
        if graph.id != graph_id:
            raise HTTPException(status_code=422, detail="body graph id does not match path")
        if if_match is None:
            raise HTTPException(status_code=428, detail="If-Match revision is required")
        try:
            expected_revision = int(if_match.strip('"'))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="If-Match must be an integer revision"
            ) from None
        if expected_revision < 0:
            raise HTTPException(status_code=400, detail="If-Match must be non-negative")
        if not isinstance(repo, VersionedRepository):
            raise HTTPException(
                status_code=501, detail="repository does not support versioned updates"
            )
        try:
            stored_graph = graph.model_copy(
                update={"id": storage_graph_id(request_tenant(http_request), graph_id)}
            )
            with repository_tracer.start_as_current_span(
                "methodos.repository.save_graph_if_revision"
            ) as span:
                span.set_attribute("methodos.repository.backend", type(repo).__name__)
                updated = await repo.save_graph_if_revision(stored_graph, expected_revision)
            return updated.model_copy(update={"id": graph_id})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="graph not found") from None
        except GraphRevisionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/graphs/{graph_id}/rollback/{tag}", response_model=ProceduralGraph)
    async def rollback_graph(
        graph_id: str,
        tag: str,
        http_request: Request,
        if_match: str | None = Header(default=None, alias="If-Match"),
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        """Restore a named graph snapshot with optimistic concurrency control."""
        validate_graph_id(graph_id)
        require_role(http_request, "writer")
        if if_match is None:
            raise HTTPException(status_code=428, detail="If-Match revision is required")
        try:
            expected_revision = int(if_match.strip('"'))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="If-Match must be an integer revision"
            ) from None
        if expected_revision < 0:
            raise HTTPException(status_code=400, detail="If-Match must be non-negative")
        if not isinstance(repo, SnapshotRepository):
            raise HTTPException(status_code=501, detail="repository does not support snapshots")
        try:
            restored = await repo.restore_snapshot(
                storage_graph_id(request_tenant(http_request), graph_id),
                tag,
                expected_revision=expected_revision,
            )
            return restored.model_copy(update={"id": graph_id})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="graph or snapshot not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GraphRevisionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/evolution/jobs/{job_id}", response_model=EvolutionJobStatus)
    async def evolution_job_status(
        job_id: str,
        http_request: Request,
    ) -> EvolutionJobStatus:
        """Return status for a host-owned evolution job."""
        validate_graph_id(job_id)
        require_role(http_request, "operator")
        if evolution_runner is None:
            raise HTTPException(status_code=501, detail="evolution runner is not configured")
        tenant_id = request_tenant(http_request)
        if tenant_keys:
            if not isinstance(evolution_runner, TenantScopedEvolutionRunner):
                raise HTTPException(
                    status_code=501,
                    detail="configured evolution runner does not support tenant-scoped status",
                )
            status = await evolution_runner.status_for_tenant(
                tenant_id=tenant_id,
                job_id=job_id,
            )
        else:
            status = await evolution_runner.status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="evolution job not found")
        return status

    @app.post(
        "/v1/graphs/{graph_id}/evolve",
        response_model=EvolveResponse,
        status_code=202,
    )
    async def evolve_graph(
        graph_id: str,
        request: EvolveRequest,
        http_request: Request,
        repo: Repository = Depends(get_repo),
    ) -> EvolveResponse:
        validate_graph_id(graph_id)
        require_role(http_request, "operator")
        try:
            await load_scoped_graph(repo, request_tenant(http_request), graph_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="graph not found") from None
        del repo
        if evolution_runner is None:
            raise HTTPException(
                status_code=501,
                detail=(
                    "Evolution requires an injected host EvolutionRunner; "
                    "use the Python SDK or configure a job runner."
                ),
            )
        request_id = getattr(http_request.state, "request_id", "unknown")
        tenant_id = request_tenant(http_request)
        idempotency_key = validate_idempotency_key(http_request.headers.get("idempotency-key"))
        try:
            tenant_runner: TenantScopedEvolutionRunner | None = None
            if tenant_keys and not isinstance(evolution_runner, TenantScopedEvolutionRunner):
                raise HTTPException(
                    status_code=501,
                    detail="configured evolution runner does not support tenant-scoped jobs",
                )
            if tenant_keys:
                tenant_runner = cast(TenantScopedEvolutionRunner, evolution_runner)
            if idempotency_key is not None:
                if tenant_runner is not None:
                    job_id = await tenant_runner.start_idempotent_for_tenant(
                        tenant_id=tenant_id,
                        graph_id=graph_id,
                        k_rounds=request.k_rounds,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                    )
                elif not isinstance(evolution_runner, IdempotentEvolutionRunner):
                    raise HTTPException(
                        status_code=501,
                        detail="configured evolution runner does not support idempotency",
                    )
                else:
                    job_id = await evolution_runner.start_idempotent(
                        graph_id=graph_id,
                        k_rounds=request.k_rounds,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                    )
            else:
                if tenant_runner is not None:
                    job_id = await tenant_runner.start_for_tenant(
                        tenant_id=tenant_id,
                        graph_id=graph_id,
                        k_rounds=request.k_rounds,
                        request_id=request_id,
                    )
                else:
                    job_id = await evolution_runner.start(
                        graph_id=graph_id,
                        k_rounds=request.k_rounds,
                        request_id=request_id,
                    )
        except JobIdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobQuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return EvolveResponse(detail="evolution queued", job_id=job_id, status="queued")

    return app


# Module-level app for `uvicorn methodos.service:app`.
app = create_app()


__all__ = [
    "ErrorResponse",
    "EvolutionJobStatus",
    "EvolutionRunner",
    "EvolveRequest",
    "EvolveResponse",
    "GraphCreateRequest",
    "GuidanceRequest",
    "GuidanceResponse",
    "HealthResponse",
    "IdempotentEvolutionRunner",
    "ReadinessResponse",
    "TenantScopedEvolutionRunner",
    "app",
    "create_app",
]
