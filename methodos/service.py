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

import hashlib
import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint

from methodos.graph import match_node, neighborhood
from methodos.guidance import generate_guidance
from methodos.llm import LiteLLMClient, LLMClient
from methodos.repo import (
    AtomicRepository,
    GraphAlreadyExistsError,
    GraphRevisionConflictError,
    Repository,
    VersionedRepository,
    build_repository,
)
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)
GRAPH_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
TENANT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
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

    Evolution requires a host Solver, which cannot be supplied over HTTP
    (the solver is application-specific). This endpoint is reserved for
    future expansion and currently returns 501; use the Python SDK.
    """

    model_config = ConfigDict(extra="forbid")
    k_rounds: int = Field(default=10, ge=1, le=100)


class EvolveResponse(BaseModel):
    """Response body for the evolution endpoint (reserved)."""

    model_config = ConfigDict(extra="forbid")
    detail: str


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


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------


def create_app(
    *,
    repo: Repository | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Build a FastAPI app wired to the given (or default-built) backend.

    Args:
        repo: Repository implementation. Defaults to `build_repository()`
            which selects based on environment variables.
        llm: LLMClient implementation. Defaults to LiteLLMClient with
            model `gpt-4o-mini` (override via `OPENAI_API_KEY` env var).
    """
    backend_repo: Repository = repo if repo is not None else build_repository()
    backend_llm: LLMClient = (
        llm
        if llm is not None
        else LiteLLMClient(
            model="gpt-4o-mini",
        )
    )
    configured_api_key = os.environ.get("METHODOS_API_KEY", "").strip()
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
    configured_rate_limit = int(os.environ.get("METHODOS_RATE_LIMIT_PER_MINUTE", "0"))
    if configured_rate_limit < 0:
        raise ValueError("METHODOS_RATE_LIMIT_PER_MINUTE must be non-negative")
    request_windows: defaultdict[str, deque[float]] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.repo = backend_repo
        app.state.llm = backend_llm
        logger.info("methodos service started")
        yield
        logger.info("methodos service stopped")

    app = FastAPI(title="methodos", version="0.1.0", lifespan=lifespan)

    def error_payload(request: Request, detail: object, code: str) -> dict[str, object]:
        """Build the documented error envelope without leaking internals."""
        safe_detail: str | list[dict[str, object]]
        if isinstance(detail, str):
            safe_detail = detail
        elif isinstance(detail, list) and all(isinstance(item, dict) for item in detail):
            safe_detail = [dict(item) for item in detail]
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
            501: "not_implemented",
            503: "unavailable",
        }.get(exc.status_code, "request_failed")
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
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("METHODOS_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT"],
            allow_headers=["content-type", "x-api-key", "x-request-id", "x-tenant-id"],
        )

    @app.middleware("http")
    async def request_hardening(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Attach request correlation and baseline browser security headers."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        started_at = time.perf_counter()
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
            window = request_windows[rate_key or client_host]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= configured_rate_limit:
                limited_response: Response = JSONResponse(
                    status_code=429,
                    content=error_payload(request, "rate limit exceeded", "rate_limited"),
                    headers={"retry-after": "60"},
                )
                HTTP_REQUESTS.labels(request.method, "429").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                limited_response.headers["x-request-id"] = request_id
                return limited_response
            window.append(now)
        if (configured_api_key or tenant_keys) and not public_path:
            try:
                tenant_id, _ = tenant_auth(request)
            except HTTPException as exc:
                code = "forbidden" if exc.status_code == 403 else "unauthorized"
                auth_response = JSONResponse(
                    status_code=exc.status_code,
                    content=error_payload(request, exc.detail, code),
                    headers={"www-authenticate": "ApiKey"}
                    if exc.status_code == 401
                    else {},
                )
                HTTP_REQUESTS.labels(request.method, str(exc.status_code)).inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                auth_response.headers["x-request-id"] = request_id
                return auth_response
            request.state.tenant_id = tenant_id
        response = await call_next(request)
        HTTP_REQUESTS.labels(request.method, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        return response

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

    def storage_graph_id(tenant_id: str, graph_id: str) -> str:
        """Map a public graph id into an isolated tenant namespace."""
        if not tenant_keys:
            return graph_id
        digest = hashlib.sha256(f"{tenant_id}:{graph_id}".encode()).hexdigest()
        return digest

    async def load_scoped_graph(
        repo: Repository, tenant_id: str, graph_id: str
    ) -> ProceduralGraph:
        """Load a graph from the authenticated tenant namespace."""
        stored = await repo.load_graph(storage_graph_id(tenant_id, graph_id))
        return stored.model_copy(update={"id": graph_id})

    async def create_scoped_graph(
        repo: Repository, tenant_id: str, graph: ProceduralGraph
    ) -> None:
        """Persist a graph under the authenticated tenant namespace."""
        stored = graph.model_copy(update={"id": storage_graph_id(tenant_id, graph.id)})
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
                    return tenant_id, supplied_key
            raise HTTPException(status_code=401, detail="missing or invalid API key")
        if configured_api_key and not secrets.compare_digest(supplied_key, configured_api_key):
            raise HTTPException(status_code=401, detail="missing or invalid API key")
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
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/graphs/{graph_id}", response_model=ProceduralGraph)
    async def get_graph(
        graph_id: str,
        request: Request,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        validate_graph_id(graph_id)
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
        if graph.id != graph_id:
            raise HTTPException(status_code=422, detail="body graph id does not match path")
        if if_match is None:
            raise HTTPException(status_code=428, detail="If-Match revision is required")
        try:
            expected_revision = int(if_match.strip('"'))
        except ValueError:
            raise HTTPException(status_code=400, detail="If-Match must be an integer revision") from None
        if expected_revision < 0:
            raise HTTPException(status_code=400, detail="If-Match must be non-negative")
        if not isinstance(repo, VersionedRepository):
            raise HTTPException(status_code=501, detail="repository does not support versioned updates")
        try:
            stored_graph = graph.model_copy(
                update={"id": storage_graph_id(request_tenant(http_request), graph_id)}
            )
            updated = await repo.save_graph_if_revision(stored_graph, expected_revision)
            return updated.model_copy(update={"id": graph_id})
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="graph not found") from None
        except GraphRevisionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/v1/graphs/{graph_id}/evolve",
        response_model=EvolveResponse,
        status_code=501,
    )
    async def evolve_graph(
        graph_id: str,
        request: EvolveRequest,
        repo: Repository = Depends(get_repo),
    ) -> EvolveResponse:
        validate_graph_id(graph_id)
        # Evolution requires a host Solver; supply one via the Python SDK.
        del graph_id, request, repo
        raise HTTPException(
            status_code=501,
            detail=(
                "Evolution requires a host Solver; use the Python SDK "
                "(methodos.evolution.EvolutionEngine)."
            ),
        )

    return app


# Module-level app for `uvicorn methodos.service:app`.
app = create_app()


__all__ = [
    "ErrorResponse",
    "EvolveRequest",
    "EvolveResponse",
    "GraphCreateRequest",
    "GuidanceRequest",
    "GuidanceResponse",
    "HealthResponse",
    "ReadinessResponse",
    "app",
    "create_app",
]
