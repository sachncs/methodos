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

import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint

from methodos.graph import match_node, neighborhood
from methodos.guidance import generate_guidance
from methodos.llm import LiteLLMClient, LLMClient
from methodos.repo import AtomicRepository, GraphAlreadyExistsError, Repository, build_repository
from methodos.schema import ProceduralGraph

logger = logging.getLogger(__name__)
GRAPH_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
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


def validate_graph_id(graph_id: str) -> str:
    """Validate a graph identifier before it reaches a repository."""
    if re.fullmatch(GRAPH_ID_PATTERN, graph_id) is None:
        raise HTTPException(status_code=422, detail="invalid graph_id")
    return graph_id


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
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "x-api-key", "x-request-id"],
        )

    @app.middleware("http")
    async def request_hardening(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Attach request correlation and baseline browser security headers."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
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
            window = request_windows[client_host]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= configured_rate_limit:
                limited_response: Response = JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                    headers={"retry-after": "60"},
                )
                HTTP_REQUESTS.labels(request.method, "429").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                limited_response.headers["x-request-id"] = request_id
                return limited_response
            window.append(now)
        if configured_api_key and not public_path:
            supplied_key = request.headers.get("x-api-key", "")
            if not secrets.compare_digest(supplied_key, configured_api_key):
                auth_response: Response = JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid API key"},
                    headers={"www-authenticate": "ApiKey"},
                )
                HTTP_REQUESTS.labels(request.method, "401").inc()
                HTTP_LATENCY.labels(request.method).observe(time.perf_counter() - started_at)
                auth_response.headers["x-request-id"] = request_id
                return auth_response
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
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        validate_graph_id(graph_id)
        try:
            return await repo.load_graph(graph_id)
        except FileNotFoundError:
            logger.info("graph not found: %s", graph_id)
            raise HTTPException(status_code=404, detail="graph not found") from None

    @app.post("/v1/graphs", response_model=ProceduralGraph, status_code=201)
    async def create_graph(
        request: GraphCreateRequest,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        if request.from_graph_id:
            try:
                base = await repo.load_graph(request.from_graph_id)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="source graph not found") from None
            new_graph = base.model_copy(update={"id": request.id})
        else:
            new_graph = ProceduralGraph(id=request.id)
        if isinstance(repo, AtomicRepository):
            try:
                await repo.create_graph(new_graph)
            except GraphAlreadyExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        else:
            try:
                await repo.load_graph(request.id)
            except FileNotFoundError:
                await repo.save_graph(new_graph)
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
        repo: Repository = Depends(get_repo),
        llm: LLMClient = Depends(get_llm),
    ) -> GuidanceResponse:
        validate_graph_id(graph_id)
        try:
            graph = await repo.load_graph(graph_id)
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
