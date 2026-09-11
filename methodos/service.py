"""FastAPI app factory for the `methodos` hosted service.

Engineering notes:
- `create_app(...)` is the public factory; everything else lives inside
  it (routers, lifespan, dependencies). This is the standard FastAPI
  pattern for testable apps.
- HTTP request/response models live in this module as Pydantic DTOs,
  distinct from the domain models in `methodos.schema`.
- No module-level globals except `app = create_app()` for the uvicorn
  entry point.
- Lifespan wires the configured `repo`, `llm`, `auth`, and rate limiter
  into `app.state`.

Hardening:
- Auth: optional `APIKeyAuth` driven by `PGRAPH_API_KEY`. When unset,
  every request is accepted (dev default). When set, all `/v1/*` routes
  require a matching bearer token or `X-API-Key` header.
- Rate limiting: optional per-process token bucket driven by
  `PGRAPH_RATE_LIMIT_*`. Subject is the API key when auth is enabled,
  otherwise the client IP.
- Logging: structured JSON via `methodos.observability.configure_logging`;
  records carry request metadata via `extra={...}`.
- Metrics: Prometheus instruments from `methodos.observability`; `/metrics`
  endpoint exposes the default registry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

import methodos
from methodos import observability
from methodos.auth import APIKeyAuth, api_key_from_env, auth_dependency
from methodos.llm import LiteLLMClient, LLMClient, LLMError
from methodos.observability import (
    CONTENT_TYPE_LATEST,
    configure_logging,
    generate_latest,
)
from methodos.rate_limit import (
    TokenBucketLimiter,
    limiter_from_env,
    rate_limit_dependency,
)
from methodos.repo import Repository, build_repository

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# HTTP DTOs (separate from domain models to keep the wire format flexible)
# ----------------------------------------------------------------------------


class GraphCreateRequest(BaseModel):
    """Request body for `POST /v1/graphs`."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    from_graph_id: str | None = Field(default=None, max_length=128)


class GuidanceRequest(BaseModel):
    """Request body for `POST /v1/graphs/{graph_id}/guidance`."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)
    trajectory: list[list[str]] = Field(default_factory=list)
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
    """Response body for `GET /health/ready`."""

    model_config = ConfigDict(extra="forbid")
    status: str
    checks: dict[str, str]


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------


def create_app(
    *,
    repo: Repository | None = None,
    llm: LLMClient | None = None,
    auth: APIKeyAuth | None = None,
    rate_limiter: TokenBucketLimiter | None = None,
    log_json: bool = False,
    log_level: str = "INFO",
) -> FastAPI:
    """Build a FastAPI app wired to the given (or default-built) backend.

    Args:
        repo: Repository implementation. Defaults to `build_repository()`
            which selects based on environment variables.
        llm: LLMClient implementation. Defaults to LiteLLMClient with
            model `gpt-4o-mini` (override via `OPENAI_API_KEY` env var).
        auth: Optional API key authenticator. Defaults to reading
            `PGRAPH_API_KEY` from the environment.
        rate_limiter: Optional token-bucket limiter. Defaults to reading
            `PGRAPH_RATE_LIMIT_*` from the environment.
        log_json: Emit JSON-formatted log lines (default: human-readable).
        log_level: Root log level name (default: `INFO`).
    """
    configure_logging(level=log_level, json_format=log_json)
    observability.INFO.labels(version=methodos.__version__, component="service").set(1)

    backend_repo: Repository = repo if repo is not None else build_repository()
    backend_llm: LLMClient = llm if llm is not None else LiteLLMClient(model="gpt-4o-mini")
    backend_auth: APIKeyAuth = (
        auth if auth is not None else APIKeyAuth(configured_key=api_key_from_env())
    )
    backend_limiter: TokenBucketLimiter | None = (
        rate_limiter if rate_limiter is not None else limiter_from_env()
    )

    require_api_key = auth_dependency(backend_auth)
    enforce_rate_limit = rate_limit_dependency(
        backend_limiter,
        api_key_provider=lambda req: getattr(req.state, "methodos_api_key", None),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.repo = backend_repo
        app.state.llm = backend_llm
        app.state.auth = backend_auth
        app.state.rate_limiter = backend_limiter
        logger.info(
            "methodos service started",
            extra={
                "version": methodos.__version__,
                "auth_enabled": backend_auth.enabled,
                "rate_limit_enabled": backend_limiter is not None,
            },
        )
        yield
        logger.info("methodos service stopped")

    app = FastAPI(title="methodos", version=methodos.__version__, lifespan=lifespan)

    # ---- exception handlers --------------------------------------------------

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        logger.warning(
            "llm call failed",
            extra={"path": request.url.path, "error": str(exc)},
        )
        return JSONResponse(
            status_code=502,
            content={"detail": "upstream LLM call failed"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled error",
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    # ---- middleware ----------------------------------------------------------

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        # Use the route template (e.g. `/v1/graphs/{graph_id}`) rather than
        # the raw path so high-cardinality IDs don't explode the metric.
        route = request.scope.get("route")
        path_template = getattr(route, "path", request.url.path)
        status_label = str(response.status_code)
        observability.REQUESTS_TOTAL.labels(
            method=request.method,
            path=path_template,
            status=status_label,
        ).inc()
        observability.REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            path=path_template,
        ).observe(elapsed)
        return response

    # ---- dependencies --------------------------------------------------------

    def get_repo() -> Repository:
        repo: Repository = app.state.repo
        return repo

    def get_llm() -> LLMClient:
        llm: LLMClient = app.state.llm
        return llm

    # ---- routes --------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    async def health_ready(
        repo: Repository = Depends(get_repo),
    ) -> Response:
        checks: dict[str, str] = {}
        try:
            # Round-trip the repo: reading trajectories for a sentinel id
            # exercises both the connection and the read path without
            # requiring a pre-existing graph.
            agen = repo.read_trajectories("__healthcheck__", "train")
            async for _ in agen:  # pragma: no cover - empty result expected
                break
            checks["repo"] = "ok"
        except Exception as exc:
            logger.warning(
                "readiness probe failed",
                extra={"component": "repo", "error": str(exc)},
            )
            checks["repo"] = "error"
        status_label = "ok" if all(v == "ok" for v in checks.values()) else "error"
        body = ReadinessResponse(status=status_label, checks=checks)
        return JSONResponse(
            status_code=200 if status_label == "ok" else 503,
            content=body.model_dump(),
        )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/v1/graphs/{graph_id}",
        response_model=methodos.schema.ProceduralGraph,
        dependencies=[Depends(require_api_key)],
    )
    async def get_graph(
        graph_id: str,
        repo: Repository = Depends(get_repo),
    ) -> methodos.schema.ProceduralGraph:
        try:
            return await repo.load_graph(graph_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/v1/graphs",
        response_model=methodos.schema.ProceduralGraph,
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    async def create_graph(
        request: GraphCreateRequest,
        repo: Repository = Depends(get_repo),
    ) -> methodos.schema.ProceduralGraph:
        if request.from_graph_id:
            try:
                base = await repo.load_graph(request.from_graph_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            new_graph = base.model_copy(update={"id": request.id})
        else:
            new_graph = methodos.schema.ProceduralGraph(id=request.id)
        await repo.save_graph(new_graph)
        return new_graph

    @app.post(
        "/v1/graphs/{graph_id}/guidance",
        response_model=GuidanceResponse,
        dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
    )
    async def get_guidance(
        graph_id: str,
        request: GuidanceRequest,
        repo: Repository = Depends(get_repo),
        llm: LLMClient = Depends(get_llm),
    ) -> GuidanceResponse:
        try:
            graph = await repo.load_graph(graph_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Match the most recent action to a node; fall back to full graph
        # on miss (paper §3.2).
        last_action = request.trajectory[-1][0] if request.trajectory else "Start"
        from methodos.graph import match_node, neighborhood

        node_id = match_node(last_action, graph.nodes)
        sub = (
            neighborhood(graph, node_id, h=request.guidance_hops) if node_id is not None else graph
        )

        # The trajectory in the HTTP body is `list[list[str]]`; convert to
        # the (action, observation) tuple form expected by the guidance
        # generator.
        steps = [(pair[0], pair[1]) for pair in request.trajectory]
        from methodos.guidance import generate_guidance

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
        dependencies=[Depends(require_api_key)],
    )
    async def evolve_graph(
        graph_id: str,
        request: EvolveRequest,
        repo: Repository = Depends(get_repo),
    ) -> EvolveResponse:
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
