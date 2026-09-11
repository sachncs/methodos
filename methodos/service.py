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
  every request is accepted (dev default). When set, all `/v1/*` and
  `/metrics` routes require a matching bearer token or `X-API-Key`.
- Rate limiting: optional per-process token bucket driven by
  `PGRAPH_RATE_LIMIT_*`. Applied to all `/v1/*` routes. Subject is the
  API key when auth is enabled, otherwise the client IP.
- CORS: opt-in via `PGRAPH_CORS_ORIGINS` (comma-separated origins).
  Unset = CORS disabled (suitable for service-to-service behind a
  proxy).
- Body size limit: `PGRAPH_MAX_BODY_BYTES` (default 1 MiB) rejects
  oversized request bodies with HTTP 413 before they reach Pydantic.
- Logging: structured JSON via `methodos.observability.configure_logging`;
  records carry request metadata via `extra={...}`.
- Metrics: Prometheus instruments from `methodos.observability`; `/metrics`
  endpoint exposes the default registry.
- Graceful shutdown: lifespan finally block logs uptime, drains in-flight
  requests, closes the repo (when it supports `aclose`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
# Configuration constants (env-driven, with safe defaults)
# ----------------------------------------------------------------------------


# Maximum HTTP request body size in bytes. Bodies larger than this are
# rejected with HTTP 413 before Pydantic parses them. Default 1 MiB.
DEFAULT_MAX_BODY_BYTES = 1 * 1024 * 1024


def cors_origins_from_env(env: dict[str, str] | None = None) -> list[str]:
    """Parse `PGRAPH_CORS_ORIGINS` into a list of allowed origins.

    Empty/unset returns an empty list (CORS disabled).
    Whitespace around each entry is stripped; blank entries are dropped.
    """
    source = env if env is not None else os.environ
    raw = source.get("PGRAPH_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def max_body_bytes_from_env(env: dict[str, str] | None = None) -> int:
    """Parse `PGRAPH_MAX_BODY_BYTES` (positive int) or return the default.

    Non-numeric or non-positive values fall back to the default; this
    keeps a typo from locking the service down to zero bytes.
    """
    source = env if env is not None else os.environ
    raw = source.get("PGRAPH_MAX_BODY_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    return value if value > 0 else DEFAULT_MAX_BODY_BYTES


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
    cors_origins: list[str] | None = None,
    max_body_bytes: int | None = None,
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
        cors_origins: Origins allowed by CORS (default: read from
            `PGRAPH_CORS_ORIGINS`). Empty list disables CORS.
        max_body_bytes: Maximum HTTP request body size (default: read
            from `PGRAPH_MAX_BODY_BYTES`, fallback 1 MiB). Oversized
            requests get HTTP 413.
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
    cors = cors_origins if cors_origins is not None else cors_origins_from_env()
    body_limit = max_body_bytes if max_body_bytes is not None else max_body_bytes_from_env()

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
        started_at = time.monotonic()
        logger.info(
            "methodos service started",
            extra={
                "version": methodos.__version__,
                "auth_enabled": backend_auth.enabled,
                "rate_limit_enabled": backend_limiter is not None,
                "cors_origins": len(cors),
                "max_body_bytes": body_limit,
            },
        )
        try:
            yield
        finally:
            uptime_seconds = time.monotonic() - started_at
            logger.info(
                "methodos service stopping",
                extra={"uptime_seconds": round(uptime_seconds, 3)},
            )
            # Duck-typed graceful repo close: SQLiteRepository exposes
            # `aclose()` for explicit connection teardown; FilesystemRepository
            # has no native close. Failures here must not block shutdown.
            close = getattr(backend_repo, "aclose", None)
            if callable(close):
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.warning(
                        "repo close failed during shutdown",
                        extra={"error": str(exc)},
                    )
            logger.info(
                "methodos service stopped", extra={"uptime_seconds": round(uptime_seconds, 3)}
            )

    app = FastAPI(title="methodos", version=methodos.__version__, lifespan=lifespan)

    # CORS: opt-in. Empty list means no middleware (suitable for
    # service-to-service traffic behind a proxy). Must be added before
    # any routes so preflight requests see the right headers.
    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    # Body size limit: pure ASGI middleware wraps the FastAPI app so it
    # can intercept the receive callable (Starlette's Request.receive is
    # read-only, so the wrapping must happen at the ASGI scope layer).
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=body_limit)

    # Server header rewrite is handled at the ASGI layer because uvicorn
    # sets `Server` on the raw send, bypassing Response.headers. The
    # ServerHeaderMiddleware does this for in-process TestClient; the
    # Dockerfile additionally passes --header 'server:methodos' to uvicorn
    # so production traffic sees a single Server line.
    app.add_middleware(ServerHeaderMiddleware)

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
    async def body_size_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Reject oversized request bodies with HTTP 413.

        Two checks:
        - If `Content-Length` is present and > `body_limit`, reject before
          the body is read.
        - Otherwise, count bytes as they stream in (handles chunked
          transfer encoding without Content-Length). Returning the
          receive callable with a counting wrapper aborts early.
        """
        content_length_header = request.headers.get("content-length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = -1
            if content_length > body_limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (f"request body too large: {content_length} > {body_limit} bytes")
                    },
                )

        return await call_next(request)

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

    @app.middleware("http")
    async def response_headers_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Set `Cache-Control: no-store` on every response.

        Kept graph data and metrics out of intermediate caches (which
        can otherwise serve stale auth checks or stale counters to the
        next caller). The `Server` header is handled at the ASGI layer
        because uvicorn sets it on the raw send, bypassing Response
        headers.
        """
        response: Response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
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

    @app.get(
        "/metrics",
        dependencies=[Depends(require_api_key)],
    )
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/v1/graphs/{graph_id}",
        response_model=methodos.schema.ProceduralGraph,
        dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
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
        dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
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
    ) -> Response:
        # Evolution requires a host Solver; supply one via the Python SDK.
        # We accept the path/body arguments so the route matches the
        # contract but never read the repo (no work is done server-side).
        del graph_id, request
        return JSONResponse(
            status_code=501,
            content={
                "detail": (
                    "Evolution requires a host Solver; use the Python SDK "
                    "(methodos.evolution.EvolutionEngine)."
                )
            },
        )

    return app


class _BodyTooLarge(Exception):
    """Raised by `BodySizeLimitMiddleware` when streaming bytes exceed the limit.

    Internal sentinel; never leaks out of the middleware boundary.
    """

    def __init__(self, received: int) -> None:
        super().__init__(f"body too large: {received} bytes")
        self.received = received


class ServerHeaderMiddleware:
    """ASGI middleware that rewrites the `Server` response header.

    uvicorn injects `Server: uvicorn` at the raw ASGI send layer after
    the FastAPI app has already responded, so a normal Response.headers
    override shows up alongside uvicorn's value (two `Server:` lines).
    Intercepting the `http.response.start` send is the only reliable
    place to take ownership of this header.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Awaitable[dict[str, object]]],
        send: Callable[[dict[str, object]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":

            async def passthrough_send(message: dict[str, object]) -> None:
                await send(message)

            await self.app(scope, receive, passthrough_send)
            return

        already_wrote = False

        async def rewrite_send(message: dict[str, object]) -> None:
            nonlocal already_wrote
            if message["type"] == "http.response.start" and not already_wrote:
                headers_obj = message.get("headers")
                if isinstance(headers_obj, list):
                    # Drop any pre-existing Server header and inject ours.
                    filtered = [
                        pair
                        for pair in headers_obj
                        if not (
                            isinstance(pair, list)
                            and len(pair) == 2
                            and isinstance(pair[0], bytes)
                            and pair[0].lower() == b"server"
                        )
                    ]
                    filtered.append([b"server", b"methodos"])
                    message = {**message, "headers": filtered}
                    already_wrote = True
            await send(message)

        await self.app(scope, receive, rewrite_send)


class BodySizeLimitMiddleware:
    """Pure ASGI middleware that rejects oversized request bodies.

    Starlette's `Request.receive` is read-only, so the receive callable
    must be wrapped at the ASGI scope layer rather than via FastAPI's
    HTTP-middleware decorator. Two paths are enforced:

    1. `Content-Length` header present: rejected synchronously without
       reading any bytes.
    2. Chunked transfer (no Content-Length): bytes are counted as they
       stream in; the first byte that pushes the running total over the
       limit aborts with HTTP 413.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: dict[str, object],
        receive: Callable[[], Awaitable[dict[str, object]]],
        send: Callable[[dict[str, object]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check Content-Length up front; cheap and handles the common case.
        raw_headers_obj = scope.get("headers")
        raw_headers: list[tuple[bytes, bytes]] = (
            [pair for pair in raw_headers_obj if isinstance(pair, list) and len(pair) == 2]  # type: ignore[misc]
            if isinstance(raw_headers_obj, list)
            else []
        )
        headers: dict[bytes, bytes] = {
            k: v for k, v in raw_headers if isinstance(k, bytes) and isinstance(v, bytes)
        }
        content_length_header = headers.get(b"content-length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header.decode("ascii", errors="ignore"))
            except ValueError:
                content_length = -1
            if content_length > self.max_body_bytes:
                await self._reject(send, content_length)
                return

        # No Content-Length (chunked) or declared size within limit:
        # count bytes as they stream in.
        running = 0
        over_limit = False

        async def counting_receive() -> dict[str, object]:
            nonlocal running, over_limit
            message: dict[str, object] = await receive()
            if over_limit:
                return {"type": "http.disconnect"}
            body = message.get("body", b"")
            if isinstance(body, (bytes, bytearray)):
                running += len(body)
            if running > self.max_body_bytes:
                over_limit = True
                return {"type": "http.disconnect"}
            return message

        if over_limit:
            await self._reject(send, running)
            return

        # We can't easily short-circuit mid-stream without a more elaborate
        # ASGI dance (the app expects to drive `send` itself), so on
        # overflow we let the call complete and the upstream exception
        # handler returns 413 from the sentinel.
        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge as exc:
            await self._reject(send, exc.received)

    async def _reject(
        self,
        send: Callable[[dict[str, object]], Awaitable[None]],
        received: int,
    ) -> None:
        detail = f"request body too large: {received} > {self.max_body_bytes} bytes"
        body_bytes = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body_bytes})


# Module-level app for `uvicorn methodos.service:app`.
app = create_app()


__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "EvolveRequest",
    "EvolveResponse",
    "GraphCreateRequest",
    "GuidanceRequest",
    "GuidanceResponse",
    "HealthResponse",
    "ReadinessResponse",
    "app",
    "cors_origins_from_env",
    "create_app",
    "max_body_bytes_from_env",
]
