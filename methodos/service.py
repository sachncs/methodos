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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from methodos.graph import match_node, neighborhood
from methodos.guidance import generate_guidance
from methodos.llm import LiteLLMClient, LLMClient
from methodos.repo import Repository, build_repository
from methodos.schema import ProceduralGraph

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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.repo = backend_repo
        app.state.llm = backend_llm
        logger.info("methodos service started")
        yield
        logger.info("methodos service stopped")

    app = FastAPI(title="methodos", version="0.1.0", lifespan=lifespan)

    def get_repo() -> Repository:
        repo: Repository = app.state.repo
        return repo

    def get_llm() -> LLMClient:
        llm: LLMClient = app.state.llm
        return llm

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/v1/graphs/{graph_id}", response_model=ProceduralGraph)
    async def get_graph(
        graph_id: str,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        try:
            return await repo.load_graph(graph_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/graphs", response_model=ProceduralGraph, status_code=201)
    async def create_graph(
        request: GraphCreateRequest,
        repo: Repository = Depends(get_repo),
    ) -> ProceduralGraph:
        if request.from_graph_id:
            try:
                base = await repo.load_graph(request.from_graph_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            new_graph = base.model_copy(update={"id": request.id})
        else:
            new_graph = ProceduralGraph(id=request.id)
        await repo.save_graph(new_graph)
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
        try:
            graph = await repo.load_graph(graph_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Match the most recent action to a node; fall back to full graph
        # on miss (paper §3.2).
        last_action = request.trajectory[-1][0] if request.trajectory else "Start"
        node_id = match_node(last_action, graph.nodes)
        sub = (
            neighborhood(graph, node_id, h=request.guidance_hops) if node_id is not None else graph
        )

        # The trajectory in the HTTP body is `list[list[str]]`; convert to
        # the (action, observation) tuple form expected by the guidance
        # generator.
        steps = [(pair[0], pair[1]) for pair in request.trajectory]
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
    "app",
    "create_app",
]
