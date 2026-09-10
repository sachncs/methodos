"""Tests for `methodos.service` (FastAPI app)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from methodos.llm import LLMClient
from methodos.repo import (
    FilesystemRepository,
    SQLiteRepository,
    Task,
    Trajectory,
)
from methodos.schema import ProceduralGraph
from methodos.service import (
    EvolveResponse,
    GraphCreateRequest,
    GuidanceResponse,
    HealthResponse,
    create_app,
)
from tests.conftest import FakeLLM


@pytest.fixture
def graph() -> ProceduralGraph:
    """Standard sample graph for service tests."""
    from methodos.schema import Node
    return ProceduralGraph(
        id="g1",
        nodes={
            "start": Node(id="start"),
            "search": Node(id="search"),
            "answer": Node(id="answer"),
        },
        edges=[],
        terminal_ids={"answer"},
    )


@pytest.fixture
def seeded_repo(graph: ProceduralGraph, tmp_path: Path) -> Iterator[SQLiteRepository]:
    """SQLite repo with one pre-saved graph."""
    repo = SQLiteRepository(db_path=tmp_path / "test.db")
    import asyncio
    asyncio.run(repo.save_graph(graph))
    yield repo


@pytest.fixture
def client(seeded_repo: SQLiteRepository) -> Iterator[TestClient]:
    """FastAPI TestClient with a seeded SQLite repo and a FakeLLM."""
    fake_llm = FakeLLM(responses=["guidance-text"])
    app = create_app(repo=seeded_repo, llm=fake_llm)
    with TestClient(app) as test_client:
        yield test_client


# ----------------------------------------------------------------------------
# /health
# ----------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ----------------------------------------------------------------------------
# GET /v1/graphs/{graph_id}
# ----------------------------------------------------------------------------


class TestGetGraph:
    def test_returns_seeded_graph(self, client: TestClient) -> None:
        response = client.get("/v1/graphs/g1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "g1"
        assert "start" in data["nodes"]

    def test_missing_graph_returns_404(self, client: TestClient) -> None:
        response = client.get("/v1/graphs/nope")
        assert response.status_code == 404


# ----------------------------------------------------------------------------
# POST /v1/graphs
# ----------------------------------------------------------------------------


class TestCreateGraph:
    def test_creates_empty_graph(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs", json={"id": "new"},
        )
        assert response.status_code == 201
        assert response.json()["id"] == "new"
        # Subsequent GET retrieves it
        assert client.get("/v1/graphs/new").status_code == 200

    def test_creates_from_existing(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs",
            json={"id": "clone", "from_graph_id": "g1"},
        )
        assert response.status_code == 201
        assert response.json()["id"] == "clone"
        # Cloned graph has the same nodes as g1
        cloned = client.get("/v1/graphs/clone").json()
        assert set(cloned["nodes"].keys()) == {"start", "search", "answer"}

    def test_clone_missing_source_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs",
            json={"id": "x", "from_graph_id": "missing"},
        )
        assert response.status_code == 404

    def test_extra_fields_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs", json={"id": "x", "extra": "bad"},
        )
        assert response.status_code == 422  # pydantic validation error

    def test_empty_id_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/graphs", json={"id": ""})
        assert response.status_code == 422


# ----------------------------------------------------------------------------
# POST /v1/graphs/{graph_id}/guidance
# ----------------------------------------------------------------------------


class TestGetGuidance:
    def test_returns_guidance_text(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "What?", "trajectory": [], "window": 3, "guidance_hops": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert "guidance" in data
        assert data["guidance"] == "guidance-text"

    def test_missing_graph_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/nope/guidance",
            json={"query": "What?"},
        )
        assert response.status_code == 404

    def test_with_trajectory(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={
                "query": "Q?",
                "trajectory": [["start", "obs1"], ["search", "obs2"]],
                "window": 2,
                "guidance_hops": 1,
            },
        )
        assert response.status_code == 200

    def test_window_validation(self, client: TestClient) -> None:
        # Negative window rejected
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "Q?", "window": -1},
        )
        assert response.status_code == 422
        # Window > 20 rejected
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "Q?", "window": 100},
        )
        assert response.status_code == 422

    def test_guidance_hops_validation(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "Q?", "guidance_hops": 100},
        )
        assert response.status_code == 422

    def test_extra_fields_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "Q?", "extra": "bad"},
        )
        assert response.status_code == 422


# ----------------------------------------------------------------------------
# POST /v1/graphs/{graph_id}/evolve (501 reserved)
# ----------------------------------------------------------------------------


class TestEvolveEndpoint:
    def test_returns_501(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/evolve", json={"k_rounds": 5},
        )
        assert response.status_code == 501
        body = response.json()
        assert "Python SDK" in body["detail"]


# ----------------------------------------------------------------------------
# DTO validation
# ----------------------------------------------------------------------------


class TestGraphCreateRequestDTO:
    def test_default_no_clone(self) -> None:
        req = GraphCreateRequest(id="x")
        assert req.from_graph_id is None

    def test_extra_forbidden(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GraphCreateRequest(id="x", extra="bad")  # type: ignore[call-arg]


class TestGuidanceResponseDTO:
    def test_construction(self) -> None:
        resp = GuidanceResponse(guidance="x")
        assert resp.guidance == "x"


class TestEvolveResponseDTO:
    def test_construction(self) -> None:
        resp = EvolveResponse(detail="not implemented")
        assert resp.detail == "not implemented"


class TestHealthResponseDTO:
    def test_construction(self) -> None:
        resp = HealthResponse(status="ok")
        assert resp.status == "ok"


# ----------------------------------------------------------------------------
# create_app factory with FilesystemRepository
# ----------------------------------------------------------------------------


class TestFilesystemRepositoryWiring:
    """`create_app` accepts any Repository implementation."""

    def test_filesystem_repo(self, tmp_path: Path) -> None:
        import asyncio

        from methodos.schema import Node
        # Pre-seed a graph in the filesystem repo
        fs_repo = FilesystemRepository(root=tmp_path / "fs_home")
        graph = ProceduralGraph(
            id="fs-g", nodes={"a": Node(id="a")}, terminal_ids={"a"},
        )
        asyncio.run(fs_repo.save_graph(graph))

        fake_llm = FakeLLM()
        app = create_app(repo=fs_repo, llm=fake_llm)
        with TestClient(app) as client:
            response = client.get("/v1/graphs/fs-g")
            assert response.status_code == 200
            assert response.json()["id"] == "fs-g"

    def test_default_repo_is_built_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.delenv("PGRAPH_BACKEND", raising=False)
        fake_llm = FakeLLM()
        app = create_app(llm=fake_llm)
        # Lifespan has run; we can hit health at minimum.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
