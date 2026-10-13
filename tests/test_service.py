"""Tests for `methodos.service` (FastAPI app)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from methodos.llm import LLMClient
from methodos.repo import (
    FilesystemRepository,
    Repository,
    SQLiteRepository,
    Task,
    Trajectory,
)
from methodos.schema import Node, ProceduralGraph
from methodos.service import (
    EvolveResponse,
    GraphCreateRequest,
    GuidanceResponse,
    HealthResponse,
    create_app,
    validate_cors_origins,
)
from tests.conftest import ScriptedLLM


@pytest.fixture
def graph() -> ProceduralGraph:
    """Standard sample graph for service tests."""
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
    asyncio.run(repo.save_graph(graph))
    yield repo


@pytest.fixture
def client(seeded_repo: SQLiteRepository) -> Iterator[TestClient]:
    """FastAPI TestClient with a seeded SQLite repo and a ScriptedLLM."""
    scripted_llm = ScriptedLLM(responses=["guidance-text"])
    app = create_app(repo=seeded_repo, llm=scripted_llm)
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

    def test_response_has_request_id_and_security_headers(self, client: TestClient) -> None:
        response = client.get("/health", headers={"x-request-id": "request-1"})
        assert response.headers["x-request-id"] == "request-1"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"

    def test_metrics_endpoint_is_available(self, client: TestClient) -> None:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text
        assert "methodos_http_requests_total" in response.text

    def test_readiness_checks_repository(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_api_key_protects_non_public_routes(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_API_KEY", "test-secret")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as protected_client:
            assert protected_client.get("/v1/graphs/g1").status_code == 401
            assert protected_client.get(
                "/v1/graphs/g1", headers={"x-api-key": "test-secret"}
            ).status_code == 200
            assert protected_client.get("/health").status_code == 200

    def test_overlapping_api_keys_support_rotation(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_API_KEYS", "old-secret,new-secret")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as protected_client:
            assert protected_client.get(
                "/v1/graphs/g1", headers={"x-api-key": "old-secret"}
            ).status_code == 200
            assert protected_client.get(
                "/v1/graphs/g1", headers={"x-api-key": "new-secret"}
            ).status_code == 200

    def test_rate_limit_returns_retryable_429(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_RATE_LIMIT_PER_MINUTE", "1")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as limited_client:
            assert limited_client.get("/v1/graphs/g1").status_code == 200
            response = limited_client.get("/v1/graphs/g1")
            assert response.status_code == 429
            assert response.headers["retry-after"] == "60"

    def test_negative_rate_limit_configuration_fails_fast(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_RATE_LIMIT_PER_MINUTE", "-1")
        with pytest.raises(ValueError, match="non-negative"):
            create_app(repo=seeded_repo, llm=ScriptedLLM())

    def test_configured_cors_allows_declared_origin(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_CORS_ORIGINS", "https://console.example")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as cors_client:
            response = cors_client.options(
                "/v1/graphs/g1",
                headers={
                    "origin": "https://console.example",
                    "access-control-request-method": "GET",
                },
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "https://console.example"

    def test_invalid_cors_origin_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="invalid CORS origin"):
            validate_cors_origins(["https://console.example/path"], authenticated=False)

    def test_tenant_keys_isolate_graph_namespaces(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("METHODOS_API_KEY", raising=False)
        monkeypatch.setenv("METHODOS_TENANT_KEYS", "alpha:key-a,beta:key-b")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as tenant_client:
            alpha_headers = {"x-api-key": "key-a", "x-tenant-id": "alpha"}
            beta_headers = {"x-api-key": "key-b", "x-tenant-id": "beta"}
            assert tenant_client.post(
                "/v1/graphs", headers=alpha_headers, json={"id": "shared"}
            ).status_code == 201
            assert tenant_client.get(
                "/v1/graphs/shared", headers=alpha_headers
            ).status_code == 200
            assert tenant_client.get(
                "/v1/graphs/shared", headers=beta_headers
            ).status_code == 404
            assert tenant_client.post(
                "/v1/graphs", headers=beta_headers, json={"id": "shared"}
            ).status_code == 201

    def test_tenant_key_cannot_claim_another_tenant(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("METHODOS_API_KEY", raising=False)
        monkeypatch.setenv("METHODOS_TENANT_KEYS", "alpha:key-a,beta:key-b")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM())
        with TestClient(app) as tenant_client:
            response = tenant_client.get(
                "/v1/graphs/g1",
                headers={"x-api-key": "key-a", "x-tenant-id": "beta"},
            )
        assert response.status_code == 403
        assert response.json()["code"] == "forbidden"


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
        assert response.json()["detail"] == "graph not found"
        assert response.json()["code"] == "not_found"
        assert response.json()["request_id"]

    def test_legacy_repository_rejects_versioned_update(
        self, seeded_repo: SQLiteRepository
    ) -> None:
        class LegacyRepository:
            async def load_graph(self, graph_id: str) -> ProceduralGraph:
                return await seeded_repo.load_graph(graph_id)

            async def save_graph(self, graph: ProceduralGraph) -> None:
                await seeded_repo.save_graph(graph)

        app = create_app(repo=cast(Repository, LegacyRepository()), llm=ScriptedLLM())
        with TestClient(app) as legacy_client:
            graph = legacy_client.get("/v1/graphs/g1").json()
            response = legacy_client.put(
                "/v1/graphs/g1",
                headers={"If-Match": "0"},
                json=graph,
            )
        assert response.status_code == 501
        assert response.json()["code"] == "not_implemented"


# ----------------------------------------------------------------------------
# POST /v1/graphs
# ----------------------------------------------------------------------------


class TestCreateGraph:
    def test_creates_empty_graph(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs",
            json={"id": "new"},
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
            "/v1/graphs",
            json={"id": "x", "extra": "bad"},
        )
        assert response.status_code == 422  # pydantic validation error

    def test_empty_id_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/graphs", json={"id": ""})
        assert response.status_code == 422

    def test_duplicate_id_returns_conflict(self, client: TestClient) -> None:
        response = client.post("/v1/graphs", json={"id": "g1"})
        assert response.status_code == 409

    def test_path_unsafe_id_rejected(self, client: TestClient) -> None:
        response = client.post("/v1/graphs", json={"id": "../escape"})
        assert response.status_code == 422

    def test_legacy_repository_uses_safe_fallback(self, seeded_repo: SQLiteRepository) -> None:
        class LegacyRepository:
            async def load_graph(self, graph_id: str) -> ProceduralGraph:
                return await seeded_repo.load_graph(graph_id)

            async def save_graph(self, graph: ProceduralGraph) -> None:
                await seeded_repo.save_graph(graph)

        app = create_app(
            repo=cast(Repository, LegacyRepository()),
            llm=ScriptedLLM(responses=["x"]),
        )
        with TestClient(app) as legacy_client:
            assert legacy_client.post("/v1/graphs", json={"id": "legacy"}).status_code == 201
            assert legacy_client.post("/v1/graphs", json={"id": "legacy"}).status_code == 409


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

    def test_malformed_trajectory_pair_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/guidance",
            json={"query": "Q?", "trajectory": [["only-action"]]},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_request"


class TestUpdateGraph:
    def test_requires_if_match(self, client: TestClient) -> None:
        response = client.put("/v1/graphs/g1", json={"id": "g1"})
        assert response.status_code == 428

    def test_compare_and_swap_updates_graph(self, client: TestClient) -> None:
        graph = client.get("/v1/graphs/g1").json()
        graph["nodes"]["answer"]["description"] = "updated"
        response = client.put(
            "/v1/graphs/g1",
            headers={"If-Match": "0"},
            json=graph,
        )
        assert response.status_code == 200
        assert response.json()["revision"] == 1
        stale = client.put(
            "/v1/graphs/g1",
            headers={"If-Match": "0"},
            json=graph,
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "conflict"

    def test_rejects_invalid_if_match(self, client: TestClient) -> None:
        graph = client.get("/v1/graphs/g1").json()
        response = client.put("/v1/graphs/g1", headers={"If-Match": "old"}, json=graph)
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_request"

    def test_rejects_negative_if_match(self, client: TestClient) -> None:
        graph = client.get("/v1/graphs/g1").json()
        response = client.put("/v1/graphs/g1", headers={"If-Match": "-1"}, json=graph)
        assert response.status_code == 400

    def test_rejects_path_body_mismatch(self, client: TestClient) -> None:
        response = client.put(
            "/v1/graphs/g1",
            headers={"If-Match": "0"},
            json={"id": "other"},
        )
        assert response.status_code == 422

    def test_missing_graph_returns_404(self, client: TestClient) -> None:
        response = client.put(
            "/v1/graphs/missing",
            headers={"If-Match": "0"},
            json={"id": "missing"},
        )
        assert response.status_code == 404


# ----------------------------------------------------------------------------
# POST /v1/graphs/{graph_id}/evolve (501 reserved)
# ----------------------------------------------------------------------------


class TestEvolveEndpoint:
    def test_returns_501(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/evolve",
            json={"k_rounds": 5},
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
        with pytest.raises(ValidationError):
            GraphCreateRequest.model_validate({"id": "x", "extra": "bad"})


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

        # Pre-seed a graph in the filesystem repo
        fs_repo = FilesystemRepository(root=tmp_path / "fs_home")
        graph = ProceduralGraph(
            id="fs-g",
            nodes={"a": Node(id="a")},
            terminal_ids={"a"},
        )
        asyncio.run(fs_repo.save_graph(graph))

        scripted_llm = ScriptedLLM()
        app = create_app(repo=fs_repo, llm=scripted_llm)
        with TestClient(app) as client:
            response = client.get("/v1/graphs/fs-g")
            assert response.status_code == 200
            assert response.json()["id"] == "fs-g"

    def test_default_repo_is_built_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("PGRAPH_HOME", str(tmp_path))
        monkeypatch.delenv("PGRAPH_BACKEND", raising=False)
        scripted_llm = ScriptedLLM()
        app = create_app(llm=scripted_llm)
        # Lifespan has run; we can hit health at minimum.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
