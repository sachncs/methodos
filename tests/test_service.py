"""Tests for `methodos.service` (FastAPI app)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from methodos.auth import APIKeyAuth
from methodos.llm import LLMClient, LLMError
from methodos.rate_limit import TokenBucketLimiter
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
            id="fs-g",
            nodes={"a": Node(id="a")},
            terminal_ids={"a"},
        )
        asyncio.run(fs_repo.save_graph(graph))

        fake_llm = FakeLLM()
        app = create_app(repo=fs_repo, llm=fake_llm)
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
        fake_llm = FakeLLM()
        app = create_app(llm=fake_llm)
        # Lifespan has run; we can hit health at minimum.
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------------
# /health/live, /health/ready, /metrics
# ----------------------------------------------------------------------------


class TestHealthLive:
    def test_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestHealthReady:
    def test_returns_ok_when_repo_healthy(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["repo"] == "ok"


class TestMetricsEndpoint:
    def test_returns_prometheus_payload(self, client: TestClient) -> None:
        # Hit a route first so the counters have a sample
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus exposition content type
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "methodos_requests_total" in body
        assert "methodos_info" in body

    def test_records_request_metric(self, client: TestClient) -> None:
        client.get("/health")
        body = client.get("/metrics").text
        # The /health GET should show up in the counter
        assert 'method="GET"' in body
        assert 'path="/health"' in body


# ----------------------------------------------------------------------------
# Exception handlers
# ----------------------------------------------------------------------------


class RaisingLLM(LLMClient):
    """LLM stub that always raises `LLMError`."""

    async def complete(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise LLMError("upstream provider unavailable")


class TestExceptionHandlers:
    def test_llm_error_maps_to_502(self, tmp_path: Path) -> None:
        repo = SQLiteRepository(db_path=tmp_path / "ex.db")
        import asyncio

        graph = ProceduralGraph(id="g1")
        asyncio.run(repo.save_graph(graph))
        llm = RaisingLLM()
        app = create_app(repo=repo, llm=llm)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs/g1/guidance",
                json={"query": "Q?"},
            )
            assert response.status_code == 502
            assert response.json()["detail"] == "upstream LLM call failed"


# ----------------------------------------------------------------------------
# Rate limiting integration
# ----------------------------------------------------------------------------


class TestRateLimitIntegration:
    def test_rate_limited_returns_429(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM(responses=["g"])
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=0.001)
        app = create_app(repo=seeded_repo, llm=fake_llm, rate_limiter=limiter)
        with TestClient(app) as client:
            first = client.post(
                "/v1/graphs/g1/guidance",
                json={"query": "Q1?"},
            )
            assert first.status_code == 200
            second = client.post(
                "/v1/graphs/g1/guidance",
                json={"query": "Q2?"},
            )
            assert second.status_code == 429
            assert "Retry-After" in second.headers

    def test_no_limiter_allows_burst(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM(responses=["g"])
        app = create_app(repo=seeded_repo, llm=fake_llm, rate_limiter=None)
        with TestClient(app) as client:
            for _ in range(3):
                response = client.post(
                    "/v1/graphs/g1/guidance",
                    json={"query": "Q?"},
                )
                assert response.status_code == 200


# ----------------------------------------------------------------------------
# Auth integration (already covered in test_auth.py; smoke test here)
# ----------------------------------------------------------------------------


class TestAuthIntegration:
    def test_auth_enabled_protects_routes(self) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="k")
        app = create_app(repo=None, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            # Health is not behind auth
            assert client.get("/health").status_code == 200
            # Graph create is
            assert client.post("/v1/graphs", json={"id": "x"}).status_code == 401


# ----------------------------------------------------------------------------
# /metrics auth gating
# ----------------------------------------------------------------------------


class TestMetricsAuthGating:
    def test_metrics_open_when_auth_disabled(self, client: TestClient) -> None:
        # No PGRAPH_API_KEY set in the fixture; auth is disabled.
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_requires_auth_when_enabled(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="secret")
        app = create_app(repo=seeded_repo, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            # No key -> 401
            response = client.get("/metrics")
            assert response.status_code == 401
            # Valid key -> 200
            response = client.get("/metrics", headers={"Authorization": "Bearer secret"})
            assert response.status_code == 200


# ----------------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------------


class TestCors:
    def test_cors_disabled_by_default(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM()
        app = create_app(repo=seeded_repo, llm=fake_llm)
        with TestClient(app) as client:
            # Preflight without CORS config: no Access-Control-Allow-Origin.
            response = client.options(
                "/v1/graphs",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert "access-control-allow-origin" not in {k.lower() for k in response.headers}

    def test_cors_enabled_with_origins(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM()
        app = create_app(
            repo=seeded_repo,
            llm=fake_llm,
            cors_origins=["https://app.example.com"],
        )
        with TestClient(app) as client:
            response = client.get(
                "/v1/graphs/g1",
                headers={"Origin": "https://app.example.com"},
            )
            assert response.headers.get("access-control-allow-origin") == "https://app.example.com"


# ----------------------------------------------------------------------------
# Body size limit
# ----------------------------------------------------------------------------


class TestBodySizeLimit:
    def test_content_length_too_large(self) -> None:
        fake_llm = FakeLLM()
        # 100-byte cap so the request body advertises itself as too big.
        app = create_app(repo=None, llm=fake_llm, max_body_bytes=100)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs",
                content=b'{"id":"x","padding":"' + b"a" * 200 + b'"}',
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 413
            assert "too large" in response.json()["detail"]

    def test_body_within_limit_passes(self) -> None:
        fake_llm = FakeLLM()
        app = create_app(repo=None, llm=fake_llm, max_body_bytes=10_000)
        with TestClient(app) as client:
            response = client.post("/v1/graphs", json={"id": "ok"})
            assert response.status_code == 201

    def test_invalid_content_length_rejected(self) -> None:
        fake_llm = FakeLLM()
        app = create_app(repo=None, llm=fake_llm, max_body_bytes=100)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs",
                content=b'{"id":"x"}',
                headers={"Content-Type": "application/json", "Content-Length": "abc"},
            )
            # Invalid Content-Length is treated as -1 (under the limit),
            # so the request is allowed; subsequent Pydantic parsing may
            # then succeed.
            assert response.status_code in (201, 422)


# ----------------------------------------------------------------------------
# Rate limit on GET/POST /v1/graphs
# ----------------------------------------------------------------------------


class TestRateLimitOnGraphRoutes:
    def test_get_graph_429_when_exhausted(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM(responses=["g"])
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=0.001)
        app = create_app(repo=seeded_repo, llm=fake_llm, rate_limiter=limiter)
        with TestClient(app) as client:
            first = client.get("/v1/graphs/g1")
            assert first.status_code == 200
            second = client.get("/v1/graphs/g1")
            assert second.status_code == 429
            assert "Retry-After" in second.headers

    def test_post_graph_429_when_exhausted(self, seeded_repo: SQLiteRepository) -> None:
        fake_llm = FakeLLM(responses=["g"])
        limiter = TokenBucketLimiter(capacity=1, refill_per_second=0.001)
        app = create_app(repo=seeded_repo, llm=fake_llm, rate_limiter=limiter)
        with TestClient(app) as client:
            first = client.post("/v1/graphs", json={"id": "a"})
            assert first.status_code == 201
            second = client.post("/v1/graphs", json={"id": "b"})
            assert second.status_code == 429


# ----------------------------------------------------------------------------
# Graceful shutdown / lifespan
# ----------------------------------------------------------------------------


class TestGracefulShutdown:
    def test_repo_aclose_called_on_shutdown(self) -> None:
        class AclosingRepo:
            def __init__(self) -> None:
                self.aclose_called = False

            async def load_graph(self, graph_id: str) -> object:  # pragma: no cover
                raise FileNotFoundError(graph_id)

            async def aclose(self) -> None:
                self.aclose_called = True

        fake_llm = FakeLLM()
        repo = AclosingRepo()
        app = create_app(repo=repo, llm=fake_llm)
        with TestClient(app) as client:
            client.get("/health/live")
        assert repo.aclose_called is True

    def test_repo_without_aclose_does_not_crash(self) -> None:
        class PlainRepo:
            async def load_graph(self, graph_id: str) -> object:  # pragma: no cover
                raise FileNotFoundError(graph_id)

        fake_llm = FakeLLM()
        repo = PlainRepo()
        app = create_app(repo=repo, llm=fake_llm)
        with TestClient(app) as client:
            # Should not raise on shutdown.
            client.get("/health/live")


# ----------------------------------------------------------------------------
# Env helpers (no FastAPI roundtrip needed)
# ----------------------------------------------------------------------------


class TestServiceEnvHelpers:
    def test_cors_origins_from_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import cors_origins_from_env

        monkeypatch.delenv("PGRAPH_CORS_ORIGINS", raising=False)
        assert cors_origins_from_env() == []

    def test_cors_origins_from_env_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import cors_origins_from_env

        monkeypatch.setenv("PGRAPH_CORS_ORIGINS", "https://a.example.com, https://b.example.com ")
        assert cors_origins_from_env() == ["https://a.example.com", "https://b.example.com"]

    def test_max_body_bytes_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import DEFAULT_MAX_BODY_BYTES, max_body_bytes_from_env

        monkeypatch.delenv("PGRAPH_MAX_BODY_BYTES", raising=False)
        assert max_body_bytes_from_env() == DEFAULT_MAX_BODY_BYTES

    def test_max_body_bytes_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import max_body_bytes_from_env

        monkeypatch.setenv("PGRAPH_MAX_BODY_BYTES", "65536")
        assert max_body_bytes_from_env() == 65536

    def test_max_body_bytes_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import DEFAULT_MAX_BODY_BYTES, max_body_bytes_from_env

        monkeypatch.setenv("PGRAPH_MAX_BODY_BYTES", "not-a-number")
        assert max_body_bytes_from_env() == DEFAULT_MAX_BODY_BYTES

    def test_max_body_bytes_zero_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from methodos.service import DEFAULT_MAX_BODY_BYTES, max_body_bytes_from_env

        monkeypatch.setenv("PGRAPH_MAX_BODY_BYTES", "0")
        assert max_body_bytes_from_env() == DEFAULT_MAX_BODY_BYTES


# ----------------------------------------------------------------------------
# Evolve handler (501 short-circuit)
# ----------------------------------------------------------------------------


class TestEvolveShortCircuit:
    def test_evolve_does_not_touch_repo(self) -> None:
        # Repo raises if load_graph is called; 501 must come back without
        # invoking it.
        class ExplodingRepo:
            async def load_graph(self, graph_id: str) -> object:  # pragma: no cover
                raise AssertionError("evolve must not load the graph")

        fake_llm = FakeLLM()
        app = create_app(repo=ExplodingRepo(), llm=fake_llm)
        with TestClient(app) as client:
            response = client.post("/v1/graphs/anything/evolve", json={"k_rounds": 3})
            assert response.status_code == 501
            assert "Python SDK" in response.json()["detail"]
