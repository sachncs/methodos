"""Tests for `methodos.service` (FastAPI app)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from starlette.types import Message, Receive, Scope, Send

from methodos.jobs import (
    EvolutionJob,
    EvolutionJobWorker,
    JobIdempotencyConflictError,
    SQLiteEvolutionJobStore,
    StoreBackedEvolutionRunner,
)
from methodos.llm import LiteLLMClient, LLMClient, ScopedTokenBudget
from methodos.repo import (
    FilesystemRepository,
    Repository,
    SQLiteRepository,
    Task,
    Trajectory,
)
from methodos.schema import Node, ProceduralGraph
from methodos.service import (
    EvolutionJobStatus,
    EvolveResponse,
    GraphCreateRequest,
    GuidanceResponse,
    HealthResponse,
    create_app,
    parse_api_key_roles,
    validate_api_key_roles,
    validate_cors_origins,
    validate_graph_id,
    validate_idempotency_key,
    validate_runtime_mode,
    validate_tenant_id,
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
    def test_production_mode_requires_authentication(self, seeded_repo: SQLiteRepository) -> None:
        with pytest.raises(ValueError, match="requires METHODOS_API_KEYS"):
            validate_runtime_mode(
                "production",
                repo=seeded_repo,
                authenticated=False,
            )

    def test_production_mode_rejects_filesystem_persistence(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires SQLite or Postgres"):
            validate_runtime_mode(
                "production",
                repo=FilesystemRepository(root=tmp_path / "files"),
                authenticated=True,
                credentials=("x" * 16,),
            )

    def test_production_mode_rejects_weak_credentials(self, seeded_repo: SQLiteRepository) -> None:
        with pytest.raises(ValueError, match="at least 16 characters"):
            validate_runtime_mode(
                "production",
                repo=seeded_repo,
                authenticated=True,
                credentials=("short",),
            )

    @pytest.mark.parametrize("mode", ["development", "test", "staging", "production"])
    def test_runtime_mode_accepts_known_modes(
        self, seeded_repo: SQLiteRepository, mode: str
    ) -> None:
        authenticated = mode in {"staging", "production"}
        credentials = ("x" * 16,) if authenticated else ()
        assert (
            validate_runtime_mode(
                mode,
                repo=seeded_repo,
                authenticated=authenticated,
                credentials=credentials,
            )
            == mode
        )

    def test_runtime_mode_rejects_unknown_mode(self, seeded_repo: SQLiteRepository) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            validate_runtime_mode("preview", repo=seeded_repo, authenticated=False)

    def test_configures_durable_scoped_budget(
        self, seeded_repo: SQLiteRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", "100")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_INPUT", "40")

        app = create_app(repo=seeded_repo)

        with TestClient(app):
            assert isinstance(app.state.llm, LiteLLMClient)
            assert isinstance(app.state.llm.budget, ScopedTokenBudget)
            assert (
                asyncio.run(
                    app.state.llm.budget.reserve_scoped_async(
                        scope="default:g1", system="1234", user="1234", output_tokens=5
                    )
                )
                == 2
            )

    @pytest.mark.parametrize("value", ["not-an-int", "0", "-1"])
    def test_rejects_invalid_token_budget_values(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", value)
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_INPUT", "40")

        with pytest.raises(ValueError, match="must be a positive integer"):
            create_app(repo=seeded_repo)

    def test_rejects_partial_token_budget_configuration(
        self, seeded_repo: SQLiteRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", "100")
        monkeypatch.delenv("METHODOS_TOKEN_BUDGET_INPUT", raising=False)

        with pytest.raises(ValueError, match="must be set together"):
            create_app(repo=seeded_repo)

    def test_configures_pricing_aware_scoped_budget(
        self, seeded_repo: SQLiteRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", "100")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_INPUT", "40")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_COST_USD", "0.50")
        monkeypatch.setenv("METHODOS_TOKEN_PRICE_INPUT_PER_1K_USD", "0.50")
        monkeypatch.setenv("METHODOS_TOKEN_PRICE_OUTPUT_PER_1K_USD", "1.00")

        app = create_app(repo=seeded_repo)

        with TestClient(app):
            assert isinstance(app.state.llm.budget, ScopedTokenBudget)

    def test_rejects_partial_pricing_configuration(
        self, seeded_repo: SQLiteRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", "100")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_INPUT", "40")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_COST_USD", "0.50")
        with pytest.raises(ValueError, match="must be set together"):
            create_app(repo=seeded_repo)

    def test_rejects_durable_budget_for_filesystem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_TOTAL", "100")
        monkeypatch.setenv("METHODOS_TOKEN_BUDGET_INPUT", "40")

        with pytest.raises(ValueError, match="require PGRAPH_BACKEND=sqlite or postgres"):
            create_app(repo=FilesystemRepository(root=tmp_path / "files"))

    def test_parse_api_key_roles(self) -> None:
        assert parse_api_key_roles("read=reader,ops=operator|writer") == {
            "read": frozenset({"reader"}),
            "ops": frozenset({"operator", "writer"}),
        }
        with pytest.raises(ValueError, match="invalid key or role"):
            parse_api_key_roles("read=unknown")
        with pytest.raises(ValueError, match="key=role"):
            parse_api_key_roles("malformed")
        with pytest.raises(ValueError, match="duplicate"):
            parse_api_key_roles("read=reader,read=writer")

    @pytest.mark.parametrize("value", ["", "bad/id", "a" * 129])
    def test_rejects_invalid_graph_ids(self, value: str) -> None:
        with pytest.raises(HTTPException, match="invalid graph_id"):
            validate_graph_id(value)

    @pytest.mark.parametrize("value", ["", "bad tenant", "a" * 65])
    def test_rejects_invalid_tenant_ids(self, value: str) -> None:
        with pytest.raises(ValueError, match="tenant identifier"):
            validate_tenant_id(value)

    @pytest.mark.parametrize("value", ["", "contains spaces", "a" * 129])
    def test_rejects_invalid_idempotency_keys(self, value: str) -> None:
        with pytest.raises(HTTPException, match="invalid Idempotency-Key"):
            validate_idempotency_key(value)
        assert validate_idempotency_key(None) is None

    def test_rejects_roles_for_unconfigured_keys(self) -> None:
        with pytest.raises(ValueError, match="not configured"):
            validate_api_key_roles({"missing": frozenset({"reader"})}, ("active",))

    def test_accepts_roles_for_configured_keys(self) -> None:
        validate_api_key_roles(
            {"reader-key": frozenset({"reader"}), "writer-key": frozenset({"writer"})},
            ("reader-key", "writer-key"),
        )

    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_response_has_request_id_and_security_headers(self, client: TestClient) -> None:
        response = client.get("/health", headers={"x-request-id": "request-1"})
        assert response.headers["x-request-id"] == "request-1"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize("request_id", ["bad request", "/unsafe", "x" * 129])
    def test_invalid_request_id_is_replaced(self, client: TestClient, request_id: str) -> None:
        response = client.get("/health", headers={"x-request-id": request_id})
        assert response.status_code == 200
        assert response.headers["x-request-id"] != request_id
        assert len(response.headers["x-request-id"]) == 36

    def test_metrics_endpoint_is_available(self, client: TestClient) -> None:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text
        assert "methodos_http_requests_total" in response.text

    def test_rejects_oversized_declared_body(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_MAX_BODY_BYTES", "10")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as size_client:
            response = size_client.post(
                "/v1/graphs",
                headers={"content-length": "11"},
                json={"id": "g2"},
            )
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"

    def test_rejects_invalid_declared_body_length(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_MAX_BODY_BYTES", "10")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as size_client:
            response = size_client.post(
                "/v1/graphs",
                headers={"content-length": "not-a-number"},
                json={"id": "g2"},
            )
        assert response.status_code == 413
        assert response.json()["code"] == "request_too_large"

    async def test_rejects_oversized_streamed_body(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_MAX_BODY_BYTES", "10")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/graphs",
            "raw_path": b"/v1/graphs",
            "query_string": b"",
            "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
            "client": ("test", 1234),
            "server": ("test", 80),
        }
        messages: list[Message] = [
            {"type": "http.request", "body": b'{"id":', "more_body": True},
            {"type": "http.request", "body": b'"body-that-is-too-large"}', "more_body": False},
        ]
        sent: list[Message] = []

        async def receive() -> Message:
            return messages.pop(0)

        async def send(message: Message) -> None:
            sent.append(message)

        async with app.router.lifespan_context(app):
            await app(scope, cast(Receive, receive), cast(Send, send))

        response_start = next(
            message for message in sent if message["type"] == "http.response.start"
        )
        assert response_start["status"] == 413

    async def test_stops_processing_when_client_disconnects_during_upload(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/graphs",
            "raw_path": b"/v1/graphs",
            "query_string": b"",
            "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
            "client": ("test", 1234),
            "server": ("test", 80),
        }
        sent: list[Message] = []

        async def receive() -> Message:
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        async with app.router.lifespan_context(app):
            await app(scope, cast(Receive, receive), cast(Send, send))

        response_start = next(
            message for message in sent if message["type"] == "http.response.start"
        )
        assert response_start["status"] == 499
        response_body = next(message for message in sent if message["type"] == "http.response.body")
        assert b"client_disconnected" in response_body["body"]

    def test_metrics_include_store_backed_evolution_queue(
        self,
        seeded_repo: SQLiteRepository,
        tmp_path: Path,
    ) -> None:
        job_store = SQLiteEvolutionJobStore(tmp_path / "jobs.db")
        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=StoreBackedEvolutionRunner(job_store),
        )
        with TestClient(app) as metrics_client:
            response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert "methodos_evolution_jobs" in response.text
        assert 'state="queued"' in response.text

    def test_metrics_include_worker_liveness(
        self,
        seeded_repo: SQLiteRepository,
        tmp_path: Path,
    ) -> None:
        job_store = SQLiteEvolutionJobStore(tmp_path / "jobs.db")

        class Executor:
            async def execute(self, job: EvolutionJob) -> None:
                del job

        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_worker=EvolutionJobWorker(job_store, Executor()),
        )
        with TestClient(app) as metrics_client:
            response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert "methodos_evolution_worker" in response.text


class TestConcurrentHttpResilience:
    async def test_parallel_reads_keep_request_correlation(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as concurrent_client:
            responses = await asyncio.gather(
                *(asyncio.to_thread(concurrent_client.get, "/v1/graphs/g1") for _ in range(24))
            )
        assert all(response.status_code == 200 for response in responses)
        request_ids = {response.headers["x-request-id"] for response in responses}
        assert len(request_ids) == len(responses)

    async def test_parallel_cas_writers_allow_only_one_revision_winner(
        self,
        seeded_repo: SQLiteRepository,
        graph: ProceduralGraph,
    ) -> None:
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as concurrent_client:
            responses = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        concurrent_client.put,
                        "/v1/graphs/g1",
                        headers={"If-Match": "0"},
                        json=graph.model_dump(mode="json"),
                    )
                    for _ in range(8)
                )
            )
        statuses = [response.status_code for response in responses]
        assert statuses.count(200) == 1
        assert statuses.count(409) == len(statuses) - 1

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
            assert (
                protected_client.get(
                    "/v1/graphs/g1", headers={"x-api-key": "test-secret"}
                ).status_code
                == 200
            )
            assert protected_client.get("/health").status_code == 200

    def test_overlapping_api_keys_support_rotation(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_API_KEYS", "old-secret,new-secret")
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as protected_client:
            assert (
                protected_client.get(
                    "/v1/graphs/g1", headers={"x-api-key": "old-secret"}
                ).status_code
                == 200
            )
            assert (
                protected_client.get(
                    "/v1/graphs/g1", headers={"x-api-key": "new-secret"}
                ).status_code
                == 200
            )

    def test_explicit_roles_gate_mutations(
        self,
        seeded_repo: SQLiteRepository,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_API_KEYS", "read-secret,write-secret")
        monkeypatch.setenv(
            "METHODOS_API_KEY_ROLES",
            "read-secret=reader,write-secret=writer",
        )
        app = create_app(repo=seeded_repo, llm=ScriptedLLM(responses=["x"]))
        with TestClient(app) as protected_client:
            assert (
                protected_client.get(
                    "/v1/graphs/g1", headers={"x-api-key": "read-secret"}
                ).status_code
                == 200
            )
            denied = protected_client.post(
                "/v1/graphs", headers={"x-api-key": "read-secret"}, json={"id": "denied"}
            )
            assert denied.status_code == 403
            metrics = protected_client.get("/metrics")
            assert 'methodos_auth_failures_total{reason="forbidden"}' in metrics.text
            created = protected_client.post(
                "/v1/graphs",
                headers={"x-api-key": "write-secret"},
                json={"id": "created"},
            )
            assert created.status_code == 201

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

    @pytest.mark.parametrize("authenticated", [True, False])
    def test_wildcard_cors_origin_fails_fast(self, authenticated: bool) -> None:
        with pytest.raises(ValueError, match=r"wildcard|explicit origin"):
            validate_cors_origins(["*"], authenticated=authenticated)

    def test_readiness_returns_503_on_repository_failure(self) -> None:
        class BrokenRepository:
            async def load_graph(self, graph_id: str) -> ProceduralGraph:
                del graph_id
                raise RuntimeError("storage unavailable")

        app = create_app(
            repo=cast(Repository, BrokenRepository()),
            llm=ScriptedLLM(responses=["x"]),
        )
        with TestClient(app) as broken_client:
            response = broken_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["code"] == "unavailable"

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
            assert (
                tenant_client.post(
                    "/v1/graphs", headers=alpha_headers, json={"id": "shared"}
                ).status_code
                == 201
            )
            assert tenant_client.get("/v1/graphs/shared", headers=alpha_headers).status_code == 200
            assert tenant_client.get("/v1/graphs/shared", headers=beta_headers).status_code == 404
            assert (
                tenant_client.post(
                    "/v1/graphs", headers=beta_headers, json={"id": "shared"}
                ).status_code
                == 201
            )

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

    def test_passes_tenant_graph_budget_scope(self, seeded_repo: SQLiteRepository) -> None:
        llm = ScriptedLLM(responses=["scoped-guidance"])
        app = create_app(repo=seeded_repo, llm=llm)
        with TestClient(app) as scoped_client:
            response = scoped_client.post(
                "/v1/graphs/g1/guidance",
                json={"query": "What?"},
            )

        assert response.status_code == 200
        assert llm.calls[0]["budget_scope"] == "default:g1"

    async def test_cancelled_asgi_request_cancels_provider_call(
        self, seeded_repo: SQLiteRepository
    ) -> None:
        class HangingLLM:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def complete(self, **kwargs: object) -> str:
                del kwargs
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
                return "unreachable"

        llm = HangingLLM()
        app = create_app(repo=seeded_repo, llm=llm)
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://test") as http_client,
        ):
            request_task = asyncio.create_task(
                http_client.post(
                    "/v1/graphs/g1/guidance",
                    json={"query": "cancel me"},
                )
            )
            await asyncio.wait_for(llm.started.wait(), timeout=2.0)
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        assert llm.cancelled.is_set()

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

    def test_rejects_evolution_for_missing_graph(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        class Runner:
            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                del graph_id, k_rounds, request_id
                return "job-1"

            async def status(self, job_id: str) -> EvolutionJobStatus | None:
                del job_id
                return None

        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=Runner(),
        )
        with TestClient(app) as runner_client:
            response = runner_client.post(
                "/v1/graphs/missing/evolve",
                json={"k_rounds": 5},
            )
        assert response.status_code == 404
        assert response.json()["detail"] == "graph not found"

    def test_idempotency_conflict_is_a_structured_conflict(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        class IdempotentRunner:
            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                del graph_id, k_rounds, request_id
                return "job-1"

            async def status(self, job_id: str) -> EvolutionJobStatus | None:
                del job_id
                return None

            async def start_idempotent(
                self,
                *,
                graph_id: str,
                k_rounds: int,
                request_id: str,
                idempotency_key: str,
            ) -> str:
                del graph_id, k_rounds, request_id, idempotency_key
                raise JobIdempotencyConflictError(
                    "idempotency key was used for a different request"
                )

        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=IdempotentRunner(),
        )
        with TestClient(app) as runner_client:
            response = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers={"Idempotency-Key": "retry-1"},
                json={"k_rounds": 5},
            )
        assert response.status_code == 409
        assert response.json()["detail"] == "idempotency key was used for a different request"

    def test_tenant_scoped_runner_hides_job_status_from_other_tenants(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_TENANT_KEYS", "alpha:key-alpha,beta:key-beta")
        monkeypatch.setenv(
            "METHODOS_API_KEY_ROLES",
            "key-alpha=operator,key-beta=operator",
        )
        repo = SQLiteRepository(db_path=tmp_path / "tenant-service.db")
        jobs = SQLiteEvolutionJobStore(tmp_path / "tenant-jobs.db")
        app = create_app(
            repo=repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=StoreBackedEvolutionRunner(jobs),
        )
        alpha_headers = {"X-API-Key": "key-alpha"}
        beta_headers = {"X-API-Key": "key-beta"}
        with TestClient(app) as runner_client:
            created = runner_client.post(
                "/v1/graphs",
                headers=alpha_headers,
                json={"id": "g1"},
            )
            assert created.status_code == 201
            queued = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers=alpha_headers,
                json={"k_rounds": 2},
            )
            assert queued.status_code == 202
            job_id = queued.json()["job_id"]
            assert (
                runner_client.get(f"/v1/evolution/jobs/{job_id}", headers=alpha_headers).status_code
                == 200
            )
            assert (
                runner_client.get(f"/v1/evolution/jobs/{job_id}", headers=beta_headers).status_code
                == 404
            )
            retried = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers={**alpha_headers, "Idempotency-Key": "tenant-retry"},
                json={"k_rounds": 2},
            )
            repeated = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers={**alpha_headers, "Idempotency-Key": "tenant-retry"},
                json={"k_rounds": 2},
            )
            assert retried.status_code == repeated.status_code == 202
            assert retried.json()["job_id"] == repeated.json()["job_id"]

    def test_tenant_active_job_quota_returns_rate_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_TENANT_KEYS", "alpha:key-alpha")
        monkeypatch.setenv("METHODOS_API_KEY_ROLES", "key-alpha=operator")
        repo = SQLiteRepository(db_path=tmp_path / "quota-service.db")
        jobs = SQLiteEvolutionJobStore(tmp_path / "quota-jobs.db")
        app = create_app(
            repo=repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=StoreBackedEvolutionRunner(jobs, max_active_jobs=1),
        )
        headers = {"X-API-Key": "key-alpha"}
        with TestClient(app) as runner_client:
            assert (
                runner_client.post("/v1/graphs", headers=headers, json={"id": "g1"}).status_code
                == 201
            )
            first = runner_client.post(
                "/v1/graphs/g1/evolve", headers=headers, json={"k_rounds": 1}
            )
            second = runner_client.post(
                "/v1/graphs/g1/evolve", headers=headers, json={"k_rounds": 1}
            )
        assert first.status_code == 202
        assert second.status_code == 429

    def test_tenant_mode_rejects_unscoped_runner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("METHODOS_TENANT_KEYS", "alpha:key-alpha")
        monkeypatch.setenv("METHODOS_API_KEY_ROLES", "key-alpha=operator")

        class Runner:
            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                del graph_id, k_rounds, request_id
                return "job-1"

            async def status(self, job_id: str) -> EvolutionJobStatus | None:
                del job_id
                return None

        app = create_app(
            repo=SQLiteRepository(db_path=tmp_path / "unscoped-service.db"),
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=Runner(),
        )
        headers = {"X-API-Key": "key-alpha"}
        with TestClient(app) as runner_client:
            assert (
                runner_client.post("/v1/graphs", headers=headers, json={"id": "g1"}).status_code
                == 201
            )
            queued = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers=headers,
                json={"k_rounds": 1},
            )
            assert queued.status_code == 501
            status = runner_client.get("/v1/evolution/jobs/job-1", headers=headers)
            assert status.status_code == 501

    async def test_injected_runner_returns_accepted_job(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        class Runner:
            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                assert graph_id == "g1"
                assert k_rounds == 5
                assert request_id
                return "job-1"

            async def status(self, job_id: str) -> EvolutionJobStatus | None:
                if job_id != "job-1":
                    return None
                return EvolutionJobStatus(job_id=job_id, status="running", detail="working")

        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=Runner(),
        )
        with TestClient(app) as runner_client:
            response = runner_client.post("/v1/graphs/g1/evolve", json={"k_rounds": 5})
        assert response.status_code == 202
        assert response.json() == {
            "detail": "evolution queued",
            "job_id": "job-1",
            "status": "queued",
        }

        with TestClient(app) as runner_client:
            status = runner_client.get("/v1/evolution/jobs/job-1")
        assert status.status_code == 200
        assert status.json()["status"] == "running"
        with TestClient(app) as runner_client:
            missing = runner_client.get("/v1/evolution/jobs/missing")
        assert missing.status_code == 404

        with TestClient(app) as runner_client:
            unsupported_retry = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers={"Idempotency-Key": "retry-1"},
                json={"k_rounds": 5},
            )
        assert unsupported_retry.status_code == 501

    def test_idempotency_key_uses_runner_capability(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        class IdempotentRunner:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                del graph_id, k_rounds, request_id
                return "job-basic"

            async def status(self, job_id: str) -> EvolutionJobStatus:
                return EvolutionJobStatus(job_id=job_id, status="queued")

            async def start_idempotent(
                self,
                *,
                graph_id: str,
                k_rounds: int,
                request_id: str,
                idempotency_key: str,
            ) -> str:
                del graph_id, k_rounds, request_id
                self.calls.append(idempotency_key)
                return "job-deduplicated"

        runner = IdempotentRunner()
        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=runner,
        )
        headers = {"Idempotency-Key": "retry-1"}
        with TestClient(app) as runner_client:
            first = runner_client.post(
                "/v1/graphs/g1/evolve", headers=headers, json={"k_rounds": 5}
            )
            second = runner_client.post(
                "/v1/graphs/g1/evolve", headers=headers, json={"k_rounds": 5}
            )
        assert first.status_code == second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"] == "job-deduplicated"
        assert runner.calls == ["retry-1", "retry-1"]

    def test_invalid_idempotency_key_is_rejected(
        self,
        seeded_repo: SQLiteRepository,
    ) -> None:
        class Runner:
            async def start(self, *, graph_id: str, k_rounds: int, request_id: str) -> str:
                del graph_id, k_rounds, request_id
                return "job-1"

            async def status(self, job_id: str) -> EvolutionJobStatus:
                return EvolutionJobStatus(job_id=job_id, status="queued")

        app = create_app(
            repo=seeded_repo,
            llm=ScriptedLLM(responses=["x"]),
            evolution_runner=Runner(),
        )
        with TestClient(app) as runner_client:
            response = runner_client.post(
                "/v1/graphs/g1/evolve",
                headers={"Idempotency-Key": "bad key"},
                json={"k_rounds": 5},
            )
        assert response.status_code == 422

    def test_status_requires_configured_runner(self, client: TestClient) -> None:
        response = client.get("/v1/evolution/jobs/job-1")
        assert response.status_code == 501


class TestRollbackEndpoint:
    async def test_rolls_back_with_revision_guard(
        self,
        client: TestClient,
        seeded_repo: SQLiteRepository,
    ) -> None:
        await seeded_repo.snapshot("g1", "release-1")
        graph = client.get("/v1/graphs/g1").json()
        graph["nodes"]["start"]["description"] = "changed"
        updated = client.put("/v1/graphs/g1", headers={"If-Match": "0"}, json=graph)
        assert updated.status_code == 200

        response = client.post(
            "/v1/graphs/g1/rollback/release-1",
            headers={"If-Match": "1"},
        )

        assert response.status_code == 200
        assert response.json()["revision"] == 2
        assert response.json()["nodes"]["start"]["description"] == ""

    def test_requires_if_match(self, client: TestClient) -> None:
        response = client.post("/v1/graphs/g1/rollback/release-1")
        assert response.status_code == 428

    def test_rejects_invalid_if_match(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/rollback/release-1",
            headers={"If-Match": "old"},
        )
        assert response.status_code == 400

    def test_rejects_negative_if_match(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/rollback/release-1",
            headers={"If-Match": "-1"},
        )
        assert response.status_code == 400

    def test_rejects_stale_if_match(
        self,
        client: TestClient,
        seeded_repo: SQLiteRepository,
    ) -> None:
        asyncio.run(seeded_repo.snapshot("g1", "release-1"))
        response = client.post(
            "/v1/graphs/g1/rollback/release-1",
            headers={"If-Match": "1"},
        )
        assert response.status_code == 409

    def test_missing_snapshot_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/v1/graphs/g1/rollback/missing",
            headers={"If-Match": "0"},
        )
        assert response.status_code == 404


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
