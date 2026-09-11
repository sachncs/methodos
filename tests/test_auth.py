"""Tests for `methodos.auth`."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from methodos.auth import APIKeyAuth, api_key_from_env, auth_dependency, extract_candidate
from methodos.service import create_app
from tests.conftest import FakeLLM


class TestExtractCandidate:
    def test_x_api_key_wins(self) -> None:
        assert extract_candidate(authorization=None, x_api_key="abc") == "abc"

    def test_bearer_scheme(self) -> None:
        assert extract_candidate(authorization="Bearer secret", x_api_key=None) == "secret"

    def test_bearer_case_insensitive(self) -> None:
        assert extract_candidate(authorization="bearer t", x_api_key=None) == "t"

    def test_non_bearer_scheme_ignored(self) -> None:
        assert extract_candidate(authorization="Basic xyz", x_api_key=None) is None

    def test_bearer_with_empty_token_ignored(self) -> None:
        assert extract_candidate(authorization="Bearer ", x_api_key=None) is None

    def test_no_headers(self) -> None:
        assert extract_candidate(authorization=None, x_api_key=None) is None


class TestAPIKeyAuth:
    def test_disabled_when_key_unset(self) -> None:
        auth = APIKeyAuth(configured_key=None)
        assert auth.enabled is False
        auth.check(authorization=None, x_api_key=None)  # no-op

    def test_disabled_when_key_empty_string(self) -> None:
        auth = APIKeyAuth(configured_key="")
        assert auth.enabled is False

    def test_enabled_when_key_set(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        assert auth.enabled is True

    def test_valid_bearer_passes(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        auth.check(authorization="Bearer secret", x_api_key=None)

    def test_valid_x_api_key_passes(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        auth.check(authorization=None, x_api_key="secret")

    def test_missing_headers_raises_401(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        with pytest.raises(HTTPException) as exc:
            auth.check(authorization=None, x_api_key=None)
        assert exc.value.status_code == 401
        assert exc.value.headers["WWW-Authenticate"] == "Bearer"

    def test_invalid_key_raises_401(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        with pytest.raises(HTTPException) as exc:
            auth.check(authorization="Bearer wrong", x_api_key=None)
        assert exc.value.status_code == 401


class TestApiKeyFromEnv:
    def test_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGRAPH_API_KEY", "my-key")
        assert api_key_from_env() == "my-key"

    def test_empty_string_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PGRAPH_API_KEY", "")
        assert api_key_from_env() is None

    def test_missing_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PGRAPH_API_KEY", raising=False)
        assert api_key_from_env() is None


class TestAuthDependencyIntegration:
    """The dependency wired into `create_app` actually enforces auth."""

    def test_no_auth_when_unset(self) -> None:
        fake_llm = FakeLLM(responses=["x"])
        app = create_app(repo=None, llm=fake_llm)
        with TestClient(app) as client:
            response = client.post("/v1/graphs", json={"id": "x"})
            assert response.status_code == 201

    def test_missing_header_returns_401(self) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="secret")
        app = create_app(repo=None, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            response = client.post("/v1/graphs", json={"id": "x"})
            assert response.status_code == 401

    def test_valid_bearer_passes(self) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="secret")
        app = create_app(repo=None, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs",
                json={"id": "x"},
                headers={"Authorization": "Bearer secret"},
            )
            assert response.status_code == 201

    def test_valid_x_api_key_passes(self) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="secret")
        app = create_app(repo=None, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs",
                json={"id": "x"},
                headers={"X-API-Key": "secret"},
            )
            assert response.status_code == 201

    def test_invalid_key_returns_401(self) -> None:
        fake_llm = FakeLLM()
        auth = APIKeyAuth(configured_key="secret")
        app = create_app(repo=None, llm=fake_llm, auth=auth)
        with TestClient(app) as client:
            response = client.post(
                "/v1/graphs",
                json={"id": "x"},
                headers={"Authorization": "Bearer wrong"},
            )
            assert response.status_code == 401

    def test_auth_dependency_callable_returns_callable(self) -> None:
        auth = APIKeyAuth(configured_key="secret")
        dep = auth_dependency(auth)
        assert callable(dep)
