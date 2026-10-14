# Testing

The repository's default test command is the authoritative local quality gate.
It runs the unit, API, persistence, worker, and service tests with branch
coverage enabled.

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check methodos tests scripts eval/hotpotqa/*.py examples
uv run mypy methodos tests scripts eval/hotpotqa/*.py examples
uv run pytest -q --cov=methodos --cov-fail-under=95
```

## Postgres integration tests

Postgres tests are skipped unless `METHODOS_POSTGRES_TEST_DSN` is set. The
following uses a disposable local container and runs the complete suite, so
the repository-wide coverage threshold remains meaningful:

```bash
container_name="methodos-test-postgres-$$"
docker run --detach --rm --name "$container_name" \
  --env POSTGRES_USER=methodos \
  --env POSTGRES_PASSWORD=methodos \
  --env POSTGRES_DB=methodos \
  --publish 55432:5432 postgres:16

METHODOS_POSTGRES_TEST_DSN=postgresql://methodos:methodos@127.0.0.1:55432/methodos \
  uv run pytest -q --cov=methodos --cov-fail-under=95

docker stop "$container_name"
```

To run only the Postgres tests, disable the global coverage gate because
deselected tests cannot provide repository-wide coverage:

```bash
METHODOS_POSTGRES_TEST_DSN=postgresql://methodos:methodos@127.0.0.1:55432/methodos \
  uv run pytest -q -m postgres --no-cov
```

CI provisions Postgres as a service and runs the complete suite with the DSN
set, so the integration path is part of the hosted gate.

## External-provider tests

Live LLM tests remain opt-in and require credentials for a supported provider.
They are intentionally not part of the unauthenticated default gate because
they incur cost and depend on external availability. Run them explicitly in a
controlled environment:

```bash
OPENAI_API_KEY=... uv run pytest -q -m live_llm --no-cov
```

Never commit provider credentials or use production keys for test runs.
