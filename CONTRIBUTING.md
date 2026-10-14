# Contributing to Methodos

Thank you for helping improve Methodos. Changes should preserve the public API,
durable-data compatibility, tenant isolation, and the failure behavior described
in the production-readiness audit.

## Local checks

Install the locked development environment and run the same checks used by CI:

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run ruff format --check methodos tests scripts eval/hotpotqa/*.py examples
uv run mypy methodos tests scripts eval/hotpotqa/*.py examples
uv run pytest --cov=methodos --cov-fail-under=95
cd site && npm ci && npm run build && npm run test:e2e
```

When changing persistence, authentication, request handling, workers, or
deployment files, add a regression test for the failure mode and update the
relevant runbook or compatibility note.

## Design and naming

Read [docs/engineering.md](docs/engineering.md) before changing core code.
Use public `snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE` names. Use one
leading underscore for implementation-only state; reserve double-leading
underscores for language protocols and name-mangling cases. Prefer injected
Protocol capabilities over global state and keep domain logic separate from
transport and persistence concerns.

## Pull requests

Keep changes focused, explain operational impact, and include test evidence.
Schema or API changes must update [docs/compatibility.md](docs/compatibility.md)
and [CHANGELOG.md](CHANGELOG.md). Security-sensitive changes should explain
authentication, authorization, secret-handling, and rollback implications.
