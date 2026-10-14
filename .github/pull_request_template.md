## Summary

<!-- What changed and why? -->

## Verification

- [ ] Tests added or updated for changed behavior and failure paths.
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check methodos tests scripts eval/hotpotqa/*.py examples`
- [ ] `uv run mypy methodos tests scripts eval/hotpotqa/*.py examples`
- [ ] `uv run pytest --cov=methodos --cov-fail-under=95`
- [ ] Site checks run when `site/` changes.

## Operational impact

- [ ] API, persistence, authentication, tenant, or compatibility impact reviewed.
- [ ] Deployment/runbook/changelog documentation updated when needed.
- [ ] No secrets, credentials, or sensitive identifiers added.
