# Engineering principles

Methodos uses a small, explicit architecture. The goal is not to add classes
for their own sake; it is to make boundaries, substitution, and failure
behavior visible.

## Boundaries

- `schema.py` owns validated wire and domain data.
- `graph.py` owns pure graph transformations and structural invariants.
- `adapter.py` owns runtime orchestration around a host solver.
- `llm.py` owns provider I/O, timeouts, bounded output, and retries.
- `repo.py` owns durable persistence behind the `Repository` protocol.
- `service.py` owns HTTP DTOs, authentication, readiness, and transport errors.

## Polymorphism

Runtime dependencies are substituted through structural protocols rather than
inheritance: `Solver`, `LLMClient`, `Repository`, and `VectorIndex` are all
replaceable contracts. This keeps the core testable with in-memory doubles and
allows SQLite, filesystem, vector, and future backends to coexist without
coupling the domain to infrastructure.

## Reliability rules

- Validate untrusted identifiers before they reach a path or query.
- Preserve atomic writes and use SQLite transactions for multi-step updates.
- Bound external calls by timeout, output budget, and retry count.
- Make liveness and readiness separate signals.
- Return stable HTTP status codes for malformed input and conflicts.
- Keep production-only configuration explicit through environment variables.

## Naming and style

Public APIs use Google-style `snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE`.
Names are intentionally public where callers need observability; implementation
details are kept behind module boundaries rather than hidden through a maze of
semi-private attributes. Ruff, mypy, and pytest are required checks.
