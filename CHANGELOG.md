# Changelog

All notable changes to methodos are documented here. Releases use Git tags
with the `vMAJOR.MINOR.PATCH` format and are published by the release workflow.

## Unreleased

- Migrated the product site to Astro and GitHub Pages artifact deployment.
- Added responsive product UI, procedural graph visual, and new methodos mark.
- Added stable API error envelopes, request correlation, and `If-Match`
  optimistic-concurrency updates.
- Added LLM circuit breaking, browser smoke tests, container SBOM generation,
  and vulnerability scanning.

## Compatibility policy

- Public Python imports and HTTP response fields are preserved within a minor
  release.
- Breaking API or persisted-schema changes require a major version and a
  migration note.
- Graph `schema_version` changes require a migration runner before the version
  is accepted by a production repository.
