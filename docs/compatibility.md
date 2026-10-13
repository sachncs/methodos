# Compatibility and release policy

Methodos follows semantic versioning for the Python package and uses explicit
`schema_version` values for persisted graphs. A release may add fields with
safe defaults, but it must not silently reinterpret existing graph data.

Breaking changes require:

1. a major version increment;
2. an entry in `CHANGELOG.md`;
3. migration code and migration tests for persisted data; and
4. updated API documentation and examples.

The CI release workflow validates tagged releases before publishing generated
release notes and the Python distribution artifacts.
