"""Small, explicit schema-migration runner for repository backends."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


class MigrationError(RuntimeError):
    """Raised when a database cannot be upgraded safely."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward-only migration and its SQL statements."""

    version: int
    statements: tuple[str, ...]


class MigrationRunner:
    """Validate and apply a contiguous forward-only migration plan."""

    def __init__(self, migrations: tuple[Migration, ...]) -> None:
        """Create a runner from a contiguous, one-based migration sequence."""
        if not migrations:
            raise ValueError("at least one migration is required")
        versions = tuple(migration.version for migration in migrations)
        if versions != tuple(range(1, len(versions) + 1)):
            raise ValueError("migration versions must be contiguous starting at 1")
        self.migrations = migrations
        self.latest_version = versions[-1]

    def pending(self, current_version: int) -> tuple[Migration, ...]:
        """Return migrations newer than `current_version`, or fail closed."""
        if current_version < 0:
            raise MigrationError("database schema version cannot be negative")
        if current_version > self.latest_version:
            raise MigrationError(
                f"unsupported database schema version {current_version}; "
                f"expected at most {self.latest_version}"
            )
        return self.migrations[current_version:]

    def apply_sqlite(
        self,
        connection: sqlite3.Connection,
        current_version: int,
        *,
        metadata_table: str = "schema_meta",
    ) -> None:
        """Apply pending migrations in one caller-owned SQLite transaction."""
        for migration in self.pending(current_version):
            for statement in migration.statements:
                connection.execute(statement)
            if migration.version == 1:
                connection.execute(
                    f"INSERT INTO {metadata_table} (version) VALUES (?)",
                    (migration.version,),
                )
            else:
                connection.execute(
                    f"UPDATE {metadata_table} SET version = ?",
                    (migration.version,),
                )

    async def apply_asyncpg(
        self,
        connection: Any,
        current_version: int,
        *,
        metadata_table: str,
    ) -> None:
        """Apply pending migrations in a caller-owned asyncpg transaction."""
        for migration in self.pending(current_version):
            if migration.statements:
                await connection.execute("\n".join(migration.statements))
            if migration.version == 1:
                await connection.execute(
                    f"INSERT INTO {metadata_table} (version) VALUES ($1)",
                    migration.version,
                )
            else:
                await connection.execute(
                    f"UPDATE {metadata_table} SET version = $1",
                    migration.version,
                )


__all__ = ["Migration", "MigrationError", "MigrationRunner"]
