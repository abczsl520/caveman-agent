"""Tests for numbered memory schema migrations."""
from __future__ import annotations

import sqlite3

from caveman.cli.migrate import run_migrate
from caveman.memory.store_helpers import get_schema_version, migrate_schema, pending_migrations


def _create_legacy_memory_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, "
        "content TEXT NOT NULL, "
        "type TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}'"
        ")"
    )
    conn.commit()


def test_memory_schema_migration_creates_version_table_and_columns(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)

    assert get_schema_version(conn) == 0
    assert pending_migrations(conn) == [
        (1, "baseline additive memory columns"),
        (2, "last_accessed first-class column"),
    ]

    applied = migrate_schema(conn)
    assert applied == [
        (1, "baseline additive memory columns"),
        (2, "last_accessed first-class column"),
    ]
    assert get_schema_version(conn) == 2

    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert {"trust_score", "retrieval_count", "helpful_count", "entities_json", "last_accessed"} <= columns
    assert pending_migrations(conn) == []
    conn.close()


def test_memory_schema_migration_dry_run_does_not_apply_columns(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)

    pending = migrate_schema(conn, dry_run=True)
    assert pending == [
        (1, "baseline additive memory columns"),
        (2, "last_accessed first-class column"),
    ]
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "trust_score" not in columns
    assert "last_accessed" not in columns
    assert get_schema_version(conn) == 0
    conn.close()


def test_memory_schema_migration_backfills_last_accessed_from_metadata(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (
            "m1",
            "old memory",
            "semantic",
            "2026-01-01T00:00:00",
            '{"last_accessed": "2026-04-24T12:00:00"}',
        ),
    )
    conn.commit()

    migrate_schema(conn)
    row = conn.execute("SELECT last_accessed FROM memories WHERE id = 'm1'").fetchone()
    assert row[0] == "2026-04-24T12:00:00"
    assert get_schema_version(conn) == 2
    conn.close()


def test_memory_schema_migration_rolls_back_failed_version(tmp_path, monkeypatch):
    """A failing numbered migration must not advance schema_version or leave partial DDL."""
    import caveman.memory.store_helpers as helpers

    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)

    def broken_migration(c: sqlite3.Connection) -> None:
        c.execute("ALTER TABLE memories ADD COLUMN partial_col TEXT")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        helpers,
        "_MIGRATIONS",
        [(1, "broken migration", broken_migration)],
    )

    try:
        helpers.migrate_schema(conn)
        assert False, "Expected migration failure"
    except RuntimeError:
        pass

    assert helpers.get_schema_version(conn) == 0
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "partial_col" not in columns
    assert "last_accessed" not in columns
    conn.close()



def test_migrate_cli_dry_run_and_apply(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)
    conn.close()

    preview = run_migrate(db_path=db, dry_run=True)
    assert "2 pending migration" in preview
    assert "v1: baseline additive memory columns" in preview
    assert "v2: last_accessed first-class column" in preview

    applied = run_migrate(db_path=db, dry_run=False)
    assert "Applied 2 migration" in applied
    assert "now v2" in applied

    after = run_migrate(db_path=db, dry_run=True)
    assert "no pending migrations" in after


def test_migrate_cli_apply_initializes_brand_new_database(tmp_path):
    db = tmp_path / "brand-new.db"
    result = run_migrate(db_path=db, dry_run=False)
    assert "Memory schema" in result

    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "memories" in tables
    assert "schema_version" in tables
    assert get_schema_version(conn) == 2
    conn.close()
