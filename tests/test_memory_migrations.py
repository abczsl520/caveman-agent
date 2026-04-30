"""Tests for numbered memory schema migrations."""
from __future__ import annotations

import json
import sqlite3

from caveman.cli.migrate import run_migrate
from caveman.memory.store_helpers import (
    get_schema_version,
    migrate_schema,
    normalize_import_metadata,
    pending_migrations,
)


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
        (3, "normalize import memory source metadata"),
    ]

    applied = migrate_schema(conn)
    assert applied == [
        (1, "baseline additive memory columns"),
        (2, "last_accessed first-class column"),
        (3, "normalize import memory source metadata"),
    ]
    assert get_schema_version(conn) == 3

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
        (3, "normalize import memory source metadata"),
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
    assert get_schema_version(conn) == 3
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
    assert "3 pending migration" in preview
    assert "v1: baseline additive memory columns" in preview
    assert "v2: last_accessed first-class column" in preview
    assert "v3: normalize import memory source metadata" in preview

    applied = run_migrate(db_path=db, dry_run=False)
    assert "Applied 3 migration" in applied
    assert "now v3" in applied

    after = run_migrate(db_path=db, dry_run=True)
    assert "no pending migrations" in after


def test_normalize_import_metadata_backfills_source_with_provenance():
    meta, changed = normalize_import_metadata(
        {"source_file": "/tmp/openclaw-memory/memory/dev-rules.md", "source": ""},
        fallback_source="import:openclaw",
        reason="migration-v3",
    )

    assert changed is True
    assert meta["source"] == "import:openclaw"
    assert meta["source_file"] == "/tmp/openclaw-memory/memory/dev-rules.md"
    assert meta["source_normalized_at"]
    assert meta["source_normalization_reason"] == "migration-v3"
    assert meta["source_normalization_previous"] == ""


def test_memory_schema_migration_backfills_missing_import_sources(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (
            "m-openclaw",
            "imported openclaw memory",
            "semantic",
            "2026-01-01T00:00:00",
            '{"source_file": "/Users/me/.hermes/openclaw-memory/memory/dev-rules.md"}',
        ),
    )
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (
            "m-user",
            "organic user memory",
            "semantic",
            "2026-01-01T00:00:00",
            "{}",
        ),
    )
    conn.commit()

    applied = migrate_schema(conn)
    assert applied == [
        (1, "baseline additive memory columns"),
        (2, "last_accessed first-class column"),
        (3, "normalize import memory source metadata"),
    ]
    assert get_schema_version(conn) == 3

    row = conn.execute("SELECT metadata_json FROM memories WHERE id = 'm-openclaw'").fetchone()
    meta = json.loads(row[0])
    assert meta["source"] == "import:openclaw"
    assert meta["source_file"].endswith("dev-rules.md")
    assert meta["source_normalization_reason"] == "migration-v3"
    assert meta["source_normalization_previous"] is None

    user_meta = json.loads(
        conn.execute("SELECT metadata_json FROM memories WHERE id = 'm-user'").fetchone()[0]
    )
    assert "source" not in user_meta
    conn.close()


def test_memory_schema_migration_backfills_legacy_task_result_sources(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (
            "legacy-task",
            "Task: What is 2+2? Result: 4",
            "episodic",
            "2026-01-01T00:00:00",
            "{}",
        ),
    )
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (
            "organic-task-note",
            "Task: describe the result-oriented roadmap for Caveman",
            "semantic",
            "2026-01-01T00:00:00",
            "{}",
        ),
    )
    conn.commit()

    migrate_schema(conn)

    meta = json.loads(
        conn.execute("SELECT metadata_json FROM memories WHERE id = 'legacy-task'").fetchone()[0]
    )
    assert meta["source"] == "legacy:task-result"
    assert meta["source_normalization_reason"] == "migration-v3"
    assert meta["source_normalization_previous"] is None

    organic_meta = json.loads(
        conn.execute("SELECT metadata_json FROM memories WHERE id = 'organic-task-note'").fetchone()[0]
    )
    assert "source" not in organic_meta
    conn.close()


def test_memory_schema_migration_preserves_malformed_metadata(tmp_path):
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _create_legacy_memory_schema(conn)
    conn.execute(
        "INSERT INTO memories(id, content, type, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
        ("bad", "malformed", "semantic", "2026-01-01T00:00:00", "{not-json"),
    )
    conn.commit()

    migrate_schema(conn)

    row = conn.execute("SELECT metadata_json FROM memories WHERE id = 'bad'").fetchone()
    assert row[0] == "{not-json"
    assert get_schema_version(conn) == 3
    conn.close()


def test_migrate_cli_apply_initializes_brand_new_database(tmp_path):
    db = tmp_path / "brand-new.db"
    result = run_migrate(db_path=db, dry_run=False)
    assert "Memory schema" in result

    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "memories" in tables
    assert "schema_version" in tables
    assert get_schema_version(conn) == 3
    conn.close()
