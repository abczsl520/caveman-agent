"""SQLite memory store helpers — extracted to keep sqlite_store.py under 400 lines.

Schema versioning (PRD §8.9.6 + §8.12):
  Current: SCHEMA_VERSION = 3
  Migration: schema_version table + numbered, transactional migration functions.
  Dry-run: inspect pending migrations without mutating the database.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Callable

from caveman.memory.sources import canonicalize_memory_source
from .types import MemoryType, MemoryEntry

__all__ = [
    "SCHEMA_VERSION",
    "row_to_entry",
    "is_quarantined",
    "quarantine_memory_sql",
    "active_memory_sql",
    "cleanup_related_refs",
    "get_schema_version",
    "pending_migrations",
    "migrate_schema",
    "normalize_import_metadata",
]


logger = logging.getLogger(__name__)

# PRD §8.9.6: Schema version. Increment when adding/changing columns.
SCHEMA_VERSION = 3


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "component TEXT PRIMARY KEY, "
        "version INTEGER NOT NULL, "
        "applied_at TEXT NOT NULL"
        ")"
    )


def get_schema_version(conn: sqlite3.Connection, component: str = "memory") -> int:
    """Return tracked schema version for a component, creating the table if needed."""
    _ensure_schema_version_table(conn)
    row = conn.execute(
        "SELECT version FROM schema_version WHERE component = ?", (component,)
    ).fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int, component: str = "memory") -> None:
    _ensure_schema_version_table(conn)
    conn.execute(
        "INSERT INTO schema_version(component, version, applied_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(component) DO UPDATE SET "
        "version = excluded.version, applied_at = excluded.applied_at",
        (component, version, datetime.now().isoformat()),
    )


def _memory_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}


def _migration_001_baseline_columns(conn: sqlite3.Connection) -> None:
    """Ensure v1 baseline additive columns exist on legacy memory DBs."""
    existing = _memory_columns(conn)
    migrations = [
        ("trust_score", "ALTER TABLE memories ADD COLUMN trust_score REAL DEFAULT 0.5"),
        ("retrieval_count", "ALTER TABLE memories ADD COLUMN retrieval_count INTEGER DEFAULT 0"),
        ("helpful_count", "ALTER TABLE memories ADD COLUMN helpful_count INTEGER DEFAULT 0"),
        ("entities_json", "ALTER TABLE memories ADD COLUMN entities_json TEXT DEFAULT '[]'"),
    ]
    for col, sql in migrations:
        if col not in existing:
            conn.execute(sql)
            logger.info("Schema migration v1: added column '%s'", col)


def _migration_002_last_accessed_column(conn: sqlite3.Connection) -> None:
    """Promote last_accessed from metadata_json to a first-class queryable column."""
    existing = _memory_columns(conn)
    if "last_accessed" not in existing:
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")
        logger.info("Schema migration v2: added column 'last_accessed'")

    rows = conn.execute("SELECT id, metadata_json FROM memories").fetchall()
    for memory_id, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            continue
        last_accessed = metadata.get("last_accessed")
        if last_accessed:
            conn.execute(
                "UPDATE memories SET last_accessed = COALESCE(last_accessed, ?) WHERE id = ?",
                (last_accessed, memory_id),
            )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)")


def _infer_import_source_from_metadata(metadata: dict) -> str | None:
    raw_source = metadata.get("source")
    if isinstance(raw_source, str) and raw_source.strip() and raw_source.strip() != "<missing>":
        return canonicalize_memory_source(raw_source)

    evidence = " ".join(
        str(metadata.get(key, ""))
        for key in ("source_file", "path", "source_path", "origin", "import_source")
    ).lower()
    if "openclaw-session" in evidence or "openclaw_sessions" in evidence:
        return "import:openclaw-session"
    if "openclaw" in evidence:
        return "import:openclaw"
    if "hermes" in evidence:
        return "import:hermes"
    if "claude" in evidence:
        return "import:claude-code"
    if "codex" in evidence:
        return "import:codex"
    return None


def _infer_import_source_from_content(content: object) -> str | None:
    if not isinstance(content, str):
        return None
    stripped = content.lstrip()
    if stripped.startswith("Task: ") and ("\nResult:" in stripped[:500] or " Result:" in stripped[:500]):
        return "legacy:task-result"
    return None


def normalize_import_metadata(
    metadata: dict,
    fallback_source: str | None = None,
    reason: str = "normalize-import-source",
) -> tuple[dict, bool]:
    """Backfill/normalize import source metadata while preserving provenance."""
    meta = dict(metadata or {})
    previous = meta.get("source")
    source = fallback_source or _infer_import_source_from_metadata(meta)
    if not source:
        return meta, False
    if isinstance(previous, str) and previous.strip() == source:
        return meta, False
    if previous not in (None, "") and previous != "<missing>":
        return meta, False

    meta["source"] = source
    meta.setdefault("source_normalization_previous", previous)
    meta.setdefault("source_normalization_reason", reason)
    meta.setdefault("source_normalized_at", datetime.now().isoformat())
    return meta, True


def _migration_003_normalize_import_sources(conn: sqlite3.Connection) -> None:
    """Backfill missing import source metadata using preserved source-file provenance."""
    rows = conn.execute("SELECT id, content, metadata_json FROM memories").fetchall()
    for memory_id, content, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metadata, changed = normalize_import_metadata(
            metadata,
            fallback_source=_infer_import_source_from_content(content),
            reason="migration-v3",
        )
        if changed:
            conn.execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), memory_id),
            )


_MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "baseline additive memory columns", _migration_001_baseline_columns),
    (2, "last_accessed first-class column", _migration_002_last_accessed_column),
    (3, "normalize import memory source metadata", _migration_003_normalize_import_sources),
]


def pending_migrations(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Return pending memory schema migrations without mutating memory data."""
    current = get_schema_version(conn)
    return [(version, name) for version, name, _ in _MIGRATIONS if version > current]


def row_to_entry(row, fts_rank: float | None = None, trust: float | None = None,
                 retrieval_count: int | None = None,
                 last_accessed: str | None = None) -> MemoryEntry:
    """Convert a DB row to MemoryEntry. Unified for all query types.

    Be deliberately tolerant: one corrupt legacy row must not make recall fail
    for the whole agent loop. Bad metadata/type/date values are logged with the
    memory id and downgraded to safe defaults.
    """
    memory_id = row[0]
    try:
        meta = json.loads(row[4]) if row[4] else {}
        if not isinstance(meta, dict):
            logger.warning("Memory row %s has non-object metadata_json; using empty metadata", memory_id)
            meta = {}
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Memory row %s has invalid metadata_json: %s", memory_id, exc)
        meta = {}

    if fts_rank is not None:
        meta["_fts_rank"] = fts_rank
    if trust is not None:
        # trust_score column is the source of truth; metadata_json may contain
        # the initial value from creation time and must not shadow live updates.
        meta["trust_score"] = trust
    elif len(row) > 5:
        meta["trust_score"] = row[5]
    if retrieval_count is not None:
        meta["retrieval_count"] = retrieval_count
    if last_accessed is None and len(row) > 7:
        last_accessed = row[7]
    if last_accessed is not None:
        meta["last_accessed"] = last_accessed

    try:
        memory_type = MemoryType(row[2])
    except ValueError:
        logger.warning("Memory row %s has invalid type %r; falling back to semantic", memory_id, row[2])
        memory_type = MemoryType.SEMANTIC

    try:
        created_at = datetime.fromisoformat(row[3])
    except (ValueError, TypeError) as exc:
        logger.warning("Memory row %s has invalid created_at %r: %s", memory_id, row[3], exc)
        created_at = datetime.now()

    return MemoryEntry(
        id=memory_id, content=row[1],
        memory_type=memory_type,
        created_at=created_at,
        metadata=meta,
    )


def is_quarantined(entry: MemoryEntry) -> bool:
    """Return True when a memory has been removed from active recall."""
    return str(entry.metadata.get("governance_state", "")).lower() == "quarantined"


def quarantine_memory_sql(metadata_column: str = "metadata_json") -> str:
    """Return a json-safe SQL predicate for memories currently in quarantine."""
    return (
        f"({metadata_column} IS NOT NULL "
        f"AND CASE WHEN json_valid({metadata_column}) THEN "
        f"lower(json_extract({metadata_column}, '$.governance_state')) = 'quarantined' "
        "ELSE 0 END)"
    )


def active_memory_sql(metadata_column: str = "metadata_json") -> str:
    """Return a json-safe SQL predicate for memories exposed to recall/search.

    Legacy rows may contain malformed metadata JSON. Guard json_extract with
    json_valid so one corrupt import row cannot crash active memory queries.
    """
    return (
        f"({metadata_column} IS NULL "
        f"OR CASE WHEN json_valid({metadata_column}) THEN "
        f"COALESCE(lower(json_extract({metadata_column}, '$.governance_state')) != 'quarantined', 1) "
        "ELSE 1 END)"
    )


def cleanup_related_refs(conn: sqlite3.Connection, memory_id: str) -> None:
    """Remove a memory id from other rows' metadata.related lists."""
    rows = conn.execute(
        "SELECT id, metadata_json FROM memories WHERE metadata_json LIKE ?",
        (f"%{memory_id}%",),
    ).fetchall()
    for row_id, metadata_json in rows:
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
            related = meta.get("related", [])
            if memory_id in related:
                related.remove(memory_id)
                meta["related"] = related
                conn.execute(
                    "UPDATE memories SET metadata_json = ? WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), row_id),
                )
        except Exception as exc:
            logger.debug("forget: suppressed %s", exc)


def migrate_schema(conn: sqlite3.Connection, dry_run: bool = False) -> list[tuple[int, str]]:
    """Apply numbered memory schema migrations transactionally.

    PRD §8.12: migrations are tracked in a `schema_version` table and can be
    inspected with dry-run before mutating the database. Each version is applied
    inside one transaction; failure rolls back that version.
    """
    pending = pending_migrations(conn)
    if dry_run:
        return pending

    for version, name, fn in _MIGRATIONS:
        if version <= get_schema_version(conn):
            continue
        try:
            conn.execute("BEGIN")
            fn(conn)
            _set_schema_version(conn, version)
            conn.commit()
            logger.info("Schema migration v%d complete: %s", version, name)
        except Exception:
            conn.rollback()
            logger.exception("Schema migration v%d failed; rolled back", version)
            raise
    return pending
