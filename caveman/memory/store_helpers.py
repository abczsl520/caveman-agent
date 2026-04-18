"""SQLite memory store helpers — extracted to keep sqlite_store.py under 400 lines.

Schema versioning (PRD §8.9.6 + §8.12):
  Current: SCHEMA_VERSION = 1 (v0.4.0 baseline)
  Migration: column-add via PRAGMA table_info detection.
  Future (v0.5.0): schema_version table + numbered migration scripts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import List

from .types import MemoryType, MemoryEntry

logger = logging.getLogger(__name__)

# PRD §8.9.6: Schema version. Increment when adding/changing columns.
# v0.5.0 will introduce a schema_version table for proper migration tracking.
SCHEMA_VERSION = 1


def row_to_entry(row, fts_rank: float | None = None, trust: float | None = None,
                 retrieval_count: int | None = None) -> MemoryEntry:
    """Convert a DB row to MemoryEntry. Unified for all query types."""
    meta = json.loads(row[4]) if row[4] else {}
    if fts_rank is not None:
        meta["_fts_rank"] = fts_rank
    if trust is not None:
        meta.setdefault("trust_score", trust)
    elif len(row) > 5:
        meta.setdefault("trust_score", row[5])
    if retrieval_count is not None:
        meta["retrieval_count"] = retrieval_count
    return MemoryEntry(
        id=row[0], content=row[1],
        memory_type=MemoryType(row[2]),
        created_at=datetime.fromisoformat(row[3]),
        metadata=meta,
    )


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns that may be missing from older schema versions.

    PRD §8.12.1: Current mechanism — detect missing columns via PRAGMA table_info,
    then ALTER TABLE ADD COLUMN. Only additive (can't modify/drop columns).
    v0.5.0 will replace this with numbered migration scripts (§8.12.2).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    migrations = [
        ("trust_score", "ALTER TABLE memories ADD COLUMN trust_score REAL DEFAULT 0.5"),
        ("retrieval_count", "ALTER TABLE memories ADD COLUMN retrieval_count INTEGER DEFAULT 0"),
        ("helpful_count", "ALTER TABLE memories ADD COLUMN helpful_count INTEGER DEFAULT 0"),
        ("entities_json", "ALTER TABLE memories ADD COLUMN entities_json TEXT DEFAULT '[]'"),
    ]
    applied = 0
    for col, sql in migrations:
        if col not in existing:
            try:
                conn.execute(sql)
                applied += 1
                logger.info("Schema migration: added column '%s'", col)
            except sqlite3.OperationalError:
                pass
    if applied:
        conn.commit()
        logger.info("Schema migration complete: %d columns added (schema v%d)", applied, SCHEMA_VERSION)
