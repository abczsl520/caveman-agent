"""Reversible quarantine lifecycle helpers for SQLite memory stores."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .metadata import validate_metadata
from .store_helpers import quarantine_memory_sql, row_to_entry
from .types import MemoryEntry

if TYPE_CHECKING:
    from .sqlite_store import SQLiteMemoryStore


SELECT_MEMORY_ROW = (
    "SELECT id, content, type, created_at, metadata_json, "
    "trust_score, retrieval_count, last_accessed FROM memories "
)


def list_quarantined(
    store: "SQLiteMemoryStore", source: str | None = None, limit: int = 50
) -> list[MemoryEntry]:
    """List quarantined memories for operator review."""
    where = quarantine_memory_sql()
    params: list = []
    if source:
        where += " AND CASE WHEN json_valid(metadata_json) THEN json_extract(metadata_json, '$.source') = ? ELSE 0 END"
        params.append(source)
    params.append(limit)
    rows = store._get_conn().execute(
        SELECT_MEMORY_ROW + f"WHERE {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()
    return [
        row_to_entry(row, trust=row[5], retrieval_count=row[6], last_accessed=row[7])
        for row in rows
    ]


async def restore_quarantined(
    store: "SQLiteMemoryStore",
    memory_id: str,
    *,
    restored_by: str = "operator",
    restore_reason: str = "manual restore",
) -> bool:
    """Restore a quarantined memory while retaining audit metadata."""
    async with store._write_lock:
        conn = store._get_conn()
        row = conn.execute(
            "SELECT metadata_json FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return False
        existing = _safe_metadata(row)
        if str(existing.get("governance_state", "")).lower() != "quarantined":
            return False
        existing.update(
            validate_metadata(
                {
                    "governance_state": "active",
                    "previous_governance_state": "quarantined",
                    "restored_at": datetime.now(timezone.utc).isoformat(),
                    "restored_by": restored_by,
                    "restore_reason": restore_reason,
                },
                context="restore_quarantined",
            )
        )
        conn.execute(
            "UPDATE memories SET metadata_json = ? WHERE id = ?",
            (json.dumps(existing, ensure_ascii=False), memory_id),
        )
        conn.commit()
    return True


def _safe_metadata(row: sqlite3.Row | tuple) -> dict:
    try:
        metadata = json.loads(row[0]) if row[0] else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}
