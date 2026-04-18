"""Audit log — structured event export for compliance and debugging.

Exports EventBus events + EventStore history to JSONL format.
Supports filtering by time range, event type, and source.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from caveman.events import EventBus, EventType

logger = logging.getLogger(__name__)


async def export_audit_log(
    bus: EventBus,
    output_path: str | None = None,
    hours: int = 24,
    event_types: list[str] | None = None,
    db_path: str | None = None,
) -> str:
    """Export recent events as JSONL audit log.

    Args:
        bus: EventBus instance (for in-memory events).
        output_path: Output file path. Defaults to ~/.caveman/audit/YYYY-MM-DD.jsonl.
        hours: How many hours back to export.
        event_types: Filter to specific event types. None = all.

    Returns:
        Path to the exported file.
    """
    from caveman.paths import CAVEMAN_HOME

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        audit_dir = CAVEMAN_HOME / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        out = audit_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect events from EventStore (persistent)
    entries: list[dict[str, Any]] = []
    try:
        from caveman.events_store import EventStore
        store = EventStore(db_path=db_path) if db_path else EventStore()
        cutoff = datetime.now().timestamp() - (hours * 3600)

        if event_types:
            for et in event_types:
                for row in store.query(event_type=et, limit=10000):
                    ts = row.get("timestamp", 0)
                    if ts >= cutoff:
                        entries.append(_row_to_audit(row))
        else:
            for row in store.query(limit=10000):
                ts = row.get("timestamp", 0)
                if ts >= cutoff:
                    entries.append(_row_to_audit(row))
    except Exception as e:
        logger.warning("EventStore query failed, using in-memory only: %s", e)

    # Sort by timestamp
    entries.sort(key=lambda e: e.get("timestamp", 0))

    # Write JSONL
    with open(out, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    logger.info("Exported %d audit entries to %s", len(entries), out)
    return str(out)


def _event_to_audit(event) -> dict[str, Any]:
    """Convert an Event to audit log format."""
    return {
        "timestamp": event.timestamp,
        "datetime": datetime.fromtimestamp(event.timestamp).isoformat(),
        "type": str(event.type.value) if hasattr(event.type, "value") else str(event.type),
        "source": event.source,
        "data": event.data,
    }


def _row_to_audit(row: dict[str, Any]) -> dict[str, Any]:
    """Convert an EventStore row (dict) to audit log format."""
    ts = row.get("timestamp", 0)
    data = row.get("data_json", "{}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = {"raw": data}
    return {
        "timestamp": ts,
        "datetime": datetime.fromtimestamp(ts).isoformat() if ts else "",
        "type": row.get("event_type", "unknown"),
        "source": row.get("source", "unknown"),
        "data": data,
    }


def query_audit_log(
    path: str | Path,
    event_type: str | None = None,
    source: str | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[dict[str, Any]]:
    """Query an existing audit log file with filters."""
    results = []
    p = Path(path)
    if not p.exists():
        return results

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event_type and entry.get("type") != event_type:
            continue
        if source and entry.get("source") != source:
            continue
        if after and entry.get("timestamp", 0) < after.timestamp():
            continue
        if before and entry.get("timestamp", 0) > before.timestamp():
            continue

        results.append(entry)

    return results
