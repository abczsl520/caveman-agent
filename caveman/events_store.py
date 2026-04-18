"""EventStore — SQLite-backed event persistence for replay and audit.

Every event emitted through the EventBus can be persisted here.
Attach via ``bus.on_all(store.handle)`` for full capture.

Replay: re-emit stored events through an EventBus for debugging/testing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from caveman.events import Event, EventBus
from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_DB = CAVEMAN_HOME / "events.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT    NOT NULL,
    data_json  TEXT    NOT NULL DEFAULT '{}',
    timestamp  REAL    NOT NULL,
    source     TEXT    NOT NULL DEFAULT '',
    session_id TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, timestamp);
"""


class EventStore:
    """SQLite-backed event persistence for replay and audit."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_EVENTS_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._session_id = ""

    def set_session(self, session_id: str) -> None:
        """Tag subsequent events with a session ID."""
        self._session_id = session_id

    def handle(self, event: Event) -> None:
        """Sync handler — called by EventBus for every event."""
        try:
            self._conn.execute(
                "INSERT INTO events (event_type, data_json, timestamp, source, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.type if isinstance(event.type, str) else event.type.value,
                    json.dumps(event.data, ensure_ascii=False, default=str),
                    event.timestamp,
                    event.source,
                    self._session_id,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.warning("EventStore write failed", exc_info=True)

    def query(
        self,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        session_id: str | None = None,
        source: str | None = None,
        limit: int = 100,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """Query stored events with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM events {where} ORDER BY id {order} LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self, event_type: str | None = None) -> int:
        if event_type is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = ?", (event_type,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0]

    def distinct_types(self) -> list[str]:
        """List all distinct event types stored."""
        rows = self._conn.execute(
            "SELECT DISTINCT event_type FROM events ORDER BY event_type"
        ).fetchall()
        return [r[0] for r in rows]

    def sessions(self) -> list[dict]:
        """List all sessions with event counts and time ranges."""
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*) as count, "
            "MIN(timestamp) as first_ts, MAX(timestamp) as last_ts "
            "FROM events WHERE session_id != '' "
            "GROUP BY session_id ORDER BY last_ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def purge(self, before: float | None = None, event_type: str | None = None) -> int:
        """Delete events matching criteria. Returns count deleted."""
        clauses: list[str] = []
        params: list[Any] = []
        if before is not None:
            clauses.append("timestamp < ?")
            params.append(before)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if not clauses:
            return 0  # Safety: refuse to purge everything without criteria
        where = " AND ".join(clauses)
        cursor = self._conn.execute(f"DELETE FROM events WHERE {where}", params)
        self._conn.commit()
        return cursor.rowcount

    # ── Replay ──

    async def replay(
        self,
        bus: EventBus,
        session_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        speed: float = 1.0,
        limit: int = 10000,
    ) -> int:
        """Replay stored events through an EventBus.

        Args:
            bus: Target EventBus to emit events into.
            session_id: Filter by session.
            event_type: Filter by event type.
            since: Start timestamp.
            until: End timestamp.
            speed: Replay speed multiplier (1.0 = real-time, 0 = instant).
            limit: Max events to replay.

        Returns:
            Number of events replayed.
        """
        events = self.query(
            event_type=event_type,
            since=since,
            until=until,
            session_id=session_id,
            limit=limit,
            ascending=True,
        )

        if not events:
            return 0

        count = 0
        prev_ts = events[0]["timestamp"]

        for row in events:
            # Timing: wait proportional to original time gaps
            if speed > 0 and count > 0:
                gap = (row["timestamp"] - prev_ts) / speed
                if gap > 0:
                    await asyncio.sleep(min(gap, 5.0))  # Cap at 5s per gap
            prev_ts = row["timestamp"]

            # Reconstruct and emit event
            try:
                data = json.loads(row["data_json"]) if isinstance(row["data_json"], str) else row["data_json"]
            except json.JSONDecodeError:
                data = {}

            await bus.emit(
                row["event_type"],
                data,
                source=f"replay:{row.get('source', '')}",
            )
            count += 1

        return count

    async def replay_iter(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        limit: int = 10000,
    ) -> AsyncIterator[dict]:
        """Iterate over stored events without emitting (for analysis)."""
        events = self.query(
            event_type=event_type,
            since=since,
            session_id=session_id,
            limit=limit,
            ascending=True,
        )
        for row in events:
            try:
                row["data"] = json.loads(row["data_json"]) if isinstance(row["data_json"], str) else row["data_json"]
            except json.JSONDecodeError:
                row["data"] = {}
            yield row

    # ── Summary ──

    def summary(self, session_id: str | None = None) -> dict:
        """Get a summary of stored events."""
        if session_id:
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) as count FROM events "
                "WHERE session_id = ? GROUP BY event_type ORDER BY count DESC",
                (session_id,),
            ).fetchall()
            total = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        else:
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) as count FROM events "
                "GROUP BY event_type ORDER BY count DESC"
            ).fetchall()
            total = self.count()

        return {
            "total": total,
            "by_type": {r[0]: r[1] for r in rows},
            "session_id": session_id,
        }

    def close(self) -> None:
        self._conn.close()
