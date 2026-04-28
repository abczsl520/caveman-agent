"""Retrieval log — SQLite-backed memory search log for embedding training.

Every time Recall or memory_search runs, we log:
  - query: what the user/system searched for
  - results: which memories were returned (with scores)
  - latency_ms: how long the retrieval took (when caller can provide it)
  - adopted: which results were actually used (if trackable)

This log is the correct data source for embedding training pairs,
NOT conversation Q&A (which was the previous incorrect approach).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from caveman.db import connect as db_connect

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    results_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'recall',
    adopted_ids_json TEXT NOT NULL DEFAULT '[]',
    latency_ms REAL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_timestamp ON retrieval_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_source ON retrieval_log(source);
"""


@dataclass
class RetrievalEntry:
    """A single retrieval event."""

    query: str
    results: list[dict]  # [{"memory_id": str, "content": str, "score": float}]
    source: str = "recall"  # "recall" | "memory_search" | "nudge" | "adoption"
    adopted_ids: list[str] = field(default_factory=list)  # IDs user actually used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _LegacyRetrievalRow(NamedTuple):
    query: str
    results_json: str
    source: str
    adopted_ids_json: str
    latency_ms: float | None
    timestamp: str


class RetrievalLog:
    """Append-only SQLite log of memory retrieval events.

    The public API intentionally matches the old JSONL implementation so the
    training/eval pipeline remains stable while the storage becomes queryable by
    `caveman doctor` and future data tooling.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        if log_path is None:
            from caveman.paths import TRAINING_DIR
            log_path = TRAINING_DIR / "retrieval_log.sqlite"
        self._path = Path(log_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = db_connect(self._path)
            self._conn.executescript(_SCHEMA)
            self._migrate_legacy_jsonl_once()
        return self._conn

    def _migrate_legacy_jsonl_once(self) -> None:
        """Import same-stem legacy JSONL retrieval logs into new SQLite storage.

        The JSONL→SQLite pivot must not strand historical retrieval data. This
        best-effort migration is intentionally idempotent: it only runs when the
        SQLite table is empty, so normal writes and repeated startups do not
        duplicate legacy rows.
        """
        if self._path.suffix == ".jsonl":
            return
        legacy_path = self._path.with_suffix(".jsonl")
        if not legacy_path.exists():
            return
        conn = self._conn
        if conn is None:
            return
        try:
            existing_row = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()
            existing = int(existing_row[0]) if existing_row is not None else 0
            if existing > 0:
                return
            rows: list[_LegacyRetrievalRow] = []
            with legacy_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = RetrievalEntry.from_dict(json.loads(line))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Skip malformed legacy retrieval log row: %s", e)
                        continue
                    rows.append(_LegacyRetrievalRow(
                        entry.query,
                        json.dumps(entry.results, ensure_ascii=False),
                        entry.source,
                        json.dumps(entry.adopted_ids, ensure_ascii=False),
                        entry.latency_ms,
                        entry.timestamp,
                    ))
            if rows:
                conn.executemany(
                    "INSERT INTO retrieval_log(query, results_json, source, adopted_ids_json, latency_ms, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                logger.info("Migrated %d legacy retrieval log rows from %s", len(rows), legacy_path)
        except Exception as e:
            logger.warning("Failed to migrate legacy retrieval log %s: %s", legacy_path, e)

    @property
    def path(self) -> Path:
        return self._path

    def log(self, entry: RetrievalEntry) -> None:
        """Append a retrieval event to the SQLite log."""
        try:
            self._get_conn().execute(
                "INSERT INTO retrieval_log(query, results_json, source, adopted_ids_json, latency_ms, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.query,
                    json.dumps(entry.results, ensure_ascii=False),
                    entry.source,
                    json.dumps(entry.adopted_ids, ensure_ascii=False),
                    entry.latency_ms,
                    entry.timestamp,
                ),
            )
            self._get_conn().commit()
        except Exception as e:
            logger.warning("Failed to write retrieval log: %s", e)

    def log_search(
        self,
        query: str,
        results: list[tuple[float, Any]],
        source: str = "recall",
        latency_ms: float | None = None,
    ) -> None:
        """Convenience: log from (score, MemoryEntry) tuples."""
        result_dicts = []
        for score, entry in results:
            result_dicts.append({
                "memory_id": getattr(entry, "id", ""),
                "content": getattr(entry, "content", str(entry))[:500],
                "score": round(score, 4),
            })
        self.log(RetrievalEntry(
            query=query, results=result_dicts, source=source, latency_ms=latency_ms,
        ))

    def mark_adopted(self, query: str, adopted_ids: list[str]) -> None:
        """Mark which results from a query were actually used.

        This creates a follow-up entry that can be joined with the original
        during training pair generation.
        """
        self.log(RetrievalEntry(
            query=query,
            results=[],
            source="adoption",
            adopted_ids=adopted_ids,
        ))

    def read_all(self) -> list[RetrievalEntry]:
        """Read all entries from the SQLite log."""
        if not self._path.exists():
            return []
        try:
            rows = self._get_conn().execute(
                "SELECT query, results_json, source, adopted_ids_json, timestamp, latency_ms "
                "FROM retrieval_log ORDER BY id"
            ).fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("Failed to read retrieval log: %s", e)
            return []

        entries: list[RetrievalEntry] = []
        for row in rows:
            try:
                entries.append(RetrievalEntry(
                    query=row[0],
                    results=json.loads(row[1]) if row[1] else [],
                    source=row[2] or "recall",
                    adopted_ids=json.loads(row[3]) if row[3] else [],
                    timestamp=row[4],
                    latency_ms=row[5],
                ))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skip malformed retrieval log row: %s", e)
        return entries

    def count(self) -> int:
        """Count entries without loading all into memory."""
        if not self._path.exists():
            return 0
        return int(self._get_conn().execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        """Return doctor-friendly retrieval log metrics."""
        if not self._path.exists():
            return {"count": 0, "avg_latency_ms": None, "by_source": {}}
        conn = self._get_conn()
        count = int(conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0])
        avg = conn.execute(
            "SELECT AVG(latency_ms) FROM retrieval_log WHERE latency_ms IS NOT NULL"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT source, COUNT(*) FROM retrieval_log GROUP BY source ORDER BY source"
        ).fetchall()
        return {
            "count": count,
            "avg_latency_ms": float(avg) if avg is not None else None,
            "by_source": {row[0]: row[1] for row in rows},
            "path": str(self._path),
        }

    def generate_training_pairs(self) -> list[dict]:
        """Generate query-positive pairs from retrieval log for embedding training.

        Logic:
        - For each search entry with results scored > 0.5: query → top result = positive pair
        - For entries with adopted_ids: adopted = positive, non-adopted = hard negative
        """
        entries = self.read_all()
        pairs = []

        # Index adoption events by query
        adoptions: dict[str, list[str]] = {}
        for e in entries:
            if e.source == "adoption" and e.adopted_ids:
                adoptions[e.query] = e.adopted_ids

        for entry in entries:
            if entry.source == "adoption" or not entry.results:
                continue

            adopted_ids = adoptions.get(entry.query, [])

            for result in entry.results:
                score = result.get("score", 0)
                content = result.get("content", "")
                mid = result.get("memory_id", "")

                if not content or len(content) < 10:
                    continue

                # If we have adoption data, use it
                if adopted_ids:
                    if mid in adopted_ids:
                        pairs.append({
                            "query": entry.query,
                            "positive": content,
                            "source": "adopted",
                        })
                elif score >= 0.5:
                    # No adoption data — use score as proxy
                    pairs.append({
                        "query": entry.query,
                        "positive": content,
                        "source": "score",
                    })

        return pairs

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
