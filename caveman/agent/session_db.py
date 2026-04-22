"""SQLite session store — replaces JSON file storage.

Drop-in replacement for SessionStore with:
- Atomic writes (no data loss on concurrent access)
- Queryable history (SQL instead of full-file reads)
- System prompt persistence (for cache stability)
- Usage analytics (cost, tokens, tool calls per session)

Schema designed for Hermes-level capability with Caveman's simplicity.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from caveman.agent.session_store import SessionMeta
from caveman.timeouts import SQLITE_BUSY, SQLITE_CONNECT

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    model TEXT DEFAULT '',
    started_at REAL DEFAULT 0,
    last_active_at REAL DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    compaction_count INTEGER DEFAULT 0,
    title TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    surface TEXT DEFAULT 'cli',
    system_prompt_hash TEXT DEFAULT '',
    system_prompt TEXT DEFAULT '',
    loop_snapshot TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    extra TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS compactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    turns_compressed INTEGER DEFAULT 0,
    ts REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_session ON transcript(session_id, id);
CREATE INDEX IF NOT EXISTS idx_transcript_ts ON transcript(ts);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active_at DESC);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    content, session_id UNINDEXED, role UNINDEXED,
    content='transcript', content_rowid='id',
    tokenize='unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS transcript_ai AFTER INSERT ON transcript BEGIN
    INSERT INTO transcript_fts(rowid, content, session_id, role)
    VALUES (new.id, new.content, new.session_id, new.role);
END;
CREATE TRIGGER IF NOT EXISTS transcript_ad AFTER DELETE ON transcript BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, content, session_id, role)
    VALUES ('delete', old.id, old.content, old.session_id, old.role);
END;
"""


class SessionDB:
    """SQLite-backed session store. Same interface as SessionStore."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=SQLITE_CONNECT,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY)}")
        self._conn.executescript(_SCHEMA)
        try:
            self._conn.executescript(_FTS_SCHEMA)
            # Rebuild FTS index if empty but transcript has data
            fts_count = self._conn.execute(
                "SELECT COUNT(*) FROM transcript_fts"
            ).fetchone()[0]
            if fts_count == 0:
                tx_count = self._conn.execute(
                    "SELECT COUNT(*) FROM transcript"
                ).fetchone()[0]
                if tx_count > 0:
                    self._conn.execute(
                        "INSERT INTO transcript_fts(transcript_fts) VALUES('rebuild')"
                    )
                    self._conn.commit()
                    logger.info("Rebuilt FTS index for %d transcript entries", tx_count)
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 setup skipped: %s", e)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SessionDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # --- Metadata ---

    def save_meta(self, meta: SessionMeta) -> None:
        d = meta.to_dict()
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, model, started_at, last_active_at, turn_count,
                total_tokens, total_cost_usd, compaction_count, title, tags,
                surface, system_prompt_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (d["session_id"], d["model"], d["started_at"], d["last_active_at"],
             d["turn_count"], d["total_tokens"], d["total_cost_usd"],
             d["compaction_count"], d["title"], json.dumps(d["tags"]),
             d["surface"], d.get("system_prompt_hash", "")),
        )
        self._conn.commit()

    def load_meta(self, session_id: str) -> SessionMeta | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_meta(row)

    def update_meta(self, session_id: str, **fields: Any) -> None:
        """Update specific metadata fields without full rewrite.

        Safety: field names in set_clause come from the ``allowed`` whitelist
        below, NOT from user input, so the f-string SQL is safe from injection.
        """
        allowed = {"last_active_at", "turn_count", "total_tokens",
                    "total_cost_usd", "compaction_count", "title", "system_prompt_hash"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        _ALLOWED_COLUMNS = {
            "model", "started_at", "last_active_at", "turn_count",
            "total_tokens", "total_cost_usd", "compaction_count",
            "title", "tags", "surface", "system_prompt_hash",
            "system_prompt", "loop_snapshot",
        }
        bad_keys = set(updates) - _ALLOWED_COLUMNS
        if bad_keys:
            raise ValueError(f"Invalid column names: {bad_keys}")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
            (*updates.values(), session_id),
        )
        self._conn.commit()

    def list_sessions(self, limit: int = 50) -> list[SessionMeta]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY last_active_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_meta(r) for r in rows]

    # --- Transcript ---

    def append_turn(self, session_id: str, role: str, content: str,
                     **extra: Any) -> None:
        ts = time.time()
        self._conn.execute(
            "INSERT INTO transcript (session_id, role, content, ts, extra) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, ts, json.dumps(extra) if extra else "{}"),
        )
        self._conn.execute(
            "UPDATE sessions SET last_active_at = ?, turn_count = turn_count + 1 WHERE session_id = ?",
            (ts, session_id),
        )
        self._conn.commit()

    def load_transcript(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content, ts, extra FROM transcript WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        result = []
        for r in reversed(rows):  # Reverse to get chronological order
            entry = {"role": r["role"], "content": r["content"], "ts": r["ts"]}
            extra = json.loads(r["extra"]) if r["extra"] != "{}" else {}
            entry.update(extra)
            result.append(entry)
        return result

    def transcript_turn_count(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM transcript WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def search_transcripts(
        self, query: str, limit: int = 20, session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search across all transcripts. Returns matching turns."""
        try:
            if session_id:
                rows = self._conn.execute(
                    """SELECT t.session_id, t.role, t.content, t.ts,
                              snippet(transcript_fts, 0, '>>>', '<<<', '...', 32) as snippet
                       FROM transcript_fts f
                       JOIN transcript t ON t.id = f.rowid
                       WHERE transcript_fts MATCH ? AND f.session_id = ?
                       ORDER BY rank LIMIT ?""",
                    (query, session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT t.session_id, t.role, t.content, t.ts,
                              snippet(transcript_fts, 0, '>>>', '<<<', '...', 32) as snippet
                       FROM transcript_fts f
                       JOIN transcript t ON t.id = f.rowid
                       WHERE transcript_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS table might not exist yet
            return []

    # --- Compaction ---

    def save_compaction(self, session_id: str, summary: str,
                         turns_compressed: int = 0) -> None:
        self._conn.execute(
            "INSERT INTO compactions (session_id, summary, turns_compressed, ts) VALUES (?, ?, ?, ?)",
            (session_id, summary, turns_compressed, time.time()),
        )
        self._conn.execute(
            "UPDATE sessions SET compaction_count = compaction_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def load_compactions(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT summary, turns_compressed, ts FROM compactions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- System Prompt Persistence ---

    def save_system_prompt(self, session_id: str, prompt: str, prompt_hash: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET system_prompt = ?, system_prompt_hash = ? WHERE session_id = ?",
            (prompt, prompt_hash, session_id),
        )
        self._conn.commit()

    def load_system_prompt(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["system_prompt"] if row and row["system_prompt"] else None

    # --- Loop Snapshot ---

    def save_snapshot(self, session_id: str, snapshot: dict) -> None:
        """Save loop snapshot (turn state, system prompt, etc.)."""
        self._conn.execute(
            "UPDATE sessions SET loop_snapshot = ? WHERE session_id = ?",
            (json.dumps(snapshot), session_id),
        )
        self._conn.commit()

    def load_snapshot(self, session_id: str) -> dict:
        """Load loop snapshot."""
        row = self._conn.execute(
            "SELECT loop_snapshot FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row and row["loop_snapshot"]:
            try:
                return json.loads(row["loop_snapshot"])
            except json.JSONDecodeError:
                pass  # intentional: Exception suppressed
        return {}

    # --- Analytics (new capability unlocked by SQL) ---

    def usage_summary(self) -> dict[str, Any]:
        """Aggregate usage stats across all sessions."""
        row = self._conn.execute("""
            SELECT COUNT(*) as total_sessions,
                   SUM(turn_count) as total_turns,
                   SUM(total_tokens) as total_tokens,
                   SUM(total_cost_usd) as total_cost,
                   AVG(turn_count) as avg_turns_per_session
            FROM sessions
        """).fetchone()
        return dict(row) if row else {}

    def recent_sessions(self, hours: int = 24) -> list[dict[str, Any]]:
        """Sessions active in the last N hours."""
        cutoff = time.time() - hours * 3600
        rows = self._conn.execute(
            "SELECT session_id, surface, turn_count, total_tokens, total_cost_usd, last_active_at "
            "FROM sessions WHERE last_active_at > ? ORDER BY last_active_at DESC",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Cleanup ---

    def delete_session(self, session_id: str) -> bool:
        self._conn.execute("DELETE FROM transcript WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM compactions WHERE session_id = ?", (session_id,))
        cursor = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def export_session(self, session_id: str) -> dict[str, Any] | None:
        """Export a complete session as a JSON-serializable dict."""
        meta = self.load_meta(session_id)
        if not meta:
            return None
        transcript = self.load_transcript(session_id, limit=10000)
        compactions = self.load_compactions(session_id)
        prompt = self.load_system_prompt(session_id)
        snapshot = self.load_snapshot(session_id)
        return {
            "session_id": session_id,
            "meta": {k: v for k, v in meta.__dict__.items() if v is not None},
            "transcript": transcript,
            "compactions": compactions,
            "system_prompt": prompt,
            "snapshot": snapshot,
            "message_count": len(transcript),
        }

    # --- Migration ---
    def migrate_from_json(self, json_store_dir: Path) -> int:
        """Import sessions from JSON file store. Returns count of migrated sessions."""
        if not json_store_dir.exists():
            return 0
        migrated = 0
        for d in json_store_dir.iterdir():
            if not d.is_dir() or not (d / "meta.json").exists():
                continue
            try:
                meta_data = json.loads((d / "meta.json").read_text())
                meta = SessionMeta.from_dict(meta_data)
                # Skip if already in DB
                if self.load_meta(meta.session_id):
                    continue
                self.save_meta(meta)
                # Migrate transcript
                transcript_path = d / "transcript.jsonl"
                if transcript_path.exists():
                    for line in transcript_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            role = entry.pop("role", "user")
                            content = entry.pop("content", "")
                            ts = entry.pop("ts", time.time())
                            self._conn.execute(
                                "INSERT INTO transcript (session_id, role, content, ts, extra) VALUES (?, ?, ?, ?, ?)",
                                (meta.session_id, role, content, ts, json.dumps(entry) if entry else "{}"),
                            )
                        except json.JSONDecodeError:
                            continue
                # Migrate compactions
                compactions_path = d / "compactions.jsonl"
                if compactions_path.exists():
                    for line in compactions_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            self._conn.execute(
                                "INSERT INTO compactions (session_id, summary, turns_compressed, ts) VALUES (?, ?, ?, ?)",
                                (meta.session_id, entry.get("summary", ""),
                                 entry.get("turns_compressed", 0), entry.get("ts", time.time())),
                            )
                        except json.JSONDecodeError:
                            continue
                self._conn.commit()
                migrated += 1
                logger.info("Migrated session %s (%d turns)", meta.session_id,
                            self.transcript_turn_count(meta.session_id))
            except Exception as e:
                logger.warning("Failed to migrate session %s: %s", d.name, e)
        return migrated
    # --- Internal ---
    @staticmethod
    def _row_to_meta(row: sqlite3.Row) -> SessionMeta:
        tags = json.loads(row["tags"]) if row["tags"] else []
        return SessionMeta(
            session_id=row["session_id"],
            model=row["model"],
            started_at=row["started_at"],
            last_active_at=row["last_active_at"],
            turn_count=row["turn_count"],
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            compaction_count=row["compaction_count"],
            title=row["title"] or "",
            tags=tags,
            surface=row["surface"],
            system_prompt_hash=row["system_prompt_hash"] or "",
        )
