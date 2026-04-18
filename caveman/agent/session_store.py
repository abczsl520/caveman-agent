"""Session persistence — transcript storage and restoration.

Learned from OpenClaw's session management:
- Full transcript stored on disk (JSON lines)
- Compaction summaries preserved alongside raw history
- Session metadata (model, start time, turn count, cost)
- Resume from any point in history
"""
from __future__ import annotations

__all__ = ["SessionStore", "SessionMeta"]

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionMeta:
    """Metadata for a session."""

    session_id: str
    model: str = ""
    started_at: float = 0.0
    last_active_at: float = 0.0
    turn_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    compaction_count: int = 0
    title: str = ""
    tags: list[str] = field(default_factory=list)
    surface: str = "cli"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "started_at": self.started_at,
            "last_active_at": self.last_active_at,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "compaction_count": self.compaction_count,
            "title": self.title,
            "tags": self.tags,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        return cls(
            session_id=data.get("session_id", ""),
            model=data.get("model", ""),
            started_at=data.get("started_at", 0.0),
            last_active_at=data.get("last_active_at", 0.0),
            turn_count=data.get("turn_count", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            compaction_count=data.get("compaction_count", 0),
            title=data.get("title", ""),
            tags=data.get("tags", []),
            surface=data.get("surface", "cli"),
        )


class SessionStore:
    """Persistent session storage using JSON lines.

    Directory layout:
        sessions/
        ├── {session_id}/
        │   ├── meta.json        — session metadata
        │   ├── transcript.jsonl — full message history
        │   └── compactions.jsonl — compaction summaries
    """

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        d = (self._base / session_id).resolve()
        if not str(d).startswith(str(self._base.resolve())):
            raise ValueError(f"Invalid session_id: path traversal detected: {session_id!r}")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- Metadata ---

    def save_meta(self, meta: SessionMeta) -> Path:
        """Save session metadata."""
        path = self._session_dir(meta.session_id) / "meta.json"
        path.write_text(json.dumps(meta.to_dict(), indent=2))
        return path

    def load_meta(self, session_id: str) -> SessionMeta | None:
        """Load session metadata."""
        path = self._session_dir(session_id) / "meta.json"
        if not path.exists():
            return None
        try:
            return SessionMeta.from_dict(json.loads(path.read_text()))
        except Exception as e:
            logger.warning("Failed to load meta for %s: %s", session_id, e)
            return None

    def list_sessions(self) -> list[SessionMeta]:
        """List all sessions, sorted by last active (newest first)."""
        sessions = []
        for d in self._base.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                meta = self.load_meta(d.name)
                if meta:
                    sessions.append(meta)
        sessions.sort(key=lambda m: m.last_active_at, reverse=True)
        return sessions

    # --- Transcript ---

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        **extra: Any,
    ) -> None:
        """Append a turn to the transcript."""
        path = self._session_dir(session_id) / "transcript.jsonl"
        entry = {"role": role, "content": content, "ts": time.time(), **extra}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Load full transcript."""
        path = self._session_dir(session_id) / "transcript.jsonl"
        if not path.exists():
            return []
        turns = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return turns

    def transcript_turn_count(self, session_id: str) -> int:
        """Count turns without loading full transcript."""
        path = self._session_dir(session_id) / "transcript.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    # --- Compaction ---

    def save_compaction(
        self,
        session_id: str,
        summary: str,
        turns_compressed: int = 0,
    ) -> None:
        """Save a compaction summary."""
        path = self._session_dir(session_id) / "compactions.jsonl"
        entry = {
            "summary": summary,
            "turns_compressed": turns_compressed,
            "ts": time.time(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_compactions(self, session_id: str) -> list[dict[str, Any]]:
        """Load all compaction summaries."""
        path = self._session_dir(session_id) / "compactions.jsonl"
        if not path.exists():
            return []
        compactions = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    compactions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return compactions

    # --- Cleanup ---

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        import shutil
        d = (self._base / session_id).resolve()
        if not str(d).startswith(str(self._base.resolve())):
            raise ValueError(f"Invalid session_id: path traversal detected: {session_id!r}")
        if d.exists() and (d / "meta.json").exists():
            shutil.rmtree(d)
            return True
        if d.exists():
            shutil.rmtree(d)
        return False
