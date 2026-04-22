"""ACP Lifecycle Depth — thread binding, persistent sessions, cleanup.

Supplements acp_lifecycle.py with thread-bound sessions, persistence,
and advanced cleanup. Extracted from OpenClaw ACP + sessions.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ThreadBinding",
    "ThreadBindingStore",
    "PersistentSession",
    "PersistentSessionStore",
]


logger = logging.getLogger("caveman.gateway.acp_lifecycle_depth")


@dataclass
class ThreadBinding:
    """Binds an ACP session to a chat thread."""
    session_id: str
    thread_id: str
    channel: str = ""
    created_at: float = 0
    last_activity: float = 0
    message_count: int = 0
    agent_id: str = ""
    mode: str = "session"  # session | oneshot

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity if self.last_activity else 0

    def touch(self) -> None:
        self.last_activity = time.time()
        self.message_count += 1


class ThreadBindingStore:
    """Manages thread-to-session bindings."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._bindings: Dict[str, ThreadBinding] = {}
        self._persist_path = persist_path

    def bind(self, thread_id: str, session_id: str, **kwargs) -> ThreadBinding:
        binding = ThreadBinding(
            session_id=session_id,
            thread_id=thread_id,
            created_at=time.time(),
            last_activity=time.time(),
            **kwargs,
        )
        self._bindings[thread_id] = binding
        self._save()
        return binding

    def get(self, thread_id: str) -> Optional[ThreadBinding]:
        binding = self._bindings.get(thread_id)
        if binding:
            binding.touch()
            self._save()
        return binding

    def unbind(self, thread_id: str) -> bool:
        if thread_id in self._bindings:
            del self._bindings[thread_id]
            self._save()
            return True
        return False

    def list_bindings(self) -> List[ThreadBinding]:
        return list(self._bindings.values())

    def cleanup_idle(self, max_idle_seconds: float = 3600) -> int:
        """Remove bindings idle longer than threshold."""
        to_remove = [
            tid for tid, b in self._bindings.items()
            if b.idle_seconds > max_idle_seconds
        ]
        for tid in to_remove:
            del self._bindings[tid]
        if to_remove:
            self._save()
        return len(to_remove)

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                tid: {
                    "session_id": b.session_id,
                    "thread_id": b.thread_id,
                    "channel": b.channel,
                    "created_at": b.created_at,
                    "last_activity": b.last_activity,
                    "message_count": b.message_count,
                    "agent_id": b.agent_id,
                    "mode": b.mode,
                }
                for tid, b in self._bindings.items()
            }
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8",
            )
        except Exception as e:
            logger.debug("Failed to save bindings: %s", e)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for tid, d in data.items():
                self._bindings[tid] = ThreadBinding(**d)
        except Exception as e:
            logger.debug("Failed to load bindings: %s", e)


# ── Persistent Sessions ──

@dataclass
class PersistentSession:
    """A persistent ACP session that survives restarts."""
    session_id: str
    agent_id: str
    created_at: float = 0
    last_activity: float = 0
    state: str = "active"  # active | paused | completed | failed
    context: Dict[str, Any] = field(default_factory=dict)
    history_path: str = ""

    @property
    def is_active(self) -> bool:
        return self.state == "active"


class PersistentSessionStore:
    """Manages persistent ACP sessions."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir or Path.home() / ".caveman" / "acp_sessions"
        self._sessions: Dict[str, PersistentSession] = {}

    def create(self, session_id: str, agent_id: str, **kwargs) -> PersistentSession:
        session = PersistentSession(
            session_id=session_id,
            agent_id=agent_id,
            created_at=time.time(),
            last_activity=time.time(),
            **kwargs,
        )
        self._sessions[session_id] = session
        self._save_session(session)
        return session

    def get(self, session_id: str) -> Optional[PersistentSession]:
        return self._sessions.get(session_id)

    def update_state(self, session_id: str, state: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.state = state
        session.last_activity = time.time()
        self._save_session(session)
        return True

    def list_active(self) -> List[PersistentSession]:
        return [s for s in self._sessions.values() if s.is_active]

    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            sid for sid, s in self._sessions.items()
            if not s.is_active and s.last_activity < cutoff
        ]
        for sid in to_remove:
            del self._sessions[sid]
            # Remove persisted file
            path = self._base_dir / f"{sid}.json"
            path.unlink(missing_ok=True)
        return len(to_remove)

    def _save_session(self, session: PersistentSession) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / f"{session.session_id}.json"
        try:
            path.write_text(json.dumps({
                "session_id": session.session_id,
                "agent_id": session.agent_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "state": session.state,
                "context": session.context,
                "history_path": session.history_path,
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to save session %s: %s", session.session_id, e)

    def load_all(self) -> int:
        """Load all persisted sessions from disk."""
        if not self._base_dir.exists():
            return 0
        count = 0
        for path in self._base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._sessions[data["session_id"]] = PersistentSession(**data)
                count += 1
            except Exception as exc:
                logger.debug("load_all: suppressed %s", exc)
        return count
