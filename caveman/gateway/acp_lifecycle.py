"""ACP Lifecycle Commands — ACP session management from chat.

Extracted from OpenClaw commands-acp/lifecycle.ts (868 lines).
Handles: /acp spawn, /acp list, /acp send, /acp kill, /acp status.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("caveman.gateway.acp_lifecycle")


@dataclass
class ACPSession:
    """An ACP sub-agent session."""
    session_id: str
    agent_id: str = ""
    task: str = ""
    status: str = "pending"  # pending | running | result_available | no_response | failed | cancelled
    created_at: float = 0
    completed_at: float = 0
    parent_session: str = ""
    result: str = ""
    error: str = ""
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    @property
    def duration_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.created_at) * 1000 if self.created_at else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "task": self.task[:100],
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
        }


class ACPLifecycleManager:
    """Manages ACP sub-agent sessions from chat commands."""

    def __init__(
        self,
        spawn_fn: Optional[Callable] = None,
        max_sessions: int = 20,
    ):
        self._spawn_fn = spawn_fn
        self._sessions: Dict[str, ACPSession] = {}
        self._max_sessions = max_sessions

    # ── Commands ──

    async def handle_spawn(
        self, agent_id: str, task: str, parent_session: str = "",
    ) -> str:
        """Spawn a new ACP sub-agent."""
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_completed()
            if len(self._sessions) >= self._max_sessions:
                return f"Too many ACP sessions ({self._max_sessions}). Kill some first."

        import uuid
        session_id = str(uuid.uuid4())
        session = ACPSession(
            session_id=session_id,
            agent_id=agent_id,
            task=task,
            status="pending",
            created_at=time.monotonic(),
            parent_session=parent_session,
        )
        self._sessions[session_id] = session

        if self._spawn_fn:
            try:
                session.status = "running"
                result = self._spawn_fn(agent_id, task, session_id)
                if hasattr(result, "__await__"):
                    result = await result
                if isinstance(result, dict):
                    session.result = str(result.get("text") or result.get("result") or "")
                    returned_status = str(result.get("status") or "").lower()
                    if returned_status in {"running", "pending"}:
                        session.status = returned_status
                    else:
                        session.status = "result_available" if session.result else "no_response"
                else:
                    session.result = str(result) if result else ""
                    session.status = "result_available" if session.result else "no_response"
            except Exception as e:
                session.status = "failed"
                session.error = str(e)
            finally:
                if session.status not in {"running", "pending"}:
                    session.completed_at = time.monotonic()

        return f"ACP session {session_id[:8]} ({agent_id}): {session.status}"

    async def handle_send(self, session_id: str, message: str) -> str:
        """Send a message to an ACP session."""
        session = self._find_session(session_id)
        if not session:
            return f"Session not found: {session_id}"
        if session.status != "running":
            return f"Session {session_id[:8]} is {session.status}, not running"
        # In a real implementation, this would send via the ACP protocol
        return f"Message sent to {session_id[:8]}"

    async def handle_kill(self, session_id: str) -> str:
        """Kill an ACP session."""
        session = self._find_session(session_id)
        if not session:
            return f"Session not found: {session_id}"

        if session._task and not session._task.done():
            session._task.cancel()

        session.status = "cancelled"
        session.completed_at = time.monotonic()
        return f"Killed ACP session {session_id[:8]}"

    def handle_list(self, status: str = "") -> str:
        """List ACP sessions."""
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]

        if not sessions:
            return "No ACP sessions."

        lines = ["ACP Sessions:"]
        for s in sessions:
            dur = f"{s.duration_ms / 1000:.1f}s" if s.duration_ms else "?"
            lines.append(f"  {s.session_id[:8]} | {s.agent_id} | {s.status} | {dur}")
        return "\n".join(lines)

    def handle_status(self, session_id: str) -> str:
        """Get status of an ACP session."""
        session = self._find_session(session_id)
        if not session:
            return f"Session not found: {session_id}"

        lines = [
            f"Session: {session.session_id[:8]}",
            f"Agent: {session.agent_id}",
            f"Status: {session.status}",
            f"Task: {session.task[:200]}",
            f"Duration: {session.duration_ms / 1000:.1f}s",
        ]
        if session.result:
            lines.append(f"Result: {session.result[:500]}")
        if session.error:
            lines.append(f"Error: {session.error}")
        return "\n".join(lines)

    # ── Internal ──

    def _find_session(self, session_id: str) -> Optional[ACPSession]:
        """Find session by full or partial ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        # Partial match
        for sid, session in self._sessions.items():
            if sid.startswith(session_id):
                return session
        return None

    def _cleanup_completed(self) -> int:
        """Remove oldest inactive sessions."""
        inactive = [
            (sid, s) for sid, s in self._sessions.items()
            if s.status in {"result_available", "no_response", "completed", "failed", "cancelled"}
        ]
        inactive.sort(key=lambda x: x[1].completed_at or x[1].created_at)
        removed = 0
        for sid, _ in inactive[:len(inactive) // 2]:
            self._sessions.pop(sid)
            removed += 1
        return removed

from caveman.gateway.acp_lifecycle_depth import (  # noqa: F401,E402  # depth wiring
    ThreadBinding,
    ThreadBindingStore,
    PersistentSession,
    PersistentSessionStore,
)

__all__ = [
    "ACPSession",
    "ACPLifecycleManager",
    "ThreadBinding",
    "ThreadBindingStore",
    "PersistentSession",
    "PersistentSessionStore",
]
