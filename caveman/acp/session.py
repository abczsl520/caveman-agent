"""ACP Session Manager — persistent, thread-safe session lifecycle.

Maps ACP sessions to Caveman AgentLoop instances. Sessions survive
process restarts via SQLite persistence.

Learned from: Hermes acp_adapter/session.py (475 lines)
Our version: Async-native, integrated with Caveman's SessionDB.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("caveman.acp")


@dataclass
class ACPSessionState:
    """Per-session state for an ACP-managed agent."""

    session_id: str
    cwd: str = "."
    model: str = ""
    mode: str = "agent"  # agent | edit | chat
    history: List[Dict[str, Any]] = field(default_factory=list)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _agent: Any = field(default=None, repr=False)  # AgentLoop instance
    created_at: str = ""
    last_active: str = ""

    @property
    def agent(self) -> Any:
        return self._agent

    @agent.setter
    def agent(self, value: Any) -> None:
        self._agent = value


class ACPSessionManager:
    """Async session manager for ACP protocol.

    Features:
    - Create/get/remove/fork/list sessions
    - SQLite persistence (via Caveman's session_db)
    - Session restore on process restart
    - CWD binding per session
    """

    def __init__(
        self,
        agent_factory: Optional[Callable] = None,
        db: Optional[Any] = None,
        max_sessions: int = 100,
    ):
        self._sessions: Dict[str, ACPSessionState] = {}
        self._agent_factory = agent_factory
        self._db = db
        self._max_sessions = max_sessions
        self._lock = asyncio.Lock()

    # ── Public API ──

    async def create_session(
        self, cwd: str = ".", model: str = "", mode: str = "agent",
    ) -> ACPSessionState:
        """Create a new session with a fresh agent."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        session_id = str(uuid.uuid4())
        agent = await self._make_agent(session_id, cwd, model)

        state = ACPSessionState(
            session_id=session_id,
            cwd=cwd,
            model=model or getattr(agent, "model", ""),
            mode=mode,
            cancel_event=asyncio.Event(),
            _agent=agent,
            created_at=now,
            last_active=now,
        )

        async with self._lock:
            await self._evict_if_full()
            self._sessions[session_id] = state

        await self._persist(state)
        logger.info("Created ACP session %s (cwd=%s, model=%s)", session_id, cwd, model)
        return state

    async def get_session(self, session_id: str) -> Optional[ACPSessionState]:
        """Get session by ID. Restores from DB if not in memory."""
        async with self._lock:
            state = self._sessions.get(session_id)
        if state is not None:
            return state
        return await self._restore(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove session from memory and DB."""
        async with self._lock:
            state = self._sessions.pop(session_id, None)
        if state and state.agent:
            try:
                if hasattr(state.agent, "shutdown"):
                    await state.agent.shutdown()
            except Exception:
                logger.debug("Error shutting down agent for session %s", session_id, exc_info=True)
        await self._delete_persisted(session_id)
        return state is not None

    async def fork_session(
        self, session_id: str, cwd: str = ".",
    ) -> Optional[ACPSessionState]:
        """Deep-copy a session's history into a new session."""
        original = await self.get_session(session_id)
        if original is None:
            return None

        new_state = await self.create_session(
            cwd=cwd, model=original.model, mode=original.mode,
        )
        new_state.history = copy.deepcopy(original.history)
        await self._persist(new_state)
        logger.info("Forked ACP session %s → %s", session_id, new_state.session_id)
        return new_state

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions (memory + DB)."""
        async with self._lock:
            seen = set(self._sessions.keys())
            results = [
                {
                    "session_id": s.session_id,
                    "cwd": s.cwd,
                    "model": s.model,
                    "mode": s.mode,
                    "history_len": len(s.history),
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                }
                for s in self._sessions.values()
            ]

        # Merge DB sessions not in memory
        db = self._get_db()
        if db:
            try:
                rows = db.search_sessions(source="acp", limit=self._max_sessions)
                for row in rows:
                    sid = row.get("id", "")
                    if sid in seen:
                        continue
                    results.append({
                        "session_id": sid,
                        "cwd": ".",
                        "model": row.get("model", ""),
                        "mode": "agent",
                        "history_len": 0,
                        "created_at": row.get("created_at", ""),
                        "last_active": row.get("updated_at", ""),
                    })
            except Exception:
                logger.debug("Failed to list DB sessions", exc_info=True)

        return results

    async def cancel_session(self, session_id: str) -> bool:
        """Signal cancellation for a running session."""
        state = await self.get_session(session_id)
        if state is None:
            return False
        state.cancel_event.set()
        return True

    # ── Agent factory ──

    async def _make_agent(
        self, session_id: str, cwd: str, model: str = "",
    ) -> Any:
        """Create an agent instance for a session."""
        if self._agent_factory:
            result = self._agent_factory(session_id=session_id, cwd=cwd, model=model)
            if asyncio.iscoroutine(result):
                return await result
            return result
        # Default: create a Caveman AgentLoop
        try:
            from caveman.agent.loop import AgentLoop
            from caveman.config.loader import load_config
            config = load_config()
            return AgentLoop(
                config=config,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("Failed to create AgentLoop: %s", e)
            return None

    # ── Persistence ──

    def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        try:
            from caveman.agent.session_db import SessionDB
            self._db = SessionDB()
            return self._db
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return None

    async def _persist(self, state: ACPSessionState) -> None:
        db = self._get_db()
        if not db:
            return
        try:
            db.upsert_session(
                session_id=state.session_id,
                source="acp",
                model=state.model,
                model_config=json.dumps({"cwd": state.cwd, "mode": state.mode}),
                history=json.dumps(state.history[-50:]),  # Keep last 50 turns
            )
        except Exception:
            logger.debug("Failed to persist session %s", state.session_id, exc_info=True)

    async def _restore(self, session_id: str) -> Optional[ACPSessionState]:
        db = self._get_db()
        if not db:
            return None
        try:
            row = db.get_session(session_id)
            if not row:
                return None
            mc = json.loads(row.get("model_config", "{}") or "{}")
            history = json.loads(row.get("history", "[]") or "[]")
            agent = await self._make_agent(session_id, mc.get("cwd", "."), row.get("model", ""))
            state = ACPSessionState(
                session_id=session_id,
                cwd=mc.get("cwd", "."),
                model=row.get("model", ""),
                mode=mc.get("mode", "agent"),
                history=history,
                _agent=agent,
                created_at=row.get("created_at", ""),
                last_active=row.get("updated_at", ""),
            )
            async with self._lock:
                self._sessions[session_id] = state
            return state
        except Exception:
            logger.debug("Failed to restore session %s", session_id, exc_info=True)
            return None

    async def _delete_persisted(self, session_id: str) -> bool:
        db = self._get_db()
        if not db:
            return False
        try:
            db.delete_session(session_id)
            return True
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return False

    async def _evict_if_full(self) -> None:
        """Remove oldest sessions if at capacity."""
        while len(self._sessions) >= self._max_sessions:
            oldest_id = next(iter(self._sessions))
            state = self._sessions.pop(oldest_id)
            if state.agent and hasattr(state.agent, "shutdown"):
                try:
                    await state.agent.shutdown()
                except Exception:
                    pass  # intentional: Exception suppressed
