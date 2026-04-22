"""Session Manager — deep session lifecycle with context management.

Extracted from OpenClaw auto-reply/reply/session.ts (792 lines) and
Hermes gateway/session_manager.py patterns.

Features:
- Context window tracking and auto-compaction trigger
- Session persistence (SQLite)
- Model override per session (user-initiated vs auto-fallback)
- Session forking and restoration
- TTL-based expiration with knowledge extraction
- Concurrent session isolation
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULT_SESSION_TTL",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_CONTEXT_BUDGET",
    "SessionEntry",
    "GatewaySessionManager",
]


logger = logging.getLogger("caveman.gateway.session_mgr")

DEFAULT_SESSION_TTL = 3600  # 1 hour
DEFAULT_MAX_SESSIONS = 200
DEFAULT_CONTEXT_BUDGET = 0.6  # Use 60% of context window


@dataclass
class SessionEntry:
    """A single gateway session."""
    session_key: str
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    model: str = ""
    model_override: str = ""
    model_override_source: str = ""  # "user" | "auto" | ""
    provider: str = ""
    mode: str = "agent"  # agent | edit | chat
    history: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float = 0.0
    turn_count: int = 0
    compaction_count: int = 0
    system_prompt: str = ""
    system_prompt_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def touch(self) -> None:
        self.last_active = time.monotonic()
        self.turn_count += 1

    @property
    def effective_model(self) -> str:
        return self.model_override or self.model

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active

    def set_model_override(self, model: str, source: str = "user") -> None:
        """Set model override. source='user' survives fallback resets."""
        self.model_override = model
        self.model_override_source = source

    def clear_model_override(self) -> None:
        self.model_override = ""
        self.model_override_source = ""

    def should_compact(self, context_limit: int) -> bool:
        """Check if session needs compaction based on token usage."""
        if context_limit <= 0:
            return False
        return self.total_tokens > context_limit * DEFAULT_CONTEXT_BUDGET

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "model": self.effective_model,
            "mode": self.mode,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "compaction_count": self.compaction_count,
            "idle_seconds": round(self.idle_seconds, 1),
            "age_seconds": round(self.age_seconds, 1),
        }


class GatewaySessionManager:
    """Manages gateway sessions with lifecycle, persistence, and cleanup."""

    def __init__(
        self,
        ttl: float = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        on_session_end: Optional[Any] = None,
        db: Optional[Any] = None,
    ):
        self._sessions: Dict[str, SessionEntry] = {}
        self._ttl = ttl
        self._max_sessions = max_sessions
        self._on_session_end = on_session_end  # async (session) → None
        self._db = db
        self._lock = asyncio.Lock()

    # ── Session Lifecycle ──

    async def get_or_create(
        self, session_key: str, model: str = "", provider: str = "",
    ) -> SessionEntry:
        """Get existing session or create new one."""
        async with self._lock:
            entry = self._sessions.get(session_key)
            if entry:
                entry.touch()
                return entry

            # Try restore from DB
            entry = await self._restore(session_key)
            if entry:
                entry.touch()
                self._sessions[session_key] = entry
                return entry

            # Create new
            await self._evict_if_full()
            entry = SessionEntry(
                session_key=session_key,
                model=model,
                provider=provider,
            )
            self._sessions[session_key] = entry
            return entry

    async def get(self, session_key: str) -> Optional[SessionEntry]:
        return self._sessions.get(session_key)

    async def remove(self, session_key: str) -> bool:
        """Remove session, calling on_session_end hook."""
        entry = self._sessions.pop(session_key, None)
        if entry:
            if self._on_session_end:
                try:
                    await self._on_session_end(entry)
                except Exception:
                    logger.debug("on_session_end failed for %s", session_key, exc_info=True)
            await self._delete_persisted(session_key)
            return True
        return False

    async def reset(self, session_key: str, reason: str = "") -> Optional[SessionEntry]:
        """Reset session: clear history but keep metadata."""
        entry = self._sessions.get(session_key)
        if not entry:
            return None
        logger.info("Resetting session %s (reason: %s)", session_key, reason)
        entry.history.clear()
        entry.total_tokens = 0
        entry.compaction_count = 0
        entry.turn_count = 0
        entry.system_prompt = ""
        entry.system_prompt_hash = ""
        # Preserve user model override
        if entry.model_override_source != "user":
            entry.clear_model_override()
        return entry

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._sessions.values()]

    # ── Context Management ──

    async def compact_session(
        self, session_key: str, compact_fn: Optional[Any] = None,
    ) -> bool:
        """Trigger compaction for a session."""
        entry = self._sessions.get(session_key)
        if not entry:
            return False

        if compact_fn:
            try:
                result = await compact_fn(entry.history)
                if result:
                    entry.history = result if isinstance(result, list) else [result]
                    entry.compaction_count += 1
                    logger.info("Compacted session %s (count: %d)",
                                session_key, entry.compaction_count)
                    return True
            except Exception as e:
                logger.warning("Compaction failed for %s: %s", session_key, e)
        return False

    def update_tokens(self, session_key: str, tokens: int, cost: float = 0) -> None:
        """Update token usage for a session."""
        entry = self._sessions.get(session_key)
        if entry:
            entry.total_tokens += tokens
            entry.cost_usd += cost

    # ── Cleanup ──

    async def reap_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        now = time.monotonic()
        expired = [
            k for k, e in self._sessions.items()
            if now - e.last_active > self._ttl
        ]
        for key in expired:
            await self.remove(key)
        if expired:
            logger.info("Reaped %d expired sessions", len(expired))
        return len(expired)

    async def _evict_if_full(self) -> None:
        """Evict oldest sessions if at capacity."""
        while len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda e: e.last_active)
            await self.remove(oldest.session_key)

    # ── Persistence ──

    async def _restore(self, session_key: str) -> Optional[SessionEntry]:
        if not self._db:
            return None
        try:
            row = self._db.get_session(session_key)
            if not row:
                return None
            history = json.loads(row.get("history", "[]") or "[]")
            return SessionEntry(
                session_key=session_key,
                model=row.get("model", ""),
                history=history,
                total_tokens=row.get("total_tokens", 0),
            )
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return None

    async def persist(self, session_key: str) -> None:
        entry = self._sessions.get(session_key)
        if not entry or not self._db:
            return
        try:
            self._db.upsert_session(
                session_id=session_key,
                source="gateway",
                model=entry.effective_model,
                history=json.dumps(entry.history[-50:]),
            )
        except Exception:
            logger.debug("Failed to persist session %s", session_key, exc_info=True)

    async def _delete_persisted(self, session_key: str) -> None:
        if not self._db:
            return
        try:
            self._db.delete_session(session_key)
        except Exception:
            pass  # intentional: Exception suppressed
