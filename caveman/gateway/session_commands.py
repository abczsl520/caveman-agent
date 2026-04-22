"""Session Commands — session lifecycle management.

Extracted from OpenClaw commands-session.ts (669 lines).
Handles: /reset, /compact, /session idle, /session max-age, session binding.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.session_commands")


@dataclass
class SessionBinding:
    """A session binding record."""
    session_key: str
    channel_id: str = ""
    user_id: str = ""
    bound_by: str = ""  # user | system | thread
    bound_at: float = 0
    expires_at: float = 0
    idle_timeout_ms: float = 0
    max_age_ms: float = 0
    last_activity: float = 0

    @property
    def is_expired(self) -> bool:
        now = time.time()
        if self.expires_at > 0 and now > self.expires_at:
            return True
        if self.idle_timeout_ms > 0 and self.last_activity > 0:
            idle_ms = (now - self.last_activity) * 1000
            if idle_ms > self.idle_timeout_ms:
                return True
        return False

    def touch(self) -> None:
        self.last_activity = time.time()


def parse_duration_ms(raw: str) -> int:
    """Parse a duration string like '30m', '2h', '1d' into milliseconds."""
    raw = raw.strip().lower()
    multipliers = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
    for suffix, mult in multipliers.items():
        if raw.endswith(suffix):
            try:
                return int(float(raw[:-1]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(raw) * 1000)  # Assume seconds
    except ValueError:
        return 0


class SessionCommandHandler:
    """Handles session-related slash commands."""

    def __init__(
        self,
        reset_fn: Optional[Callable] = None,
        compact_fn: Optional[Callable] = None,
        get_session_fn: Optional[Callable] = None,
    ):
        self._reset_fn = reset_fn
        self._compact_fn = compact_fn
        self._get_session_fn = get_session_fn
        self._bindings: Dict[str, SessionBinding] = {}

    # ── Commands ──

    async def handle_reset(self, session_key: str, **kwargs) -> str:
        """Reset a session (clear history)."""
        if self._reset_fn:
            try:
                result = self._reset_fn(session_key)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                return f"Reset failed: {e}"

        self._bindings.pop(session_key, None)
        return "Session reset. Starting fresh."

    async def handle_compact(self, session_key: str, **kwargs) -> str:
        """Compact session history."""
        if self._compact_fn:
            try:
                result = self._compact_fn(session_key)
                if hasattr(result, "__await__"):
                    result = await result
                tokens_after = result.get("tokens_after", "unknown") if isinstance(result, dict) else "unknown"
                return f"Session compacted. Tokens after: {tokens_after}"
            except Exception as e:
                return f"Compaction failed: {e}"
        return "No compaction handler configured."

    async def handle_session_idle(self, session_key: str, duration: str = "", **kwargs) -> str:
        """Set session idle timeout."""
        if not duration:
            binding = self._bindings.get(session_key)
            if binding and binding.idle_timeout_ms > 0:
                return f"Idle timeout: {binding.idle_timeout_ms / 60000:.0f}m"
            return "No idle timeout set."

        ms = parse_duration_ms(duration)
        if ms <= 0:
            return f"Invalid duration: {duration}"

        binding = self._bindings.setdefault(
            session_key, SessionBinding(session_key=session_key),
        )
        binding.idle_timeout_ms = ms
        return f"Idle timeout set to {ms / 60000:.0f}m"

    async def handle_session_max_age(self, session_key: str, duration: str = "", **kwargs) -> str:
        """Set session max age."""
        if not duration:
            binding = self._bindings.get(session_key)
            if binding and binding.max_age_ms > 0:
                return f"Max age: {binding.max_age_ms / 3600000:.1f}h"
            return "No max age set."

        ms = parse_duration_ms(duration)
        if ms <= 0:
            return f"Invalid duration: {duration}"

        binding = self._bindings.setdefault(
            session_key, SessionBinding(session_key=session_key),
        )
        binding.max_age_ms = ms
        binding.expires_at = time.time() + ms / 1000
        return f"Max age set to {ms / 3600000:.1f}h"

    # ── Binding Management ──

    def get_binding(self, session_key: str) -> Optional[SessionBinding]:
        binding = self._bindings.get(session_key)
        if binding and binding.is_expired:
            self._bindings.pop(session_key, None)
            return None
        return binding

    def touch(self, session_key: str) -> None:
        binding = self._bindings.get(session_key)
        if binding:
            binding.touch()

    def reap_expired(self) -> List[str]:
        """Remove expired bindings. Returns list of reaped session keys."""
        expired = [k for k, b in self._bindings.items() if b.is_expired]
        for k in expired:
            self._bindings.pop(k, None)
        return expired

from caveman.gateway.session_commands_depth import (  # noqa: F401,E402  # depth wiring
    SessionExport,
    export_session,
    import_session,
    search_sessions,
    bulk_delete_sessions,
)

__all__ = [
    "SessionBinding",
    "parse_duration_ms",
    "SessionCommandHandler",
    "SessionExport",
    "export_session",
    "import_session",
    "search_sessions",
    "bulk_delete_sessions",
]

