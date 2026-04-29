"""Permission tiering — controls agent action permissions."""
from __future__ import annotations
import logging
from enum import Enum
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Access control level for agent actions."""
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


DEFAULT_PERMISSIONS: dict[str, PermissionLevel] = {
    "file_read": PermissionLevel.AUTO,
    "file_write": PermissionLevel.ASK,
    "file_delete": PermissionLevel.ASK,
    "bash_safe": PermissionLevel.AUTO,
    "bash_write": PermissionLevel.ASK,
    "bash_sudo": PermissionLevel.DENY,
    "web_search": PermissionLevel.AUTO,
    "http_get": PermissionLevel.AUTO,
    "http_post": PermissionLevel.ASK,
    "skill_create": PermissionLevel.AUTO,
    "memory_read": PermissionLevel.AUTO,
    "memory_write": PermissionLevel.AUTO,
    "openclaw_spawn": PermissionLevel.ASK,
    "hermes_delegate": PermissionLevel.ASK,
}


class PermissionManager:
    """Manages permission checks and approval flows for agent actions."""
    def __init__(self, permissions: dict[str, PermissionLevel] | None = None, bus=None):
        self._permissions = permissions or DEFAULT_PERMISSIONS.copy()
        self._approval_callback: Callable[[str, str], Awaitable[bool]] | None = None
        self._bus = bus

    @staticmethod
    def _normalize_level(level) -> PermissionLevel:
        """Normalize permission levels across config strings and hot-reloaded Enums.

        SIGUSR2 can reload caveman.security.permissions while existing gateway
        sessions still hold old PermissionLevel enum instances. Comparing enum
        identity then misclassifies old AUTO as ASK. Normalize by semantic value.
        """
        if isinstance(level, PermissionLevel):
            return level
        raw = getattr(level, "value", level)
        if isinstance(raw, str):
            try:
                return PermissionLevel(raw.lower())
            except ValueError:
                logger.warning("Unknown permission level %r; defaulting to ASK", raw)
        return PermissionLevel.ASK

    def set_approval_callback(self, callback: Callable[[str, str], Awaitable[bool]]) -> None:
        self._approval_callback = callback

    def check(self, action: str) -> PermissionLevel:
        return self._normalize_level(self._permissions.get(action, PermissionLevel.ASK))

    async def request(self, action: str, description: str) -> bool:
        from caveman.events import EventType
        level = self.check(action)
        if level == PermissionLevel.AUTO:
            if self._bus:
                await self._bus.emit(
                    EventType.PERMISSION_CHECK,
                    {"action": action, "level": "auto", "granted": True},
                )
            return True
        if level == PermissionLevel.DENY:
            logger.warning("Permission DENIED for action '%s': %s", action, description[:100])
            if self._bus:
                await self._bus.emit(
                    EventType.PERMISSION_CHECK,
                    {"action": action, "level": "deny", "granted": False},
                )
            return False
        if self._approval_callback:
            approved = await self._approval_callback(action, description)
            return approved is True
        # P1 #3 fix: ASK without callback → DENY (fail-closed, not fail-open)
        logger.warning(
            "Permission DENIED for '%s' — ASK mode with no approval callback installed. "
            "Install a callback via set_approval_callback() to enable interactive approval.",
            action,
        )
        return False
