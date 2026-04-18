"""Permission tiering — controls agent action permissions."""
from __future__ import annotations
import logging
from enum import Enum
from typing import Callable

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
    def __init__(self, permissions: dict[str, PermissionLevel] | None = None):
        self._permissions = permissions or DEFAULT_PERMISSIONS.copy()
        self._approval_callback: Callable | None = None

    def set_approval_callback(self, callback: Callable) -> None:
        self._approval_callback = callback

    def check(self, action: str) -> PermissionLevel:
        return self._permissions.get(action, PermissionLevel.ASK)

    async def request(self, action: str, description: str) -> bool:
        level = self.check(action)
        if level == PermissionLevel.AUTO:
            return True
        if level == PermissionLevel.DENY:
            logger.warning("Permission DENIED for action '%s': %s", action, description[:100])
            return False
        if self._approval_callback:
            return await self._approval_callback(action, description)
        # P1 #3 fix: ASK without callback → DENY (fail-closed, not fail-open)
        logger.warning(
            "Permission DENIED for '%s' — ASK mode with no approval callback installed. "
            "Install a callback via set_approval_callback() to enable interactive approval.",
            action,
        )
        return False
