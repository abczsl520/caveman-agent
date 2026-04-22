"""Command Registry Depth — aliases, permissions, cooldowns, help.

Supplements command_registry.py with command aliases, permission
levels, cooldown tracking, and auto-help generation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

__all__ = [
    "CommandPermission",
    "CommandCooldown",
    "CommandAlias",
    "EnhancedCommand",
    "EnhancedCommandRegistry",
]


logger = logging.getLogger("caveman.gateway.command_registry_depth")


@dataclass
class CommandPermission:
    """Permission level for a command."""
    level: str = "user"  # user | admin | owner
    allowlist: Set[str] = field(default_factory=set)
    denylist: Set[str] = field(default_factory=set)

    def check(self, sender_id: str, sender_level: str = "user") -> bool:
        if sender_id in self.denylist:
            return False
        if sender_id in self.allowlist:
            return True
        levels = {"user": 0, "admin": 1, "owner": 2}
        return levels.get(sender_level, 0) >= levels.get(self.level, 0)


@dataclass
class CommandCooldown:
    """Cooldown tracking for a command."""
    per_user_seconds: float = 0
    global_seconds: float = 0
    _user_last: Dict[str, float] = field(default_factory=dict)
    _global_last: float = 0

    def check(self, sender_id: str) -> Optional[float]:
        """Check cooldown. Returns remaining seconds or None if ready."""
        now = time.time()
        if self.global_seconds > 0:
            remaining = self._global_last + self.global_seconds - now
            if remaining > 0:
                return remaining
        if self.per_user_seconds > 0:
            last = self._user_last.get(sender_id, 0)
            remaining = last + self.per_user_seconds - now
            if remaining > 0:
                return remaining
        return None

    def record(self, sender_id: str) -> None:
        now = time.time()
        self._global_last = now
        self._user_last[sender_id] = now


@dataclass
class CommandAlias:
    """An alias for a command."""
    alias: str
    target: str
    args_transform: Optional[Callable] = None


@dataclass
class EnhancedCommand:
    """A command with full metadata."""
    name: str
    handler: Optional[Callable] = None
    description: str = ""
    usage: str = ""
    examples: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    permission: CommandPermission = field(default_factory=CommandPermission)
    cooldown: CommandCooldown = field(default_factory=CommandCooldown)
    hidden: bool = False
    category: str = "general"


class EnhancedCommandRegistry:
    """Command registry with aliases, permissions, cooldowns, and help."""

    def __init__(self):
        self._commands: Dict[str, EnhancedCommand] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, cmd: EnhancedCommand) -> None:
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._aliases[alias] = cmd.name

    def resolve(self, name: str) -> Optional[EnhancedCommand]:
        """Resolve a command name or alias."""
        if name in self._commands:
            return self._commands[name]
        target = self._aliases.get(name)
        if target:
            return self._commands.get(target)
        return None

    def can_execute(self, name: str, sender_id: str, sender_level: str = "user") -> Dict[str, Any]:
        """Check if a command can be executed."""
        cmd = self.resolve(name)
        if not cmd:
            return {"allowed": False, "reason": "unknown_command"}
        if not cmd.permission.check(sender_id, sender_level):
            return {"allowed": False, "reason": "permission_denied", "required": cmd.permission.level}
        remaining = cmd.cooldown.check(sender_id)
        if remaining is not None:
            return {"allowed": False, "reason": "cooldown", "remaining_seconds": round(remaining, 1)}
        return {"allowed": True}

    def execute(self, name: str, sender_id: str, **kwargs) -> Any:
        """Execute a command (records cooldown)."""
        cmd = self.resolve(name)
        if not cmd or not cmd.handler:
            return None
        cmd.cooldown.record(sender_id)
        return cmd.handler(**kwargs)

    def generate_help(self, sender_level: str = "user", category: Optional[str] = None) -> str:
        """Generate help text for available commands."""
        lines = ["Available commands:"]
        categories: Dict[str, List[EnhancedCommand]] = {}
        for cmd in self._commands.values():
            if cmd.hidden:
                continue
            if category and cmd.category != category:
                continue
            # Check permission visibility
            levels = {"user": 0, "admin": 1, "owner": 2}
            if levels.get(sender_level, 0) < levels.get(cmd.permission.level, 0):
                continue
            categories.setdefault(cmd.category, []).append(cmd)

        for cat, cmds in sorted(categories.items()):
            lines.append(f"\n{cat.title()}:")
            for cmd in sorted(cmds, key=lambda c: c.name):
                alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                lines.append(f"  /{cmd.name}{alias_str} — {cmd.description}")
                if cmd.usage:
                    lines.append(f"    Usage: {cmd.usage}")

        return "\n".join(lines)

    def list_categories(self) -> List[str]:
        return sorted({cmd.category for cmd in self._commands.values()})
