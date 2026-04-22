"""Command Registry — slash command system with auth, categories, and dispatch.

Extracted from OpenClaw commands-registry.shared.ts (837 lines) and
command-auth.ts (730 lines).

Features:
- Declarative command definition with categories
- Argument parsing (positional, named, capture-remaining)
- Permission levels per command
- Alias support
- Help generation
- Command dispatch pipeline
"""
from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.commands")


class CommandCategory(str, Enum):
    """Grouping category for slash commands in help display."""
    STATUS = "status"
    SESSION = "session"
    TOOLS = "tools"
    MEDIA = "media"
    MANAGEMENT = "management"
    MODEL = "model"
    DEBUG = "debug"


@dataclass
class CommandArg:
    """A command argument definition."""
    name: str
    description: str = ""
    type: str = "string"
    required: bool = False
    choices: Optional[List[str]] = None
    capture_remaining: bool = False


@dataclass
class CommandDefinition:
    """A slash command definition."""
    key: str
    description: str = ""
    category: CommandCategory = CommandCategory.STATUS
    aliases: List[str] = field(default_factory=list)
    args: List[CommandArg] = field(default_factory=list)
    min_access: str = "user"  # "guest" | "user" | "admin" | "owner"
    handler: Optional[Callable] = None
    hidden: bool = False

    @property
    def all_triggers(self) -> List[str]:
        return [f"/{self.key}"] + [f"/{a}" for a in self.aliases]


@dataclass
class ParsedCommand:
    """Result of parsing a command string."""
    key: str
    raw: str
    args: Dict[str, str] = field(default_factory=dict)
    positional: List[str] = field(default_factory=list)
    remaining: str = ""
    valid: bool = True
    error: str = ""


class CommandRegistry:
    """Registry and dispatcher for slash commands."""

    def __init__(self):
        self._commands: Dict[str, CommandDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias → key

    # ── Registration ──

    def register(self, cmd: CommandDefinition) -> None:
        self._commands[cmd.key] = cmd
        for alias in cmd.aliases:
            self._aliases[alias] = cmd.key

    def register_handler(self, key: str, handler: Callable) -> None:
        cmd = self._commands.get(key)
        if cmd:
            cmd.handler = handler

    def unregister(self, key: str) -> None:
        cmd = self._commands.pop(key, None)
        if cmd:
            for alias in cmd.aliases:
                self._aliases.pop(alias, None)

    # ── Parsing ──

    def parse(self, text: str) -> Optional[ParsedCommand]:
        """Parse a command string. Returns None if not a command."""
        text = text.strip()
        if not text.startswith("/"):
            return None

        parts = text.split(maxsplit=1)
        trigger = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        # Resolve key
        key = trigger[1:]  # Remove /
        if key in self._aliases:
            key = self._aliases[key]
        if key not in self._commands:
            return ParsedCommand(key=key, raw=text, valid=False,
                                 error=f"Unknown command: /{key}")

        cmd = self._commands[key]
        parsed = ParsedCommand(key=key, raw=text)

        # Parse arguments
        if cmd.args and rest:
            try:
                tokens = shlex.split(rest)
            except ValueError:
                tokens = rest.split()

            arg_idx = 0
            for i, arg_def in enumerate(cmd.args):
                if arg_def.capture_remaining:
                    parsed.remaining = " ".join(tokens[arg_idx:])
                    parsed.args[arg_def.name] = parsed.remaining
                    break
                if arg_idx < len(tokens):
                    val = tokens[arg_idx]
                    if arg_def.choices and val not in arg_def.choices:
                        parsed.error = f"Invalid value for {arg_def.name}: {val} (choices: {', '.join(arg_def.choices)})"
                    parsed.args[arg_def.name] = val
                    arg_idx += 1
                elif arg_def.required:
                    parsed.valid = False
                    parsed.error = f"Missing required argument: {arg_def.name}"
        elif rest:
            parsed.remaining = rest

        return parsed

    # ── Dispatch ──

    async def dispatch(self, text: str, context: Optional[Dict] = None) -> Optional[str]:
        """Parse and dispatch a command. Returns response text or None."""
        parsed = self.parse(text)
        if not parsed:
            return None
        if not parsed.valid:
            return parsed.error

        cmd = self._commands.get(parsed.key)
        if not cmd or not cmd.handler:
            return f"Command /{parsed.key} has no handler."

        try:
            result = cmd.handler(parsed, context or {})
            if hasattr(result, "__await__"):
                result = await result
            return str(result) if result is not None else None
        except Exception as e:
            logger.warning("Command /%s failed: %s", parsed.key, e)
            return f"Error: {e}"

    # ── Query ──

    def get(self, key: str) -> Optional[CommandDefinition]:
        return self._commands.get(key)

    def list_commands(self, category: Optional[CommandCategory] = None,
                      include_hidden: bool = False) -> List[CommandDefinition]:
        cmds = list(self._commands.values())
        if category:
            cmds = [c for c in cmds if c.category == category]
        if not include_hidden:
            cmds = [c for c in cmds if not c.hidden]
        return sorted(cmds, key=lambda c: (c.category.value, c.key))

    def build_help(self, category: Optional[CommandCategory] = None) -> str:
        """Generate help text for all commands."""
        cmds = self.list_commands(category)
        if not cmds:
            return "No commands available."

        lines = ["Available commands:"]
        current_cat = None
        for cmd in cmds:
            if cmd.category != current_cat:
                current_cat = cmd.category
                lines.append(f"\n{current_cat.value.upper()}:")
            args_str = ""
            if cmd.args:
                args_str = " " + " ".join(
                    f"<{a.name}>" if a.required else f"[{a.name}]"
                    for a in cmd.args
                )
            aliases = f" (aliases: {', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"  /{cmd.key}{args_str} — {cmd.description}{aliases}")

        return "\n".join(lines)


def build_builtin_commands() -> List[CommandDefinition]:
    """Build the default set of slash commands."""
    return [
        CommandDefinition(key="help", description="Show available commands", category=CommandCategory.STATUS),
        CommandDefinition(key="status", description="Show current status", category=CommandCategory.STATUS),
        CommandDefinition(key="whoami", description="Show your sender ID", category=CommandCategory.STATUS),
        CommandDefinition(key="tools", description="List available tools", category=CommandCategory.TOOLS,
                          args=[CommandArg(name="mode", choices=["compact", "verbose"])]),
        CommandDefinition(key="model", description="Get or set the model", category=CommandCategory.MODEL,
                          args=[CommandArg(name="model", capture_remaining=True)],
                          aliases=["models"]),
        CommandDefinition(key="reset", description="Reset session", category=CommandCategory.SESSION,
                          aliases=["new"]),
        CommandDefinition(key="compact", description="Compact conversation history", category=CommandCategory.SESSION),
        CommandDefinition(key="stop", description="Stop current processing", category=CommandCategory.SESSION,
                          aliases=["cancel"]),
        CommandDefinition(key="approve", description="Approve pending action", category=CommandCategory.MANAGEMENT,
                          args=[CommandArg(name="code", capture_remaining=True)]),
        CommandDefinition(key="deny", description="Deny pending action", category=CommandCategory.MANAGEMENT),
        CommandDefinition(key="allowlist", description="Manage allowlist", category=CommandCategory.MANAGEMENT,
                          args=[CommandArg(name="action", choices=["add", "remove", "list"]),
                                CommandArg(name="target", capture_remaining=True)],
                          min_access="admin"),
        CommandDefinition(key="tts", description="Text-to-speech control", category=CommandCategory.MEDIA,
                          args=[CommandArg(name="action", choices=["on", "off", "status"])]),
        CommandDefinition(key="voice", description="Voice mode control", category=CommandCategory.MEDIA,
                          args=[CommandArg(name="action", choices=["on", "off"])]),
        CommandDefinition(key="skill", description="Run a skill", category=CommandCategory.TOOLS,
                          args=[CommandArg(name="name", required=True),
                                CommandArg(name="input", capture_remaining=True)]),
        CommandDefinition(key="context", description="Show context info", category=CommandCategory.DEBUG),
        CommandDefinition(key="reasoning", description="Toggle reasoning mode", category=CommandCategory.DEBUG,
                          args=[CommandArg(name="mode", choices=["on", "off", "stream"])]),
        CommandDefinition(key="verbose", description="Toggle verbose mode", category=CommandCategory.DEBUG,
                          args=[CommandArg(name="level", choices=["off", "on", "debug"])]),
    ]

from caveman.gateway.command_registry_depth import (  # noqa: F401,E402  # depth wiring
    CommandPermission,
    CommandCooldown,
    CommandAlias,
    EnhancedCommand,
    EnhancedCommandRegistry,
)

__all__ = [
    "CommandCategory",
    "CommandArg",
    "CommandDefinition",
    "ParsedCommand",
    "CommandRegistry",
    "build_builtin_commands",
    "CommandPermission",
    "CommandCooldown",
    "CommandAlias",
    "EnhancedCommand",
    "EnhancedCommandRegistry",
]

