"""Directives Depth — /approve, /config, /help, /export, /import.

Supplements directives.py with additional slash commands.
Extracted from OpenClaw get-reply-directives.ts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ApproveDirective",
    "ConfigDirective",
    "EXTENDED_DIRECTIVES",
    "ParsedDirective",
    "parse_extended_directives",
    "generate_help",
    "ApprovalStore",
]


logger = logging.getLogger("caveman.gateway.directives_depth")


@dataclass
class ApproveDirective:
    """Parsed /approve command."""
    command_hash: str = ""
    policy: str = "allow-once"  # allow-once | allow-always | deny
    scope: str = ""  # Optional scope restriction

    @classmethod
    def parse(cls, args: str) -> "ApproveDirective":
        parts = args.strip().split()
        directive = cls()
        if parts:
            directive.command_hash = parts[0]
        if len(parts) > 1:
            directive.policy = parts[1]
        if len(parts) > 2:
            directive.scope = parts[2]
        return directive


@dataclass
class ConfigDirective:
    """Parsed /config command."""
    action: str = "get"  # get | set | reset | list
    key: str = ""
    value: str = ""

    @classmethod
    def parse(cls, args: str) -> "ConfigDirective":
        parts = args.strip().split(None, 2)
        directive = cls()
        if parts:
            directive.action = parts[0]
        if len(parts) > 1:
            directive.key = parts[1]
        if len(parts) > 2:
            directive.value = parts[2]
        return directive


# ── Extended Directive Parser ──

EXTENDED_DIRECTIVES = {
    "/approve": "approve",
    "/config": "config",
    "/help": "help",
    "/export": "export",
    "/import": "import",
    "/pin": "pin",
    "/unpin": "unpin",
    "/tag": "tag",
    "/untag": "untag",
    "/alias": "alias",
    "/whoami": "whoami",
    "/uptime": "uptime",
    "/version": "version",
}


@dataclass
class ParsedDirective:
    """A parsed directive with its arguments."""
    name: str
    args: str = ""
    raw: str = ""


def parse_extended_directives(text: str) -> Tuple[List[ParsedDirective], str]:
    """Parse extended directives from message text.

    Returns (directives, remaining_text).
    """
    directives = []
    remaining_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        matched = False
        for prefix, name in EXTENDED_DIRECTIVES.items():
            if stripped.lower().startswith(prefix):
                args = stripped[len(prefix):].strip()
                directives.append(ParsedDirective(name=name, args=args, raw=stripped))
                matched = True
                break
        if not matched:
            remaining_lines.append(line)

    return directives, "\n".join(remaining_lines).strip()


# ── Help Generator ──

def generate_help(
    commands: Optional[Dict[str, str]] = None,
    surface: str = "cli",
) -> str:
    """Generate help text for all available directives."""
    base_commands = {
        "/model <name>": "Switch model (e.g., /model opus)",
        "/reasoning [on|off|stream]": "Toggle extended thinking",
        "/verbose [on|off]": "Toggle verbose mode",
        "/status": "Show session status",
        "/reset": "Reset session context",
        "/new": "Start new conversation",
        "/compact": "Compact conversation history",
        "/stop": "Stop current generation",
        "/cancel": "Cancel current operation",
        "/skill <name>": "Load a skill",
        "/approve <hash> [policy]": "Approve a pending command",
        "/config [get|set|reset] <key> [value]": "View or change config",
        "/help": "Show this help",
        "/export": "Export current session",
        "/whoami": "Show your identity info",
        "/uptime": "Show session uptime",
        "/version": "Show version info",
    }

    if commands:
        base_commands.update(commands)

    lines = ["Directives:"]
    for cmd, desc in sorted(base_commands.items()):
        lines.append(f"  {cmd:40s} {desc}")

    return "\n".join(lines)


# ── Approval Store ──

class ApprovalStore:
    """Tracks command approvals."""

    def __init__(self):
        self._approvals: Dict[str, ApproveDirective] = {}

    def add(self, directive: ApproveDirective) -> None:
        self._approvals[directive.command_hash] = directive

    def check(self, command_hash: str) -> Optional[ApproveDirective]:
        """Check if a command is approved. Consumes allow-once approvals."""
        approval = self._approvals.get(command_hash)
        if not approval:
            return None
        if approval.policy == "allow-once":
            del self._approvals[command_hash]
        elif approval.policy == "deny":
            return None
        return approval

    def list_approvals(self) -> List[Dict[str, str]]:
        return [
            {"hash": a.command_hash, "policy": a.policy, "scope": a.scope}
            for a in self._approvals.values()
        ]

    def clear(self) -> int:
        count = len(self._approvals)
        self._approvals.clear()
        return count
