"""Caveman slash command system — registry architecture with 50+ commands."""
from __future__ import annotations

from caveman.commands.types import CommandDef, CommandContext
from caveman.commands.registry import COMMAND_REGISTRY, resolve_command, get_by_category
from caveman.commands.dispatcher import dispatch, parse_command

__all__ = [
    "CommandDef",
    "CommandContext",
    "COMMAND_REGISTRY",
    "resolve_command",
    "get_by_category",
    "dispatch",
    "parse_command",
]
