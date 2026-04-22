"""Reply Directives — system directive parsing and resolution.

Extracted from OpenClaw get-reply-directives.ts (581 lines).
Parses inline directives from messages: /model, /reasoning, /verbose, etc.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("caveman.gateway.directives")


@dataclass
class ParsedDirectives:
    """Result of parsing inline directives from a message."""
    cleaned: str = ""  # Message with directives removed
    model_override: str = ""
    provider_override: str = ""
    reasoning_mode: str = ""  # "" | "on" | "off" | "stream"
    verbose_level: str = ""  # "" | "off" | "on" | "debug"
    has_status_directive: bool = False
    has_elevated_directive: bool = False
    has_reset_directive: bool = False
    has_compact_directive: bool = False
    has_stop_directive: bool = False
    skill_name: str = ""
    skill_input: str = ""
    raw_directives: List[str] = field(default_factory=list)


# Directive patterns
_DIRECTIVE_PATTERNS = {
    "model": re.compile(r"/model\s+(\S+)", re.IGNORECASE),
    "reasoning": re.compile(r"/reasoning\s+(on|off|stream)", re.IGNORECASE),
    "verbose": re.compile(r"/verbose\s+(off|on|debug)", re.IGNORECASE),
    "status": re.compile(r"^/status\s*$", re.IGNORECASE | re.MULTILINE),
    "reset": re.compile(r"^/(?:reset|new)\s*$", re.IGNORECASE | re.MULTILINE),
    "compact": re.compile(r"^/compact\s*$", re.IGNORECASE | re.MULTILINE),
    "stop": re.compile(r"^/(?:stop|cancel)\s*$", re.IGNORECASE | re.MULTILINE),
    "skill": re.compile(r"/skill\s+(\S+)(?:\s+(.+))?", re.IGNORECASE),
}


def parse_inline_directives(
    text: str,
    model_aliases: Optional[List[str]] = None,
    allow_status: bool = True,
) -> ParsedDirectives:
    """Parse inline directives from message text."""
    result = ParsedDirectives(cleaned=text)
    aliases = set(model_aliases or [])

    # Model directive
    m = _DIRECTIVE_PATTERNS["model"].search(text)
    if m:
        model_val = m.group(1)
        result.model_override = model_val
        result.raw_directives.append(m.group(0))
        result.cleaned = result.cleaned.replace(m.group(0), "").strip()
        result.has_elevated_directive = True

    # Check for bare model alias at start of message
    if not result.model_override and aliases:
        first_word = text.strip().split()[0] if text.strip() else ""
        if first_word.lower() in {a.lower() for a in aliases}:
            result.model_override = first_word
            result.cleaned = text.strip()[len(first_word):].strip()
            result.has_elevated_directive = True

    # Reasoning directive
    m = _DIRECTIVE_PATTERNS["reasoning"].search(text)
    if m:
        result.reasoning_mode = m.group(1).lower()
        result.raw_directives.append(m.group(0))
        result.cleaned = result.cleaned.replace(m.group(0), "").strip()

    # Verbose directive
    m = _DIRECTIVE_PATTERNS["verbose"].search(text)
    if m:
        result.verbose_level = m.group(1).lower()
        result.raw_directives.append(m.group(0))
        result.cleaned = result.cleaned.replace(m.group(0), "").strip()

    # Status directive
    if allow_status and _DIRECTIVE_PATTERNS["status"].search(text):
        result.has_status_directive = True
        result.raw_directives.append("/status")
        result.cleaned = _DIRECTIVE_PATTERNS["status"].sub("", result.cleaned).strip()

    # Reset directive
    if _DIRECTIVE_PATTERNS["reset"].search(text):
        result.has_reset_directive = True
        result.raw_directives.append("/reset")
        result.cleaned = _DIRECTIVE_PATTERNS["reset"].sub("", result.cleaned).strip()

    # Compact directive
    if _DIRECTIVE_PATTERNS["compact"].search(text):
        result.has_compact_directive = True
        result.raw_directives.append("/compact")
        result.cleaned = _DIRECTIVE_PATTERNS["compact"].sub("", result.cleaned).strip()

    # Stop directive
    if _DIRECTIVE_PATTERNS["stop"].search(text):
        result.has_stop_directive = True
        result.raw_directives.append("/stop")
        result.cleaned = _DIRECTIVE_PATTERNS["stop"].sub("", result.cleaned).strip()

    # Skill directive
    m = _DIRECTIVE_PATTERNS["skill"].search(text)
    if m:
        result.skill_name = m.group(1)
        result.skill_input = (m.group(2) or "").strip()
        result.raw_directives.append(m.group(0))
        result.cleaned = result.cleaned.replace(m.group(0), "").strip()

    return result


@dataclass
class DirectiveResolution:
    """Resolved directives with final provider/model/settings."""
    provider: str = ""
    model: str = ""
    reasoning: str = ""
    verbose: str = ""
    is_command: bool = False
    command_key: str = ""
    command_args: str = ""
    body: str = ""  # Cleaned message body
    should_skip_agent: bool = False
    skip_reason: str = ""


def resolve_directives(
    text: str,
    default_provider: str = "",
    default_model: str = "",
    model_aliases: Optional[Dict[str, str]] = None,
    is_authorized: bool = True,
    is_group: bool = False,
    was_mentioned: bool = False,
) -> DirectiveResolution:
    """Resolve all directives from a message into final settings."""
    aliases_list = list((model_aliases or {}).keys())
    parsed = parse_inline_directives(text, aliases_list, allow_status=is_authorized)

    resolution = DirectiveResolution(
        provider=default_provider,
        model=default_model,
        body=parsed.cleaned,
    )

    # Resolve model override
    if parsed.model_override:
        alias_map = model_aliases or {}
        resolved = alias_map.get(parsed.model_override.lower(), parsed.model_override)
        if "/" in resolved:
            parts = resolved.split("/", 1)
            resolution.provider = parts[0]
            resolution.model = parts[1]
        else:
            resolution.model = resolved

    # Reasoning/verbose
    resolution.reasoning = parsed.reasoning_mode
    resolution.verbose = parsed.verbose_level

    # Command detection
    if parsed.has_status_directive:
        resolution.is_command = True
        resolution.command_key = "status"
        resolution.should_skip_agent = True
    elif parsed.has_reset_directive:
        resolution.is_command = True
        resolution.command_key = "reset"
        resolution.should_skip_agent = True
    elif parsed.has_compact_directive:
        resolution.is_command = True
        resolution.command_key = "compact"
        resolution.should_skip_agent = True
    elif parsed.has_stop_directive:
        resolution.is_command = True
        resolution.command_key = "stop"
        resolution.should_skip_agent = True
    elif parsed.skill_name:
        resolution.is_command = True
        resolution.command_key = "skill"
        resolution.command_args = f"{parsed.skill_name} {parsed.skill_input}".strip()

    # In group chats, elevated directives require mention
    if is_group and parsed.has_elevated_directive and not was_mentioned:
        resolution.model = default_model
        resolution.provider = default_provider

    return resolution

from caveman.gateway.directives_depth import (  # noqa: F401,E402  # depth wiring
    ApproveDirective,
    ConfigDirective,
    EXTENDED_DIRECTIVES,
    ParsedDirective,
    parse_extended_directives,
    generate_help,
    ApprovalStore,
)

__all__ = [
    "ParsedDirectives",
    "parse_inline_directives",
    "DirectiveResolution",
    "resolve_directives",
    "ApproveDirective",
    "ConfigDirective",
    "EXTENDED_DIRECTIVES",
    "ParsedDirective",
    "parse_extended_directives",
    "generate_help",
    "ApprovalStore",
]

