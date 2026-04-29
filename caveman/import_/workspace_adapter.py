"""Workspace file adapter — transforms OpenClaw workspace files for Caveman.

When importing from OpenClaw, workspace files (SOUL.md, AGENTS.md, etc.)
contain references to OpenClaw-specific tools, concepts, and patterns.
This module rewrites them to use Caveman equivalents.

Design principle: adaptation happens at import time, not runtime.
The imported files should work natively in Caveman without any
"translation layer" at prompt injection time.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─── Tool Name Mapping ───────────────────────────────────────────────────────
# OpenClaw tool → Caveman tool
_TOOL_MAP: dict[str, str | None] = {
    "message": "progress",
    "exec": "bash",
    "read": "file_read",
    "write": "file_write",
    "edit": "file_edit",
    "web_fetch": "url_check",
    "memory_get": "memory_recent",
    "sessions_spawn": "delegate",
    "sessions_send": "gateway_send",
    "tts": None,  # No equivalent
    "canvas": None,
    "nodes": None,
    "cron": None,
}

# Concepts that need rewriting
_CONCEPT_MAP: dict[str, str] = {
    "sessions_spawn": "delegate tool",
    "sessions_yield": "(not needed in Caveman)",
    "subagents": "delegate tool",
    "session_status": "metrics tool",
    "NO_REPLY": "(Caveman handles this automatically)",
}

# ─── Patterns to detect and rewrite ─────────────────────────────────────────

# Backtick-quoted tool names: `message`, `exec`, etc.
_BACKTICK_TOOL_RE = re.compile(r'`(' + '|'.join(re.escape(k) for k in _TOOL_MAP) + r')`')

# "use X tool" / "X tool" / "the X tool"
_PROSE_TOOL_RE = re.compile(
    r'\b(the\s+)?(' + '|'.join(re.escape(k) for k in _TOOL_MAP) + r')\s+tool\b',
    re.IGNORECASE,
)

# OpenClaw-specific parameters: `background: true`, `exec + background`
_BACKGROUND_RE = re.compile(r'`(?:exec|bash)`\s*\+\s*`background:\s*true`', re.IGNORECASE)
_BACKGROUND_PROSE_RE = re.compile(r'(?:exec|bash).*background.*true', re.IGNORECASE)


def adapt_workspace_content(filename: str, content: str) -> str:
    """Transform OpenClaw workspace file content for Caveman compatibility.

    Args:
        filename: The workspace file name (e.g., "AGENTS.md", "SOUL.md")
        content: Raw file content from OpenClaw

    Returns:
        Adapted content suitable for Caveman
    """
    adaptations: list[str] = []

    # 1. Replace compound patterns FIRST (before individual tool names change)
    # `exec` + `background: true` → `process_start`
    content = re.sub(
        r'预计\s*>60s[：:]\s*`exec`\s*\+\s*`background:\s*true`[^。\n]*[。.]?',
        '预计 >60s：用 `process_start` 启动后台进程，用 `process_output` 轮询结果。',
        content,
    )
    content = _BACKGROUND_RE.sub('`process_start`', content)

    # 2. Replace backtick-quoted tool names
    def _replace_backtick(m: re.Match) -> str:
        old = m.group(1)
        new = _TOOL_MAP.get(old)
        if new is None:
            return f"`{old}` (not available)"
        if new != old:
            adaptations.append(f"`{old}` → `{new}`")
        return f"`{new}`"

    content = _BACKTICK_TOOL_RE.sub(_replace_backtick, content)

    # 2. Replace prose tool references
    def _replace_prose(m: re.Match) -> str:
        prefix = m.group(1) or ""
        old = m.group(2)
        new = _TOOL_MAP.get(old)
        if new is None:
            return f"{prefix}{old} tool (not available)"
        return f"{prefix}{new} tool"

    content = _PROSE_TOOL_RE.sub(_replace_prose, content)

    # 4. Remove OpenClaw-specific sections that don't apply
    content = _strip_openclaw_sections(content)

    # 5. Deduplicate reporting rules (keep only one copy)
    if filename == "SOUL.md":
        content = _strip_reporting_rules_from_soul(content)

    if adaptations:
        logger.info("Adapted %s: %d tool name replacements", filename, len(adaptations))
        logger.debug("Adaptations: %s", ", ".join(adaptations))

    return content


def _strip_openclaw_sections(content: str) -> str:
    """Remove sections that reference OpenClaw-only features."""
    # Remove lines about gateway tool (Caveman doesn't expose it to agents)
    lines = content.split('\n')
    filtered = []
    skip_until_blank = False

    for line in lines:
        # Skip lines about unavailable tools
        if '`gateway`' in line and ('禁止' in line or 'forbidden' in line.lower()):
            # Keep prohibition rules — they're still valid as general guidance
            filtered.append(line.replace('`gateway`', '`gateway_send`'))
            continue
        if skip_until_blank:
            if line.strip() == '':
                skip_until_blank = False
                filtered.append(line)
            continue
        filtered.append(line)

    return '\n'.join(filtered)


def _strip_reporting_rules_from_soul(content: str) -> str:
    """Remove reporting rules from SOUL.md (they belong in AGENTS.md only).

    Prevents the "said 3 times" problem: reporting rules should exist
    in exactly ONE place (AGENTS.md), not scattered across SOUL + AGENTS + prompt.py.
    """
    lines = content.split('\n')
    result = []
    in_reporting_section = False

    for line in lines:
        # Detect reporting rule sections
        if re.match(r'^##\s*⚠️\s*实时汇报', line) or \
           re.match(r'^##\s*Gateway Communication', line, re.IGNORECASE) or \
           re.match(r'^##\s*Reporting Rules', line, re.IGNORECASE):
            in_reporting_section = True
            continue

        # End of section: next heading
        if in_reporting_section and re.match(r'^##\s', line):
            in_reporting_section = False

        if not in_reporting_section:
            result.append(line)

    return '\n'.join(result)


def validate_adapted_content(filename: str, content: str) -> list[str]:
    """Check adapted content for remaining OpenClaw references.

    Returns list of warnings (empty = clean).
    """
    warnings: list[str] = []

    # Check for remaining OpenClaw tool names in backticks
    for tool in _TOOL_MAP:
        if _TOOL_MAP[tool] is not None and tool != _TOOL_MAP[tool]:
            if f'`{tool}`' in content:
                warnings.append(f"Still contains OpenClaw tool name: `{tool}`")

    # Check for OpenClaw-specific concepts
    if 'NO_REPLY' in content and filename != "AGENTS.md":
        warnings.append("Contains NO_REPLY (OpenClaw concept)")

    if 'sessions_spawn' in content or 'sessions_yield' in content:
        warnings.append("Contains OpenClaw session management references")

    return warnings
