"""Helpers for cleaning legacy gateway transcript metadata."""
from __future__ import annotations

import re

_TOOL_COUNT_PREFIX = re.compile(r"^(\[使用了\s*\d+\s*个工具调用\]\s*)+", re.MULTILINE)
_FORMAT_REMINDER = re.compile(r"\n?\[Format:\s*\w+\s*—[^\]]*\]\s*$")
_STYLE_RESET = re.compile(r"^\[Style reset\]\s*")
_COMPACTION_NOTE = re.compile(r"\n*\[Note: Earlier turns compacted.*\]\s*$")


def clean_transcript_message(role: str, content: str) -> str | None:
    """Clean legacy metadata injections from a transcript message."""
    if not content:
        return content
    if role == "assistant":
        content = _TOOL_COUNT_PREFIX.sub("", content).lstrip()
    elif role == "user":
        content = _FORMAT_REMINDER.sub("", content).rstrip()
    elif role == "system":
        if _STYLE_RESET.match(content):
            return None
        content = _COMPACTION_NOTE.sub("", content).rstrip()
    return content
