"""Message Splitting — platform-aware message chunking.

Splits long messages to fit platform character limits while
preserving code blocks, markdown, and natural break points.
"""
from __future__ import annotations

import re
from typing import List, Optional

__all__ = [
    "PLATFORM_LIMITS",
    "split_message",
    "estimate_message_count",
]


# Platform character limits
PLATFORM_LIMITS = {
    "discord": 2000,
    "telegram": 4096,
    "whatsapp": 65536,
    "slack": 40000,
    "signal": 65536,
    "sms": 160,
    "default": 4096,
}


def split_message(
    text: str,
    platform: str = "default",
    max_length: Optional[int] = None,
) -> List[str]:
    """Split a message to fit platform limits.

    Preserves code blocks, tries to break at natural points.
    """
    limit = max_length or PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS["default"])

    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Find best break point
        chunk = remaining[:limit]
        break_idx = _find_break_point(chunk, remaining, limit)

        chunks.append(remaining[:break_idx].rstrip())
        remaining = remaining[break_idx:].lstrip("\n")

    # Post-process: ensure every chunk has balanced fences.
    # If a chunk ends with an open fence, close it and re-open in the next chunk.
    result = [c for c in chunks if c.strip()]
    balanced: List[str] = []
    pending_lang = ""  # language tag to re-open in next chunk
    for chunk in result:
        if pending_lang:
            chunk = f"```{pending_lang}\n{chunk}"
            pending_lang = ""
        # Count fences in this chunk
        fence_count = 0
        last_fence_line = ""
        for line in chunk.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                fence_count += 1
                last_fence_line = stripped
        if fence_count % 2 == 1:
            # Odd = unclosed fence — extract language tag and close it
            lang = last_fence_line[3:].strip().split()[0] if len(last_fence_line) > 3 else ""
            chunk += "\n```"
            pending_lang = lang
        balanced.append(chunk)
    return balanced


def _find_break_point(chunk: str, full_text: str, limit: int) -> int:
    """Find the best break point within the limit.

    Uses fence-parity counting to correctly handle any number of
    code fences (```) — never breaks inside an open code block.
    """
    # Scan all ``` positions and track open/close state
    fence_positions: list[int] = []
    search_start = 0
    while True:
        idx = chunk.find("```", search_start)
        if idx < 0:
            break
        fence_positions.append(idx)
        search_start = idx + 3

    if fence_positions:
        # Odd count = chunk ends inside an unclosed fence
        in_fence = len(fence_positions) % 2 == 1
        if in_fence:
            # Break before the last (unclosed) fence opener
            last_fence = fence_positions[-1]
            # Try to break at a newline just before the fence
            nl = chunk.rfind("\n", 0, last_fence)
            if nl > limit * 0.2:
                return nl + 1
            # Fall back to right before the fence
            if last_fence > 0:
                return last_fence
        else:
            # All fences are paired — try to break after the last closed block
            last_close = fence_positions[-1]
            after_block = chunk.find("\n", last_close + 3)
            if after_block > 0:
                return after_block + 1

    # Try paragraph break
    idx = chunk.rfind("\n\n")
    if idx > limit * 0.3:
        return idx + 2

    # Try line break
    idx = chunk.rfind("\n")
    if idx > limit * 0.3:
        return idx + 1

    # Try sentence break
    for pattern in (r'\. ', r'! ', r'\? '):
        match = None
        for m in re.finditer(pattern, chunk):
            if m.end() > limit * 0.3:
                match = m
        if match:
            return match.end()

    # Hard break at limit
    return limit


def estimate_message_count(text: str, platform: str = "default") -> int:
    """Estimate how many messages a text will be split into."""
    limit = PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS["default"])
    if len(text) <= limit:
        return 1
    return len(split_message(text, platform))
