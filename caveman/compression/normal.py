"""Compression — normal level: truncate verbose tool results."""
from __future__ import annotations


async def compress(messages: list[dict], target_ratio: float = 0.75) -> list[dict]:
    """Truncate tool results >500 chars to first/last 200 + ellipsis."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 500 and msg.get("role") == "tool":
            truncated = content[:200] + "\n...[truncated]...\n" + content[-200:]
            result.append({**msg, "content": truncated})
        else:
            result.append(msg)
    return result
