"""Compression — micro level: remove near-duplicate messages."""
from __future__ import annotations


async def compress(messages: list[dict], target_ratio: float = 0.9) -> list[dict]:
    """Remove near-duplicate consecutive messages."""
    if len(messages) <= 2:
        return messages
    result = [messages[0]]
    for i in range(1, len(messages)):
        prev = str(messages[i - 1].get("content", ""))
        curr = str(messages[i].get("content", ""))
        if prev == curr:
            continue  # skip exact duplicate
        result.append(messages[i])
    return result
