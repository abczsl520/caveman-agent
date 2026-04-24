"""Convert Caveman internal message format → OpenAI Chat API format.

Caveman stores messages in Anthropic-style content blocks:
  assistant: [{"type": "text", ...}, {"type": "tool_use", ...}]
  user/tool: [{"type": "tool_result", "tool_use_id": ..., "content": ...}]

OpenAI expects:
  assistant: {"content": "text", "tool_calls": [{"id": ..., "type": "function", ...}]}
  tool:      {"role": "tool", "content": "result", "tool_call_id": "..."}
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["convert_to_openai_messages"]


def convert_to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert internal messages to OpenAI API format.

    Handles:
    - Plain string content (pass-through)
    - Anthropic-style content blocks (tool_use, tool_result, text, image)
    - Already-OpenAI-format messages (idempotent)
    """
    result: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        if role == "system":
            result.append({"role": "system", "content": _ensure_string(content)})
            continue

        if role == "assistant":
            result.extend(_convert_assistant(m))
            continue

        # Already-OpenAI tool message (has tool_call_id) — pass through
        if role == "tool" and "tool_call_id" in m:
            result.append(m)
            continue

        # user role with tool_result blocks (Anthropic format)
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            result.extend(_extract_tool_results(content))
            continue

        # Plain user message
        result.append({"role": role, "content": _ensure_string(content)})

    return result


def _convert_assistant(m: dict) -> list[dict]:
    """Convert an assistant message, splitting text and tool_calls."""
    content = m.get("content")

    # Already OpenAI format (has tool_calls key)
    if "tool_calls" in m:
        msg = {"role": "assistant"}
        if content is not None:
            msg["content"] = _ensure_string(content)
        msg["tool_calls"] = m["tool_calls"]
        return [msg]

    # Plain string — pass through
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]

    if not isinstance(content, list):
        return [{"role": "assistant", "content": _ensure_string(content)}]

    # Anthropic-style content blocks
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        btype = block.get("type", "")

        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tc_input = block.get("input", {})
            if isinstance(tc_input, str):
                args_str = tc_input
            else:
                args_str = json.dumps(tc_input, ensure_ascii=False)
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": args_str,
                },
            })
        elif btype == "image_url":
            # Vision — keep as content array for OpenAI vision
            text_parts.append("[image]")
        else:
            text_parts.append(block.get("text", str(block)))

    text = "\n".join(t for t in text_parts if t).strip()

    msg: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        # OpenAI: content can be null when there are tool_calls
        msg["content"] = text or None
        msg["tool_calls"] = tool_calls
    else:
        msg["content"] = text or ""

    return [msg]


def _extract_tool_results(blocks: list) -> list[dict]:
    """Convert Anthropic tool_result blocks to OpenAI tool messages."""
    results: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                # Nested content blocks — flatten to string
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", str(part)))
                    else:
                        parts.append(str(part))
                content = "\n".join(parts)
            elif not isinstance(content, str):
                content = str(content) if content is not None else ""
            results.append({
                "role": "tool",
                "content": content,
                "tool_call_id": block.get("tool_use_id", ""),
            })
    return results


def _ensure_string(content: Any) -> str:
    """Coerce content to a non-null string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)
