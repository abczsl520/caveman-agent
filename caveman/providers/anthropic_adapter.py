"""Anthropic message format adapter — converts between OpenAI and Anthropic formats.

Ported from Hermes anthropic_adapter.py (MIT, Nous Research).
Handles: role alternation, tool_use/tool_result pairing, thinking block management,
orphan cleanup, content block normalization.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Anthropic model output caps (tokens)
_MODEL_MAX_OUTPUT: dict[str, int] = {
    "claude-opus-4-6": 128_000,
    "claude-sonnet-4-6": 64_000,
    "claude-opus-4-5": 32_000,
    "claude-sonnet-4-5": 16_000,
    "claude-haiku-3-5": 8_192,
}
_DEFAULT_MAX_OUTPUT = 8_192

# Thinking budget by effort level
THINKING_BUDGET: dict[str, int] = {
    "low": 2_000,
    "medium": 8_000,
    "high": 32_000,
}

ADAPTIVE_EFFORT_MAP: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# Models that support adaptive thinking (Claude 4.6+)
_ADAPTIVE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

_THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))

# Tool ID sanitization: Anthropic requires alphanumeric + underscore
_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_]")


def get_max_output(model: str) -> int:
    """Get model's max output token cap."""
    for key, val in _MODEL_MAX_OUTPUT.items():
        if key in model:
            return val
    return _DEFAULT_MAX_OUTPUT


def supports_adaptive_thinking(model: str) -> bool:
    return any(m in model for m in _ADAPTIVE_MODELS)


def _sanitize_tool_id(tool_id: str) -> str:
    """Ensure tool ID matches Anthropic's format requirements."""
    sanitized = _TOOL_ID_RE.sub("_", tool_id)
    return sanitized[:64] if sanitized else "tool_0"


def convert_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get("function", t)  # Support both wrapped and unwrapped
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", fn.get("input_schema", {
                "type": "object", "properties": {},
            })),
        })
    return result


def _convert_content_part(part: Any) -> Optional[dict]:
    """Convert a single content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type", "text")
    if ptype in ("text", "input_text"):
        return {"type": "text", "text": part.get("text", "")}
    if ptype in ("image_url", "input_image"):
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = "image/jpeg"
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    return dict(part)


def _convert_content(content: Any) -> Any:
    """Convert content field (string or list of parts) to Anthropic format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = []
        for part in content:
            c = _convert_content_part(part)
            if c:
                converted.append(c)
        return converted or [{"type": "text", "text": "(empty)"}]
    return str(content) if content else "(empty)"


def _convert_assistant_message(m: dict) -> dict:
    """Convert an assistant message, including tool_calls → tool_use blocks."""
    content = m.get("content", "")
    blocks: list[dict] = []

    if content:
        if isinstance(content, list):
            converted = _convert_content(content)
            if isinstance(converted, list):
                blocks.extend(converted)
        else:
            blocks.append({"type": "text", "text": str(content)})

    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        blocks.append({
            "type": "tool_use",
            "id": _sanitize_tool_id(tc.get("id", "")),
            "name": fn.get("name", ""),
            "input": parsed,
        })

    return {"role": "assistant", "content": blocks or [{"type": "text", "text": "(empty)"}]}


def _convert_tool_message(m: dict, result: list[dict]) -> None:
    """Convert a tool result message, merging consecutive results into one user message."""
    content = m.get("content", "")
    result_content = content if isinstance(content, str) else json.dumps(content)
    if not result_content:
        result_content = "(no output)"

    tool_result = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": result_content,
    }

    # Merge consecutive tool results into one user message
    if (result and result[-1]["role"] == "user"
            and isinstance(result[-1]["content"], list)
            and result[-1]["content"]
            and result[-1]["content"][0].get("type") == "tool_result"):
        result[-1]["content"].append(tool_result)
    else:
        result.append({"role": "user", "content": [tool_result]})


def _convert_user_message(content: Any) -> dict:
    """Convert a user message to Anthropic format."""
    if isinstance(content, list):
        converted = _convert_content(content)
        has_non_text = any(
            isinstance(b, dict) and b.get("type") not in ("text", "input_text")
            for b in converted
        )
        if not has_non_text and (not converted or all(
            b.get("text", "").strip() == "" for b in converted
            if isinstance(b, dict) and b.get("type") == "text"
        )):
            converted = [{"type": "text", "text": "(empty message)"}]
        return {"role": "user", "content": converted}
    else:
        if not content or (isinstance(content, str) and not content.strip()):
            content = "(empty message)"
        return {"role": "user", "content": content}


def _extract_system(m: dict) -> Any:
    """Extract system prompt from a system message."""
    content = m.get("content", "")
    if isinstance(content, list):
        has_cache = any(p.get("cache_control") for p in content if isinstance(p, dict))
        if has_cache:
            return [p for p in content if isinstance(p, dict)]
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return content


def _cleanup_orphan_tool_use(result: list[dict]) -> None:
    """Remove tool_use blocks without matching tool_result."""
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))

    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            m["content"] = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") == "tool_use")
                or b.get("id") in tool_result_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]


def _cleanup_orphan_tool_result(result: list[dict]) -> None:
    """Remove tool_result blocks without matching tool_use."""
    tool_use_ids = set()
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_ids.add(block.get("id"))

    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            m["content"] = [
                b for b in m["content"]
                if not (isinstance(b, dict) and b.get("type") == "tool_result")
                or b.get("tool_use_id") in tool_use_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool result removed)"}]


def _enforce_role_alternation(result: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages to enforce strict alternation."""
    fixed: list[dict] = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            prev = fixed[-1]["content"]
            curr = m["content"]
            if isinstance(prev, str) and isinstance(curr, str):
                fixed[-1]["content"] = prev + "\n" + curr
            elif isinstance(prev, list) and isinstance(curr, list):
                fixed[-1]["content"] = prev + curr
            else:
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                if isinstance(curr, str):
                    curr = [{"type": "text", "text": curr}]
                fixed[-1]["content"] = prev + curr
        else:
            fixed.append(m)
    return fixed


def _manage_thinking_blocks(result: list[dict]) -> None:
    """Strip/downgrade thinking blocks per Anthropic rules."""
    last_assistant_idx = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        if idx != last_assistant_idx:
            stripped = [b for b in m["content"]
                        if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)]
            m["content"] = stripped or [{"type": "text", "text": "(thinking elided)"}]
        else:
            new_content = []
            for b in m["content"]:
                if not isinstance(b, dict) or b.get("type") not in _THINKING_TYPES:
                    new_content.append(b)
                    continue
                if b.get("type") == "redacted_thinking" and b.get("data"):
                    new_content.append(b)
                elif b.get("signature"):
                    new_content.append(b)
                else:
                    text = b.get("thinking", "")
                    if text:
                        new_content.append({"type": "text", "text": text})
            m["content"] = new_content or [{"type": "text", "text": "(empty)"}]

        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                b.pop("cache_control", None)


def convert_messages(
    messages: list[dict],
) -> tuple[Optional[Any], list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    Handles: system extraction, role alternation, tool pairing, orphan cleanup.
    """
    system = None
    result: list[dict] = []

    for m in messages:
        role = m.get("role", "user")

        if role == "system":
            system = _extract_system(m)
        elif role == "assistant":
            result.append(_convert_assistant_message(m))
        elif role == "tool":
            _convert_tool_message(m, result)
        else:  # user
            result.append(_convert_user_message(m.get("content", "")))

    _cleanup_orphan_tool_use(result)
    _cleanup_orphan_tool_result(result)
    result = _enforce_role_alternation(result)
    _manage_thinking_blocks(result)

    return system, result


def build_api_kwargs(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
    system: str | None = None,
    thinking: dict | None = None,
    tool_choice: str | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Build kwargs for anthropic.messages.create() / .stream().

    If system is provided directly, uses it. Otherwise extracts from messages.
    """
    if system is not None:
        # System provided separately — just convert messages
        _, anthropic_messages = convert_messages(messages)
        api_system = system
    else:
        api_system, anthropic_messages = convert_messages(messages)

    anthropic_tools = convert_tools(tools) if tools else []
    effective_max = max_tokens or get_max_output(model)

    # Clamp output to context window
    if context_length and effective_max > context_length:
        effective_max = max(context_length - 1, 1)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max,
    }

    if api_system:
        kwargs["system"] = api_system

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            kwargs.pop("tools", None)
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Thinking / reasoning config
    if thinking and isinstance(thinking, dict):
        if thinking.get("enabled") is not False and "haiku" not in model.lower():
            effort = str(thinking.get("effort", "medium")).lower()
            budget = THINKING_BUDGET.get(effort, 8000)
            if supports_adaptive_thinking(model):
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": ADAPTIVE_EFFORT_MAP.get(effort, "medium")}
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                kwargs["temperature"] = 1
                kwargs["max_tokens"] = max(effective_max, budget + 4096)

    return kwargs
