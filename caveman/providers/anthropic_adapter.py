"""Anthropic message format adapter — converts between OpenAI and Anthropic formats.

Ported from Hermes anthropic_adapter.py (MIT, Nous Research).
Handles: role alternation, tool_use/tool_result pairing, thinking block management,
orphan cleanup, content block normalization.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional
from caveman.providers.anthropic_messages import (_THINKING_TYPES, _sanitize_tool_id, convert_tools, _convert_content_part, _convert_content, _convert_assistant_message, _convert_tool_message, _convert_user_message, _extract_system, _cleanup_orphan_tool_use, _cleanup_orphan_tool_result, _enforce_role_alternation, _manage_thinking_blocks, convert_messages)

__all__ = [
    "CACHE_BOUNDARY",
    "get_max_output",
    "supports_adaptive_thinking",
    "build_api_kwargs",
]


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


# Tool ID sanitization: Anthropic requires alphanumeric + underscore
_TOOL_ID_RE = re.compile(r"[^a-zA-Z0-9_]")

# Cache boundary marker — shared with loop.py for prompt cache optimization
CACHE_BOUNDARY = "\n<!-- CAVEMAN_CACHE_BOUNDARY -->\n"


def get_max_output(model: str) -> int:
    """Get model's max output token cap."""
    for key, val in _MODEL_MAX_OUTPUT.items():
        if key in model:
            return val
    return _DEFAULT_MAX_OUTPUT


def supports_adaptive_thinking(model: str) -> bool:
    """Check if the current model supports adaptive thinking (extended thinking)."""
    return any(m in model for m in _ADAPTIVE_MODELS)





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
        # Use structured system prompt for cache optimization.
        # If system contains a cache boundary marker, split into stable (cached)
        # and dynamic (uncached) parts. Otherwise cache the entire prompt.
        if CACHE_BOUNDARY in api_system:
            parts = api_system.split(CACHE_BOUNDARY, 1)
            kwargs["system"] = [
                {"type": "text", "text": parts[0].rstrip(),
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": parts[1].lstrip()},
            ]
        else:
            kwargs["system"] = [
                {"type": "text", "text": api_system,
                 "cache_control": {"type": "ephemeral"}},
            ]

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        # Cache tool definitions — they rarely change between turns
        if anthropic_tools:
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            kwargs.pop("tools", None)
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Message-level cache control: mark last user turn as cache breakpoint
    # This allows Anthropic to cache everything up to and including the last user message
    if anthropic_messages:
        for msg in anthropic_messages:
            # Remove stale cache markers from previous turns
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
            elif isinstance(msg.get("content"), str):
                pass  # strings don't have cache_control
        # Mark the last user message as cache breakpoint
        last_user_idx = None
        for i in range(len(anthropic_messages) - 1, -1, -1):
            if anthropic_messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            msg = anthropic_messages[last_user_idx]
            if isinstance(msg.get("content"), str):
                msg["content"] = [{"type": "text", "text": msg["content"],
                                   "cache_control": {"type": "ephemeral"}}]
            elif isinstance(msg.get("content"), list) and msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}

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
