"""Preemptive compaction — proactive context management before overflow.

Ported from OpenClaw's preemptive-compaction.ts + tool-result-truncation.ts
with Caveman adaptations.

Strategy (3-tier, cheapest first):
  1. Truncate oversized tool results (no LLM call, instant)
  2. Prune old images from history (no LLM call, instant)
  3. Full LLM-assisted compression (expensive, last resort)

Called before each LLM call to ensure the prompt fits within budget.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from caveman.utils import estimate_tokens as estimate_str_tokens

__all__ = [
    "RESPONSE_RESERVE_TOKENS",
    "MAX_TOOL_RESULT_SHARE",
    "MIN_KEEP_CHARS",
    "COMPACTION_TIMEOUT",
    "PREEMPTIVE_THRESHOLD",
    "CompactionRoute",
    "PreemptiveResult",
    "truncate_tool_result_text",
    "calculate_max_tool_result_chars",
    "count_prunable_images",
    "prune_images",
    "should_preemptively_compact",
    "apply_tool_result_truncation",
    "apply_image_pruning",
]


if TYPE_CHECKING:
    from caveman.agent.context import AgentContext

logger = logging.getLogger(__name__)

# --- Constants ---

# Reserve tokens for the LLM response
RESPONSE_RESERVE_TOKENS = 4096

# Single tool result should not exceed this share of context
MAX_TOOL_RESULT_SHARE = 0.3

# Minimum chars to keep when truncating tool results
MIN_KEEP_CHARS = 2000

# Head+tail truncation marker
_OMISSION_MARKER = "\n\n⚠️ [... middle content omitted — showing head and tail ...]\n\n"

# Compaction safety timeout (seconds)
COMPACTION_TIMEOUT = 60

# Threshold: if estimated prompt tokens exceed this fraction of budget, act
PREEMPTIVE_THRESHOLD = 0.85


class CompactionRoute(Enum):
    """What action to take before the next LLM call."""
    FITS = "fits"                          # No action needed
    TRUNCATE_TOOL_RESULTS = "truncate"     # Truncate oversized tool results
    PRUNE_IMAGES = "prune_images"          # Remove old images
    COMPRESS = "compress"                  # Full LLM compression
    TRUNCATE_THEN_COMPRESS = "truncate_then_compress"


@dataclass
class PreemptiveResult:
    """Result of preemptive compaction check."""
    route: CompactionRoute
    estimated_tokens: int
    budget_tokens: int
    overflow_tokens: int
    truncatable_chars: int = 0
    prunable_images: int = 0


# --- Tool result truncation ---

def _has_important_tail(text: str) -> bool:
    """Check if text has error/diagnostic content near the end."""
    tail = text[-2000:].lower() if len(text) > 2000 else text.lower()
    return bool(re.search(
        r'\b(error|exception|failed|fatal|traceback|panic|exit code|total|summary|result)\b',
        tail
    )) or tail.rstrip().endswith('}')


def truncate_tool_result_text(text: str, max_chars: int) -> str:
    """Truncate a tool result, preserving head+tail if tail has errors."""
    if len(text) <= max_chars:
        return text

    suffix = f"\n\n⚠️ [{len(text) - max_chars:,} chars truncated]"
    budget = max(MIN_KEEP_CHARS, max_chars - len(suffix))

    # Head+tail strategy if tail looks important
    if _has_important_tail(text) and budget > MIN_KEEP_CHARS * 2:
        tail_budget = min(int(budget * 0.3), 4000)
        head_budget = budget - tail_budget - len(_OMISSION_MARKER)

        if head_budget > MIN_KEEP_CHARS:
            # Find clean cut points at newline boundaries
            head_cut = head_budget
            nl = text.rfind('\n', 0, head_budget)
            if nl > head_budget * 0.8:
                head_cut = nl

            tail_start = len(text) - tail_budget
            nl = text.find('\n', tail_start)
            if nl != -1 and nl < tail_start + tail_budget * 0.2:
                tail_start = nl + 1

            return text[:head_cut] + _OMISSION_MARKER + text[tail_start:] + suffix

    # Default: keep the beginning
    cut = budget
    nl = text.rfind('\n', 0, budget)
    if nl > budget * 0.8:
        cut = nl
    return text[:cut] + suffix


def calculate_max_tool_result_chars(context_window: int) -> int:
    """Max chars for a single tool result based on context window."""
    max_tokens = int(context_window * MAX_TOOL_RESULT_SHARE)
    # ~4 chars per token for English, ~1 for CJK — use conservative 2
    return min(max_tokens * 2, 40_000)


# --- Image pruning ---

def count_prunable_images(messages: list[dict], protect_last_n: int = 3) -> int:
    """Count images that can be removed from older messages."""
    count = 0
    cutoff = max(0, len(messages) - protect_last_n)
    for i in range(cutoff):
        msg = messages[i]
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    count += 1
    return count


def prune_images(messages: list[dict], protect_last_n: int = 3) -> tuple[list[dict], int]:
    """Remove image blocks from older messages. Returns (new_messages, pruned_count)."""
    result = []
    pruned = 0
    cutoff = max(0, len(messages) - protect_last_n)

    for i, msg in enumerate(messages):
        if i >= cutoff:
            result.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                pruned += 1
                new_content.append({
                    "type": "text",
                    "text": "[image removed to save context space]"
                })
            else:
                new_content.append(block)

        result.append({**msg, "content": new_content})

    return result, pruned


# --- Preemptive check ---

def should_preemptively_compact(context: "AgentContext") -> PreemptiveResult:
    """Check if we need to act before the next LLM call.

    Returns a PreemptiveResult with the recommended route.
    """
    try:
        budget = int(context.max_tokens) - RESPONSE_RESERVE_TOKENS
        estimated = int(context.total_tokens)
    except (TypeError, ValueError):
        return PreemptiveResult(
            route=CompactionRoute.FITS,
            estimated_tokens=0, budget_tokens=0, overflow_tokens=0,
        )
    overflow = max(0, estimated - int(budget * PREEMPTIVE_THRESHOLD))

    if overflow <= 0:
        return PreemptiveResult(
            route=CompactionRoute.FITS,
            estimated_tokens=estimated,
            budget_tokens=budget,
            overflow_tokens=0,
        )

    # Check how much we can save by truncating tool results
    max_chars = calculate_max_tool_result_chars(context.max_tokens)
    truncatable = 0
    messages = [m.__dict__ if hasattr(m, '__dict__') else m for m in context.messages]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = getattr(msg, 'role', None) or msg.get('role', '')
        content = getattr(msg, 'content', None) or msg.get('content', '')
        if role == 'tool' and isinstance(content, str) and len(content) > max_chars:
            truncatable += len(content) - max_chars

    # Check prunable images
    prunable = count_prunable_images(
        [m.__dict__ if hasattr(m, '__dict__') else m for m in context.messages]
    )

    # Estimate savings from truncation (rough: 2 chars ≈ 1 token)
    truncation_savings_tokens = truncatable // 2
    image_savings_tokens = prunable * 500  # ~500 tokens per image

    if truncation_savings_tokens >= overflow:
        route = CompactionRoute.TRUNCATE_TOOL_RESULTS
    elif truncation_savings_tokens + image_savings_tokens >= overflow:
        route = CompactionRoute.PRUNE_IMAGES
    elif truncation_savings_tokens > 0:
        route = CompactionRoute.TRUNCATE_THEN_COMPRESS
    else:
        route = CompactionRoute.COMPRESS

    return PreemptiveResult(
        route=route,
        estimated_tokens=estimated,
        budget_tokens=budget,
        overflow_tokens=overflow,
        truncatable_chars=truncatable,
        prunable_images=prunable,
    )


def apply_tool_result_truncation(context: "AgentContext") -> int:
    """Truncate oversized tool results in-place. Returns count truncated."""
    max_chars = calculate_max_tool_result_chars(context.max_tokens)
    truncated = 0

    for msg in context.messages:
        role = getattr(msg, 'role', msg.get('role', '')) if isinstance(msg, dict) else getattr(msg, 'role', '')
        content = getattr(msg, 'content', msg.get('content', '')) if isinstance(msg, dict) else getattr(msg, 'content', '')

        if role == 'tool' and isinstance(content, str) and len(content) > max_chars:
            new_content = truncate_tool_result_text(content, max_chars)
            if isinstance(msg, dict):
                msg['content'] = new_content
            else:
                msg.content = new_content
            # Recalculate tokens
            new_tokens = estimate_str_tokens(new_content)
            if isinstance(msg, dict):
                msg['tokens'] = new_tokens
            elif hasattr(msg, 'tokens'):
                msg.tokens = new_tokens
            truncated += 1

    return truncated


def apply_image_pruning(context: "AgentContext") -> int:
    """Remove old images from context in-place. Returns count pruned."""
    pruned = 0
    protect_last = 3
    cutoff = max(0, len(context.messages) - protect_last)

    for i in range(cutoff):
        msg = context.messages[i]
        content = getattr(msg, 'content', None)
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    pruned += 1
                    new_content.append({
                        "type": "text",
                        "text": "[image removed to save context space]"
                    })
                else:
                    new_content.append(block)
            if isinstance(msg, dict):
                msg['content'] = new_content
            else:
                msg.content = new_content

    return pruned
