"""Compression utilities — tool pair sanitization, boundary alignment, estimation.

Extracted from smart.py to comply with NFR-502 (max 400 lines per file).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- Constants ---

_CHARS_PER_TOKEN = 4
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# OpenClaw identifier preservation (MIT, Peter Steinberger)
IDENTIFIER_PRESERVATION = (
    "Preserve all opaque identifiers exactly as written (no shortening or "
    "reconstruction), including UUIDs, hashes, IDs, tokens, API keys, "
    "hostnames, IPs, ports, URLs, and file names."
)

# Truncation limits for summarizer input
_CONTENT_MAX = 6000
_CONTENT_HEAD = 4000
_CONTENT_TAIL = 1500
_TOOL_ARGS_MAX = 1500
_TOOL_ARGS_HEAD = 1200


# --- Token estimation ---

def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // _CHARS_PER_TOKEN + 10
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", ""))) // _CHARS_PER_TOKEN
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                total += len(args) // _CHARS_PER_TOKEN
    return max(total, 1)


# --- Tool pair sanitization ---

def _get_tool_call_id(tc: Any) -> str:
    """Extract call ID from a tool_call entry (dict or object)."""
    if isinstance(tc, dict):
        return tc.get("id", "")
    return getattr(tc, "id", "") or ""


def sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
    """Fix orphaned tool_call / tool_result pairs after compression.

    Removes orphaned results and inserts stub results for orphaned calls.
    Ported from Hermes ContextCompressor (MIT, Nous Research).
    """
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = _get_tool_call_id(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # Remove orphaned tool results
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        logger.info("Sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

    # Add stub results for missing tool results
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: list[dict] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = _get_tool_call_id(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "content": "[Result from earlier — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        logger.info("Sanitizer: added %d stub tool result(s)", len(missing_results))

    return messages


# --- Boundary alignment ---

def align_forward(messages: list[dict], idx: int) -> int:
    """Push boundary forward past orphan tool results."""
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


def align_backward(messages: list[dict], idx: int) -> int:
    """Pull boundary backward to avoid splitting tool_call/result groups."""
    if idx <= 0 or idx >= len(messages):
        return idx
    check = idx - 1
    while check >= 0 and messages[check].get("role") == "tool":
        check -= 1
    if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
        idx = check
    return idx


# --- Serialization for summarizer ---

def serialize_turns(turns: list[dict]) -> str:
    """Serialize turns into labeled text for the summarizer."""
    parts = []
    for msg in turns:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""

        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            if len(content) > _CONTENT_MAX:
                content = content[:_CONTENT_HEAD] + "\n...[truncated]...\n" + content[-_CONTENT_TAIL:]
            parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            continue

        if role == "assistant":
            if len(content) > _CONTENT_MAX:
                content = content[:_CONTENT_HEAD] + "\n...[truncated]...\n" + content[-_CONTENT_TAIL:]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > _TOOL_ARGS_MAX:
                            args = args[:_TOOL_ARGS_HEAD] + "..."
                        tc_parts.append(f"  {name}({args})")
                    else:
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "?") if fn else "?"
                        tc_parts.append(f"  {name}(...)")
                content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
            parts.append(f"[ASSISTANT]: {content}")
            continue

        if len(content) > _CONTENT_MAX:
            content = content[:_CONTENT_HEAD] + "\n...[truncated]...\n" + content[-_CONTENT_TAIL:]
        parts.append(f"[{role.upper()}]: {content}")

    return "\n\n".join(parts)


# --- Structured summary template ---

def build_template(budget: int, focus_topic: str | None = None) -> str:
    """Build the structured summary template."""
    template = f"""## Goal
[What the user is trying to accomplish]

## Constraints/Rules
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — include specific file paths, commands, results]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues]

## Key Decisions
[Important technical decisions and why]

## Resolved Questions
[Questions already answered — include the answer]

## Pending User Asks
[Unanswered questions or unfulfilled requests. "None." if empty]

## Relevant Files
[Files read, modified, or created — with brief note]

## Remaining Work
[What remains — framed as context, not instructions]

## Critical Context
[Specific values, error messages, config details that would be lost]

## Exact Identifiers
[UUIDs, commit hashes, IPs, URLs, file paths — preserved verbatim]

Target ~{budget} tokens. Be specific — include file paths, command outputs,
error messages, and concrete values.
{IDENTIFIER_PRESERVATION}
Write only the summary body. No preamble or prefix."""

    if focus_topic:
        template += f"""\n\nFOCUS TOPIC: \"{focus_topic}\"
Prioritise preserving all information related to this topic (60-70% of budget).
For unrelated content, summarise aggressively."""

    return template
