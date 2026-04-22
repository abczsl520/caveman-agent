"""Pre-API message sanitization — fix orphaned tool calls before every LLM call.

Ported from Hermes _sanitize_api_messages (MIT, Nous Research).
Runs unconditionally before API calls to catch orphans from:
- Context compression dropping messages
- Session restore with partial history
- Manual message manipulation

Works with OpenAI-format messages (role: assistant/tool with tool_calls/tool_call_id).
Anthropic-format messages are handled by anthropic_adapter.py.
"""
from __future__ import annotations
import logging
import re

__all__ = [
    "sanitize_messages",
    "deduplicate_tool_calls",
    "sanitize_surrogates",
    "strip_reasoning_tags",
]


logger = logging.getLogger(__name__)

_VALID_ROLES = {"system", "user", "assistant", "tool"}
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')
_THINK_RE = re.compile(
    r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>.*?'
    r'</(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>',
    re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(
    r'</?(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>\s*',
    re.IGNORECASE,
)


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Fix orphaned tool_call/tool_result pairs. Returns cleaned list."""
    # Sanitize surrogates first (prevents json.dumps crashes)
    sanitize_surrogates(messages)

    # Drop invalid roles
    filtered = [m for m in messages if m.get("role") in _VALID_ROLES]
    if len(filtered) < len(messages):
        logger.debug("Sanitizer: dropped %d messages with invalid roles",
                      len(messages) - len(filtered))
    messages = filtered

    # Collect surviving tool_call IDs
    call_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") or (tc.get("function", {}).get("name", "") + "_stub")
                call_ids.add(cid)

    # Collect tool result IDs
    result_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                result_ids.add(cid)

    # 1. Drop orphaned tool results (no matching call)
    orphaned_results = result_ids - call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        logger.debug("Sanitizer: removed %d orphaned tool results", len(orphaned_results))

    # 2. Inject stub results for calls without results
    missing_results = call_ids - result_ids
    if missing_results:
        patched: list[dict] = []
        for m in messages:
            patched.append(m)
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id")
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
                        missing_results.discard(cid)
        messages = patched
        logger.debug("Sanitizer: added %d stub tool results", len(call_ids - result_ids))

    return messages


def deduplicate_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Remove duplicate (name, arguments) pairs within a single turn.

    Ported from Hermes _deduplicate_tool_calls.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for tc in tool_calls:
        name = tc.get("name", tc.get("function", {}).get("name", ""))
        args = tc.get("input", tc.get("function", {}).get("arguments", ""))
        if isinstance(args, dict):
            import json
            args = json.dumps(args, sort_keys=True)
        key = (name, str(args))
        if key not in seen:
            seen.add(key)
            unique.append(tc)
        else:
            logger.warning("Removed duplicate tool call: %s", name)
    return unique if len(unique) < len(tool_calls) else tool_calls


def sanitize_surrogates(messages: list[dict]) -> bool:
    """Replace lone surrogate code points with U+FFFD in all message strings.

    Surrogates crash json.dumps() in the OpenAI SDK.
    Ported from Hermes _sanitize_messages_surrogates.
    Returns True if any surrogates were found and replaced.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        for field in ("name", "tool_call_id"):
            val = msg.get(field)
            if isinstance(val, str) and _SURROGATE_RE.search(val):
                msg[field] = _SURROGATE_RE.sub('\ufffd', val)
                found = True
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            for key in ("name", "arguments"):
                val = fn.get(key)
                if isinstance(val, str) and _SURROGATE_RE.search(val):
                    fn[key] = _SURROGATE_RE.sub('\ufffd', val)
                    found = True
    if found:
        logger.debug("Sanitizer: replaced surrogate characters in messages")
    return found


def strip_reasoning_tags(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning> blocks from assistant content.

    Ported from Hermes _strip_think_blocks. These tags are internal
    reasoning that should not be shown to users or stored in transcripts.
    """
    if not text:
        return text
    text = _THINK_RE.sub('', text)
    text = _THINK_TAG_RE.sub('', text)
    return text.strip()
