"""Copilot ACP protocol helpers — message formatting, tool call extraction.

Extracted from copilot_client.py to keep modules under 450 lines.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_command() -> str:
    return (
        os.getenv("CAVEMAN_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> List[str]:
    raw = os.getenv("CAVEMAN_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def _jsonrpc_error(message_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    """Validate that a path is absolute and within the session cwd."""
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path '{resolved}' is outside the session cwd '{root}'."
        ) from exc
    return resolved


# ── Message formatting ─────────────────────────────────────────────────────

def _render_message_content(content: Any) -> str:
    """Render message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
) -> str:
    """Format conversation messages into a single prompt for ACP."""
    sections: List[str] = [
        "You are being used as the active ACP agent backend for Caveman.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, output tool calls using "
        "<tool_call>{...}</tool_call> blocks with JSON in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append({
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call>.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            label = "Tool"
        elif role in {"system", "user", "assistant"}:
            label = role.title()
        else:
            label = "Context"
        rendered = _render_message_content(message.get("content"))
        if rendered:
            transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


# ── Tool call extraction ───────────────────────────────────────────────────

def _extract_tool_calls_from_text(text: str) -> Tuple[List[SimpleNamespace], str]:
    """Extract tool calls from response text, return (calls, cleaned_text)."""
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: List[SimpleNamespace] = []
    consumed_spans: List[Tuple[int, int]] = []

    def _try_add(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted) + 1}"
        extracted.append(SimpleNamespace(
            id=call_id, call_id=call_id, response_item_id=None,
            type="function",
            function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
        ))

    # Try XML-wrapped blocks first
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add(m.group(1))
        consumed_spans.append((m.start(), m.end()))

    # Fallback: bare JSON (only if no XML blocks found)
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add(m.group(0))
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    # Remove consumed spans from text
    consumed_spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned

