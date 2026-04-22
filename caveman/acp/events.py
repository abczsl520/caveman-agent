"""ACP Event Callbacks — bridge AgentLoop events to ACP protocol updates.

Translates Caveman's internal events (tool calls, thinking, messages)
into ACP-compatible session updates for connected editors/clients.

Learned from: Hermes acp_adapter/events.py (175 lines)
Our version: Async-native, uses Caveman's EventBus.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "get_tool_kind",
    "make_tool_call_id",
    "build_tool_title",
    "ACPEvent",
    "ACPEventEmitter",
]


logger = logging.getLogger("caveman.acp")


# ── Tool Kind Mapping ──

TOOL_KIND_MAP: Dict[str, str] = {
    # File operations
    "file_read": "read",
    "file_write": "edit",
    "file_edit": "edit",
    "file_search": "search",
    # Execution
    "bash": "execute",
    "terminal": "execute",
    # Web
    "web_search": "fetch",
    "web_fetch": "fetch",
    # Browser
    "browser": "fetch",
    # Agent
    "acp_delegate": "execute",
    "moa": "execute",
    # Meta
    "_thinking": "think",
}


def get_tool_kind(tool_name: str) -> str:
    """Map Caveman tool name to ACP ToolKind."""
    return TOOL_KIND_MAP.get(tool_name, "other")


def make_tool_call_id() -> str:
    return f"tc-{uuid.uuid4().hex[:12]}"


def build_tool_title(tool_name: str, args: Dict[str, Any]) -> str:
    """Human-readable title for a tool call."""
    if tool_name == "bash":
        cmd = str(args.get("command", ""))
        return f"bash: {cmd[:77]}..." if len(cmd) > 80 else f"bash: {cmd}"
    if tool_name in ("file_read", "file_write", "file_edit"):
        return f"{tool_name}: {args.get('path', '?')}"
    if tool_name == "web_search":
        return f"search: {args.get('query', '?')}"
    if tool_name == "browser":
        return f"browser: {args.get('action', '?')}"
    return tool_name


# ── Event Data Structures ──

class ACPEvent:
    """A single ACP event for streaming to clients."""

    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.event_type, **self.data}

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


# ── Callback Factories ──

class ACPEventEmitter:
    """Collects and emits ACP events for a session.

    Used by the ACP server to bridge AgentLoop events to SSE/WebSocket.
    """

    def __init__(self, session_id: str, send_fn: Optional[Callable] = None):
        self.session_id = session_id
        self._send_fn = send_fn  # async callable(event: ACPEvent)
        self._events: List[ACPEvent] = []
        self._tool_call_ids: Dict[str, List[str]] = {}

    @property
    def events(self) -> List[ACPEvent]:
        return self._events

    async def _emit(self, event: ACPEvent) -> None:
        self._events.append(event)
        if self._send_fn:
            try:
                await self._send_fn(event)
            except Exception:
                logger.debug("Failed to send ACP event", exc_info=True)

    # ── Tool events ──

    async def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Emit tool call start. Returns the tool_call_id."""
        tc_id = make_tool_call_id()
        self._tool_call_ids.setdefault(tool_name, []).append(tc_id)

        await self._emit(ACPEvent("tool_call_start", {
            "tool_call_id": tc_id,
            "tool_name": tool_name,
            "kind": get_tool_kind(tool_name),
            "title": build_tool_title(tool_name, args),
            "arguments": args,
        }))
        return tc_id

    async def on_tool_complete(
        self, tool_name: str, result: Optional[str] = None,
    ) -> None:
        """Emit tool call completion."""
        ids = self._tool_call_ids.get(tool_name, [])
        tc_id = ids.pop(0) if ids else make_tool_call_id()
        if not ids:
            self._tool_call_ids.pop(tool_name, None)

        display = result or ""
        if len(display) > 5000:
            display = display[:4900] + f"\n... ({len(result)} chars, truncated)"

        await self._emit(ACPEvent("tool_call_complete", {
            "tool_call_id": tc_id,
            "tool_name": tool_name,
            "kind": get_tool_kind(tool_name),
            "status": "completed",
            "result": display,
        }))

    # ── Thinking events ──

    async def on_thinking(self, text: str) -> None:
        if text:
            await self._emit(ACPEvent("thinking", {"text": text}))

    # ── Message events ──

    async def on_message(self, text: str) -> None:
        if text:
            await self._emit(ACPEvent("message", {"text": text}))

    async def on_message_delta(self, delta: str) -> None:
        """Streaming text delta."""
        if delta:
            await self._emit(ACPEvent("message_delta", {"delta": delta}))

    # ── Status events ──

    async def on_status(self, status: str, detail: str = "") -> None:
        await self._emit(ACPEvent("status", {"status": status, "detail": detail}))

    async def on_error(self, error: str) -> None:
        await self._emit(ACPEvent("error", {"error": error}))

    async def on_done(self, result: Optional[str] = None) -> None:
        await self._emit(ACPEvent("done", {"result": result or ""}))
