"""Tool execution helpers — extracted from loop.py."""
from __future__ import annotations

import json
import logging
from typing import Any

from caveman.agent.context import AgentContext
from caveman.agent.display import show_tool_call, show_tool_result, show_skill_nudge
from caveman.events import EventBus, EventType
from caveman.security.permissions import PermissionManager
from caveman.tools.registry import ToolRegistry
from caveman.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)

# Declarative permission mapping — tools declare their risk level
TOOL_PERMISSIONS = {
    "bash": "bash_write",
    "file_read": "file_read",
    "file_write": "file_write",
    "file_edit": "file_write",
    "file_list": "file_read",
    "web_search": "web_search",
    "browser": "browser",
}


async def execute_tool(
    name: str,
    args: dict,
    call_id: str,
    tool_registry: ToolRegistry,
    permission_manager: PermissionManager,
) -> dict:
    """Execute a single tool call with permission check."""
    from caveman.result import Ok, Err

    show_tool_call(name, args)

    action = TOOL_PERMISSIONS.get(name, "bash_write")
    approved = await permission_manager.request(action, f"{name}({args})")
    if not approved:
        result = Err(f"Not approved: {name}")
        show_tool_result(name, result.error, False)
        return {
            "type": "tool_result", "tool_use_id": call_id,
            "content": result.to_content(), "is_error": True,
        }

    try:
        raw = await tool_registry.dispatch(name, args)
        # Handle ToolResult objects
        from caveman.result import ToolResult
        if isinstance(raw, ToolResult):
            s = raw.to_content()
            is_error = raw.is_error
        elif isinstance(raw, dict):
            s = json.dumps(raw)
            is_error = "error" in raw or raw.get("success") is False
        else:
            s = str(raw)
            is_error = False
        show_tool_result(name, s[:200], not is_error)
        # Truncate large tool outputs to prevent context explosion
        # 30K chars ≈ 7500 tokens — leaves room for other context
        if len(s) > 30_000:
            s = s[:30_000] + f"\n... (truncated, {len(s)} chars total)"
        return {
            "type": "tool_result", "tool_use_id": call_id,
            "content": s, **({"is_error": True} if is_error else {}),
        }
    except Exception as e:
        result = Err(f"{type(e).__name__}: {e}")
        show_tool_result(name, result.error, False)
        return {
            "type": "tool_result", "tool_use_id": call_id,
            "content": result.to_content(), "is_error": True,
        }


async def phase_tool_execution(
    context: AgentContext,
    tool_calls: list,
    tool_registry: ToolRegistry,
    permission_manager: PermissionManager,
    trajectory_recorder: TrajectoryRecorder,
    bus: EventBus,
    tool_call_count: int,
    bg_skill_nudge_fn=None,
) -> int:
    """Execute all tool calls, record results. Returns updated tool_call_count.

    bg_skill_nudge_fn: async callable that the caller (AgentLoop) wraps
    via _safe_bg for proper lifecycle tracking. We just await it inline
    or let the caller schedule it — no orphan ensure_future here.
    """
    results = []
    for tc in tool_calls:
        await bus.emit(EventType.TOOL_CALL, {
            "name": tc["name"], "call_id": tc["id"],
            "args_keys": list(tc["input"].keys()),
        }, source="tool")

        r = await execute_tool(
            tc["name"], tc["input"], tc["id"],
            tool_registry, permission_manager,
        )
        results.append(r)
        tool_call_count += 1

        is_error = r.get("is_error", False)
        event_type = EventType.TOOL_ERROR if is_error else EventType.TOOL_RESULT
        await bus.emit(event_type, {
            "name": tc["name"], "call_id": tc["id"],
            "is_error": is_error,
            "result_len": len(r.get("content", "")),
        }, source="tool")

    context.add_message("user", results)
    await trajectory_recorder.record_turn("tool", json.dumps(results))

    # Skill nudge every 10 tool calls — signal caller to schedule it
    if tool_call_count % 10 == 0 and bg_skill_nudge_fn:
        show_skill_nudge()
        await bus.emit(EventType.SKILL_NUDGE, {
            "tool_calls": tool_call_count,
        }, source="skill")
        # Return a sentinel so caller can schedule via _safe_bg
        # The caller (AgentLoop.run) already calls _bg_skill_nudge
        # through _safe_bg after phase_tool_execution returns.

    return tool_call_count
