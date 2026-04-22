"""Tool execution helpers — extracted from loop.py."""
from __future__ import annotations

import asyncio
import time
import json
import logging

from caveman.agent.context import AgentContext
from caveman.agent.display import show_tool_call, show_tool_result, show_skill_nudge
from caveman.events import EventBus, EventType
from caveman.security.permissions import PermissionManager
from caveman.tools.registry import ToolRegistry
from caveman.trajectory.recorder import TrajectoryRecorder

__all__ = [
    "TOOL_PERMISSIONS",
    "execute_tool",
    "phase_tool_execution",
]


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

# Tools that are safe to run in parallel (no side effects on each other)
_PARALLELIZABLE = frozenset({
    "file_read", "file_list", "web_search", "memory_search", "memory_recall",
    "url_fetch", "vision", "transcribe",
})

# Tools that must run sequentially (side effects, state mutation)
_SEQUENTIAL = frozenset({
    "bash", "file_write", "file_edit", "gateway_send", "progress",
    "delegate", "coding_agent",
})


async def execute_tool(
    name: str,
    args: dict,
    call_id: str,
    tool_registry: ToolRegistry,
    permission_manager: PermissionManager,
) -> dict:
    """Execute a single tool call with permission check."""
    from caveman.result import Err

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
        # Cache check for idempotent tools (file_read, web_search, etc.)
        from caveman.tools.builtin.tool_result_storage import ToolResultStore
        _cache = ToolResultStore()
        _CACHEABLE_TOOLS = {"file_read", "file_search", "analyze_document"}
        cached = None
        if name in _CACHEABLE_TOOLS:
            cached = _cache.get(name, args)
        if cached is not None:
            show_tool_result(name, cached[:200], True)
            return {
                "type": "tool_result", "tool_use_id": call_id,
                "content": cached, "is_error": False,
            }

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
        # Layer 2: Persist large tool outputs to file, keep preview in context
        from caveman.tools.result_storage import persist_tool_result
        s = persist_tool_result(s, name, call_id)
        # Layer 2b: Cache idempotent tool results
        if name in _CACHEABLE_TOOLS and not is_error:
            _cache.store(name, args, s)
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


# Tools that can be parallelized if they target different paths
_PATH_SCOPED = frozenset({"file_read", "file_list", "file_write", "file_edit"})


def _extract_path(name: str, args: dict) -> str | None:
    """Extract the file path from a tool call's arguments."""
    if name not in _PATH_SCOPED:
        return None
    raw = args.get("path") or args.get("file_path") or args.get("input", {}).get("path")
    return str(raw) if raw else None


def _paths_overlap(a: str, b: str) -> bool:
    """Check if two paths might refer to the same file/subtree."""
    from pathlib import Path
    pa, pb = Path(a).resolve(), Path(b).resolve()
    parts_a, parts_b = pa.parts, pb.parts
    common = min(len(parts_a), len(parts_b))
    return parts_a[:common] == parts_b[:common]


def _deterministic_call_id(name: str, args: dict, index: int = 0) -> str:
    """Generate deterministic call_id when API doesn't provide one.

    Prevents prompt cache invalidation — random UUIDs make every call unique.
    Ported from Hermes.
    """
    import hashlib
    seed = f"{name}:{json.dumps(args, sort_keys=True)}:{index}"
    return f"call_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"


def _ensure_call_id(tc: dict, index: int) -> str:
    """Get or generate a stable tool call ID."""
    cid = tc.get("id")
    if cid:
        return cid
    return _deterministic_call_id(tc.get("name", ""), tc.get("input", {}), index)


def _can_parallelize(tool_calls: list) -> bool:
    """Check if all tool calls in a batch can run in parallel.

    Upgraded from simple set check to path-aware overlap detection (Hermes pattern).
    """
    if len(tool_calls) <= 1:
        return False
    names = [tc.get("name", "") for tc in tool_calls]
    # Any sequential tool → no parallel
    if any(n in _SEQUENTIAL for n in names):
        return False
    # All in safe set → parallel
    if all(n in _PARALLELIZABLE for n in names):
        return True
    # Path-scoped tools: check for overlap
    if all(n in _PARALLELIZABLE | _PATH_SCOPED for n in names):
        paths: list[str] = []
        for tc in tool_calls:
            p = _extract_path(tc.get("name", ""), tc.get("input", {}))
            if p:
                if any(_paths_overlap(p, existing) for existing in paths):
                    return False
                paths.append(p)
        return True
    return False


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
    """Execute tool calls (parallel when safe). Returns updated tool_call_count."""
    # Deduplicate tool calls within this turn
    from caveman.providers.message_sanitizer import deduplicate_tool_calls
    tool_calls = deduplicate_tool_calls(tool_calls)

    # Cap delegate calls per turn (prevent model from spawning too many sub-agents)
    delegate_count = sum(1 for tc in tool_calls if tc.get("name") == "delegate")
    if delegate_count > 3:
        kept = 0
        capped = []
        for tc in tool_calls:
            if tc.get("name") == "delegate":
                if kept < 3:
                    capped.append(tc)
                    kept += 1
            else:
                capped.append(tc)
        logger.warning("Capped %d excess delegate calls to 3", delegate_count - 3)
        tool_calls = capped

    parallel = _can_parallelize(tool_calls)

    if parallel:
        # Emit all events first, then execute in parallel
        for i, tc in enumerate(tool_calls):
            await bus.emit(EventType.TOOL_CALL, {
                "name": tc["name"], "call_id": _ensure_call_id(tc, i),
                "args_keys": list(tc["input"].keys()),
            }, source="tool")

        tasks = [
            execute_tool(tc["name"], tc["input"], _ensure_call_id(tc, i),
                         tool_registry, permission_manager)
            for i, tc in enumerate(tool_calls)
        ]
        results = await asyncio.gather(*tasks)
        results = list(results)
        tool_call_count += len(tool_calls)
        logger.info("Parallel execution: %d tools (%s)",
                     len(tool_calls), ", ".join(tc["name"] for tc in tool_calls))

        for i, (tc, r) in enumerate(zip(tool_calls, results)):
            is_error = r.get("is_error", False)
            event_type = EventType.TOOL_ERROR if is_error else EventType.TOOL_RESULT
            await bus.emit(event_type, {
                "name": tc["name"], "call_id": _ensure_call_id(tc, i),
                "is_error": is_error,
                "result_len": len(r.get("content", "")),
            }, source="tool")
    else:
        # Sequential execution (default for side-effect tools)
        results = []
        for i, tc in enumerate(tool_calls):
            await bus.emit(EventType.TOOL_CALL, {
                "name": tc["name"], "call_id": _ensure_call_id(tc, i),
                "args_keys": list(tc["input"].keys()),
            }, source="tool")

            _t0 = time.perf_counter()
            r = await execute_tool(
                tc["name"], tc["input"], _ensure_call_id(tc, i),
                tool_registry, permission_manager,
            )
            _dur_ms = round((time.perf_counter() - _t0) * 1000, 1)
            results.append(r)
            tool_call_count += 1

            is_error = r.get("is_error", False)
            event_type = EventType.TOOL_ERROR if is_error else EventType.TOOL_RESULT
            await bus.emit(event_type, {
                "name": tc["name"], "call_id": _ensure_call_id(tc, i),
                "is_error": is_error,
                "result_len": len(r.get("content", "")),
                "duration_ms": _dur_ms,
            }, source="tool")

    # Layer 3: Enforce per-turn aggregate budget
    from caveman.tools.result_storage import enforce_turn_budget
    results = enforce_turn_budget(results)

    context.add_message("user", results)
    # Record each tool call as function_call (request) + function_response (result)
    # in ShareGPT format. This fixes the role mismatch bug where "tool" role
    # was never counted by the recorder (which checks for "function_call").
    for tc, r in zip(tool_calls, results):
        call_payload = {"name": tc["name"], "arguments": tc["input"]}
        await trajectory_recorder.record_turn("function_call", json.dumps(call_payload))
        await trajectory_recorder.record_turn("function_response", json.dumps(r))

    if tool_call_count % 10 == 0 and bg_skill_nudge_fn:
        show_skill_nudge()
        await bus.emit(EventType.SKILL_NUDGE, {
            "tool_calls": tool_call_count,
        }, source="skill")

    return tool_call_count
