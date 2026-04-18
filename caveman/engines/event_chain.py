"""Event chain — wires the inner flywheel: Shield → Nudge → Ripple → Lint → Recall.

PRD §5.3: "This is the Agent OS kernel heartbeat. Users don't see it,
but without it the outer flywheel can't turn."

PRD §4.3 Nudge triggers (event-driven, not polling):
  1. SHIELD_UPDATE → Nudge extraction (if due)
  2. TOOL_ERROR → immediate Nudge (error context is high-value)
  3. LOOP_END → final extraction (task completion)
  4. USER_PREFERENCE detected → extract user-type memory
  5. NEW_FACT detected → extract project-type memory

The chain:
  SHIELD_UPDATE → triggers Nudge extraction (if due)
  NUDGE_EXTRACT → Ripple propagation happens automatically via memory.store()
  LINT_SCAN     → demotes trust for flagged memories (already in lint.py)
  MEMORY_RECALL → confidence feedback (already in phase_finalize)

This module registers the event handlers that connect the engines.
Without it, each engine runs independently (the flywheel is broken).
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from caveman.events import EventBus, EventType, Event

if TYPE_CHECKING:
    from caveman.engines.manager import EngineSet

logger = logging.getLogger(__name__)


def wire_inner_flywheel(
    bus: EventBus,
    engines: "EngineSet",
    get_turns: Any = None,
    get_task: Any = None,
) -> list:
    """Register event handlers that connect engines into the inner flywheel.

    Args:
        bus: The event bus to subscribe to.
        engines: EngineSet with shield, nudge, ripple, lint, recall.
        get_turns: Callable that returns current conversation turns.
        get_task: Callable that returns current task string.

    Returns:
        List of (event_type, handler) tuples for cleanup.
    """
    handlers: list[tuple[EventType, Any]] = []

    # Chain 1: SHIELD_UPDATE → Nudge extraction
    # PRD §5.3: "Shield → 触发 Nudge + Ripple"
    # PRD §11.4: "事件驱动，不是定时轮询" — we trigger on Shield delta,
    # not on turn count. Minimum 3-turn gap prevents over-extraction.
    if engines.nudge:
        nudge = engines.nudge
        _last_nudge_turn = [0]  # mutable for closure

        async def _on_shield_update(event: Event) -> None:
            """Shield updated → Nudge extraction if enough new content."""
            turn_count = event.data.get("turn_count", 0)
            # Minimum gap: at least 3 turns since last nudge (not polling, just throttle)
            if turn_count - _last_nudge_turn[0] < 3:
                return
            turns = get_turns() if get_turns else []
            task = get_task() if get_task else ""
            if not turns:
                return
            try:
                created = await nudge.run(turns, task=task)
                if created:
                    _last_nudge_turn[0] = turn_count
                    await bus.emit(EventType.NUDGE_EXTRACT, {
                        "count": len(created),
                        "types": [e.memory_type.value for e in created],
                        "trigger": "shield_update",
                    }, source="nudge")
                    logger.info(
                        "Inner flywheel: Shield → Nudge extracted %d memories",
                        len(created),
                    )
            except Exception as e:
                logger.debug("Shield→Nudge chain failed: %s", e)

        bus.on(EventType.SHIELD_UPDATE, _on_shield_update)
        handlers.append((EventType.SHIELD_UPDATE, _on_shield_update))

    # Chain 2: NUDGE_EXTRACT → Ripple propagation
    # Note: Ripple is already wired into memory.store() via MemoryManager.set_ripple().
    # This handler is for logging/metrics only — the actual propagation happens
    # automatically when Nudge calls memory.store().
    if engines.ripple:
        def _on_nudge_extract(event: Event) -> None:
            """Log that Nudge→Ripple chain is active."""
            count = event.data.get("count", 0)
            trigger = event.data.get("trigger", "unknown")
            logger.debug(
                "Inner flywheel: Nudge(%s) → Ripple auto-propagated %d entries",
                trigger, count,
            )

        bus.on(EventType.NUDGE_EXTRACT, _on_nudge_extract)
        handlers.append((EventType.NUDGE_EXTRACT, _on_nudge_extract))

    # Chain 3: TOOL_ERROR → Nudge immediate extraction
    # PRD §4.3: "事件驱动，不是定时轮询" — errors are high-value events
    if engines.nudge:
        nudge_ref = engines.nudge

        async def _on_tool_error_nudge(event: Event) -> None:
            """Tool error → immediate Nudge extraction (error context is high-value)."""
            turns = get_turns() if get_turns else []
            task = get_task() if get_task else ""
            if not turns:
                return
            try:
                created = await nudge_ref.run(turns, task=task)
                if created:
                    await bus.emit(EventType.NUDGE_EXTRACT, {
                        "count": len(created),
                        "types": [e.memory_type.value for e in created],
                        "trigger": "tool_error",
                    }, source="nudge")
            except Exception as e:
                logger.debug("ToolError→Nudge chain failed: %s", e)

        bus.on(EventType.TOOL_ERROR, _on_tool_error_nudge)
        handlers.append((EventType.TOOL_ERROR, _on_tool_error_nudge))

    # Chain 4: MEMORY_STORE → Lint incremental check (if from nudge)
    # PRD §5.3: Nudge → Ripple → Lint feedback loop
    # When new memories are stored, Lint can do a quick validation
    if engines.lint:
        lint_ref = engines.lint

        async def _on_memory_store_lint(event: Event) -> None:
            """New memory stored → Lint can validate it."""
            source = event.data.get("source", "")
            if source != "nudge":
                return  # Only lint nudge-extracted memories
            # Don't run full scan, just note that new memories exist
            # The actual scan happens on session end or periodically
            logger.debug("Inner flywheel: new nudge memory → Lint will check on next scan")

        bus.on(EventType.MEMORY_STORE, _on_memory_store_lint)
        handlers.append((EventType.MEMORY_STORE, _on_memory_store_lint))

    logger.info(
        "Inner flywheel wired: %d event chains registered",
        len(handlers),
    )
    return handlers


def unwire_inner_flywheel(bus: EventBus, handlers: list) -> None:
    """Unsubscribe all inner flywheel handlers."""
    for event_type, handler in handlers:
        bus.off(event_type, handler)
