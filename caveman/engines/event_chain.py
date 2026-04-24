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
from datetime import datetime
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
    memory_manager: Any = None,
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

    # Chain 4: MEMORY_STORE → Lint single-entry check (if from nudge)
    # PRD §5.3: Nudge → Ripple → Lint feedback loop
    # When nudge stores a new memory, lint it immediately to catch low-quality entries.
    if engines.lint:
        lint_ref = engines.lint

        async def _on_memory_store_lint(event: Event) -> None:
            """New nudge memory stored → lint it immediately."""
            source = event.data.get("source", "")
            if source != "nudge":
                return  # Only lint nudge-extracted memories
            memory_id = event.data.get("memory_id", "")
            content = event.data.get("content", "")
            mem_type = event.data.get("type", "semantic")
            if not memory_id or not content:
                return
            try:
                from caveman.memory.types import MemoryEntry, MemoryType
                entry = MemoryEntry(
                    id=memory_id, content=content,
                    memory_type=MemoryType(mem_type) if isinstance(mem_type, str) else mem_type,
                    created_at=datetime.now(),
                    metadata=event.data.get("metadata", {}),
                )
                report = await lint_ref.lint_single(entry)
                if report.issues:
                    logger.info(
                        "Inner flywheel: Nudge → Lint caught %d issues in new memory %s",
                        len(report.issues), memory_id[:8],
                    )
            except Exception as e:
                logger.debug("MEMORY_STORE→Lint chain failed: %s", e)

        bus.on(EventType.MEMORY_STORE, _on_memory_store_lint)
        handlers.append((EventType.MEMORY_STORE, _on_memory_store_lint))


    # Chain 5: LOOP_END → Outcome scoring + RL Router feedback
    # PRD §5.2 Ring 3: "Skills don't just get created — they improve."
    # This is the critical feedback signal that closes the outer flywheel.
    if engines.outcome:
        outcome_ref = engines.outcome

        async def _on_loop_end_outcome(event: Event) -> None:
            """Task completed → score outcome, feed RL Router + memory trust."""
            task = event.data.get("task", "")
            result = event.data.get("result", "")
            recalled_ids = event.data.get("recalled_ids", [])
            matched_skills = event.data.get("matched_skills", [])
            if not task:
                return
            try:
                await outcome_ref.score_and_propagate(
                    task=task, result=result,
                    matched_skills=matched_skills,
                    recalled_ids=recalled_ids,
                )
            except Exception as e:
                logger.warning("LOOP_END→Outcome chain failed: %s", e)

        bus.on(EventType.LOOP_END, _on_loop_end_outcome)
        handlers.append((EventType.LOOP_END, _on_loop_end_outcome))

    # Chain 6: LOOP_END → Reflect (post-task skill evolution)
    # PRD §5.3: Reflect analyzes what worked/failed and evolves skills.
    # Previously in post_task_engines (gated by engine_flags), now event-driven.
    if engines.reflect:
        reflect_ref = engines.reflect

        async def _on_loop_end_reflect(event: Event) -> None:
            """Task completed → Reflect on what worked/failed."""
            task = event.data.get("task", "")
            result = event.data.get("result", "")
            if not task:
                return
            turns = get_turns() if get_turns else []
            if len(turns) < 2:
                return  # Not enough context to reflect on
            try:
                await reflect_ref.reflect(task, turns, task_result=result)
            except Exception as e:
                logger.warning("LOOP_END→Reflect chain failed: %s", e)

        bus.on(EventType.LOOP_END, _on_loop_end_reflect)
        handlers.append((EventType.LOOP_END, _on_loop_end_reflect))


    # Chain 7: NUDGE_EXTRACT → Wiki auto-trigger
    # PRD §5.2 Ring 5: "Wiki is the crystallized knowledge layer."
    # When enough memories accumulate, auto-compile wiki for structured knowledge.
    try:
        from caveman.wiki.auto_trigger import WikiAutoTrigger
        from caveman.wiki.compiler import WikiCompiler
        _wiki_trigger = WikiAutoTrigger(compiler=WikiCompiler(), memory_manager=memory_manager, threshold=5, cooldown=300)

        async def _on_nudge_wiki(event: Event) -> None:
            """Nudge extracted memories → check if wiki should compile."""
            count = event.data.get("count", 1)
            _wiki_trigger.on_nudge_extract(count)

        bus.on(EventType.NUDGE_EXTRACT, _on_nudge_wiki)
        handlers.append((EventType.NUDGE_EXTRACT, _on_nudge_wiki))
    except Exception as e:
        logger.debug("Wiki auto-trigger unavailable: %s", e)


    # Chain 8: LOOP_END → Memory Decay (periodic trust erosion)
    # PRD §5.2 Ring 2: "Memories that aren't used should fade."
    # Run decay every N tasks to prevent stale memories from polluting retrieval.
    _decay_task_count = [0]
    _DECAY_INTERVAL = 10  # run decay every 10 tasks
    # Reuse single instance to preserve internal state across invocations
    _decay_instance = [None]

    async def _on_loop_end_decay(event: Event) -> None:
        """Periodically run memory decay after N tasks."""
        _decay_task_count[0] += 1
        if _decay_task_count[0] < _DECAY_INTERVAL:
            return
        _decay_task_count[0] = 0
        try:
            from caveman.memory.decay import MemoryDecay
            if _decay_instance[0] is None:
                _decay_instance[0] = MemoryDecay()
            result = _decay_instance[0].run()
            if result.memories_decayed > 0 or result.memories_pruned > 0:
                logger.info(
                    "Memory decay: %d decayed, %d pruned",
                    result.memories_decayed, result.memories_pruned,
                )
        except Exception as e:
            logger.warning("LOOP_END→Decay failed: %s", e)

    bus.on(EventType.LOOP_END, _on_loop_end_decay)
    handlers.append((EventType.LOOP_END, _on_loop_end_decay))

    logger.info(
        "Inner flywheel wired: %d event chains registered",
        len(handlers),
    )
    return handlers


def unwire_inner_flywheel(bus: EventBus, handlers: list) -> None:
    """Unsubscribe all inner flywheel handlers."""
    for event_type, handler in handlers:
        bus.off(event_type, handler)
