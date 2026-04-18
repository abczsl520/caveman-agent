"""Engine Manager — unified lifecycle for all cognitive engines.

Replaces the scattered engine initialization in AgentLoop.__init__
and factory.py with a single manager that handles:
  - Engine creation from config
  - Dependency wiring (memory_manager, llm_fn, etc.)
  - LLM Scheduler integration (priority-based rate limiting)
  - Enable/disable via EngineFlags
  - Health checks

Engines managed: Shield, Nudge, Reflect, Ripple, Lint, Recall
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from caveman.engines.flags import EngineFlags

logger = logging.getLogger(__name__)


@dataclass
class EngineSet:
    """Container for all engine instances."""
    shield: Any = None
    nudge: Any = None
    reflect: Any = None
    ripple: Any = None
    lint: Any = None
    recall: Any = None
    scheduler: Any = None  # LLMScheduler instance (if created)

    def active_names(self) -> list[str]:
        """Return names of engines that are instantiated."""
        return [
            name for name in ("shield", "nudge", "reflect", "ripple", "lint", "recall")
            if getattr(self, name) is not None
        ]


def _make_scheduled_llm_fn(scheduler, caller: str, priority):
    """Create a scheduled llm_fn wrapper for an engine.

    The engine calls `await llm_fn(prompt)` as before, but the call
    goes through the LLMScheduler with the engine's priority level.
    Shield (P0) always gets served before Lint (P3).
    """
    async def scheduled_fn(prompt: str) -> str:
        return await scheduler.request(caller, priority, prompt)
    return scheduled_fn


class EngineManager:
    """Unified engine lifecycle manager.

    Usage:
        manager = EngineManager(flags, memory_manager, skill_manager, llm_fn)
        engines = manager.create_all()
        # engines.shield, engines.recall, etc. are ready to use
    """

    def __init__(
        self,
        flags: EngineFlags,
        memory_manager: Any,
        skill_manager: Any = None,
        llm_fn: Any = None,
        session_id: str | None = None,
        enable_scheduler: bool = True,
        max_rpm: int = 30,
    ) -> None:
        self._flags = flags
        self._memory = memory_manager
        self._skills = skill_manager
        self._llm_fn = llm_fn
        self._session_id = session_id
        self._enable_scheduler = enable_scheduler
        self._max_rpm = max_rpm
        self._engines = EngineSet()

    @property
    def engines(self) -> EngineSet:
        return self._engines

    def create_all(self) -> EngineSet:
        """Create all enabled engines with proper dependency wiring.

        If llm_fn is provided and scheduler is enabled, each engine gets
        a priority-wrapped llm_fn that goes through the LLMScheduler.
        Shield (CRITICAL) > Reflect (HIGH) > Nudge/Ripple (NORMAL) > Lint (LOW).
        """
        import uuid

        sid = self._session_id or uuid.uuid4().hex[:12]

        # Create LLM Scheduler for priority-based rate limiting
        scheduler = None
        if self._llm_fn and self._enable_scheduler:
            from caveman.engines.scheduler import LLMScheduler, Priority
            scheduler = LLMScheduler(
                llm_fn=self._llm_fn, max_rpm=self._max_rpm,
            )
            self._engines.scheduler = scheduler

            # Create priority-wrapped llm_fn for each engine
            shield_llm = _make_scheduled_llm_fn(scheduler, "shield", Priority.CRITICAL)
            nudge_llm = _make_scheduled_llm_fn(scheduler, "nudge", Priority.LOW)
            reflect_llm = _make_scheduled_llm_fn(scheduler, "reflect", Priority.HIGH)
            ripple_llm = _make_scheduled_llm_fn(scheduler, "ripple", Priority.NORMAL)
            lint_llm = _make_scheduled_llm_fn(scheduler, "lint", Priority.LOW)
        else:
            # No scheduler — all engines use raw llm_fn
            shield_llm = nudge_llm = reflect_llm = ripple_llm = lint_llm = self._llm_fn

        # Shield — always created (core to agent identity)
        from caveman.engines.shield import CompactionShield
        self._engines.shield = CompactionShield(
            session_id=sid, llm_fn=shield_llm,
        )

        # Recall — always created (needed for context restoration)
        from caveman.engines.recall import RecallEngine
        self._engines.recall = RecallEngine(
            memory_manager=self._memory,
            retrieval_log=getattr(self._memory, '_retrieval_log', None),
        )

        # Nudge — extracts knowledge from conversations
        if self._flags.is_enabled("nudge"):
            from caveman.memory.nudge import MemoryNudge
            self._engines.nudge = MemoryNudge(
                memory_manager=self._memory,
                llm_fn=nudge_llm,
                interval=10, first_nudge=3,
            )

        # Reflect — post-task skill evolution
        if self._flags.is_enabled("reflect"):
            from caveman.engines.reflect import ReflectEngine
            self._engines.reflect = ReflectEngine(
                skill_manager=self._skills, llm_fn=reflect_llm,
            )

        # Ripple — knowledge propagation on write
        if self._flags.is_enabled("ripple"):
            from caveman.engines.ripple import RippleEngine
            self._engines.ripple = RippleEngine(
                memory_manager=self._memory, llm_fn=ripple_llm,
            )
            # Wire ripple into memory manager for write-time propagation
            self._memory.set_ripple(self._engines.ripple)

        # Lint — knowledge quality audit
        if self._flags.is_enabled("lint"):
            from caveman.engines.lint import LintEngine
            self._engines.lint = LintEngine(
                memory_manager=self._memory, llm_fn=lint_llm,
            )

        active = self._engines.active_names()
        sched_status = "with scheduler" if scheduler else "no scheduler"
        logger.info(
            "Engines initialized (%s): %s", sched_status, ", ".join(active)
        )
        return self._engines
