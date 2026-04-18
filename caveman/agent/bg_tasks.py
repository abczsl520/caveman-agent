"""Background task helpers for AgentLoop.

Extracted from loop.py to keep it under 400 lines (NFR-502).
All methods here are called by AgentLoop but don't need direct
access to the main run() pipeline state.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskMixin:
    """Mixin providing background task management for AgentLoop.

    Expects the host class to have:
      - _bg_tasks: set[asyncio.Task]
      - memory_manager, skill_manager, trajectory_recorder
      - _llm_fn, _nudge, _lint, engine_flags, bus
    """

    def _safe_bg(self, coro) -> None:
        """Launch a background coroutine with tracking and error logging."""
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._bg_tasks.discard(t)
            if not t.cancelled() and t.exception():
                logger.warning("BG task failed: %s", t.exception())

        task.add_done_callback(_on_done)

    async def drain_background(self, timeout: float = 10.0) -> None:
        """Wait for all background tasks to complete (for clean shutdown)."""
        if not self._bg_tasks:
            return
        pending = list(self._bg_tasks)
        logger.debug("Draining %d background tasks...", len(pending))
        done, not_done = await asyncio.wait(pending, timeout=timeout)
        for t in not_done:
            t.cancel()
        # Await cancellation to avoid "Task was destroyed but it is pending" warnings
        if not_done:
            await asyncio.wait(not_done, timeout=2.0)

    async def _boost_trust(self, memory_ids: list[str]) -> None:
        """Boost trust_score for memories that contributed to success."""
        backend = getattr(self.memory_manager, '_backend', None)
        if backend and hasattr(backend, 'mark_helpful'):
            for mid in memory_ids:
                try:
                    await backend.mark_helpful(mid, helpful=True)
                except Exception as e:
                    logger.debug("Suppressed in bg_tasks: %s", e)

    async def _end_nudge(self, task: str) -> None:
        if not self.engine_flags.is_enabled("nudge"):
            return
        # PRD §4.3: "任务完成 → 提取 procedural 记忆"
        # Always extract at loop end — this is event-driven (task completion),
        # not polling. The dedup layer prevents duplicate memories.
        try:
            created = await self._nudge.run(
                self.trajectory_recorder.to_sharegpt()[-20:], task=task,
            )
            if created:
                from caveman.events import EventType
                await self.bus.emit(EventType.NUDGE_EXTRACT, {
                    "count": len(created), "trigger": "loop_end",
                }, source="nudge")
        except Exception as e:
            logger.warning("End nudge failed: %s", e)

    async def _bg_skill_nudge(self) -> None:
        try:
            traj = self.trajectory_recorder.to_sharegpt()[-20:]
            await self.skill_manager.auto_create(traj, llm_fn=self._llm_fn)
        except Exception as e:
            logger.debug("Suppressed in _bg_skill_nudge: %s", e)

    async def _on_tool_error_nudge(self, event) -> None:
        if not self.engine_flags.is_enabled("nudge"):
            return
        try:
            turns = self.trajectory_recorder.to_sharegpt()[-10:]
            created = await self._nudge.run(turns, task=self._nudge_task_ref)
            if created:
                from caveman.events import EventType
                await self.bus.emit(EventType.NUDGE_EXTRACT, {
                    "count": len(created), "trigger": "tool_error",
                }, source="nudge")
        except Exception as e:
            logger.debug("Tool-error nudge failed: %s", e)

    async def _run_lint(self) -> None:
        try:
            await self._lint.scan()
            logger.info("Lint scan completed (background)")
        except Exception as e:
            logger.warning("Lint scan failed: %s", e)

    async def _check_save_skill(self, task: str) -> None:
        """Check if completed task should become a skill."""
        try:
            traj = self.trajectory_recorder.to_sharegpt()
            if len(traj) >= 4:
                created = await self.skill_manager.auto_create(
                    traj, task=task, llm_fn=self._llm_fn,
                )
                if created:
                    logger.info("Saved new skill: %s", created.name)
        except Exception as e:
            logger.debug("Skill save check failed: %s", e)
