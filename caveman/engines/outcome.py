"""Outcome Signal Engine — score task completion, feed RL Router + memory trust.

After each task, analyzes the trajectory to determine success/failure,
then propagates that signal to:
  1. RL Router — update arm stats for matched skills
  2. Memory trust — boost trust for recalled memories that contributed to success
  3. EventBus — emit SKILL_OUTCOME for downstream consumers

This is the critical feedback loop that makes the flywheel self-improving:
  good outcome → boost skill/memory → better future selection → better outcomes

PRD §5.2 Ring 3: "Skills don't just get created — they improve."
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["OutcomeEngine"]


class OutcomeEngine:
    """Score task outcomes and propagate feedback signals."""

    def __init__(
        self,
        rl_router: Any = None,
        memory_manager: Any = None,
        bus: Any = None,
    ) -> None:
        self._router = rl_router
        self._memory = memory_manager
        self._bus = bus

    async def score_and_propagate(
        self,
        task: str,
        result: str,
        matched_skills: list | None = None,
        recalled_ids: list[str] | None = None,
        trajectory: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Score task outcome and propagate feedback to all subsystems.

        Returns:
            Dict with outcome, score, and propagation results.
        """
        from caveman.utils import detect_outcome
        outcome = detect_outcome(result)
        success = outcome == "success"
        score = {"success": 1.0, "partial": 0.5, "failure": 0.0}[outcome]

        report: dict[str, Any] = {
            "outcome": outcome,
            "score": score,
            "skills_updated": 0,
            "memories_boosted": 0,
        }

        # 1. Feed RL Router — update arm stats for matched skills
        if self._router and matched_skills:
            for skill in matched_skills:
                name = skill.name if hasattr(skill, "name") else str(skill)
                try:
                    self._router.update(name, success)
                    report["skills_updated"] += 1
                except Exception as e:
                    logger.debug("RL Router update failed for '%s': %s", name, e)

        # 2. Boost/penalize recalled memories based on outcome
        if self._memory and recalled_ids:
            for mid in recalled_ids:
                try:
                    await self._memory.feedback(mid, helpful=success)
                    report["memories_boosted"] += 1
                except Exception as e:
                    logger.debug("Memory feedback failed for '%s': %s", mid, e)

        # 3. Emit SKILL_OUTCOME event for downstream consumers
        if self._bus:
            try:
                from caveman.events import EventType
                await self._bus.emit(EventType.SKILL_OUTCOME, {
                    "task": task[:200],
                    "outcome": outcome,
                    "score": score,
                    "skills_updated": report["skills_updated"],
                    "memories_boosted": report["memories_boosted"],
                    "recalled_ids": recalled_ids or [],
                }, source="outcome")
            except Exception as e:
                logger.debug("SKILL_OUTCOME emit failed: %s", e)

        logger.info(
            "Outcome: %s (score=%.1f) — %d skills updated, %d memories boosted",
            outcome, score, report["skills_updated"], report["memories_boosted"],
        )
        return report
