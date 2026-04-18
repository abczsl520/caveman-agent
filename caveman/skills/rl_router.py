"""Skill RL Router — reinforcement learning-based skill selection.

Replaces semantic similarity matching with a learned router that
improves over time based on success/failure feedback.

Phase 1 (this file): Multi-armed bandit (Thompson Sampling)
  - Each skill is an "arm" with Beta(success, failure) distribution
  - Sample from each arm, pick highest
  - Update on outcome
  - Cold start: fall back to keyword matching

Phase 2 (future): Contextual bandit with task embeddings
Phase 3 (future): Full RL with state = (task, memory, history)
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Beta distribution parameters for a skill arm."""
    alpha: float = 1.0  # successes + 1 (prior)
    beta: float = 1.0   # failures + 1 (prior)
    total: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta distribution."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1
        else:
            self.beta += 1
        self.total += 1


class SkillRLRouter:
    """RL-based skill router using Thompson Sampling.

    Usage:
        router = SkillRLRouter()
        skill_name = router.select(task, available_skills)
        # ... execute skill ...
        router.update(skill_name, success=True)
    """

    def __init__(self, state_path: Path | None = None):
        self._arms: dict[str, ArmStats] = {}
        self._state_path = state_path
        if state_path and state_path.exists():
            self._load_state()

    def select(
        self,
        task: str,
        available_skills: list[str],
        explore_rate: float = 0.1,
    ) -> str | None:
        """Select best skill for a task.

        Args:
            task: Task description
            available_skills: List of skill names to choose from
            explore_rate: Probability of random exploration

        Returns:
            Selected skill name, or None if no skills available
        """
        if not available_skills:
            return None

        # Exploration: random selection
        if random.random() < explore_rate:
            choice = random.choice(available_skills)
            logger.debug("RL Router: exploring with '%s'", choice)
            return choice

        # Exploitation: Thompson Sampling
        best_score = -1.0
        best_skill = available_skills[0]

        for skill_name in available_skills:
            arm = self._arms.get(skill_name)
            if arm is None:
                # Cold start: optimistic prior (encourage trying new skills)
                score = random.betavariate(1.0, 1.0)
            else:
                score = arm.sample()

            if score > best_score:
                best_score = score
                best_skill = skill_name

        logger.debug("RL Router: selected '%s' (score=%.3f)", best_skill, best_score)
        return best_skill

    def update(self, skill_name: str, success: bool) -> None:
        """Update arm statistics after skill execution."""
        if skill_name not in self._arms:
            self._arms[skill_name] = ArmStats()
        self._arms[skill_name].update(success)

        if self._state_path:
            self._save_state()

    def get_stats(self) -> dict[str, dict]:
        """Get all arm statistics."""
        return {
            name: {
                "mean": arm.mean,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "total": arm.total,
            }
            for name, arm in self._arms.items()
        }

    def get_rankings(self) -> list[tuple[str, float]]:
        """Get skills ranked by estimated success rate."""
        rankings = [(name, arm.mean) for name, arm in self._arms.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _save_state(self) -> None:
        """Persist arm stats to disk."""
        if not self._state_path:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {"alpha": arm.alpha, "beta": arm.beta, "total": arm.total}
            for name, arm in self._arms.items()
        }
        self._state_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _load_state(self) -> None:
        """Load arm stats from disk."""
        if not self._state_path or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            for name, stats in data.items():
                self._arms[name] = ArmStats(
                    alpha=stats.get("alpha", 1.0),
                    beta=stats.get("beta", 1.0),
                    total=stats.get("total", 0),
                )
        except Exception as e:
            logger.warning("Failed to load RL router state: %s", e)
