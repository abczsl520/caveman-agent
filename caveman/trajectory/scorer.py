# DEPRECATED: Not wired into runtime. See Round 38 audit.
"""Trajectory quality scorer — automatic quality assessment for training data.

Combines heuristic scoring (always available) with optional LLM judge.
Quality score determines if a trajectory enters the SFT training set.

Scoring dimensions:
  - Completeness: Did the task get done?
  - Tool usage: Demonstrates capability
  - Error handling: Recovered from errors gracefully
  - Efficiency: Reasonable number of turns
  - Verification: Passed verification checks (if available)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class TrajectoryScorer:
    """Auto-score trajectories for training data quality.

    Usage:
        scorer = TrajectoryScorer(llm_fn=my_llm)
        score = await scorer.score(trajectory_data)
        # score is 0.0-1.0
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
        min_quality: float = 0.7,
    ):
        self.llm_fn = llm_fn
        self.min_quality = min_quality

    async def score(self, trajectory: dict) -> float:
        """Score a trajectory. Returns 0.0-1.0."""
        heuristic = self._score_heuristic(trajectory)

        if self.llm_fn:
            llm_score = await self._score_llm(trajectory)
            if llm_score is not None:
                # Weighted: 40% heuristic + 60% LLM
                return heuristic * 0.4 + llm_score * 0.6

        return heuristic

    def _score_heuristic(self, trajectory: dict) -> float:
        """Rule-based quality scoring."""
        convs = trajectory.get("conversations", [])
        meta = trajectory.get("metadata", {})

        if not convs:
            return 0.0

        score = 0.5  # baseline

        # Completeness: last turn is assistant with substantial content
        if convs and convs[-1].get("from") == "gpt":
            last_len = len(convs[-1].get("value", ""))
            if last_len > 50:
                score += 0.15
            elif last_len > 10:
                score += 0.05

        # Tool usage
        tool_calls = meta.get("tool_calls", 0)
        if tool_calls > 0:
            score += min(0.15, tool_calls * 0.03)

        # Multi-turn depth
        turn_count = len(convs)
        if turn_count >= 6:
            score += 0.1
        elif turn_count >= 4:
            score += 0.05

        # Error handling: errors exist but task still completed
        errors = meta.get("errors", 0)
        if errors > 0 and convs[-1].get("from") == "gpt":
            score += 0.05  # Bonus: recovered from errors
        elif errors > 0:
            error_ratio = errors / max(turn_count, 1)
            score -= min(error_ratio * 0.3, 0.2)

        # Verification result (if available)
        verification = meta.get("verification")
        if verification == "pass":
            score += 0.1
        elif verification == "fail":
            score -= 0.15

        # Duration sanity: not too fast (trivial) or too slow (stuck)
        duration = meta.get("duration_seconds", 0)
        if 10 < duration < 600:
            score += 0.05
        elif duration > 1800:
            score -= 0.05

        return max(0.0, min(1.0, score))

    async def _score_llm(self, trajectory: dict) -> float | None:
        """LLM-based quality assessment."""
        convs = trajectory.get("conversations", [])
        task = trajectory.get("task", "unknown")

        # Build summary for LLM (limit context)
        summary_turns = []
        for t in convs[:10]:  # First 10 turns
            role = t.get("from", "?")
            value = t.get("value", "")[:200]
            summary_turns.append(f"[{role}] {value}")
        conv_text = "\n".join(summary_turns)

        prompt = f"""Rate this agent interaction for training data quality (0-10).

Task: {task}
Turns: {len(convs)}

Conversation (first 10 turns, truncated):
{conv_text}

Criteria:
- Did the agent complete the task?
- Was the approach efficient?
- Did it handle errors well?
- Is this a good example for training?

Respond with just a number 0-10."""

        try:
            response = await self.llm_fn(prompt)
            import re
            match = re.match(r'(\d+)', response.strip())
            if match:
                return min(int(match.group(1)) / 10.0, 1.0)
        except Exception as e:
            logger.warning("LLM trajectory scoring failed: %s", e)
        return None

    async def score_batch(self, trajectory_dir: Path) -> dict:
        """Score all trajectories in a directory."""
        results = {"scored": 0, "above_threshold": 0, "below_threshold": 0, "scores": []}

        for p in sorted(Path(trajectory_dir).glob("*.json")):
            if p.name.startswith("training_"):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                s = await self.score(data)
                data.setdefault("metadata", {})["quality_score"] = s
                p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

                results["scored"] += 1
                results["scores"].append(s)
                if s >= self.min_quality:
                    results["above_threshold"] += 1
                else:
                    results["below_threshold"] += 1
            except Exception as e:
                logger.warning("Failed to score %s: %s", p.name, e)

        return results
