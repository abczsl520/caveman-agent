"""Reflect Engine — post-task reflection and skill evolution.

Inspired by Memento-Skills (arxiv 2603.18743) Read-Execute-Reflect-Write cycle.
After task completion, analyzes execution trajectory to:
1. Extract effective patterns → create/update skills
2. Identify anti-patterns → record lessons learned
3. Update RL Router weights → improve future skill selection

This is the 6th engine in the Caveman flywheel:
  Shield → Nudge → Reflect → Ripple → Lint → Recall
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from caveman.skills.types import Skill, SkillTrigger
from caveman.skills.manager import SkillManager

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """Result of reflecting on a task execution."""
    task: str
    outcome: str  # "success" | "partial" | "failure"
    effective_patterns: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    skill_updates: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    reflected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "outcome": self.outcome,
            "effective_patterns": self.effective_patterns,
            "anti_patterns": self.anti_patterns,
            "lessons": self.lessons,
            "skill_updates": self.skill_updates,
            "confidence": self.confidence,
            "reflected_at": self.reflected_at.isoformat(),
        }


_TOOL_PATTERNS = {
    "bash": r"(?:ran|executed|running)\s+(?:command|bash|shell)",
    "file_write": r"(?:created|wrote|saved)\s+(?:file|to)\s+(\S+)",
    "file_read": r"(?:read|loaded|checked)\s+(\S+)",
    "web_search": r"(?:searched|googled|looked up)",
}

_ANTI_PATTERN_SIGNALS = [
    (r"(?:tried|attempt)\s+(\d+)\s+times", "retry_loop"),
    (r"(?:reverted|rolled back|undid)", "revert"),
    (r"(?:wrong approach|bad idea|shouldn't have)", "wrong_approach"),
    (r"(?:workaround|hack|temporary fix)", "workaround"),
]


class ReflectEngine:
    """Post-task reflection — extract patterns and evolve skills."""

    def __init__(
        self,
        skill_manager: Optional[SkillManager] = None,
        llm_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        max_reflections: int = 50,
    ) -> None:
        self._skill_manager = skill_manager
        self._llm_fn = llm_fn
        self._reflections: list[Reflection] = []
        self._max_reflections = max_reflections

    @property
    def reflections(self) -> list[Reflection]:
        return list(self._reflections)

    async def reflect(
        self,
        task: str,
        trajectory: list[dict[str, str]],
        task_result: str = "",
    ) -> Reflection:
        """Reflect on a completed task execution.

        Args:
            task: The original task description.
            trajectory: ShareGPT-format conversation turns.
            task_result: Final output/result of the task.

        Returns:
            Reflection with patterns, anti-patterns, and lessons.
        """
        if self._llm_fn:
            try:
                reflection = await self._reflect_with_llm(
                    task, trajectory, task_result,
                )
            except Exception as e:
                logger.warning("LLM reflection failed, using heuristic: %s", e)
                reflection = self._reflect_heuristic(task, trajectory, task_result)
        else:
            reflection = self._reflect_heuristic(task, trajectory, task_result)

        self._reflections.append(reflection)
        # Cap to prevent unbounded memory growth (视角14)
        if len(self._reflections) > self._max_reflections:
            self._reflections = self._reflections[-self._max_reflections:]

        # Apply reflection: update skills
        if self._skill_manager and reflection.skill_updates:
            self._apply_skill_updates(reflection)

        # Auto-evolve degraded skills (PRD §5.2 Ring 3)
        if self._skill_manager and reflection.outcome == "failure":
            await self._auto_evolve_degraded(reflection, trajectory)

        return reflection

    def _reflect_heuristic(
        self, task: str, trajectory: list[dict], result: str,
    ) -> Reflection:
        """Rule-based reflection from trajectory analysis."""
        all_text = " ".join(
            t.get("value", t.get("content", ""))
            for t in trajectory
        )
        combined = f"{all_text} {result}"

        # Detect outcome
        outcome = self._detect_outcome(combined)

        tool_calls = self._extract_tool_calls(trajectory)

        # Extract effective patterns
        effective = []
        if tool_calls:
            unique_order = list(dict.fromkeys(tool_calls))
            counts: dict[str, int] = {}
            for tool_name in tool_calls:
                counts[tool_name] = counts.get(tool_name, 0) + 1
            ordered_counts = ", ".join(
                f"{tool_name} x{counts[tool_name]}" for tool_name in unique_order
            )
            effective.append(
                "Procedure: use tools in this order when the task requires inspection "
                f"and iteration: {' -> '.join(tool_calls[:8])}"
            )
            effective.append(
                "Procedure: prefer this tool stack for similar tasks: "
                f"{' -> '.join(unique_order[:6])}"
            )
            effective.append(f"Tool usage frequency: {ordered_counts}")
        else:
            for turn in trajectory:
                text = turn.get("value", turn.get("content", ""))
                role = turn.get("from", turn.get("role", ""))
                if role in ("gpt", "assistant"):
                    for tool_name, pattern in _TOOL_PATTERNS.items():
                        if re.search(pattern, text, re.IGNORECASE):
                            effective.append(f"Used {tool_name}")

        # Extract anti-patterns
        anti = []
        for pattern, label in _ANTI_PATTERN_SIGNALS:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                anti.append(f"{label}: {match.group(0)[:80]}")

        # Extract lessons
        lessons = []
        if outcome == "success" and effective:
            lessons.append(
                f"For '{task[:60]}': {', '.join(effective[:3])} worked well"
            )
        if anti:
            lessons.append(
                f"Avoid: {', '.join(a.split(':')[0] for a in anti[:3])}"
            )

        # Suggest skill updates
        skill_updates = []
        if outcome == "success" and len(trajectory) >= 4:
            tool_calls = self._extract_tool_calls(trajectory)
            unique_order = list(dict.fromkeys(tool_calls))
            skill_updates.append({
                "action": "create_or_update",
                "name": self._task_to_skill_name(task),
                "description": f"Pattern for: {task[:80]}",
                "outcome": outcome,
                "steps": [
                    {
                        "tool": tool_name,
                        "args_template": {},
                        "description": f"Call {tool_name} as part of the successful tool sequence",
                    }
                    for tool_name in unique_order[:8]
                ],
            })

        confidence = 0.7 if outcome == "success" else 0.4

        return Reflection(
            task=task,
            outcome=outcome,
            effective_patterns=effective[:10],
            anti_patterns=anti[:5],
            lessons=lessons[:5],
            skill_updates=skill_updates,
            confidence=confidence,
        )

    async def _reflect_with_llm(
        self, task: str, trajectory: list[dict], result: str,
    ) -> Reflection:
        """LLM-powered deep reflection."""
        assert self._llm_fn is not None

        recent = trajectory[-15:] if len(trajectory) > 15 else trajectory
        conv = "\n".join(
            f"[{t.get('from', t.get('role', '?'))}]: "
            f"{t.get('value', t.get('content', ''))[:200]}"
            for t in recent
        )

        prompt = f"""Reflect on this task execution and extract learnings.

Task: {task}
Result: {result[:300]}

Execution trajectory:
{conv}

Analyze and respond as JSON:
{{
  "outcome": "success|partial|failure",
  "effective_patterns": ["what worked well"],
  "anti_patterns": ["what didn't work or was wasteful"],
  "lessons": ["actionable lessons for next time"],
  "confidence": 0.0-1.0
}}

Rules:
- Be specific and actionable
- Focus on reusable patterns, not task-specific details
- Anti-patterns should explain WHY something was bad
- Lessons should be prescriptive ("Do X" or "Avoid Y")"""

        response = await self._llm_fn(prompt)

        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])

            skill_updates = []
            if data.get("outcome") == "success":
                skill_updates.append({
                    "action": "create_or_update",
                    "name": self._task_to_skill_name(task),
                    "description": f"Pattern for: {task[:80]}",
                    "outcome": data["outcome"],
                })

            return Reflection(
                task=task,
                outcome=data.get("outcome", "partial"),
                effective_patterns=data.get("effective_patterns", [])[:10],
                anti_patterns=data.get("anti_patterns", [])[:5],
                lessons=data.get("lessons", [])[:5],
                skill_updates=skill_updates,
                confidence=data.get("confidence", 0.5),
            )

        raise ValueError("Could not parse LLM reflection as JSON")

    def _detect_outcome(self, text: str) -> str:
        """Detect task outcome from text signals. Uses shared detection."""
        from caveman.utils import detect_outcome
        return detect_outcome(text)

    def _extract_tool_calls(self, trajectory: list[dict]) -> list[str]:
        """Extract ordered list of tool names from function_call turns.

        Parses ShareGPT-format trajectory where tool invocations appear as
        turns with from='function_call' and value containing JSON with a
        'name' field. Falls back to scanning 'tool_calls' lists in
        assistant turns (OpenAI format).

        Returns:
            Ordered list of tool names as they were called.
        """
        tools: list[str] = []
        for turn in trajectory:
            role = turn.get("from", turn.get("role", ""))

            # ShareGPT format: dedicated function_call turn
            if role == "function_call":
                try:
                    call = json.loads(turn.get("value", "{}"))
                    name = call.get("name", "")
                    if name:
                        tools.append(name)
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            # OpenAI format: assistant turn with tool_calls list
            if role in ("assistant", "gpt"):
                for tc in turn.get("tool_calls", []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else ""
                    if name:
                        tools.append(name)
        return tools

    def _task_to_skill_name(self, task: str) -> str:
        """Convert task description to a skill name."""
        words = re.sub(r"[^\w\s]", "", task.lower()).split()[:4]
        return "-".join(words) if words else "unnamed-skill"

    def _apply_skill_updates(self, reflection: Reflection) -> None:
        """Apply reflection results to skill manager."""
        if not self._skill_manager:
            return

        for update in reflection.skill_updates:
            name = update.get("name", "")
            if not name:
                continue

            existing = self._skill_manager.get(name)
            if existing:
                # Update existing skill
                if reflection.outcome == "success":
                    existing.success_count += 1
                else:
                    existing.fail_count += 1
                existing.version += 1

                # Append lessons to content (capped to prevent unbounded growth)
                if reflection.lessons:
                    new_lessons = "\n".join(f"- {l}" for l in reflection.lessons)
                    existing.content = (existing.content + "\n\nLessons:\n" + new_lessons)[-2000:]

                self._skill_manager.save(existing)
                logger.info("Updated skill %s (v%d)", name, existing.version)
            else:
                # Create new skill
                skill = Skill(
                    name=name,
                    description=update.get("description", ""),
                    trigger=SkillTrigger.AUTO,
                    content="\n".join(f"- {l}" for l in reflection.lessons),
                    source="reflect",
                    success_count=1 if reflection.outcome == "success" else 0,
                    fail_count=0 if reflection.outcome == "success" else 1,
                )
                self._skill_manager.save(skill)
                logger.info("Created skill %s from reflection", name)

    async def _auto_evolve_degraded(
        self, reflection: Reflection, trajectory: list[dict],
    ) -> None:
        """Auto-evolve skills flagged for evolution after failure.

        PRD §5.2 Ring 3: Skills don't just get created — they improve.
        When a task fails and matched skills have needs_evolution=True,
        trigger LLM-powered evolution with the failure trajectory as context.
        """
        for skill in self._skill_manager.list_all():
            meta = getattr(skill, 'metadata', None) or {}
            if not meta.get("needs_evolution"):
                continue
            try:
                feedback = "; ".join(reflection.anti_patterns[:3])
                await self._skill_manager.evolve(
                    skill.name, feedback=feedback,
                    trajectory=trajectory, llm_fn=self._llm_fn,
                )
            except Exception as e:
                logger.debug("Auto-evolve failed for '%s': %s", skill.name, e)
