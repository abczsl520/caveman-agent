"""Skill manager v2 — load, save, match, auto-create, evolve."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

from .types import Skill, SkillTrigger, SkillStep, QualityGate


def _validate_skill_name(name: str) -> None:
    """Reject skill names that could escape the skills directory."""
    if not name or ".." in name or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid skill name: {name!r}")


class SkillManager:
    """Manages agent skills with matching, auto-creation, and evolution."""

    def __init__(self, skills_dir: Path | str | None = None):
        from caveman.paths import SKILLS_DIR
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir else SKILLS_DIR
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._loaded = False  # Cache: only load from disk once

    def load_all(self) -> None:
        """Load all YAML skill files from disk (cached after first load)."""
        if self._loaded:
            return
        if yaml is None:
            return
        for f in self.skills_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
                if data and "name" in data:
                    self._skills[data["name"]] = Skill.from_dict(data)
            except Exception as e:
                logger.warning("Failed to load skill %s: %s", f.name, e)
        self._loaded = True

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def save(self, skill: Skill) -> None:
        """Persist skill to disk as YAML."""
        if yaml is None:
            raise ImportError("pyyaml required")
        _validate_skill_name(skill.name)
        skill.updated_at = datetime.now()
        path = self.skills_dir / f"{skill.name}.yaml"
        path.write_text(
            yaml.safe_dump(skill.to_dict(), allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        self._skills[skill.name] = skill

    def delete(self, name: str) -> bool:
        _validate_skill_name(name)
        path = self.skills_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()
        return self._skills.pop(name, None) is not None

    def match(self, task: str) -> list[Skill]:
        """Find skills matching a task. Returns sorted by relevance.

        Supports cross-language matching: '上线' matches trigger_pattern '部署'.
        """
        from caveman.memory.retrieval import expand_query_cross_lang
        expanded_task = expand_query_cross_lang(task)
        matches: list[tuple[float, Skill]] = []

        for skill in self._skills.values():
            score = 0.0

            # Always-inject skills
            if skill.trigger == SkillTrigger.ALWAYS:
                matches.append((1.0, skill))
                continue

            # Pattern matching (with cross-language expansion)
            if skill.trigger == SkillTrigger.PATTERN:
                for pattern in skill.trigger_patterns:
                    try:
                        if re.search(pattern, expanded_task, re.IGNORECASE):
                            score = max(score, 0.9)
                    except re.error:
                        if pattern.lower() in expanded_task.lower():
                            score = max(score, 0.7)

            # Keyword matching from description
            if skill.trigger in (SkillTrigger.AUTO, SkillTrigger.PATTERN):
                desc_words = skill.description.lower().split()
                task_lower = expanded_task.lower()
                hits = sum(1 for w in desc_words if w in task_lower)
                if desc_words and hits > 0:
                    kw_score = hits / len(desc_words) * 0.6
                    score = max(score, kw_score)

            # Boost by success rate
            if skill.success_rate > 0:
                score *= (1.0 + skill.success_rate * 0.3)

            if score > 0.1:
                matches.append((score, skill))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in matches]

    async def auto_create(
        self, trajectory: list[dict], task: str = "", llm_fn=None
    ) -> Optional[Skill]:
        """Analyze trajectory for repeatable patterns and create skill.

        With LLM: extracts intent, generalizes args, writes description.
        Without LLM: uses heuristic tool-frequency analysis.
        """
        if len(trajectory) < 4:
            return None

        # Count tool call frequency
        tool_counts: dict[str, int] = {}
        tool_args: dict[str, list[dict]] = {}

        for turn in trajectory:
            if turn.get("from") == "function_call":
                try:
                    import json
                    call = json.loads(turn.get("value", "{}"))
                    name = call.get("name", "")
                    if name:
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        tool_args.setdefault(name, []).append(call.get("arguments", {}))
                except (json.JSONDecodeError, KeyError):
                    continue

        repeated = {name: count for name, count in tool_counts.items() if count >= 3}
        if not repeated:
            return None

        # LLM-powered skill creation
        if llm_fn:
            skill = await self._auto_create_with_llm(
                trajectory, task, repeated, tool_args, llm_fn
            )
            if skill:
                self.save(skill)
                return skill

        # Heuristic fallback
        steps = []
        for name in repeated:
            steps.append(SkillStep(
                tool=name,
                args_template={},
                description=f"Call {name} (repeated {repeated[name]}x)",
            ))

        skill = Skill(
            name=f"auto_{hash(task) % 10000:04d}",
            description=f"Auto-created: {task[:80]}" if task else "Auto-created from trajectory",
            trigger=SkillTrigger.AUTO,
            steps=steps,
            source="auto_created",
            quality_gates=[
                QualityGate(name="basic_check", check="no_errors", severity="warn"),
            ],
        )
        self.save(skill)
        return skill

    async def _auto_create_with_llm(
        self,
        trajectory: list[dict],
        task: str,
        repeated: dict[str, int],
        tool_args: dict[str, list[dict]],
        llm_fn,
    ) -> Optional[Skill]:
        """LLM-powered skill creation from trajectory analysis."""
        # Build trajectory summary
        turns_text = []
        for t in trajectory[:15]:
            role = t.get("from", "?")
            value = str(t.get("value", ""))[:200]
            turns_text.append(f"[{role}] {value}")
        traj_summary = "\n".join(turns_text)

        tools_text = ", ".join(f"{n}({c}x)" for n, c in repeated.items())

        prompt = f"""Analyze this agent trajectory and create a reusable skill.

Task: {task}
Repeated tools: {tools_text}

Trajectory (first 15 turns):
{traj_summary}

Create a skill definition as JSON:
{{
  "name": "short-kebab-name",
  "description": "What this skill does (1-2 sentences)",
  "trigger_patterns": ["keyword1", "keyword2"],
  "steps": [
    {{"tool": "tool_name", "description": "what this step does", "args_template": {{}}}}
  ],
  "constraints": ["important constraint 1"]
}}"""

        try:
            response = await llm_fn(prompt)
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                steps = [
                    SkillStep(
                        tool=s.get("tool", ""),
                        args_template=s.get("args_template", {}),
                        description=s.get("description", ""),
                    )
                    for s in data.get("steps", [])
                ]
                return Skill(
                    name=data.get("name", f"auto_{hash(task) % 10000:04d}"),
                    description=data.get("description", task[:80]),
                    trigger=SkillTrigger.PATTERN,
                    trigger_patterns=data.get("trigger_patterns", []),
                    steps=steps,
                    content="\n".join(data.get("constraints", [])),
                    source="auto_created",
                    quality_gates=[
                        QualityGate(name="basic_check", check="no_errors", severity="warn"),
                    ],
                )
        except Exception as e:
            logger.warning("LLM skill creation failed: %s", e)
        return None

    def record_outcome(self, skill_name: str, success: bool) -> None:
        """Record success/failure for skill evolution.

        When a skill accumulates enough failures (≥3 consecutive or
        success_rate < 40%), it's flagged for evolution. The actual
        evolution happens in evolve() with LLM assistance.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return
        if success:
            skill.success_count += 1
        else:
            skill.fail_count += 1
        # Flag for evolution if quality is degrading
        if skill.total_uses >= 3 and skill.success_rate < 0.4:
            skill.metadata = skill.metadata or {}
            skill.metadata["needs_evolution"] = True
            skill.metadata["evolution_reason"] = (
                f"Success rate {skill.success_rate:.0%} below 40% "
                f"after {skill.total_uses} uses"
            )
            logger.info(
                "Skill '%s' flagged for evolution: %s",
                skill_name, skill.metadata["evolution_reason"],
            )
        self.save(skill)

    async def evolve(
        self, skill_name: str, feedback: str = "",
        trajectory: list[dict] | None = None, llm_fn=None,
    ) -> Optional[Skill]:
        """Evolve a skill based on feedback and recent trajectory.

        PRD §5.2 Ring 3: Skills don't just get created — they improve.
        With LLM: analyzes failures, rewrites steps/constraints.
        Without LLM: bumps version, clears evolution flag.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return None

        if llm_fn and (feedback or trajectory):
            improved = await self._evolve_with_llm(skill, feedback, trajectory, llm_fn)
            if improved:
                return improved

        # Fallback: just bump version and clear flag
        skill.version += 1
        skill.updated_at = datetime.now()
        if hasattr(skill, 'metadata') and skill.metadata:
            skill.metadata.pop("needs_evolution", None)
            skill.metadata.pop("evolution_reason", None)
        self.save(skill)
        return skill

    async def _evolve_with_llm(
        self, skill: Skill, feedback: str,
        trajectory: list[dict] | None, llm_fn,
    ) -> Optional[Skill]:
        """LLM-powered skill evolution — the core of the learning flywheel."""
        # Build context for LLM
        current_steps = "\n".join(
            f"  {i+1}. [{s.tool}] {s.description}"
            for i, s in enumerate(skill.steps)
        )
        stats = (
            f"Uses: {skill.total_uses}, "
            f"Success: {skill.success_count}, "
            f"Fail: {skill.fail_count}, "
            f"Rate: {skill.success_rate:.0%}"
        )

        traj_text = ""
        if trajectory:
            recent = trajectory[-10:]
            traj_text = "\n".join(
                f"[{t.get('from', '?')}] {str(t.get('value', ''))[:150]}"
                for t in recent
            )

        prompt = f"""This skill needs improvement. Analyze and suggest changes.

Skill: {skill.name} (v{skill.version})
Description: {skill.description}
Stats: {stats}
Current steps:
{current_steps}

{f'Recent failure trajectory:{chr(10)}{traj_text}' if traj_text else ''}
{f'User feedback: {feedback}' if feedback else ''}

Respond as JSON with ONLY the fields that should change:
{{
  "description": "improved description (if needed)",
  "steps": [
    {{"tool": "tool_name", "description": "what this step does", "args_template": {{}}}}
  ],
  "constraints": ["new constraint based on failure analysis"],
  "trigger_patterns": ["improved patterns"]
}}

Keep what works. Fix what doesn't. Be specific about WHY each change helps."""

        try:
            response = await llm_fn(prompt)
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start < 0 or end <= start:
                return None

            data = json.loads(response[start:end])

            # Apply improvements selectively
            if "description" in data:
                skill.description = data["description"]
            if "steps" in data and data["steps"]:
                skill.steps = [
                    SkillStep(
                        tool=s.get("tool", ""),
                        args_template=s.get("args_template", {}),
                        description=s.get("description", ""),
                    )
                    for s in data["steps"]
                ]
            if "trigger_patterns" in data:
                skill.trigger_patterns = data["trigger_patterns"]
            if "constraints" in data:
                skill.content = "\n".join(data["constraints"])

            skill.version += 1
            skill.updated_at = datetime.now()
            if hasattr(skill, 'metadata') and skill.metadata:
                skill.metadata.pop("needs_evolution", None)
                skill.metadata["last_evolved"] = datetime.now().isoformat()
                skill.metadata["evolution_feedback"] = feedback[:200] if feedback else "auto"

            self.save(skill)
            logger.info("Skill '%s' evolved to v%d", skill.name, skill.version)
            return skill

        except Exception as e:
            logger.warning("LLM skill evolution failed for '%s': %s", skill.name, e)
            return None
