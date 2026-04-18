"""Skill executor — run skills with quality gate enforcement."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .types import Skill, SkillStep, QualityGate


@dataclass
class StepResult:
    """Result of executing one skill step."""
    step: SkillStep
    output: str = ""
    success: bool = True
    error: str | None = None


@dataclass
class SkillResult:
    """Result of executing a complete skill."""
    skill_name: str
    step_results: list[StepResult] = field(default_factory=list)
    gate_results: list[dict] = field(default_factory=list)
    success: bool = True
    blocked_by: str | None = None  # Gate name that blocked

    @property
    def output(self) -> str:
        return "\n".join(r.output for r in self.step_results if r.output)


class SkillExecutor:
    """Execute skills step-by-step with quality gate checks."""

    def __init__(self, tool_dispatch_fn=None):
        """
        Args:
            tool_dispatch_fn: async fn(tool_name, args) -> str
                The agent's tool dispatch function for executing skill steps.
        """
        self._dispatch = tool_dispatch_fn

    async def execute(self, skill: Skill, context: dict[str, Any] = None) -> SkillResult:
        """Execute a skill's steps and check quality gates."""
        context = context or {}
        result = SkillResult(skill_name=skill.name)

        # If skill has structured steps, execute them
        if skill.steps:
            for step in skill.steps:
                step_result = await self._execute_step(step, context)
                result.step_results.append(step_result)
                if not step_result.success:
                    result.success = False
                    break
                # Store step output in context for next step
                context[f"step_{len(result.step_results)}"] = step_result.output

        # Check quality gates
        for gate in skill.quality_gates:
            gate_ok, msg = self._check_gate(gate, result)
            result.gate_results.append({
                "gate": gate.name, "passed": gate_ok, "message": msg,
            })
            if not gate_ok and gate.severity == "block":
                result.success = False
                result.blocked_by = gate.name
                break

        return result

    async def _execute_step(self, step: SkillStep, context: dict) -> StepResult:
        """Execute a single skill step."""
        if not self._dispatch:
            return StepResult(step=step, success=False, error="No tool dispatch function configured")

        # Resolve arg templates
        resolved_args = self._resolve_templates(step.args_template, context)

        # Check condition
        if step.condition and not self._eval_condition(step.condition, context):
            return StepResult(step=step, output="[skipped: condition not met]")

        try:
            output = await self._dispatch(step.tool, resolved_args)
            return StepResult(step=step, output=str(output), success=True)
        except Exception as e:
            return StepResult(step=step, success=False, error=str(e))

    def _resolve_templates(self, template: dict, context: dict) -> dict:
        """Replace {{var}} placeholders in args with context values."""
        resolved = {}
        for k, v in template.items():
            if isinstance(v, str) and "{{" in v:
                for ctx_key, ctx_val in context.items():
                    v = v.replace(f"{{{{{ctx_key}}}}}", str(ctx_val))
            resolved[k] = v
        return resolved

    def _eval_condition(self, condition: str, context: dict) -> bool:
        """Simple condition evaluation. Only supports 'exists' checks."""
        # e.g., "step_1" checks if step_1 output exists in context
        return condition in context and bool(context[condition])

    def _check_gate(self, gate: QualityGate, result: SkillResult) -> tuple[bool, str]:
        """Check a quality gate against execution results."""
        output = result.output

        if gate.check == "no_errors":
            has_errors = any(not r.success for r in result.step_results)
            return (not has_errors, "All steps succeeded" if not has_errors else "Some steps failed")

        if gate.check == "output_contains":
            found = gate.expected.lower() in output.lower()
            return (found, f"Found '{gate.expected}'" if found else f"Missing '{gate.expected}'")

        if gate.check == "output_not_empty":
            ok = len(output.strip()) > 0
            return (ok, "Output non-empty" if ok else "Output is empty")

        if gate.check == "min_steps":
            try:
                min_n = int(gate.expected)
                ok = len(result.step_results) >= min_n
                return (ok, f"{len(result.step_results)}/{min_n} steps" if ok else f"Only {len(result.step_results)}/{min_n} steps")
            except ValueError:
                return (True, "Invalid min_steps value")

        return (True, f"Unknown gate check: {gate.check}")
