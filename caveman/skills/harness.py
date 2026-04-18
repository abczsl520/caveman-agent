"""Skill Harness — Guide + Sensor abstraction for skill execution.

Guide: Pre-execution instructions injected into the agent's context.
  - System prompt additions
  - Tool restrictions
  - Output format requirements

Sensor: Post-execution checks that evaluate skill output quality.
  - Output validation (format, content, completeness)
  - Side-effect verification (files created, APIs called)
  - Quality scoring for trajectory recording

Together they form a "harness" that wraps skill execution:
  Guide → Agent executes → Sensor evaluates → QualityGate decides
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class GuideConfig:
    """Pre-execution guidance for a skill."""
    system_prompt_addition: str = ""
    allowed_tools: list[str] | None = None  # None = all tools
    disallowed_tools: list[str] = field(default_factory=list)
    output_format: str = ""  # e.g., "json", "markdown", "code"
    max_iterations: int | None = None
    constraints: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert guide to prompt text for injection."""
        parts = []
        if self.system_prompt_addition:
            parts.append(self.system_prompt_addition)
        if self.output_format:
            parts.append(f"Output format: {self.output_format}")
        if self.constraints:
            parts.append("Constraints:")
            for c in self.constraints:
                parts.append(f"  - {c}")
        if self.disallowed_tools:
            parts.append(f"Do NOT use these tools: {', '.join(self.disallowed_tools)}")
        return "\n".join(parts)


@dataclass
class SensorCheck:
    """A single sensor check result."""
    name: str
    passed: bool
    detail: str = ""
    score: float = 1.0  # 0-1


@dataclass
class SensorResult:
    """Aggregated sensor evaluation."""
    checks: list[SensorCheck] = field(default_factory=list)
    overall_score: float = 1.0
    passed: bool = True

    def add(self, check: SensorCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.passed = False
        # Weighted average
        if self.checks:
            self.overall_score = sum(c.score for c in self.checks) / len(self.checks)


class Sensor:
    """Post-execution evaluation engine.

    Runs checks against skill output to determine quality.
    """

    def __init__(self, llm_fn: Callable[[str], Awaitable[str]] | None = None):
        self.llm_fn = llm_fn
        self._checks: list[Callable] = [
            self._check_non_empty,
            self._check_no_errors,
            self._check_format,
        ]

    async def evaluate(
        self,
        output: str,
        expected_format: str = "",
        quality_gates: list[dict] | None = None,
    ) -> SensorResult:
        """Evaluate skill output."""
        result = SensorResult()

        # Built-in checks
        result.add(self._check_non_empty(output))
        result.add(self._check_no_errors(output))
        if expected_format:
            result.add(self._check_format(output, expected_format))

        # Quality gate checks
        if quality_gates:
            for gate in quality_gates:
                result.add(self._check_quality_gate(output, gate))

        # LLM quality check (if available and output is substantial)
        if self.llm_fn and len(output) > 100:
            llm_check = await self._check_quality_llm(output)
            if llm_check:
                result.add(llm_check)

        return result

    def _check_non_empty(self, output: str) -> SensorCheck:
        if not output or not output.strip():
            return SensorCheck("non_empty", False, "Output is empty", 0.0)
        return SensorCheck("non_empty", True, f"{len(output)} chars", 1.0)

    def _check_no_errors(self, output: str) -> SensorCheck:
        error_patterns = [
            r"(?:error|exception|traceback):",
            r"(?:failed|failure)\s+(?:to|with)",
            r"exit\s+code\s+[1-9]",
        ]
        for pattern in error_patterns:
            if re.search(pattern, output.lower()):
                return SensorCheck(
                    "no_errors", False,
                    "Error detected in output", 0.3,
                )
        return SensorCheck("no_errors", True, "No errors detected", 1.0)

    def _check_format(self, output: str, expected: str) -> SensorCheck:
        if expected == "json":
            try:
                import json
                json.loads(output)
                return SensorCheck("format_json", True, "Valid JSON", 1.0)
            except (json.JSONDecodeError, ValueError):
                return SensorCheck("format_json", False, "Invalid JSON", 0.2)
        elif expected == "code":
            if "```" in output or "def " in output or "class " in output:
                return SensorCheck("format_code", True, "Contains code", 1.0)
            return SensorCheck("format_code", False, "No code detected", 0.5)
        elif expected == "markdown":
            if "#" in output or "- " in output or "**" in output:
                return SensorCheck("format_md", True, "Contains markdown", 1.0)
            return SensorCheck("format_md", False, "No markdown detected", 0.7)
        return SensorCheck("format_check", True, "No format requirement", 1.0)

    def _check_quality_gate(self, output: str, gate: dict) -> SensorCheck:
        """Check a skill-defined quality gate."""
        check_type = gate.get("check", "")
        expected = gate.get("expected", "")
        name = gate.get("name", check_type)

        if check_type == "output_contains":
            if expected.lower() in output.lower():
                return SensorCheck(name, True, f"Contains '{expected}'", 1.0)
            return SensorCheck(name, False, f"Missing '{expected}'", 0.0)

        elif check_type == "output_length_min":
            min_len = int(expected) if expected else 10
            if len(output) >= min_len:
                return SensorCheck(name, True, f"Length {len(output)} >= {min_len}", 1.0)
            return SensorCheck(name, False, f"Length {len(output)} < {min_len}", 0.3)

        elif check_type == "no_errors":
            return self._check_no_errors(output)

        return SensorCheck(name, True, f"Unknown check type: {check_type}", 0.5)

    async def _check_quality_llm(self, output: str) -> SensorCheck | None:
        """LLM-based quality assessment."""
        prompt = f"""Rate the quality of this output on a scale of 0-10.
Consider: completeness, accuracy, clarity, actionability.

Output (first 1000 chars):
{output[:1000]}

Respond with just a number (0-10) and one sentence explanation.
Example: 8 — Clear and complete with good examples."""

        try:
            response = await self.llm_fn(prompt)
            # Parse score from response
            match = re.match(r'(\d+)', response.strip())
            if match:
                score = int(match.group(1))
                normalized = min(score / 10.0, 1.0)
                return SensorCheck(
                    "llm_quality", normalized >= 0.5,
                    response.strip()[:100], normalized,
                )
        except Exception as e:
            logger.warning("LLM quality check failed: %s", e)
        return None


@dataclass
class SkillHarness:
    """Complete harness wrapping a skill execution.

    Usage:
        harness = SkillHarness(guide=my_guide, sensor=my_sensor)
        # Before execution:
        prompt_addition = harness.guide.to_prompt()
        # After execution:
        eval_result = await harness.evaluate(output)
    """
    guide: GuideConfig = field(default_factory=GuideConfig)
    sensor: Sensor = field(default_factory=Sensor)
    quality_gates: list[dict] = field(default_factory=list)

    async def evaluate(self, output: str) -> SensorResult:
        """Run sensor evaluation with configured quality gates."""
        return await self.sensor.evaluate(
            output,
            expected_format=self.guide.output_format,
            quality_gates=self.quality_gates,
        )
