"""Verification Agent — anti-rationalization checks for Coordinator tasks.

Inspired by Claude Code's 6-excuse blocking pattern:
  1. "I already checked" → Verify independently
  2. "It looks correct" → Run actual tests
  3. "The user approved" → Check against spec
  4. "It's a minor change" → Assess blast radius
  5. "It worked before" → Test in current context
  6. "I'll fix it later" → Block until fixed

The Verifier runs as a post-task check in the Coordinator pipeline.
It uses a separate LLM call (or heuristic rules) to challenge the
primary agent's output.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class VerifyResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class VerificationReport:
    """Result of verifying a task output."""
    result: VerifyResult
    checks: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    confidence: float = 1.0  # 0-1

    @property
    def passed(self) -> bool:
        return self.result in (VerifyResult.PASS, VerifyResult.SKIP)


# Excuse patterns to detect and challenge
_EXCUSE_PATTERNS = [
    (r"(?:already|previously)\s+(?:checked|verified|tested|confirmed)", "excuse_already_checked"),
    (r"(?:looks?|seems?|appears?)\s+(?:correct|fine|good|ok)", "excuse_looks_correct"),
    (r"(?:user|they)\s+(?:approved|confirmed|said)", "excuse_user_approved"),
    (r"(?:minor|small|trivial|simple)\s+(?:change|fix|update)", "excuse_minor_change"),
    (r"(?:worked|works)\s+(?:before|previously|last time)", "excuse_worked_before"),
    (r"(?:fix|address|handle)\s+(?:later|next|soon|eventually)", "excuse_fix_later"),
]


class VerificationAgent:
    """Anti-rationalization verification for Coordinator tasks.

    Usage:
        verifier = VerificationAgent(llm_fn=my_llm)
        report = await verifier.verify(task_desc, task_output)
        if not report.passed:
            # Reject or flag the output
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
        strict: bool = False,
    ):
        self.llm_fn = llm_fn
        self.strict = strict  # strict mode: WARN → FAIL

    async def verify(
        self,
        task: str,
        output: Any,
        context: dict | None = None,
    ) -> VerificationReport:
        """Verify a task's output for correctness and completeness."""
        output_text = str(output) if output else ""
        report = VerificationReport(result=VerifyResult.PASS)

        # Rule-based checks (always run, zero cost)
        self._check_excuses(output_text, report)
        self._check_completeness(task, output_text, report)
        self._check_error_signals(output_text, report)

        # LLM verification (if available and task is non-trivial)
        if self.llm_fn and len(output_text) > 50:
            await self._verify_with_llm(task, output_text, context, report)

        # Determine final result
        has_fail = any(c["result"] == "fail" for c in report.checks)
        has_warn = any(c["result"] == "warn" for c in report.checks)

        if has_fail:
            report.result = VerifyResult.FAIL
        elif has_warn and self.strict:
            report.result = VerifyResult.FAIL
        elif has_warn:
            report.result = VerifyResult.WARN
        else:
            report.result = VerifyResult.PASS

        report.summary = self._build_summary(report)
        return report

    def _check_excuses(self, output: str, report: VerificationReport) -> None:
        """Detect rationalization excuses in output."""
        output_lower = output.lower()
        for pattern, excuse_name in _EXCUSE_PATTERNS:
            if re.search(pattern, output_lower):
                report.checks.append({
                    "name": excuse_name,
                    "result": "warn",
                    "detail": f"Detected excuse pattern: {excuse_name}",
                })

    def _check_completeness(
        self, task: str, output: str, report: VerificationReport
    ) -> None:
        """Check if output addresses the task."""
        if not output.strip():
            report.checks.append({
                "name": "empty_output",
                "result": "fail",
                "detail": "Task produced empty output",
            })
            return

        # Check for common incomplete signals
        incomplete_signals = [
            "todo", "fixme", "hack", "placeholder", "not implemented",
            "will implement", "coming soon", "tbd",
        ]
        output_lower = output.lower()
        found = [s for s in incomplete_signals if s in output_lower]
        if found:
            report.checks.append({
                "name": "incomplete_signals",
                "result": "warn",
                "detail": f"Found incomplete markers: {', '.join(found)}",
            })

    def _check_error_signals(self, output: str, report: VerificationReport) -> None:
        """Check for error indicators in output."""
        error_patterns = [
            r"(?:error|exception|traceback|failed|failure):",
            r"(?:cannot|could not|unable to)\s+\w+",
            r"exit\s+code\s+[1-9]",
        ]
        output_lower = output.lower()
        for pattern in error_patterns:
            if re.search(pattern, output_lower):
                report.checks.append({
                    "name": "error_signal",
                    "result": "warn",
                    "detail": f"Error signal detected in output",
                })
                break  # One warning is enough

    async def _verify_with_llm(
        self,
        task: str,
        output: str,
        context: dict | None,
        report: VerificationReport,
    ) -> None:
        """LLM-powered verification."""
        ctx_text = ""
        if context:
            ctx_text = f"\nContext: {str(context)[:500]}"

        prompt = f"""You are a verification agent. Your job is to challenge the output
and find problems. Be skeptical but fair.

TASK: {task}
{ctx_text}
OUTPUT (first 2000 chars):
{output[:2000]}

Check:
1. Does the output actually complete the task? (not just claim to)
2. Are there any errors, incomplete parts, or shortcuts?
3. Would this output cause problems downstream?

Respond as JSON:
{{"result": "pass|warn|fail", "issues": ["..."], "confidence": 0.0-1.0}}"""

        try:
            response = await self.llm_fn(prompt)
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                llm_result = parsed.get("result", "pass")
                issues = parsed.get("issues", [])
                confidence = parsed.get("confidence", 0.8)

                report.confidence = confidence
                for issue in issues:
                    report.checks.append({
                        "name": "llm_verification",
                        "result": llm_result if issue else "pass",
                        "detail": str(issue),
                    })
        except Exception as e:
            logger.warning("LLM verification failed: %s", e)

    def _build_summary(self, report: VerificationReport) -> str:
        """Build human-readable summary."""
        total = len(report.checks)
        fails = sum(1 for c in report.checks if c["result"] == "fail")
        warns = sum(1 for c in report.checks if c["result"] == "warn")
        return (
            f"Verification: {report.result.value} "
            f"({total} checks, {fails} fail, {warns} warn, "
            f"confidence={report.confidence:.0%})"
        )
