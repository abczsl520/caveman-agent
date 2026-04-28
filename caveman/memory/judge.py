"""LLM judge for memory quality and helpfulness feedback.

PRD Round 14: replace fragile heuristics with an auditable LLM judge while
keeping deterministic fallbacks for offline/test environments.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .types import MemoryEntry

logger = logging.getLogger(__name__)

__all__ = [
    "JudgeResult",
    "MemoryJudge",
    "heuristic_helpfulness",
]

LLMFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class JudgeResult:
    """Auditable memory judge decision."""

    helpful: bool
    confidence: float
    reason: str
    mode: str = "heuristic"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "judge_helpful": self.helpful,
            "judge_confidence": self.confidence,
            "judge_reason": self.reason[:300],
            "judge_mode": self.mode,
        }


_STOPWORDS = {
    "with", "from", "that", "this", "then", "than", "when", "what", "were",
    "been", "being", "have", "has", "had", "into", "onto", "over", "under",
    "your", "their", "there", "here", "task", "final", "answer", "result",
    "fixed", "running", "using", "passed", "exit", "code", "health", "checks",
}


def _significant_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9_\-]{4,}|[\u4e00-\u9fff]{2,}", text)
        if len(w.strip()) >= 2 and w.lower() not in _STOPWORDS
    }


def heuristic_helpfulness(task: str, final: str, memory: MemoryEntry, success: bool) -> JudgeResult:
    """Deterministic fallback when no LLM judge is available.

    This intentionally stays conservative: a globally successful task is not
    enough to mark every recalled memory helpful. The memory must overlap with
    either the task or final answer.
    """
    mem_words = _significant_words(memory.content[:500])
    final_words = _significant_words(final or "")
    task_words = _significant_words(task or "")
    overlap_final = mem_words & final_words
    overlap_task = mem_words & task_words

    helpful = bool(success and (overlap_final or len(overlap_task) >= 2))
    reason = (
        f"heuristic overlap final={len(overlap_final)} task={len(overlap_task)} "
        f"success={success}"
    )
    confidence = 0.45
    if overlap_final:
        confidence = min(0.85, 0.55 + len(overlap_final) * 0.05)
    elif helpful:
        confidence = 0.55
    return JudgeResult(helpful=helpful, confidence=confidence, reason=reason, mode="heuristic")


class MemoryJudge:
    """Judge whether a recalled memory actually helped the final answer.

    The LLM response is constrained to JSON for auditability. On any provider
    failure or malformed response, callers can fall back to the deterministic
    heuristic instead of blocking the agent.
    """

    def __init__(self, llm_fn: LLMFn | None = None, *, enabled: bool = True):
        self.llm_fn = llm_fn
        self.enabled = enabled

    async def judge_helpfulness(
        self,
        *,
        task: str,
        final: str,
        memory: MemoryEntry,
        success: bool,
    ) -> JudgeResult:
        if not self.enabled or self.llm_fn is None:
            return heuristic_helpfulness(task, final, memory, success)

        prompt = self._build_prompt(task=task, final=final, memory=memory, success=success)
        try:
            raw = await self.llm_fn(prompt)
            return self._parse_response(raw)
        except Exception as exc:
            logger.debug("MemoryJudge LLM failed; falling back to heuristic: %s", exc)
            return heuristic_helpfulness(task, final, memory, success)

    def _build_prompt(self, *, task: str, final: str, memory: MemoryEntry, success: bool) -> str:
        return (
            "You are an audit judge for an AI agent memory flywheel.\n"
            "Decide whether the recalled memory was actually helpful for the final answer.\n"
            "Return ONLY compact JSON with keys: helpful(boolean), confidence(number 0-1), reason(string).\n"
            "Be strict: task success alone is not enough. The memory must materially support the answer.\n\n"
            f"TASK:\n{task[:1200]}\n\n"
            f"FINAL_ANSWER:\n{final[:2000]}\n\n"
            f"TASK_SUCCESS_HEURISTIC: {success}\n\n"
            f"RECALLED_MEMORY:\n{memory.content[:1200]}\n"
        )

    def _parse_response(self, raw: str) -> JudgeResult:
        text = (raw or "").strip()
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group(0)
        data = json.loads(text)
        helpful = bool(data["helpful"])
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", "llm judge decision"))
        return JudgeResult(helpful=helpful, confidence=confidence, reason=reason, mode="llm")
