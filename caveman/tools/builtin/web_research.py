"""Web Research Environment — multi-step web research with scoring.

Provides a structured environment for multi-step web research tasks
with reward signals for correctness, diversity, and efficiency.
Extracted from Hermes environments/web_research_env.py (719 lines).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from urllib.parse import urlparse

__all__ = [
    "ResearchStep",
    "ResearchScore",
    "ResearchSession",
    "WebResearchRunner",
    "SAMPLE_QUESTIONS",
]


logger = logging.getLogger("caveman.tools.web_research")


@dataclass
class ResearchStep:
    """A single step in a research session."""
    action: str  # search | extract | analyze | answer
    query: str = ""
    urls: List[str] = field(default_factory=list)
    result: str = ""
    tokens_used: int = 0
    duration_ms: float = 0
    timestamp: float = 0


@dataclass
class ResearchScore:
    """Scoring for a research session."""
    correctness: float = 0  # 0-1, LLM judge
    source_diversity: float = 0  # 0-1, unique domains
    efficiency: float = 0  # 0-1, fewer steps = better
    tool_usage: float = 0  # 0-1, bonus for using tools
    total: float = 0

    def compute(self) -> float:
        self.total = (
            self.correctness * 0.5
            + self.source_diversity * 0.2
            + self.efficiency * 0.2
            + self.tool_usage * 0.1
        )
        return self.total


@dataclass
class ResearchSession:
    """A web research session."""
    question: str
    steps: List[ResearchStep] = field(default_factory=list)
    answer: str = ""
    score: ResearchScore = field(default_factory=ResearchScore)
    started_at: float = 0
    completed_at: float = 0
    max_steps: int = 10

    @property
    def domains_used(self) -> set:
        domains = set()
        for step in self.steps:
            for url in step.urls:
                try:
                    domains.add(urlparse(url).netloc)
                except Exception as exc:
                    logger.debug("domains_used: suppressed %s", exc)
        return domains

    @property
    def is_complete(self) -> bool:
        return bool(self.answer) or len(self.steps) >= self.max_steps

    def add_step(self, step: ResearchStep) -> None:
        step.timestamp = time.time()
        self.steps.append(step)

    def compute_score(self, reference_answer: str = "") -> ResearchScore:
        """Compute research score."""
        # Source diversity
        domains = self.domains_used
        self.score.source_diversity = min(len(domains) / 3, 1.0)

        # Efficiency (fewer steps = better, max 10)
        self.score.efficiency = max(0, 1.0 - (len(self.steps) / self.max_steps))

        # Tool usage (bonus for actually searching/extracting)
        tool_steps = sum(1 for s in self.steps if s.action in ("search", "extract"))
        self.score.tool_usage = min(tool_steps / 2, 1.0)

        # Correctness (simple string overlap if reference provided)
        if reference_answer and self.answer:
            self.score.correctness = _simple_correctness(self.answer, reference_answer)

        self.score.compute()
        return self.score


def _simple_correctness(answer: str, reference: str) -> float:
    """Simple correctness score based on keyword overlap."""
    answer_words = set(answer.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0
    overlap = len(answer_words & ref_words)
    return min(overlap / max(len(ref_words) * 0.3, 1), 1.0)


# ── Research Runner ──

class WebResearchRunner:
    """Runs structured web research sessions."""

    def __init__(
        self,
        search_fn: Optional[Callable] = None,
        extract_fn: Optional[Callable] = None,
        max_steps: int = 10,
    ):
        self._search_fn = search_fn
        self._extract_fn = extract_fn
        self._max_steps = max_steps

    async def research(
        self,
        question: str,
        reference_answer: str = "",
    ) -> ResearchSession:
        """Run a research session."""
        session = ResearchSession(
            question=question,
            started_at=time.time(),
            max_steps=self._max_steps,
        )

        # Step 1: Search
        if self._search_fn:
            start = time.monotonic()
            results = self._search_fn(question)
            if hasattr(results, "__await__"):
                results = await results
            duration = (time.monotonic() - start) * 1000

            urls = []
            if isinstance(results, list):
                urls = [r.get("url", "") for r in results if r.get("url")]
            elif isinstance(results, dict):
                for r in results.get("data", {}).get("web", []):
                    if r.get("url"):
                        urls.append(r["url"])

            session.add_step(ResearchStep(
                action="search",
                query=question,
                urls=urls,
                result=str(results)[:2000],
                duration_ms=duration,
            ))

            # Step 2: Extract top results
            if self._extract_fn and urls:
                top_urls = urls[:3]
                start = time.monotonic()
                content = self._extract_fn(top_urls)
                if hasattr(content, "__await__"):
                    content = await content
                duration = (time.monotonic() - start) * 1000

                session.add_step(ResearchStep(
                    action="extract",
                    urls=top_urls,
                    result=str(content)[:5000],
                    duration_ms=duration,
                ))

        session.completed_at = time.time()
        if reference_answer:
            session.compute_score(reference_answer)

        return session


# ── Sample Questions ──

SAMPLE_QUESTIONS = [
    {
        "question": "What is the population of Tokyo as of 2024?",
        "reference": "approximately 14 million in the city proper, 37 million in the metropolitan area",
    },
    {
        "question": "Who won the Nobel Prize in Physics in 2024?",
        "reference": "John Hopfield and Geoffrey Hinton for foundational discoveries in machine learning",
    },
    {
        "question": "What programming language is used most in AI/ML development?",
        "reference": "Python is the most widely used programming language for AI and machine learning",
    },
]
