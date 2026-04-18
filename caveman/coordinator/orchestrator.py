"""Sub-agent Orchestration — spawn, monitor, and learn from sub-agents.

Builds on top of Coordinator to provide:
  - Agent registry with capabilities and cost profiles
  - Smart routing: match tasks to best-fit agents
  - Knowledge learning: extract lessons from sub-agent results
  - Session hooks: on_delegation_complete integration

Inspired by OpenClaw's sessions_spawn + Hermes's bridge pattern.

MIT License
"""
from __future__ import annotations

__all__ = ["AgentProfile", "AgentRegistry", "SubAgentOrchestrator"]

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    """Describes a sub-agent's capabilities and characteristics."""

    name: str
    capabilities: list[str] = field(default_factory=list)  # e.g. ["coding", "web", "analysis"]
    cost_per_call: float = 0.0  # estimated cost in USD
    avg_latency_s: float = 10.0  # average response time
    success_rate: float = 1.0  # historical success rate [0, 1]
    max_concurrent: int = 1
    active_count: int = 0
    total_calls: int = 0
    total_failures: int = 0

    def update_stats(self, success: bool, latency: float) -> None:
        """Update running statistics after a call."""
        self.total_calls += 1
        if not success:
            self.total_failures += 1
        # Exponential moving average for latency
        alpha = 0.3
        self.avg_latency_s = alpha * latency + (1 - alpha) * self.avg_latency_s
        # Success rate
        if self.total_calls > 0:
            self.success_rate = 1.0 - (self.total_failures / self.total_calls)

    def is_available(self) -> bool:
        return self.active_count < self.max_concurrent

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "cost_per_call": self.cost_per_call,
            "avg_latency_s": round(self.avg_latency_s, 1),
            "success_rate": round(self.success_rate, 3),
            "total_calls": self.total_calls,
            "available": self.is_available(),
        }


# --- Capability keywords for routing ---

_CAPABILITY_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "implement", "refactor", "debug", "fix", "build", "test",
        "写代码", "实现", "重构", "调试", "修复", "构建",
    ],
    "web": [
        "browse", "scrape", "fetch", "website", "url", "page",
        "浏览", "抓取", "网页", "网站",
    ],
    "analysis": [
        "analyze", "review", "audit", "compare", "evaluate",
        "分析", "审查", "审计", "对比", "评估",
    ],
    "writing": [
        "write", "draft", "document", "summarize", "translate",
        "写", "草拟", "文档", "总结", "翻译",
    ],
}


def _match_capabilities(task: str) -> list[str]:
    """Detect required capabilities from task description."""
    task_lower = task.lower()
    matched = []
    for cap, keywords in _CAPABILITY_KEYWORDS.items():
        if any(kw in task_lower for kw in keywords):
            matched.append(cap)
    return matched or ["general"]


class AgentRegistry:
    """Registry of available sub-agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentProfile] = {}
        self._handlers: dict[str, Callable[[str, dict], Awaitable[Any]]] = {}

    def register(
        self,
        profile: AgentProfile,
        handler: Callable[[str, dict], Awaitable[Any]],
    ) -> None:
        """Register a sub-agent with its handler."""
        self._agents[profile.name] = profile
        self._handlers[profile.name] = handler

    def get(self, name: str) -> AgentProfile | None:
        return self._agents.get(name)

    def list_agents(self) -> list[AgentProfile]:
        return list(self._agents.values())

    def find_best(self, capabilities: list[str]) -> AgentProfile | None:
        """Find the best available agent for given capabilities.

        Scoring: capability_match * success_rate / (cost + 0.01)
        """
        candidates = []
        for agent in self._agents.values():
            if not agent.is_available():
                continue
            cap_match = len(set(capabilities) & set(agent.capabilities))
            if cap_match == 0 and "general" not in agent.capabilities:
                continue
            score = (cap_match + 0.5) * agent.success_rate / (agent.cost_per_call + 0.01)
            candidates.append((score, agent))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


@dataclass
class DelegationResult:
    """Result of a sub-agent delegation."""

    agent_name: str
    task: str
    success: bool
    result: Any = None
    error: str | None = None
    latency_s: float = 0.0
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "task": self.task[:100],
            "success": self.success,
            "latency_s": round(self.latency_s, 1),
            "lessons_count": len(self.lessons),
            "error": self.error,
        }


class SubAgentOrchestrator:
    """High-level orchestrator for sub-agent delegation.

    Usage:
        orch = SubAgentOrchestrator()
        orch.registry.register(
            AgentProfile("claude-code", capabilities=["coding"]),
            my_claude_code_handler,
        )
        result = await orch.delegate("Implement a REST API", context={})
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        learn_from_results: bool = True,
    ) -> None:
        self.registry = registry or AgentRegistry()
        self._learn = learn_from_results
        self._history: list[DelegationResult] = []

    async def delegate(
        self,
        task: str,
        context: dict | None = None,
        agent_name: str | None = None,
        timeout: float = 300.0,
    ) -> DelegationResult:
        """Delegate a task to a sub-agent.

        If agent_name is not specified, auto-routes to best-fit agent.
        """
        ctx = context or {}

        # Route to agent
        if agent_name:
            profile = self.registry.get(agent_name)
            if not profile:
                return DelegationResult(
                    agent_name=agent_name or "unknown", task=task,
                    success=False, error=f"Agent '{agent_name}' not registered",
                )
        else:
            caps = _match_capabilities(task)
            profile = self.registry.find_best(caps)
            if not profile:
                return DelegationResult(
                    agent_name="none", task=task,
                    success=False, error=f"No agent available for capabilities: {caps}",
                )

        handler = self.registry._handlers.get(profile.name)
        if not handler:
            return DelegationResult(
                agent_name=profile.name, task=task,
                success=False, error="No handler registered",
            )

        # Execute
        profile.active_count += 1
        start = time.monotonic()
        try:
            raw_result = await asyncio.wait_for(handler(task, ctx), timeout=timeout)
            latency = time.monotonic() - start
            profile.update_stats(True, latency)

            dr = DelegationResult(
                agent_name=profile.name, task=task,
                success=True, result=raw_result, latency_s=latency,
            )
        except asyncio.TimeoutError:
            latency = time.monotonic() - start
            profile.update_stats(False, latency)
            dr = DelegationResult(
                agent_name=profile.name, task=task,
                success=False, error=f"Timeout after {timeout}s", latency_s=latency,
            )
        except Exception as e:
            latency = time.monotonic() - start
            profile.update_stats(False, latency)
            dr = DelegationResult(
                agent_name=profile.name, task=task,
                success=False, error=str(e), latency_s=latency,
            )
        finally:
            profile.active_count -= 1

        # Learn from result
        if self._learn:
            dr.lessons = self._extract_lessons(dr)

        self._history.append(dr)
        return dr

    async def delegate_parallel(
        self,
        tasks: list[dict[str, Any]],
        max_concurrent: int = 3,
    ) -> list[DelegationResult]:
        """Delegate multiple tasks in parallel.

        Each task dict: {"task": str, "agent": str | None, "context": dict | None}
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _run(t: dict) -> DelegationResult:
            async with sem:
                return await self.delegate(
                    task=t["task"],
                    context=t.get("context"),
                    agent_name=t.get("agent"),
                )

        return await asyncio.gather(*[_run(t) for t in tasks])

    def _extract_lessons(self, result: DelegationResult) -> list[str]:
        """Extract lessons from a delegation result (heuristic)."""
        lessons = []
        if result.success:
            lessons.append(
                f"{result.agent_name} completed '{result.task[:50]}' "
                f"in {result.latency_s:.1f}s"
            )
        else:
            lessons.append(
                f"{result.agent_name} failed '{result.task[:50]}': {result.error}"
            )
            # Suggest fallback if success rate drops
            profile = self.registry.get(result.agent_name)
            if profile and profile.success_rate < 0.7:
                lessons.append(
                    f"Consider alternative to {result.agent_name} "
                    f"(success rate: {profile.success_rate:.0%})"
                )
        return lessons

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent delegation history."""
        return [r.to_dict() for r in self._history[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        """Get orchestration statistics."""
        total = len(self._history)
        successes = sum(1 for r in self._history if r.success)
        return {
            "total_delegations": total,
            "success_rate": successes / max(total, 1),
            "agents": {
                name: profile.to_dict()
                for name, profile in self.registry._agents.items()
            },
            "total_lessons": sum(len(r.lessons) for r in self._history),
        }
