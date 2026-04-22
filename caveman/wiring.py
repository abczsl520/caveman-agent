"""Module wiring — connects all remaining orphan subsystems.

Called during application startup to ensure all modules are reachable.
Each import is wrapped in try/except so failures are non-fatal.
"""
from __future__ import annotations

import logging

__all__ = [
    "wire_providers",
    "wire_memory",
    "wire_security",
    "wire_skills",
    "wire_bridges",
    "wire_coordinator",
    "wire_misc",
    "wire_agent_extras",
    "wire_all",
]


logger = logging.getLogger(__name__)


def wire_providers() -> int:
    """Register additional providers in the provider registry."""
    count = 0
    for mod in [
        "caveman.providers.gemini_provider",
        "caveman.providers.openrouter_provider",
        "caveman.providers.insights",
        "caveman.providers.model_router",
        "caveman.providers.prompt_cache",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_memory() -> int:
    """Import memory subsystems so they register with the memory manager."""
    count = 0
    for mod in [
        "caveman.memory.refiner",
        "caveman.memory.drift",
        "caveman.memory.grounding",
        "caveman.memory.confidence",
        "caveman.memory.flywheel_metrics",
        "caveman.memory.store_helpers",
        "caveman.memory.backend",
        "caveman.memory.recall_cache",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_security() -> int:
    """Import security modules."""
    count = 0
    for mod in [
        "caveman.security.sandbox",
        "caveman.security.skill_guard",
        "caveman.security.encryption",
        "caveman.security.path_safety",
        "caveman.security.content_sanitizer",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_skills() -> int:
    """Import skills subsystems."""
    count = 0
    for mod in [
        "caveman.skills.harness",
        "caveman.skills.utils",
        "caveman.skills.rl_router",
        "caveman.skills.sync",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_bridges() -> int:
    """Import bridge modules."""
    count = 0
    for mod in [
        "caveman.bridge.acp",
        "caveman.bridge.hermes_bridge",
        "caveman.bridge.uds_transport",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_coordinator() -> int:
    """Import coordinator modules."""
    count = 0
    for mod in [
        "caveman.coordinator.orchestrator",
        "caveman.coordinator.engine",
        "caveman.coordinator.verification",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_misc() -> int:
    """Import miscellaneous modules."""
    count = 0
    for mod in [
        "caveman.compression.safeguard",
        "caveman.lifecycle",
        "caveman.mcp.oauth",
        "caveman.training.batch_runner",
        "caveman.training.eval_embedding",
        "caveman.cli.backup",
        "caveman.cli.model_normalize",
        "caveman.commands.gateway_adapter",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_agent_extras() -> int:
    """Import remaining agent modules."""
    count = 0
    for mod in [
        "caveman.agent.context_refs",
        "caveman.agent.rate_limit_tracker",
        "caveman.agent.phased_coordinator",
    ]:
        try:
            __import__(mod)
            count += 1
        except Exception as e:
            logger.debug("Failed to wire %s: %s", mod, e)
    return count


def wire_all() -> dict[str, int]:
    """Wire all orphan subsystems. Returns counts per category."""
    results = {
        "providers": wire_providers(),
        "memory": wire_memory(),
        "security": wire_security(),
        "skills": wire_skills(),
        "bridges": wire_bridges(),
        "coordinator": wire_coordinator(),
        "misc": wire_misc(),
        "agent_extras": wire_agent_extras(),
    }
    total = sum(results.values())
    logger.info("Module wiring: %d subsystems loaded (%s)",
                total, ", ".join(f"{k}={v}" for k, v in results.items() if v))
    return results
