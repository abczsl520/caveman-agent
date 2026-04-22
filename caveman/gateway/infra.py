"""Gateway infrastructure — hooks, task registry, and pipeline integration.

Extracted from runner.py to keep it under 450 lines.
Provides lazy-loaded singletons for hooks and task registry.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("caveman.gateway")


class GatewayInfra:
    """Lazy-loaded gateway infrastructure components."""

    def __init__(self) -> None:
        self._hooks_registry = None
        self._task_registry = None

    def load_hooks(self) -> None:
        """Discover and load user-defined hooks."""
        try:
            from caveman.gateway.hooks import HookRegistry
            self._hooks_registry = HookRegistry()
            count = self._hooks_registry.discover_and_load()
            if count:
                logger.info("Loaded %d user hook(s)", count)
        except Exception as e:
            logger.debug("Hook loading skipped: %s", e)

    def load_task_registry(self) -> None:
        """Initialize the task registry."""
        try:
            from caveman.gateway.task_registry import TaskRegistry
            self._task_registry = TaskRegistry()
            self._task_registry.load()
            logger.debug("Task registry loaded (%d tasks)", len(self._task_registry.list_tasks()))
        except Exception as e:
            logger.debug("Task registry init skipped: %s", e)

    async def emit_hook(self, event_type: str, context: dict[str, Any] | None = None) -> None:
        """Emit a hook event (no-op if hooks not loaded)."""
        if self._hooks_registry:
            try:
                await self._hooks_registry.emit(event_type, context or {})
            except Exception as e:
                logger.debug("Hook emit error: %s", e)

    @property
    def hooks(self) -> Any:
        return self._hooks_registry

    @property
    def task_registry(self) -> Any:
        return self._task_registry
