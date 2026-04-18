"""Lifecycle manager — unified startup, shutdown, and resource cleanup.

Problem: Resources scattered across modules (SQLite, browser, UDS server, gateways)
with no coordinated startup/shutdown order. Ctrl+C kills without cleanup.

Solution: Central lifecycle manager that:
1. Registers all resources with their cleanup functions
2. Starts them in dependency order
3. Shuts them down in reverse order (LIFO)
4. Handles signals (SIGINT, SIGTERM) for graceful shutdown

Usage:
    lifecycle = Lifecycle()
    lifecycle.register("database", db, startup=db.connect, shutdown=db.close)
    lifecycle.register("server", server, startup=server.start, shutdown=server.stop)

    async with lifecycle:
        await run_forever()
    # All resources cleaned up, even on Ctrl+C
"""
from __future__ import annotations
import asyncio
import logging
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class _Resource:
    """A managed resource with lifecycle hooks."""
    name: str
    instance: Any
    startup: Callable[[], Awaitable[None]] | Callable[[], None] | None = None
    shutdown: Callable[[], Awaitable[None]] | Callable[[], None] | None = None
    started: bool = False


class Lifecycle:
    """Manages startup/shutdown of all framework resources.

    Resources are started in registration order and shut down in reverse (LIFO).
    Handles SIGINT/SIGTERM for graceful shutdown.
    """

    def __init__(self) -> None:
        self._resources: list[_Resource] = []
        self._running = False
        self._shutdown_event: asyncio.Event | None = None

    def register(
        self,
        name: str,
        instance: Any = None,
        startup: Callable | None = None,
        shutdown: Callable | None = None,
    ) -> None:
        """Register a resource for lifecycle management."""
        self._resources.append(_Resource(
            name=name, instance=instance,
            startup=startup, shutdown=shutdown,
        ))
        logger.debug("Lifecycle: registered %s", name)

    async def start_all(self) -> None:
        """Start all registered resources in order."""
        logger.info("Lifecycle: starting %d resources...", len(self._resources))
        for resource in self._resources:
            if resource.startup:
                try:
                    result = resource.startup()
                    if asyncio.iscoroutine(result):
                        await result
                    resource.started = True
                    logger.info("Lifecycle: started %s", resource.name)
                except Exception as e:
                    logger.error("Lifecycle: failed to start %s: %s", resource.name, e)
                    # Shut down already-started resources
                    await self.shutdown_all()
                    raise
        self._running = True

    async def shutdown_all(self) -> None:
        """Shut down all started resources in reverse order (LIFO)."""
        if not self._running and not any(r.started for r in self._resources):
            return

        logger.info("Lifecycle: shutting down...")
        for resource in reversed(self._resources):
            if resource.started and resource.shutdown:
                try:
                    result = resource.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                    resource.started = False
                    logger.info("Lifecycle: stopped %s", resource.name)
                except Exception as e:
                    logger.warning("Lifecycle: error stopping %s: %s", resource.name, e)
        self._running = False
        logger.info("Lifecycle: all resources stopped")

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        self._shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _signal_handler(sig: signal.Signals) -> None:
            logger.info("Lifecycle: received %s, initiating graceful shutdown...", sig.name)
            if self._shutdown_event:
                self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler, sig)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown signal received."""
        if self._shutdown_event:
            await self._shutdown_event.wait()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> dict[str, bool]:
        """Status of all registered resources."""
        return {r.name: r.started for r in self._resources}

    # Context manager support
    async def __aenter__(self) -> "Lifecycle":
        self.install_signal_handlers()
        await self.start_all()
        return self

    async def __aexit__(self, *args) -> None:
        await self.shutdown_all()
