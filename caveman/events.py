"""Event bus — decouple everything from the agent loop.

Core pattern: producers emit events, consumers subscribe.
The agent loop ONLY emits. Display, logging, metrics, auditing, plugins
all subscribe independently. Zero coupling.

Usage:
    bus = EventBus()
    bus.on("tool_call", my_handler)        # Subscribe
    await bus.emit("tool_call", data)      # Emit
    bus.off("tool_call", my_handler)       # Unsubscribe

Events are typed dataclasses for safety. Raw dicts also supported for plugins.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ── Event Types ──

class EventType(str, Enum):
    """All events the agent loop can emit."""
    # Lifecycle
    LOOP_START = "loop.start"
    LOOP_END = "loop.end"
    ITERATION_START = "iteration.start"
    ITERATION_END = "iteration.end"

    # LLM
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_DELTA = "llm.stream.delta"
    LLM_ERROR = "llm.error"

    # Tools
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Memory
    MEMORY_RECALL = "memory.recall"
    MEMORY_STORE = "memory.store"
    MEMORY_NUDGE = "memory.nudge"

    # Skills
    SKILL_MATCH = "skill.match"
    SKILL_NUDGE = "skill.nudge"
    SKILL_OUTCOME = "skill.outcome"

    # Context
    CONTEXT_COMPRESS = "context.compress"
    CONTEXT_UTILIZATION = "context.utilization"

    # Trajectory
    TRAJECTORY_TURN = "trajectory.turn"
    TRAJECTORY_SAVE = "trajectory.save"

    # Shield
    SHIELD_UPDATE = "shield.update"

    # Nudge
    NUDGE_EXTRACT = "nudge.extract"

    # Ripple
    RIPPLE_PROPAGATE = "ripple.propagate"

    # Lint
    LINT_SCAN = "lint.scan"

    # Security
    PERMISSION_CHECK = "permission.check"
    SECRET_DETECTED = "secret.detected"


@dataclass
class Event:
    """A typed event with structured data."""
    type: EventType | str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Which module emitted this


# ── Subscriber types ──

SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
Handler = SyncHandler | AsyncHandler


# ── Event Bus ──

class EventBus:
    """Central event bus — async-first, sync-compatible.

    Thread-safe for subscription. Emission is async.
    Handlers are called in subscription order.
    A failing handler does NOT block other handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._global_handlers: list[Handler] = []  # Receive ALL events
        self._metrics = _BusMetrics()

    def on(self, event_type: EventType | str, handler: Handler) -> None:
        """Subscribe to a specific event type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        self._handlers.setdefault(key, []).append(handler)

    def on_all(self, handler: Handler) -> None:
        """Subscribe to ALL events (for logging, metrics, audit)."""
        self._global_handlers.append(handler)

    def off(self, event_type: EventType | str, handler: Handler) -> None:
        """Unsubscribe from a specific event type."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        handlers = self._handlers.get(key, [])
        if handler in handlers:
            handlers.remove(handler)

    def off_all(self, handler: Handler) -> None:
        """Unsubscribe a global handler."""
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    async def emit(self, event_type: EventType | str, data: dict[str, Any] | None = None, source: str = "") -> None:
        """Emit an event to all subscribers. Non-blocking, fault-tolerant."""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        event = Event(type=key, data=data or {}, source=source)

        self._metrics.events_emitted += 1

        handlers = self._handlers.get(key, []) + self._global_handlers
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                self._metrics.handlers_succeeded += 1
            except Exception as e:
                self._metrics.handlers_failed += 1
                logger.warning("Event handler failed for %s: %s", key, e)

    @property
    def stats(self) -> dict[str, int]:
        """Bus health metrics."""
        return {
            "events_emitted": self._metrics.events_emitted,
            "handlers_succeeded": self._metrics.handlers_succeeded,
            "handlers_failed": self._metrics.handlers_failed,
            "subscriptions": sum(len(h) for h in self._handlers.values()) + len(self._global_handlers),
        }


@dataclass
class _BusMetrics:
    events_emitted: int = 0
    handlers_succeeded: int = 0
    handlers_failed: int = 0


# ── Built-in subscribers ──

def logging_subscriber(event: Event) -> None:
    """Log all events at DEBUG level (structured)."""
    logger.debug(
        "event=%s source=%s data_keys=%s",
        event.type, event.source, list(event.data.keys()),
    )


class MetricsCollector:
    """Lightweight in-process metrics — no external dependencies.

    Tracks: event counts, tool call durations, LLM latencies, error rates.
    Query via .snapshot() for dashboards/health checks.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}
        self._start_times: dict[str, float] = {}

    def handle(self, event: Event) -> None:
        """Process an event and update metrics."""
        etype = event.type

        # Count every event
        self._counters[etype] = self._counters.get(etype, 0) + 1

        # Track timing for paired start/end events
        if etype == EventType.TOOL_CALL.value:
            call_id = event.data.get("call_id", "")
            self._start_times[f"tool:{call_id}"] = event.timestamp
        elif etype == EventType.TOOL_RESULT.value:
            call_id = event.data.get("call_id", "")
            start = self._start_times.pop(f"tool:{call_id}", None)
            if start:
                duration = event.timestamp - start
                self._timings.setdefault("tool_duration", []).append(duration)

        if etype == EventType.LLM_REQUEST.value:
            self._start_times["llm"] = event.timestamp
        elif etype == EventType.LLM_RESPONSE.value:
            start = self._start_times.pop("llm", None)
            if start:
                duration = event.timestamp - start
                self._timings.setdefault("llm_latency", []).append(duration)

        # Track errors
        if etype in (EventType.TOOL_ERROR.value, EventType.LLM_ERROR.value):
            self._counters["total_errors"] = self._counters.get("total_errors", 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """Current metrics snapshot."""
        result: dict[str, Any] = {"counters": dict(self._counters)}
        for name, values in self._timings.items():
            if values:
                sorted_vals = sorted(values)
                result[f"{name}_avg"] = sum(values) / len(values)
                p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
                result[f"{name}_p95"] = sorted_vals[p95_idx]
                result[f"{name}_count"] = len(values)
        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._timings.clear()
        self._start_times.clear()


# ── Convenience: create a pre-wired bus ──

def create_default_bus(
    enable_logging: bool = True,
    enable_metrics: bool = True,
    enable_persistence: bool = False,
) -> tuple[EventBus, MetricsCollector | None]:
    """Create a bus with standard subscribers attached.

    When enable_persistence is True, the EventStore is attached as bus.event_store.
    """
    bus = EventBus()
    metrics = None

    if enable_logging:
        bus.on_all(logging_subscriber)

    if enable_metrics:
        metrics = MetricsCollector()
        bus.on_all(metrics.handle)

    if enable_persistence:
        from caveman.events_store import EventStore
        store = EventStore()
        bus.on_all(store.handle)
        bus.event_store = store  # type: ignore[attr-defined]

    return bus, metrics
