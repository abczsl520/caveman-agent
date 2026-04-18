"""Tests for the event bus and event-driven agent loop."""
import asyncio

import pytest


# ── EventBus core ──

def test_event_bus_subscribe_emit():
    from caveman.events import EventBus, EventType

    received = []

    def handler(event):
        received.append(event.type)

    bus = EventBus()
    bus.on(EventType.TOOL_CALL, handler)

    asyncio.run(bus.emit(EventType.TOOL_CALL, {"name": "bash"}))
    assert received == [EventType.TOOL_CALL.value]


def test_event_bus_unsubscribe():
    from caveman.events import EventBus, EventType

    received = []

    def handler(event):
        received.append(event.type)

    bus = EventBus()
    bus.on(EventType.TOOL_CALL, handler)
    bus.off(EventType.TOOL_CALL, handler)

    asyncio.run(bus.emit(EventType.TOOL_CALL))
    assert received == []


def test_event_bus_global_handler():
    from caveman.events import EventBus, EventType

    received = []

    def handler(event):
        received.append(event.type)

    bus = EventBus()
    bus.on_all(handler)

    asyncio.run(bus.emit(EventType.TOOL_CALL))
    asyncio.run(bus.emit(EventType.MEMORY_STORE))
    assert len(received) == 2


def test_event_bus_async_handler():
    from caveman.events import EventBus, EventType

    received = []

    async def handler(event):
        received.append(event.type)

    bus = EventBus()
    bus.on(EventType.LLM_REQUEST, handler)

    asyncio.run(bus.emit(EventType.LLM_REQUEST))
    assert received == [EventType.LLM_REQUEST.value]


def test_event_bus_handler_failure_non_blocking():
    from caveman.events import EventBus, EventType

    received = []

    def bad_handler(event):
        raise ValueError("Boom!")

    def good_handler(event):
        received.append("ok")

    bus = EventBus()
    bus.on(EventType.TOOL_CALL, bad_handler)
    bus.on(EventType.TOOL_CALL, good_handler)

    # Should not raise, and good_handler should still run
    asyncio.run(bus.emit(EventType.TOOL_CALL))
    assert received == ["ok"]
    assert bus.stats["handlers_failed"] == 1


def test_event_bus_stats():
    from caveman.events import EventBus, EventType

    bus = EventBus()
    bus.on(EventType.TOOL_CALL, lambda e: None)

    asyncio.run(bus.emit(EventType.TOOL_CALL))
    asyncio.run(bus.emit(EventType.TOOL_CALL))

    stats = bus.stats
    assert stats["events_emitted"] == 2
    assert stats["handlers_succeeded"] == 2
    assert stats["subscriptions"] == 1


def test_event_bus_string_events():
    """Support custom string events for plugins."""
    from caveman.events import EventBus

    received = []
    bus = EventBus()
    bus.on("custom.plugin.event", lambda e: received.append(e.data))

    asyncio.run(bus.emit("custom.plugin.event", {"key": "value"}))
    assert received == [{"key": "value"}]


# ── MetricsCollector ──

def test_metrics_collector_counts():
    from caveman.events import EventBus, EventType, MetricsCollector

    metrics = MetricsCollector()
    bus = EventBus()
    bus.on_all(metrics.handle)

    asyncio.run(bus.emit(EventType.TOOL_CALL, {"call_id": "1"}))
    asyncio.run(bus.emit(EventType.TOOL_RESULT, {"call_id": "1"}))
    asyncio.run(bus.emit(EventType.TOOL_ERROR, {"call_id": "2"}))

    snap = metrics.snapshot()
    assert snap["counters"][EventType.TOOL_CALL.value] == 1
    assert snap["counters"][EventType.TOOL_RESULT.value] == 1
    assert snap["counters"]["total_errors"] == 1


def test_metrics_collector_timing():
    import time
    from caveman.events import Event, EventType, MetricsCollector

    metrics = MetricsCollector()
    t0 = time.time()

    metrics.handle(Event(type=EventType.TOOL_CALL.value, data={"call_id": "x"}, timestamp=t0))
    metrics.handle(Event(type=EventType.TOOL_RESULT.value, data={"call_id": "x"}, timestamp=t0 + 0.5))

    snap = metrics.snapshot()
    assert "tool_duration_avg" in snap
    assert abs(snap["tool_duration_avg"] - 0.5) < 0.01


def test_metrics_reset():
    from caveman.events import EventBus, EventType, MetricsCollector

    metrics = MetricsCollector()
    bus = EventBus()
    bus.on_all(metrics.handle)

    asyncio.run(bus.emit(EventType.TOOL_CALL))
    metrics.reset()
    assert metrics.snapshot()["counters"] == {}


# ── create_default_bus ──

def test_create_default_bus():
    from caveman.events import create_default_bus

    bus, metrics = create_default_bus()
    assert metrics is not None
    # Should have logging + metrics subscribers
    assert bus.stats["subscriptions"] == 2


# ── AgentLoop has bus ──

def test_agent_loop_has_event_bus():
    from caveman.agent.loop import AgentLoop

    loop = AgentLoop()
    assert hasattr(loop, "bus")
    assert loop.bus is not None
    assert loop.bus.stats["subscriptions"] >= 1  # At least logging


def test_agent_loop_custom_bus():
    from caveman.agent.loop import AgentLoop
    from caveman.events import EventBus

    custom_bus = EventBus()
    loop = AgentLoop(event_bus=custom_bus)
    assert loop.bus is custom_bus


# ── Event types completeness ──

def test_all_event_types_are_strings():
    from caveman.events import EventType

    for et in EventType:
        assert isinstance(et.value, str)
        assert "." in et.value  # All events follow namespace.action pattern


def test_event_dataclass():
    from caveman.events import Event, EventType
    import time

    event = Event(type=EventType.TOOL_CALL.value, data={"name": "bash"}, source="test")
    assert event.type == "tool.call"
    assert event.data["name"] == "bash"
    assert event.source == "test"
    assert event.timestamp <= time.time()
