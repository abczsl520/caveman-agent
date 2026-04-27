"""Regression tests for hot-reload safety across long-lived gateway objects."""
from __future__ import annotations

import asyncio
import importlib
import os
import signal
from enum import Enum


class ReloadedEventType(str, Enum):
    TOOL_CALL = "tool.call"


class ReloadedTaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


class ReloadedDeliveryStatus(str, Enum):
    PENDING = "pending"


async def _emit(bus, event_type):
    await bus.emit(event_type, {"call_id": "abc"}, source="test")


def test_event_bus_routes_semantically_equivalent_reloaded_event_type():
    from caveman.events import EventBus, EventType

    bus = EventBus()
    seen = []
    bus.on(EventType.TOOL_CALL, lambda event: seen.append(event.type))

    asyncio.run(_emit(bus, ReloadedEventType.TOOL_CALL))

    assert seen == ["tool.call"]


def test_event_bus_can_unsubscribe_with_reloaded_event_type():
    from caveman.events import EventBus, EventType

    bus = EventBus()
    seen = []

    def handler(event):
        seen.append(event.type)

    bus.on(EventType.TOOL_CALL, handler)
    bus.off(ReloadedEventType.TOOL_CALL, handler)
    asyncio.run(_emit(bus, EventType.TOOL_CALL))

    assert seen == []


def test_task_registry_accepts_semantically_equivalent_reloaded_statuses():
    from caveman.gateway.task_registry import TaskRecord, TaskRegistry

    registry = TaskRegistry()
    task = TaskRecord(task_id="t1", title="old enum task")
    task.status = ReloadedTaskStatus.CREATED
    registry._tasks[task.task_id] = task

    assert registry.start_task("t1") is True
    assert task.status.value == "running"


def test_task_registry_pending_delivery_accepts_reloaded_statuses():
    from caveman.gateway.task_registry import TaskRecord, TaskRegistry

    registry = TaskRegistry()
    task = TaskRecord(task_id="t2", title="old enum delivery")
    task.status = ReloadedTaskStatus.COMPLETED
    task.delivery_status = ReloadedDeliveryStatus.PENDING
    registry._tasks[task.task_id] = task

    pending = registry.get_pending_deliveries()

    assert pending == [task]


def test_sigusr2_is_full_restart_request_not_in_process_module_reload(monkeypatch):
    """SIGUSR2 used to importlib.reload every caveman.* module in-process.

    That is unsafe for long-lived gateway sessions: old loops, registries,
    engines, and Enums survive while modules/classes change underneath them.
    The safe long-term behavior is to request the same graceful full restart as
    SIGUSR1, never broad reload modules in place.
    """
    import caveman.gateway.gateway_lifecycle as lifecycle

    callbacks = {}

    class FakeLoop:
        def add_signal_handler(self, sig, cb):
            callbacks[sig] = cb

    async def fake_run_gateway(_config_path):
        while not lifecycle._restart_requested:
            await asyncio.sleep(0)

    class FakeHealthServer:
        def __init__(self, *args, **kwargs):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    class FakeConfigWatcher:
        def __init__(self, *args, **kwargs):
            self.callback = None

        def on_change(self, callback):
            self.callback = callback

        def start(self):
            pass

        def stop(self):
            pass

    reloaded = []
    os_execv_calls = []
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr("caveman.gateway.runner.run_gateway", fake_run_gateway)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.write_pid_file", lambda: None, raising=False)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.write_runtime_state", lambda **_: None, raising=False)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.remove_pid_file", lambda: None, raising=False)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.get_running_pid", lambda: None, raising=False)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.write_restart_sentinel", lambda **_: None, raising=False)
    monkeypatch.setattr("caveman.gateway.gateway_lifecycle.drain_active_sessions", lambda *_, **__: (0, False), raising=False)
    monkeypatch.setattr("caveman.gateway.health.HealthServer", FakeHealthServer)
    monkeypatch.setattr("caveman.config.watcher.ConfigWatcher", FakeConfigWatcher)
    monkeypatch.setattr(importlib, "reload", lambda mod: reloaded.append(mod))
    monkeypatch.setattr(os, "execv", lambda *args: (_ for _ in ()).throw(SystemExit(os_execv_calls.append(args) or 0)))

    async def trigger_and_run():
        task = asyncio.create_task(lifecycle.run_gateway_forever(max_restarts=1))
        await asyncio.sleep(0)
        try:
            assert signal.SIGUSR2 in callbacks
            callbacks[signal.SIGUSR2]()
            assert lifecycle._restart_requested is True
            assert reloaded == []
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(trigger_and_run())
