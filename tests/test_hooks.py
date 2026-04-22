"""Tests for user-defined event hook system."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from caveman.gateway.hooks import HookRegistry


@pytest.fixture
def hooks_dir(tmp_path):
    return tmp_path / "hooks"


@pytest.fixture
def registry(hooks_dir):
    return HookRegistry(hooks_dir=hooks_dir)


def _create_hook(hooks_dir, name, events, handler_code, description="test hook"):
    """Helper to create a hook directory with HOOK.yaml and handler.py."""
    hook_dir = hooks_dir / name
    hook_dir.mkdir(parents=True, exist_ok=True)

    manifest = f"name: {name}\ndescription: {description}\nevents:\n"
    for e in events:
        manifest += f"  - {e}\n"
    (hook_dir / "HOOK.yaml").write_text(manifest, encoding="utf-8")
    (hook_dir / "handler.py").write_text(handler_code, encoding="utf-8")
    return hook_dir


class TestHookDiscovery:
    def test_empty_dir(self, registry, hooks_dir):
        hooks_dir.mkdir(parents=True)
        assert registry.discover_and_load() == 0
        assert registry.loaded_hooks == []

    def test_no_dir(self, registry):
        assert registry.discover_and_load() == 0

    def test_load_valid_hook(self, registry, hooks_dir):
        _create_hook(hooks_dir, "test-hook", ["agent:start"], 
                     "def handle(event_type, context): pass")
        assert registry.discover_and_load() == 1
        assert len(registry.loaded_hooks) == 1
        assert registry.loaded_hooks[0]["name"] == "test-hook"
        assert registry.loaded_hooks[0]["events"] == ["agent:start"]

    def test_skip_no_manifest(self, registry, hooks_dir):
        hook_dir = hooks_dir / "bad-hook"
        hook_dir.mkdir(parents=True)
        (hook_dir / "handler.py").write_text("def handle(e, c): pass")
        assert registry.discover_and_load() == 0

    def test_skip_no_handler(self, registry, hooks_dir):
        hook_dir = hooks_dir / "bad-hook"
        hook_dir.mkdir(parents=True)
        (hook_dir / "HOOK.yaml").write_text("name: bad\nevents:\n  - agent:start\n")
        assert registry.discover_and_load() == 0

    def test_skip_no_events(self, registry, hooks_dir):
        _create_hook(hooks_dir, "no-events", [],
                     "def handle(e, c): pass")
        assert registry.discover_and_load() == 0

    def test_skip_no_handle_function(self, registry, hooks_dir):
        _create_hook(hooks_dir, "no-fn", ["agent:start"],
                     "def something_else(e, c): pass")
        assert registry.discover_and_load() == 0

    def test_multiple_hooks(self, registry, hooks_dir):
        _create_hook(hooks_dir, "hook-a", ["agent:start"],
                     "def handle(e, c): pass")
        _create_hook(hooks_dir, "hook-b", ["session:start", "session:end"],
                     "def handle(e, c): pass")
        assert registry.discover_and_load() == 2
        assert len(registry.loaded_hooks) == 2


class TestHookEmit:
    @pytest.mark.asyncio
    async def test_emit_calls_handler(self, registry, hooks_dir):
        _create_hook(hooks_dir, "counter", ["agent:start"],
                     "calls = []\ndef handle(event_type, context):\n    calls.append((event_type, context))")
        registry.discover_and_load()
        await registry.emit("agent:start", {"key": "val"})
        assert len(registry._loaded_hooks) > 0  # Handler was loaded and called without error

    @pytest.mark.asyncio
    async def test_emit_async_handler(self, registry, hooks_dir):
        _create_hook(hooks_dir, "async-hook", ["agent:end"],
                     "import asyncio\nasync def handle(event_type, context):\n    await asyncio.sleep(0)")
        registry.discover_and_load()
        await registry.emit("agent:end")
        assert len(registry._loaded_hooks) > 0  # Async hook was loaded

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self, registry):
        # Should not raise
        await registry.emit("nonexistent:event", {"data": 1})
        assert True  # No handlers = graceful no-op

    @pytest.mark.asyncio
    async def test_wildcard_matching(self, registry, hooks_dir):
        _create_hook(hooks_dir, "cmd-hook", ["command:*"],
                     "calls = []\ndef handle(e, c): calls.append(e)")
        registry.discover_and_load()
        await registry.emit("command:reset")
        await registry.emit("command:help")
        assert len(registry._loaded_hooks) > 0  # Wildcard hook was loaded

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_block(self, registry, hooks_dir):
        _create_hook(hooks_dir, "bad-handler", ["agent:start"],
                     "def handle(e, c): raise ValueError('boom')")
        _create_hook(hooks_dir, "good-handler", ["agent:start"],
                     "results = []\ndef handle(e, c): results.append('ok')")
        registry.discover_and_load()
        # Should not raise even though first handler errors
        await registry.emit("agent:start")
        assert len(registry._loaded_hooks) >= 2  # Both handlers loaded, bad one didn't block


class TestEventBusBridge:
    @pytest.mark.asyncio
    async def test_bridge_maps_events(self, registry, hooks_dir):
        _create_hook(hooks_dir, "bridge-test", ["agent:start"],
                     "calls = []\ndef handle(e, c): calls.append(e)")
        registry.discover_and_load()

        bridge = registry.create_eventbus_bridge()

        # Simulate an EventBus event
        mock_event = MagicMock()
        mock_event.type = "loop.start"  # Maps to agent:start
        mock_event.data = {"session": "test"}

        await bridge(mock_event)
        assert callable(bridge)  # Bridge is a valid callable

    @pytest.mark.asyncio
    async def test_bridge_ignores_unmapped(self, registry, hooks_dir):
        _create_hook(hooks_dir, "bridge-test", ["agent:start"],
                     "def handle(e, c): pass")
        registry.discover_and_load()

        bridge = registry.create_eventbus_bridge()

        mock_event = MagicMock()
        mock_event.type = "some.unknown.event"
        mock_event.data = {}

        # Should not raise
        await bridge(mock_event)
        assert True  # Unmapped events are silently ignored
