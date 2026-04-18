"""Tests for Round 7: browser @tool migration, provider streaming fix, provider unification."""
import asyncio
import pytest


# ── Browser @tool migration ──

def test_browser_registered_via_auto_discover():
    """Browser should be auto-discovered as a @tool-decorated function."""
    from caveman.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry._register_builtins()
    assert "browser" in registry.list_tools()
    assert registry.tool_count >= 7  # 6 original + browser


def test_browser_schema_in_registry():
    """Browser schema should match @tool declaration."""
    from caveman.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry._register_builtins()
    schemas = {s["name"]: s for s in registry.get_schemas()}
    browser = schemas["browser"]
    assert "action" in browser["input_schema"]["properties"]
    actions = browser["input_schema"]["properties"]["action"]["enum"]
    assert "navigate" in actions
    assert "snapshot" in actions
    assert "click" in actions
    assert "close" in actions


def test_browser_dispatch_via_registry():
    """Browser should be callable through the registry dispatch."""
    async def _run():
        from caveman.tools.registry import ToolRegistry
        registry = ToolRegistry()
        registry._register_builtins()
        # Close is safe even without a real browser
        result = await registry.dispatch("browser", {"action": "close"})
        assert result["ok"]
    asyncio.run(_run())


def test_browser_bridge_mode():
    """Browser should use bridge when available."""
    async def _run():
        from caveman.tools.builtin.browser import browser_dispatch, set_bridge

        class MockBridge:
            async def call_tool(self, name, args):
                return {"result": {"snapshot": "tree..."}}

        set_bridge(MockBridge())
        result = await browser_dispatch(action="snapshot")
        assert result["ok"]
        set_bridge(None)
    asyncio.run(_run())


def test_browser_all_actions_via_bridge():
    """All browser actions should route through bridge when set."""
    async def _run():
        from caveman.tools.builtin.browser import browser_dispatch, set_bridge

        calls = []
        class SpyBridge:
            async def call_tool(self, name, args):
                calls.append(args["action"])
                return {"result": "ok"}

        set_bridge(SpyBridge())

        await browser_dispatch(action="navigate", url="https://x.com")
        await browser_dispatch(action="snapshot")
        await browser_dispatch(action="click", ref="e1")
        await browser_dispatch(action="type", ref="e2", text="hello")
        await browser_dispatch(action="screenshot")
        await browser_dispatch(action="evaluate", js="1+1")

        assert len(calls) == 6
        set_bridge(None)
    asyncio.run(_run())


def test_browser_close_clears_state():
    """Close should clear Playwright state."""
    async def _run():
        from caveman.tools.builtin.browser import browser_dispatch, _playwright_ctx
        _playwright_ctx.update(pw=None, browser=None, page=None)
        result = await browser_dispatch(action="close")
        assert result["ok"]
        assert _playwright_ctx["page"] is None
    asyncio.run(_run())


# ── Provider _build_params unification ──

def test_anthropic_build_params():
    """Anthropic provider should build params consistently."""
    from caveman.providers.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="test")
    params = p._build_params(
        messages=[{"role": "user", "content": "hi"}],
        system="You are helpful",
        tools=[{"name": "bash", "description": "Run bash", "input_schema": {}}],
    )
    assert params["model"] == p.model
    assert params["system"] == "You are helpful"
    assert params["tools"] is not None
    assert "messages" in params


def test_anthropic_build_params_no_optionals():
    """Omitted system/tools should not appear in params."""
    from caveman.providers.anthropic_provider import AnthropicProvider
    p = AnthropicProvider(api_key="test")
    params = p._build_params(messages=[{"role": "user", "content": "hi"}])
    assert "system" not in params
    assert "tools" not in params


def test_openai_build_params():
    """OpenAI provider should build params with system message prepended."""
    from caveman.providers.openai_provider import OpenAIProvider
    p = OpenAIProvider(api_key="test")
    params = p._build_params(
        messages=[{"role": "user", "content": "hi"}],
        system="You are helpful",
    )
    assert params["messages"][0]["role"] == "system"
    assert params["messages"][0]["content"] == "You are helpful"
    assert params["messages"][1]["role"] == "user"


def test_openai_build_params_tool_format():
    """OpenAI should convert tool schema to OpenAI function format."""
    from caveman.providers.openai_provider import OpenAIProvider
    p = OpenAIProvider(api_key="test")
    params = p._build_params(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "bash", "description": "Run bash", "input_schema": {"type": "object"}}],
    )
    assert params["tools"][0]["type"] == "function"
    assert params["tools"][0]["function"]["name"] == "bash"
    assert params["tool_choice"] == "auto"


def test_openai_build_params_no_tools():
    """No tools → no tools/tool_choice in params."""
    from caveman.providers.openai_provider import OpenAIProvider
    p = OpenAIProvider(api_key="test")
    params = p._build_params(messages=[{"role": "user", "content": "hi"}])
    assert "tools" not in params or params.get("tools") is None
    assert "tool_choice" not in params


# ── Provider no longer buffers stream for retry ──

def test_anthropic_complete_no_buffer_retry():
    """Anthropic complete() should NOT collect events into list for retry."""
    from pathlib import Path
    source = Path("caveman/providers/anthropic_provider.py").read_text()
    # Should NOT have the old retry-wrap-collect pattern
    assert "results = []" not in source
    assert "results.append" not in source
    # Should use smart error classification for retry decisions
    assert "classify_error" in source


def test_openai_complete_no_buffer_retry():
    """OpenAI complete() should NOT collect events into list for retry."""
    from pathlib import Path
    source = Path("caveman/providers/openai_provider.py").read_text()
    assert "results = []" not in source
    assert "results.append" not in source
    assert "classify_error" in source


# ── Provider base class ──

def test_provider_base_requires_build_params():
    """LLMProvider subclass must implement _build_params."""
    from caveman.providers.llm import LLMProvider

    class IncompleteProvider(LLMProvider):
        model = "test"
        max_tokens = 1000
        @property
        def context_length(self): return 1000
        def _get_client(self): return None
        async def complete(self, messages, **kwargs):
            yield {"type": "done"}

    with pytest.raises(TypeError, match="_build_params"):
        IncompleteProvider()


def test_provider_model_info_consistent():
    """Both providers should have same model_info keys."""
    from caveman.providers.anthropic_provider import AnthropicProvider
    from caveman.providers.openai_provider import OpenAIProvider

    a = AnthropicProvider(api_key="test")
    o = OpenAIProvider(api_key="test")
    assert set(a.model_info.keys()) == set(o.model_info.keys())
    assert a.model_info["provider"] == "AnthropicProvider"
    assert o.model_info["provider"] == "OpenAIProvider"


# ── Config validator edge cases ──

def test_config_validate_deeply_nested_missing():
    """Missing intermediate dict levels should not crash."""
    from caveman.config.validator import validate_config
    # Config with empty sections — should not crash
    config = {"agent": {}, "providers": {}}
    warnings = validate_config(config, strict=False)
    assert isinstance(warnings, list)  # No crash


def test_config_validate_completely_empty():
    """Empty config should validate cleanly (all optional)."""
    from caveman.config.validator import validate_config
    warnings = validate_config({})
    assert warnings == []


def test_config_validate_multiple_errors():
    """Multiple validation errors should all be reported."""
    from caveman.config.validator import validate_config
    config = {
        "agent": {"max_iterations": "bad", "default_model": 42},
        "memory": {"backend": "nosql"},
        "unknown_section": True,
    }
    warnings = validate_config(config, strict=False)
    assert len(warnings) >= 3  # type error + type error + choices + unknown key


# ── Event bus edge cases ──

def test_event_bus_handler_exception_isolation():
    """One failing handler should not prevent others from running."""
    async def _run():
        from caveman.events import EventBus
        results = []

        def good_handler(event):
            results.append("good")

        def bad_handler(event):
            raise RuntimeError("I fail!")

        bus = EventBus()
        bus.on("test", bad_handler)
        bus.on("test", good_handler)

        await bus.emit("test")
        assert "good" in results
    asyncio.run(_run())


def test_event_bus_off_unknown_handler():
    """Unsubscribing a non-registered handler should not crash."""
    from caveman.events import EventBus
    bus = EventBus()
    bus.off("test", lambda e: None)  # Should not raise


def test_event_bus_emit_unknown_type():
    """Emitting an event with no subscribers should work silently."""
    async def _run():
        from caveman.events import EventBus
        bus = EventBus()
        await bus.emit("nonexistent.event", {"data": "test"})
    asyncio.run(_run())


def test_event_bus_async_handler():
    """Async handlers should be properly awaited."""
    async def _run():
        from caveman.events import EventBus
        results = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            results.append(event.data.get("value"))

        bus = EventBus()
        bus.on("test", async_handler)
        await bus.emit("test", {"value": 42})
        assert results == [42]
    asyncio.run(_run())


def test_metrics_collector_snapshot():
    """MetricsCollector should track events correctly."""
    from caveman.events import MetricsCollector, Event, EventType
    import time

    mc = MetricsCollector()

    # Simulate tool call → result
    mc.handle(Event(type=EventType.TOOL_CALL.value, data={"call_id": "c1"}, timestamp=time.time()))
    mc.handle(Event(type=EventType.TOOL_RESULT.value, data={"call_id": "c1"}, timestamp=time.time() + 0.5))

    snap = mc.snapshot()
    assert snap["counters"][EventType.TOOL_CALL.value] == 1
    assert snap["counters"][EventType.TOOL_RESULT.value] == 1
    assert "tool_duration_avg" in snap

    mc.reset()
    assert mc.snapshot()["counters"] == {}


# ── Lifecycle edge cases ──

def test_lifecycle_double_shutdown():
    """Calling shutdown twice should not crash."""
    async def _run():
        from caveman.lifecycle import Lifecycle
        lc = Lifecycle()
        lc.register("x", startup=lambda: None, shutdown=lambda: None)
        await lc.start_all()
        await lc.shutdown_all()
        await lc.shutdown_all()  # Second call should be safe
    asyncio.run(_run())


def test_lifecycle_empty():
    """Lifecycle with no resources should work."""
    async def _run():
        from caveman.lifecycle import Lifecycle
        lc = Lifecycle()
        await lc.start_all()
        assert lc.is_running
        await lc.shutdown_all()
        assert not lc.is_running
    asyncio.run(_run())
