"""Tests for Round 6: result types, error hierarchy, lifecycle management."""
import asyncio
import pytest


# ── ToolResult ──

def test_ok_result():
    from caveman.result import Ok
    r = Ok({"data": "hello"})
    assert r.ok is True
    assert r.data == {"data": "hello"}
    assert r.error == ""
    assert r.is_error is False


def test_ok_result_kwargs():
    from caveman.result import Ok
    r = Ok(count=42, items=["a", "b"])
    assert r.data == {"count": 42, "items": ["a", "b"]}


def test_err_result():
    from caveman.result import Err
    r = Err("something broke")
    assert r.ok is False
    assert r.error == "something broke"
    assert r.is_error is True


def test_result_to_dict_ok():
    from caveman.result import Ok
    r = Ok({"key": "value"})
    d = r.to_dict()
    assert d["ok"] is True
    assert d["key"] == "value"
    assert "error" not in d


def test_result_to_dict_err():
    from caveman.result import Err
    r = Err("bad input")
    d = r.to_dict()
    assert d["ok"] is False
    assert d["error"] == "bad input"


def test_result_to_content():
    from caveman.result import Ok
    import json
    r = Ok({"x": 1})
    content = r.to_content()
    parsed = json.loads(content)
    assert parsed["ok"] is True
    assert parsed["x"] == 1


def test_result_frozen():
    """ToolResult should be immutable."""
    from caveman.result import Ok
    r = Ok({"x": 1})
    with pytest.raises(AttributeError):
        r.ok = False


# ── Error hierarchy ──

def test_caveman_error_base():
    from caveman.errors import CavemanError
    e = CavemanError("test error", context={"key": "val"})
    assert str(e) == "test error"
    assert e.context == {"key": "val"}
    d = e.to_dict()
    assert d["error_type"] == "CavemanError"
    assert d["message"] == "test error"


def test_error_hierarchy():
    from caveman.errors import (
        CavemanError, ConfigError, ProviderError, RateLimitError,
        AuthError, ToolError, ToolNotFoundError, ToolPermissionError,
        ToolTimeoutError, MemoryError, BridgeError, SecurityError,
    )
    # All should be CavemanError subclasses
    assert issubclass(ConfigError, CavemanError)
    assert issubclass(ProviderError, CavemanError)
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(AuthError, ProviderError)
    assert issubclass(ToolError, CavemanError)
    assert issubclass(ToolNotFoundError, ToolError)
    assert issubclass(ToolPermissionError, ToolError)
    assert issubclass(ToolTimeoutError, ToolError)
    assert issubclass(MemoryError, CavemanError)
    assert issubclass(BridgeError, CavemanError)
    assert issubclass(SecurityError, CavemanError)


def test_rate_limit_error_retry_after():
    from caveman.errors import RateLimitError
    e = RateLimitError("Too many requests", retry_after=30.0)
    assert e.retry_after == 30.0


def test_tool_not_found_error_in_registry():
    from caveman.tools.registry import ToolRegistry
    from caveman.errors import ToolNotFoundError

    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="Unknown tool"):
        asyncio.run(registry.dispatch("nonexistent", {}))


def test_config_error_from_validator():
    from caveman.config.validator import validate_config
    from caveman.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_config({"agent": {"max_iterations": "bad"}}, strict=True)


# ── Lifecycle ──

def test_lifecycle_register_and_start():
    async def _run():
        from caveman.lifecycle import Lifecycle

        started = []
        stopped = []

        lifecycle = Lifecycle()
        lifecycle.register("db", startup=lambda: started.append("db"), shutdown=lambda: stopped.append("db"))
        lifecycle.register("server", startup=lambda: started.append("server"), shutdown=lambda: stopped.append("server"))

        await lifecycle.start_all()
        assert started == ["db", "server"]  # In order
        assert lifecycle.is_running

        await lifecycle.shutdown_all()
        assert stopped == ["server", "db"]  # Reverse order (LIFO)
        assert not lifecycle.is_running

    asyncio.run(_run())


def test_lifecycle_async_hooks():
    async def _run():
        from caveman.lifecycle import Lifecycle

        log = []

        async def start_db():
            log.append("async_start")

        async def stop_db():
            log.append("async_stop")

        lifecycle = Lifecycle()
        lifecycle.register("db", startup=start_db, shutdown=stop_db)

        await lifecycle.start_all()
        await lifecycle.shutdown_all()
        assert log == ["async_start", "async_stop"]

    asyncio.run(_run())


def test_lifecycle_context_manager():
    async def _run():
        from caveman.lifecycle import Lifecycle

        log = []
        lifecycle = Lifecycle()
        lifecycle.register("r1", startup=lambda: log.append("start"), shutdown=lambda: log.append("stop"))

        async with lifecycle:
            assert lifecycle.is_running

        assert not lifecycle.is_running
        assert log == ["start", "stop"]

    asyncio.run(_run())


def test_lifecycle_startup_failure_triggers_cleanup():
    async def _run():
        from caveman.lifecycle import Lifecycle

        log = []

        def fail_start():
            raise RuntimeError("Startup failed!")

        lifecycle = Lifecycle()
        lifecycle.register("good", startup=lambda: log.append("good_start"), shutdown=lambda: log.append("good_stop"))
        lifecycle.register("bad", startup=fail_start, shutdown=lambda: log.append("bad_stop"))

        with pytest.raises(RuntimeError, match="Startup failed"):
            await lifecycle.start_all()

        # Good resource should have been cleaned up
        assert "good_start" in log
        assert "good_stop" in log

    asyncio.run(_run())


def test_lifecycle_status():
    async def _run():
        from caveman.lifecycle import Lifecycle

        lifecycle = Lifecycle()
        lifecycle.register("db", startup=lambda: None, shutdown=lambda: None)
        lifecycle.register("cache", startup=lambda: None, shutdown=lambda: None)

        await lifecycle.start_all()
        status = lifecycle.status
        assert status == {"db": True, "cache": True}

        await lifecycle.shutdown_all()
        status = lifecycle.status
        assert status == {"db": False, "cache": False}

    asyncio.run(_run())


# ── Integration: errors + events ──

def test_errors_have_context():
    """All CavemanError subclasses should carry context."""
    from caveman.errors import ToolNotFoundError

    e = ToolNotFoundError("Missing tool", context={"tool": "bash", "registry_size": 0})
    assert e.context["tool"] == "bash"
    d = e.to_dict()
    assert d["error_type"] == "ToolNotFoundError"
    assert d["context"]["tool"] == "bash"
