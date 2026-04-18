"""Tests for Round 5 optimizations: tool registry v2, config validation, provider improvements."""
import asyncio
import pytest


# ── Tool self-registration ──

def test_tool_decorator_attaches_meta():
    from caveman.tools.registry import tool

    @tool(name="test_tool", description="A test", params={"x": {"type": "string"}}, required=["x"])
    async def my_tool(x: str) -> dict:
        return {"result": x}

    assert hasattr(my_tool, "_tool_meta")
    assert my_tool._tool_meta["name"] == "test_tool"
    assert my_tool._tool_meta["schema"]["required"] == ["x"]


def test_registry_auto_discover():
    from caveman.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry._register_builtins()
    names = registry.list_tools()

    # Should have discovered all built-in tools
    assert "bash" in names
    assert "file_read" in names
    assert "file_write" in names
    assert "file_edit" in names
    assert "file_list" in names
    assert "web_search" in names
    assert registry.tool_count >= 7  # 6 original + browser, possibly more from other @tool in tests


def test_registry_manual_plus_auto():
    from caveman.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("custom_tool", lambda: None, "Custom", {"type": "object"})
    registry._register_builtins()
    assert "custom_tool" in registry.list_tools()
    assert "bash" in registry.list_tools()


def test_registry_dispatch_async():
    from caveman.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry._register_builtins()
    result = asyncio.run(registry.dispatch("file_list", {"path": "."}))
    assert "entries" in result or "error" in result


def test_registry_dispatch_unknown_tool():
    from caveman.tools.registry import ToolRegistry
    from caveman.errors import ToolNotFoundError

    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="Unknown tool"):
        asyncio.run(registry.dispatch("nonexistent", {}))


def test_tool_schema_has_required_fields():
    """Every tool schema must have name, description, input_schema."""
    from caveman.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry._register_builtins()

    for schema in registry.get_schemas():
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


# ── Config validation ──

def test_config_validate_good():
    from caveman.config.validator import validate_config

    config = {
        "agent": {"default_model": "claude-opus-4-6", "max_iterations": 50},
        "providers": {"anthropic": {"max_tokens": 8192}},
        "memory": {"backend": "local"},
    }
    warnings = validate_config(config, strict=False)
    assert warnings == []


def test_config_validate_type_error():
    from caveman.config.validator import validate_config
    from caveman.errors import ConfigError

    config = {
        "agent": {"max_iterations": "not_a_number"},  # Should be int
    }
    # strict=True (default) now raises
    with pytest.raises(ConfigError, match="expected int"):
        validate_config(config)

    # strict=False still returns warnings
    warnings = validate_config(config, strict=False)
    assert any("expected int" in w for w in warnings)


def test_config_validate_range_error():
    from caveman.config.validator import validate_config

    config = {
        "agent": {"max_iterations": 0},  # Below range [1, 1000]
    }
    warnings = validate_config(config, strict=False)
    assert any("out of range" in w for w in warnings)


def test_config_validate_choices_error():
    from caveman.config.validator import validate_config

    config = {
        "memory": {"backend": "nonexistent_backend"},
    }
    warnings = validate_config(config, strict=False)
    assert any("not in" in w for w in warnings)


def test_config_validate_unknown_key():
    from caveman.config.validator import validate_config

    config = {
        "typo_section": {"whatever": True},
    }
    warnings = validate_config(config, strict=False)
    assert any("Unknown config key" in w for w in warnings)


def test_config_validate_strict():
    from caveman.config.validator import validate_config, ConfigError

    config = {
        "agent": {"max_iterations": "bad"},
    }
    with pytest.raises(ConfigError, match="Config validation failed"):
        validate_config(config, strict=True)


def test_config_help_generation():
    from caveman.config.validator import get_config_help

    help_text = get_config_help()
    assert "agent.default_model" in help_text
    assert "memory.backend" in help_text
    assert "local" in help_text  # default value


# ── Provider model_info ──

def test_anthropic_provider_model_info():
    from caveman.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test")
    info = provider.model_info
    assert info["provider"] == "AnthropicProvider"
    assert "model" in info
    assert "context_length" in info


def test_openai_provider_model_info():
    from caveman.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test")
    info = provider.model_info
    assert info["provider"] == "OpenAIProvider"


# ── web_search uses retry_async ──

def test_web_search_no_manual_retry_loop():
    """Verify web_search no longer has a manual retry loop."""
    from pathlib import Path
    source = Path("caveman/tools/builtin/web_search.py").read_text()
    assert "for attempt" not in source
    assert "max_retries = 3" not in source
    assert "retry_async" in source


# ── Integration: ToolRegistry + AgentLoop ──

def test_agent_loop_uses_declarative_tools():
    from caveman.agent.loop import AgentLoop

    loop = AgentLoop()
    schemas = loop.tool_registry.get_schemas()
    tool_names = [s["name"] for s in schemas]
    assert "bash" in tool_names
    assert "web_search" in tool_names
