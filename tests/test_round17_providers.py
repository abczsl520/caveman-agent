"""Tests for Round 17 — Production LLM providers.

Tests the anthropic_adapter (message conversion), upgraded providers,
and error handling integration.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.providers.anthropic_adapter import (
    convert_messages,
    convert_tools,
    build_api_kwargs,
    get_max_output,
    supports_adaptive_thinking,
    _sanitize_tool_id,
    _convert_content_part,
    THINKING_BUDGET,
)
from caveman.providers.anthropic_provider import AnthropicProvider
from caveman.providers.openai_provider import OpenAIProvider


# --- Anthropic Adapter ---

class TestConvertTools:
    def test_empty(self):
        assert convert_tools([]) == []
        assert convert_tools(None) == []

    def test_openai_format(self):
        tools = [{"function": {"name": "bash", "description": "Run shell", "parameters": {"type": "object"}}}]
        result = convert_tools(tools)
        assert result[0]["name"] == "bash"
        assert result[0]["input_schema"] == {"type": "object"}

    def test_native_format(self):
        tools = [{"name": "bash", "description": "Run shell", "input_schema": {"type": "object"}}]
        result = convert_tools(tools)
        assert result[0]["name"] == "bash"


class TestConvertMessages:
    def test_system_extraction(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, result = convert_messages(msgs)
        assert system == "You are helpful"
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_use_conversion(self):
        msgs = [
            {"role": "assistant", "content": "Let me check", "tool_calls": [
                {"id": "tc_1", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "file1.py\nfile2.py"},
        ]
        _, result = convert_messages(msgs)
        assert result[0]["role"] == "assistant"
        # Should have text + tool_use blocks
        blocks = result[0]["content"]
        assert any(b["type"] == "tool_use" for b in blocks)
        assert any(b["type"] == "text" for b in blocks)

        # Tool result should be in user message
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "tool_result"

    def test_orphan_tool_use_cleanup(self):
        """tool_use without matching tool_result should be removed."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "orphan_1", "function": {"name": "bash", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "Continue"},
        ]
        _, result = convert_messages(msgs)
        # The orphan tool_use should be stripped
        assistant_blocks = result[0]["content"]
        assert not any(b.get("type") == "tool_use" for b in assistant_blocks)

    def test_orphan_tool_result_cleanup(self):
        """tool_result without matching tool_use should be removed."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "orphan_2", "content": "result"},
            {"role": "user", "content": "Continue"},
        ]
        _, result = convert_messages(msgs)
        # Should not have tool_result blocks
        for m in result:
            if isinstance(m["content"], list):
                assert not any(b.get("type") == "tool_result" for b in m["content"])

    def test_role_alternation(self):
        """Consecutive same-role messages should be merged."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        _, result = convert_messages(msgs)
        assert len(result) == 1
        assert "Hello" in str(result[0]["content"])
        assert "World" in str(result[0]["content"])

    def test_empty_content_handling(self):
        msgs = [{"role": "user", "content": ""}]
        _, result = convert_messages(msgs)
        assert result[0]["content"] == "(empty message)"

    def test_system_with_cache_control(self):
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": "Hi"},
        ]
        system, _ = convert_messages(msgs)
        assert isinstance(system, list)
        assert system[0].get("cache_control")

    def test_thinking_block_management(self):
        """Non-latest assistant thinking blocks should be stripped."""
        msgs = [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "old thought"},
                {"type": "text", "text": "old response"},
            ]},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "new thought", "signature": "abc123"},
                {"type": "text", "text": "new response"},
            ]},
        ]
        _, result = convert_messages(msgs)
        # First assistant: thinking stripped
        first_blocks = result[0]["content"]
        assert not any(b.get("type") == "thinking" for b in first_blocks)
        # Last assistant: signed thinking kept
        last_blocks = result[2]["content"]
        assert any(b.get("type") == "thinking" for b in last_blocks)

    def test_consecutive_tool_results_merged(self):
        """Multiple tool results should merge into one user message."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "t2", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "result1"},
            {"role": "tool", "tool_call_id": "t2", "content": "result2"},
        ]
        _, result = convert_messages(msgs)
        # Both tool results should be in one user message
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2
        assert all(b["type"] == "tool_result" for b in user_msg["content"])


class TestBuildApiKwargs:
    def test_basic(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert "messages" in kwargs
        assert "max_tokens" in kwargs

    def test_with_tools(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"name": "bash", "description": "Run shell", "input_schema": {}}],
        )
        assert "tools" in kwargs
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"name": "bash", "description": "Run", "input_schema": {}}],
            tool_choice="required",
        )
        assert kwargs["tool_choice"] == {"type": "any"}

    def test_thinking_adaptive(self):
        kwargs = build_api_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            thinking={"enabled": True, "effort": "high"},
        )
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"]["effort"] == "high"

    def test_thinking_manual(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            thinking={"enabled": True, "effort": "medium"},
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == THINKING_BUDGET["medium"]

    def test_context_length_clamp(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            context_length=4096,
        )
        assert kwargs["max_tokens"] <= 4096

    def test_system_provided_separately(self):
        kwargs = build_api_kwargs(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            system="You are helpful",
        )
        assert kwargs["system"] == "You are helpful"


class TestHelpers:
    def test_get_max_output(self):
        assert get_max_output("claude-opus-4-6") == 128_000
        assert get_max_output("claude-haiku-3-5") == 8_192
        assert get_max_output("unknown-model") == 8_192

    def test_supports_adaptive(self):
        assert supports_adaptive_thinking("claude-opus-4-6")
        assert supports_adaptive_thinking("claude-sonnet-4-6")
        assert not supports_adaptive_thinking("claude-sonnet-4-5")

    def test_sanitize_tool_id(self):
        assert _sanitize_tool_id("toolu_abc123") == "toolu_abc123"
        assert _sanitize_tool_id("tool-with-dashes") == "tool_with_dashes"
        assert _sanitize_tool_id("") == "tool_0"

    def test_convert_content_part_text(self):
        assert _convert_content_part("hello") == {"type": "text", "text": "hello"}

    def test_convert_content_part_image(self):
        part = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        result = _convert_content_part(part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"

    def test_convert_content_part_base64(self):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
        result = _convert_content_part(part)
        assert result["source"]["type"] == "base64"
        assert result["source"]["data"] == "abc123"


# --- Provider Integration ---

class TestAnthropicProvider:
    def test_init_defaults(self):
        p = AnthropicProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.model is not None
        assert p.context_length > 0

    def test_init_custom(self):
        p = AnthropicProvider(
            api_key="key", model="claude-haiku-3-5",
            max_tokens=4096, base_url="https://proxy.example.com",
        )
        assert p.model == "claude-haiku-3-5"
        assert p.max_tokens == 4096
        assert p.base_url == "https://proxy.example.com"

    def test_build_params(self):
        p = AnthropicProvider(api_key="key", model="claude-sonnet-4-5")
        params = p._build_params(
            [{"role": "user", "content": "Hi"}],
            system="Be helpful",
        )
        assert params["model"] == "claude-sonnet-4-5"
        assert params["system"] == "Be helpful"

    def test_usage_stats_initial(self):
        p = AnthropicProvider(api_key="key")
        stats = p.usage_stats
        assert stats["calls"] == 0
        assert stats["total_input_tokens"] == 0

    def test_model_info(self):
        p = AnthropicProvider(api_key="key", model="claude-opus-4-6")
        info = p.model_info
        assert info["model"] == "claude-opus-4-6"
        assert info["provider"] == "AnthropicProvider"


class TestOpenAIProvider:
    def test_init_defaults(self):
        p = OpenAIProvider(api_key="test-key")
        assert p.api_key == "test-key"
        assert p.model is not None

    def test_init_with_base_url(self):
        p = OpenAIProvider(api_key="key", base_url="https://api.openrouter.ai/v1")
        assert p.base_url == "https://api.openrouter.ai/v1"

    def test_build_params(self):
        p = OpenAIProvider(api_key="key")
        params = p._build_params(
            [{"role": "user", "content": "Hi"}],
            system="Be helpful",
            tools=[{"name": "bash", "description": "Run", "input_schema": {}}],
        )
        assert params["messages"][0]["role"] == "system"
        assert params["tools"] is not None
