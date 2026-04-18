"""Tests for GeminiProvider — message/tool conversion, response parsing, complete."""
from __future__ import annotations

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.providers.gemini_provider import (
    GeminiProvider,
    _convert_messages,
    _convert_tools,
    _parse_response,
)


class TestConvertMessages:
    """Test OpenAI → Gemini message format conversion."""

    def test_simple_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = _convert_messages(msgs)
        assert result == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_assistant_message(self):
        msgs = [{"role": "assistant", "content": "Hi there"}]
        result = _convert_messages(msgs)
        assert result == [{"role": "model", "parts": [{"text": "Hi there"}]}]

    def test_multi_turn(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks"},
        ]
        result = _convert_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
        assert result[2]["role"] == "user"

    def test_tool_result_message(self):
        msgs = [{"role": "tool", "content": "file contents here", "tool_call_id": "read_file"}]
        result = _convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["parts"][0]["functionResponse"]["name"] == "read_file"
        assert result[0]["parts"][0]["functionResponse"]["response"]["result"] == "file contents here"

    def test_assistant_with_tool_calls(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "tc_1",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }],
        }]
        result = _convert_messages(msgs)
        assert len(result) == 1
        fc = result[0]["parts"][0]["functionCall"]
        assert fc["name"] == "bash"
        assert fc["args"] == {"command": "ls"}

    def test_empty_content_skipped(self):
        msgs = [{"role": "user", "content": ""}]
        result = _convert_messages(msgs)
        assert result == []  # empty parts → not appended


class TestConvertTools:
    """Test Caveman tool defs → Gemini function declarations."""

    def test_basic_tool(self):
        tools = [{
            "name": "bash",
            "description": "Run a command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }]
        result = _convert_tools(tools)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "bash"
        assert decls[0]["description"] == "Run a command"
        assert "properties" in decls[0]["parameters"]

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "description": "tool a"},
            {"name": "b", "description": "tool b", "input_schema": {"type": "object"}},
        ]
        result = _convert_tools(tools)
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 2

    def test_tool_without_schema(self):
        tools = [{"name": "noop", "description": "does nothing"}]
        result = _convert_tools(tools)
        decl = result[0]["functionDeclarations"][0]
        assert "parameters" not in decl


class TestParseResponse:
    """Test Gemini response → normalized events."""

    def test_text_response(self):
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello world"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        events = _parse_response(data)
        assert len(events) == 2
        assert events[0] == {"type": "delta", "text": "Hello world"}
        assert events[1]["type"] == "done"
        assert events[1]["stop_reason"] == "end_turn"
        assert events[1]["usage"]["input_tokens"] == 10
        assert events[1]["usage"]["output_tokens"] == 5

    def test_function_call_response(self):
        data = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "bash", "args": {"command": "ls"}},
                }]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {},
        }
        events = _parse_response(data)
        assert events[0]["type"] == "tool_call"
        assert events[0]["name"] == "bash"
        assert events[0]["input"] == {"command": "ls"}

    def test_empty_candidates(self):
        data = {"candidates": []}
        events = _parse_response(data)
        assert events == []

    def test_max_tokens_finish(self):
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "partial"}]},
                "finishReason": "MAX_TOKENS",
            }],
            "usageMetadata": {},
        }
        events = _parse_response(data)
        assert events[1]["stop_reason"] == "max_tokens"


class TestGeminiComplete:
    """Test GeminiProvider.complete() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_non_stream_complete(self):
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        events = []
        async for event in provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        ):
            events.append(event)

        assert any(e["type"] == "delta" and e["text"] == "Hello!" for e in events)
        assert any(e["type"] == "done" for e in events)

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="")
        events = []
        async for event in provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "API key" in events[0]["error"]


class TestGeminiErrorHandling:
    """Test error classification and retry."""

    @pytest.mark.asyncio
    async def test_retryable_error_eventually_aborts(self):
        provider = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")

        mock_client = AsyncMock()
        # Simulate server error on every call
        import httpx
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=httpx.Request("POST", "http://test"),
            response=httpx.Response(500),
        ))
        provider._client = mock_client

        events = []
        async for event in provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        ):
            events.append(event)

        # Should eventually yield an error event after retries
        assert any(e["type"] == "error" for e in events)


class TestGeminiProperties:
    """Test provider properties."""

    def test_context_length_known_model(self):
        p = GeminiProvider(model="gemini-2.5-pro")
        assert p.context_length == 1_048_576

    def test_context_length_unknown_model(self):
        p = GeminiProvider(model="gemini-future-model")
        assert p.context_length == 1_048_576  # default

    def test_model_info(self):
        p = GeminiProvider(model="gemini-2.0-flash", api_key="k")
        info = p.model_info
        assert info["model"] == "gemini-2.0-flash"
        assert info["provider"] == "GeminiProvider"
        assert info["max_tokens"] == 8_192

    def test_max_tokens_override(self):
        p = GeminiProvider(model="gemini-2.0-flash", max_tokens=16384)
        assert p.max_tokens == 16384
