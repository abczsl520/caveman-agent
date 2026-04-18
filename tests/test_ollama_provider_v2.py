"""Tests for OllamaProvider v2 — inherits LLMProvider, normalized events."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.providers.ollama_provider import OllamaProvider
from caveman.providers.llm import LLMProvider


class TestOllamaInheritsLLMProvider:
    """Verify OllamaProvider properly inherits LLMProvider."""

    def test_is_subclass(self):
        assert issubclass(OllamaProvider, LLMProvider)

    def test_instance_check(self):
        p = OllamaProvider(model="llama3.2")
        assert isinstance(p, LLMProvider)

    def test_has_abstract_methods(self):
        p = OllamaProvider(model="llama3.2")
        # All abstract methods should be implemented
        assert hasattr(p, "complete")
        assert hasattr(p, "context_length")
        assert hasattr(p, "_get_client")
        assert hasattr(p, "_build_params")

    def test_model_info_property(self):
        p = OllamaProvider(model="llama3.2")
        info = p.model_info
        assert info["model"] == "llama3.2"
        assert info["provider"] == "OllamaProvider"
        assert info["max_tokens"] == 4096
        assert info["context_length"] == 128_000

    def test_context_length(self):
        p = OllamaProvider(model="llama3.2")
        assert p.context_length == 128_000

    def test_custom_max_tokens(self):
        p = OllamaProvider(model="llama3.2", max_tokens=8192)
        assert p.max_tokens == 8192


class TestOllamaComplete:
    """Test OllamaProvider.complete() with mocked httpx."""

    @pytest.mark.asyncio
    async def test_non_stream_complete(self):
        provider = OllamaProvider(model="llama3.2")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello from Ollama!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
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

        assert any(e["type"] == "delta" and e["text"] == "Hello from Ollama!" for e in events)
        assert any(e["type"] == "done" for e in events)
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["usage"]["input_tokens"] == 10
        assert done_event["usage"]["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_non_stream_with_tool_calls(self):
        provider = OllamaProvider(model="llama3.2")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "tc_1",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        events = []
        async for event in provider.complete(
            messages=[{"role": "user", "content": "list files"}],
            stream=False,
        ):
            events.append(event)

        tc_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        assert tc_events[0]["name"] == "bash"
        assert tc_events[0]["input"] == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_build_params_with_tools(self):
        provider = OllamaProvider(model="llama3.2")
        tools = [{
            "name": "bash",
            "description": "Run a command",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }]
        params = provider._build_params(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert "tools" in params
        assert params["tools"][0]["type"] == "function"
        assert params["tools"][0]["function"]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_build_params_with_system(self):
        provider = OllamaProvider(model="llama3.2")
        params = provider._build_params(
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][0]["content"] == "You are helpful."
        assert params["messages"][1]["role"] == "user"


class TestOllamaLegacyAPI:
    """Test backward-compatible legacy methods."""

    @pytest.mark.asyncio
    async def test_complete_text(self):
        provider = OllamaProvider(model="llama3.2")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Legacy response"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.complete_text("Hello")
        assert result == "Legacy response"
