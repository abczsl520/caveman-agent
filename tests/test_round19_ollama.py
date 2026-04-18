"""Tests for Round 19 Phase 4 — Ollama Provider (updated for LLMProvider interface)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from caveman.providers.ollama_provider import OllamaProvider, DEFAULT_OLLAMA_URL


class TestOllamaProvider:
    def test_init_defaults(self):
        p = OllamaProvider()
        assert p.model == "llama3.2"
        assert p._base_url == DEFAULT_OLLAMA_URL

    def test_init_custom(self):
        p = OllamaProvider(model="mistral", base_url="http://gpu:11434")
        assert p.model == "mistral"
        assert "gpu" in p._base_url

    @pytest.mark.asyncio
    async def test_complete_text_success(self):
        """Test legacy complete_text API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Python is a programming language."}, "finish_reason": "stop"}],
            "usage": {},
        }

        p = OllamaProvider()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        p._client = mock_client

        result = await p.complete_text("What is Python?")
        assert "Python" in result

    @pytest.mark.asyncio
    async def test_complete_text_with_system(self):
        """Test legacy complete_text API with system prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {},
        }

        p = OllamaProvider()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        p._client = mock_client

        result = await p.complete_text("Hi", system="You are helpful")
        assert result == "Hello!"

        # Verify system message was included in the params
        call_args = mock_client.post.call_args
        messages = call_args[1]["json"]["messages"]
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_is_available_true(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "mistral:latest"}]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            p = OllamaProvider(model="llama3.2")
            assert await p.is_available()

    @pytest.mark.asyncio
    async def test_is_available_false_no_model(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            p = OllamaProvider(model="llama3.2")
            assert not await p.is_available()

    @pytest.mark.asyncio
    async def test_is_available_connection_error(self):
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = ConnectionError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            p = OllamaProvider()
            assert not await p.is_available()

    @pytest.mark.asyncio
    async def test_list_models(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "mistral:7b"}]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            p = OllamaProvider()
            models = await p.list_models()
            assert len(models) == 2
            assert "llama3.2:latest" in models
