"""Tests for Mixture-of-Agents tool."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from caveman.tools.builtin.moa_tool import (
    mixture_of_agents,
    _call_model_simple,
    AGGREGATOR_SYSTEM_PROMPT,
    DEFAULT_REFERENCE_MODELS,
)


class TestMoAPipeline:
    @pytest.mark.asyncio
    async def test_successful_pipeline(self):
        """Test full MoA pipeline with mocked HTTP calls."""
        responses = [
            {"model": "m1", "content": "Answer from model 1", "error": None},
            {"model": "m2", "content": "Answer from model 2", "error": None},
            {"model": "m3", "content": "Synthesized answer", "error": None},  # aggregator
        ]
        call_count = 0

        async def mock_call(model, prompt, **kwargs):
            nonlocal call_count
            r = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return r

        with patch("caveman.tools.builtin.moa_tool._call_model_simple", side_effect=mock_call):
            result = await mixture_of_agents(
                "What is 2+2?",
                reference_models=["m1", "m2"],
                aggregator_model="m3",
            )

        assert result["content"] == "Synthesized answer"
        assert result["error"] is False
        assert result["timing_seconds"] > 0

    @pytest.mark.asyncio
    async def test_all_references_fail(self):
        async def mock_fail(model, prompt, **kwargs):
            return {"model": model, "content": "", "error": "timeout"}

        with patch("caveman.tools.builtin.moa_tool._call_model_simple", side_effect=mock_fail):
            result = await mixture_of_agents(
                "test",
                reference_models=["m1", "m2"],
            )

        assert result["error"] is True
        assert "0/2" in result["content"]

    @pytest.mark.asyncio
    async def test_partial_failure_still_works(self):
        call_idx = 0

        async def mock_partial(model, prompt, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return {"model": model, "content": "", "error": "timeout"}
            return {"model": model, "content": f"Response from {model}", "error": None}

        with patch("caveman.tools.builtin.moa_tool._call_model_simple", side_effect=mock_partial):
            result = await mixture_of_agents(
                "test",
                reference_models=["m1", "m2"],
                min_successful=1,
            )

        assert result["error"] is False

    @pytest.mark.asyncio
    async def test_custom_models(self):
        calls = []

        async def mock_track(model, prompt, **kwargs):
            calls.append(model)
            return {"model": model, "content": f"From {model}", "error": None}

        with patch("caveman.tools.builtin.moa_tool._call_model_simple", side_effect=mock_track):
            await mixture_of_agents(
                "test",
                reference_models=["custom/a", "custom/b"],
                aggregator_model="custom/agg",
            )

        assert "custom/a" in calls
        assert "custom/b" in calls
        assert "custom/agg" in calls


class TestAggregatorPrompt:
    def test_prompt_has_placeholder(self):
        assert "{responses}" in AGGREGATOR_SYSTEM_PROMPT

    def test_prompt_formats(self):
        formatted = AGGREGATOR_SYSTEM_PROMPT.format(responses="test response")
        assert "test response" in formatted


class TestDefaults:
    def test_has_reference_models(self):
        assert len(DEFAULT_REFERENCE_MODELS) >= 2

    def test_models_have_provider_prefix(self):
        for m in DEFAULT_REFERENCE_MODELS:
            assert "/" in m, f"Model {m} should have provider/ prefix"
