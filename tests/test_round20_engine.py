"""Tests for Round 20 Phase 2 — Context Engine + Auxiliary Client."""
from __future__ import annotations

import pytest
from caveman.compression.context_engine import ContextEngine
from caveman.agent.auxiliary import (
    AuxiliaryClient, generate_title, classify_intent, extract_tags,
)


class ConcreteEngine(ContextEngine):
    """Minimal concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test"

    def update_from_response(self, response):
        usage = response.get("usage", {})
        self.last_prompt_tokens = usage.get("input_tokens", 0)
        self.last_completion_tokens = usage.get("output_tokens", 0)
        self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens

    def should_compress(self) -> bool:
        return self.last_total_tokens >= self.threshold_tokens

    async def compress(self, messages, **kwargs):
        self.compression_count += 1
        return [{"role": "system", "content": "compressed"}]


class TestContextEngine:
    def test_init(self):
        engine = ConcreteEngine(context_length=200_000)
        assert engine.context_length == 200_000
        assert engine.threshold_tokens == 150_000

    def test_update_from_response(self):
        engine = ConcreteEngine()
        engine.update_from_response({"usage": {"input_tokens": 5000, "output_tokens": 1000}})
        assert engine.last_total_tokens == 6000

    def test_should_compress_false(self):
        engine = ConcreteEngine(context_length=200_000)
        engine.update_from_response({"usage": {"input_tokens": 5000, "output_tokens": 1000}})
        assert not engine.should_compress()

    def test_should_compress_true(self):
        engine = ConcreteEngine(context_length=10_000)
        engine.update_from_response({"usage": {"input_tokens": 8000, "output_tokens": 1000}})
        assert engine.should_compress()

    @pytest.mark.asyncio
    async def test_compress(self):
        engine = ConcreteEngine()
        result = await engine.compress([{"role": "user", "content": "test"}])
        assert len(result) == 1
        assert engine.compression_count == 1

    def test_utilization(self):
        engine = ConcreteEngine(context_length=100_000)
        engine.update_from_response({"usage": {"input_tokens": 50_000, "output_tokens": 0}})
        assert engine.utilization == pytest.approx(0.5)

    def test_tokens_remaining(self):
        engine = ConcreteEngine(context_length=100_000)
        engine.update_from_response({"usage": {"input_tokens": 50_000, "output_tokens": 0}})
        assert engine.tokens_remaining == 25_000  # threshold is 75K

    def test_session_lifecycle(self):
        engine = ConcreteEngine()
        engine.last_total_tokens = 5000
        engine.compression_count = 3
        engine.on_session_start("s1")
        assert engine.last_total_tokens == 0
        assert engine.compression_count == 0


class TestAuxiliaryHeuristic:
    """Test heuristic fallbacks (no LLM)."""

    @pytest.mark.asyncio
    async def test_title_heuristic(self):
        messages = [
            {"role": "user", "content": "Build a REST API with FastAPI and PostgreSQL"},
        ]
        title = await generate_title(messages)
        assert "Build" in title or "REST" in title

    @pytest.mark.asyncio
    async def test_title_empty(self):
        title = await generate_title([])
        assert title == "New conversation"

    @pytest.mark.asyncio
    async def test_classify_code(self):
        assert await classify_intent("Build a REST API") == "code"

    @pytest.mark.asyncio
    async def test_classify_question(self):
        assert await classify_intent("What is Docker?") == "question"

    @pytest.mark.asyncio
    async def test_classify_task(self):
        assert await classify_intent("Run the deployment script") == "task"

    @pytest.mark.asyncio
    async def test_classify_chat(self):
        assert await classify_intent("Hey, good morning!") == "chat"

    @pytest.mark.asyncio
    async def test_extract_tags(self):
        tags = await extract_tags("Built a FastAPI REST API with JWT authentication and PostgreSQL database")
        assert len(tags) > 0
        assert len(tags) <= 5


class TestAuxiliaryWithLLM:
    @pytest.mark.asyncio
    async def test_title_with_llm(self):
        async def mock_llm(prompt: str) -> str:
            return "FastAPI REST API Setup"

        aux = AuxiliaryClient(llm_fn=mock_llm)
        title = await aux.generate_title([{"role": "user", "content": "Build API"}])
        assert title == "FastAPI REST API Setup"

    @pytest.mark.asyncio
    async def test_classify_with_llm(self):
        async def mock_llm(prompt: str) -> str:
            return "code"

        aux = AuxiliaryClient(llm_fn=mock_llm)
        result = await aux.classify_intent("Build something")
        assert result == "code"

    @pytest.mark.asyncio
    async def test_tags_with_llm(self):
        async def mock_llm(prompt: str) -> str:
            return "fastapi, rest, jwt, postgresql"

        aux = AuxiliaryClient(llm_fn=mock_llm)
        tags = await aux.extract_tags("Build API")
        assert tags == ["fastapi", "rest", "jwt", "postgresql"]

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        aux = AuxiliaryClient(llm_fn=failing_llm)
        title = await aux.generate_title([{"role": "user", "content": "Build API"}])
        assert title  # Falls back to heuristic
