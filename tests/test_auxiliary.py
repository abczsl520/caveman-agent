"""Tests for agent/auxiliary.py — lightweight LLM client with heuristic fallbacks."""
import pytest
from caveman.agent.auxiliary import (
    AuxiliaryClient, _heuristic_title, _heuristic_intent, _heuristic_tags,
    generate_title, classify_intent, extract_tags,
)


class TestHeuristicTitle:
    def test_basic(self):
        msgs = [{"role": "user", "content": "help me fix the login page"}]
        t = _heuristic_title(msgs)
        assert t  # Should return something
        assert len(t) <= 80

    def test_empty(self):
        t = _heuristic_title([])
        assert t  # Should return a default title

    def test_long_message(self):
        msgs = [{"role": "user", "content": "word " * 100}]
        t = _heuristic_title(msgs)
        assert len(t) <= 80


class TestHeuristicIntent:
    def test_debug(self):
        assert _heuristic_intent("fix the crash in main.py") in ("debug", "fix", "code")

    def test_create(self):
        assert _heuristic_intent("create a new REST API") in ("create", "build", "code")

    def test_question(self):
        result = _heuristic_intent("what is the meaning of life?")
        assert result  # Should classify somehow


class TestHeuristicTags:
    def test_basic(self):
        tags = _heuristic_tags("fix the Python crash in the gateway module")
        assert isinstance(tags, list)
        assert len(tags) <= 5

    def test_empty(self):
        tags = _heuristic_tags("")
        assert isinstance(tags, list)


class TestAuxiliaryClient:
    def test_init(self):
        client = AuxiliaryClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_generate_title_fallback(self):
        # Without API key, should fall back to heuristic
        t = await generate_title([{"role": "user", "content": "build a web scraper"}])
        assert t is not None

    @pytest.mark.asyncio
    async def test_classify_intent_fallback(self):
        result = await classify_intent("debug the memory leak")
        assert result is not None

    @pytest.mark.asyncio
    async def test_extract_tags_fallback(self):
        tags = await extract_tags("Python gateway crash fix")
        assert isinstance(tags, list)
