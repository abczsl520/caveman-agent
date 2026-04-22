"""Tests for agent/title_generator.py — session title generation."""
import pytest
from caveman.agent.title_generator import _heuristic_title, _clean_title, generate_title


class TestHeuristicTitle:
    def test_help_pattern(self):
        assert _heuristic_title("help me fix the login bug") is not None
        assert "login bug" in _heuristic_title("help me fix the login bug")

    def test_create_pattern(self):
        t = _heuristic_title("create a REST API for users")
        assert t is not None
        assert "REST API" in t

    def test_fix_pattern(self):
        t = _heuristic_title("fix the memory leak in gateway")
        assert t is not None

    def test_chinese_help(self):
        t = _heuristic_title("帮我写一个爬虫脚本")
        assert t is not None

    def test_chinese_create(self):
        t = _heuristic_title("创建一个新的数据库表")
        assert t is not None
        assert "构建" in t

    def test_short_message_fallback(self):
        t = _heuristic_title("deploy to production server")
        assert t is not None

    def test_long_message_truncation(self):
        t = _heuristic_title("explain " + "word " * 20)
        assert t is not None
        assert len(t) <= 80

    def test_single_word_returns_none(self):
        assert _heuristic_title("x") is None

    def test_empty_returns_none(self):
        assert _heuristic_title("") is None


class TestCleanTitle:
    def test_strips_quotes(self):
        assert _clean_title('"My Title"') == "My Title"

    def test_strips_prefix(self):
        assert _clean_title("Title: My Session") == "My Session"

    def test_truncates_long(self):
        t = _clean_title("A" * 100)
        assert len(t) <= 80

    def test_empty_returns_none(self):
        assert _clean_title("") is None


class TestGenerateTitle:
    @pytest.mark.asyncio
    async def test_no_llm(self):
        t = await generate_title("help me debug the crash", use_llm=False)
        assert t is not None

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        # LLM will fail (no API key), should fall back to heuristic
        t = await generate_title("create a new microservice", use_llm=True)
        assert t is not None
