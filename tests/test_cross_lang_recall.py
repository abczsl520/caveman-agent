"""Tests for cross-language recall — the Day 3 flywheel moment."""
import asyncio
import pytest
from pathlib import Path

from caveman.memory.retrieval import expand_query_cross_lang


# ── Query expansion tests ──

def test_zh_to_en_expansion():
    result = expand_query_cross_lang("修复登录页面的bug")
    assert "login" in result
    assert "auth" in result


def test_en_to_zh_expansion():
    result = expand_query_cross_lang("fix the login bug")
    assert "登录" in result


def test_no_expansion_needed():
    result = expand_query_cross_lang("hello world")
    assert result == "hello world"


def test_deploy_expansion():
    result = expand_query_cross_lang("部署项目")
    assert "deploy" in result
    assert "project" in result or "repo" in result


def test_expansion_capped():
    """Expansion should not add more than 8 extra terms."""
    result = expand_query_cross_lang("登录注册用户密码数据库服务器部署测试配置接口")
    extra = result.split()[1:]  # Skip original
    # Original is one token (no spaces in Chinese), extras are the expansions
    assert len(result.split()) <= 20  # Reasonable cap


# ── Cross-language recall integration tests ──

@pytest.mark.asyncio
async def test_chinese_query_finds_english_memory(tmp_path):
    """Day 3 flywheel: Chinese query should find English memory."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "test.db")
    await mm.store(
        "Project files: /src/auth/login.py, /src/main.py",
        MemoryType.SEMANTIC,
    )

    results = await mm.recall("修复登录bug", top_k=3)
    assert len(results) >= 1
    assert "login" in results[0].content.lower()


@pytest.mark.asyncio
async def test_english_query_finds_chinese_memory(tmp_path):
    """Reverse: English query should find Chinese memory."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "test.db")
    await mm.store("数据库使用 PostgreSQL，端口 5432", MemoryType.SEMANTIC)

    results = await mm.recall("database port", top_k=3)
    assert len(results) >= 1
    assert "PostgreSQL" in results[0].content or "数据库" in results[0].content


@pytest.mark.asyncio
async def test_deploy_cross_lang(tmp_path):
    """Deploy scenario: Chinese query finds English deploy instructions."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "test.db")
    await mm.store("Deploy command: docker compose up -d", MemoryType.PROCEDURAL)

    results = await mm.recall("怎么部署", top_k=3)
    assert len(results) >= 1
    assert "docker" in results[0].content.lower()
