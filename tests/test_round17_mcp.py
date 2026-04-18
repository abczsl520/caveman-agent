"""Tests for Round 17 Phase 3 — MCP Server.

Tests the MCP tool definitions and their integration with Caveman engines.
Does NOT test actual MCP protocol transport (that requires a running server).
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from caveman.mcp.server import (
    mcp,
    memory_store,
    memory_search,
    memory_recall,
    shield_save,
    shield_load,
    reflect,
    skill_list,
    skill_get,
    get_status,
)


class TestMCPServerSetup:
    def test_server_name(self):
        assert mcp.name == "Caveman"

    def test_tools_registered(self):
        # FastMCP registers tools internally
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "memory_store" in tool_names
        assert "memory_search" in tool_names
        assert "memory_recall" in tool_names
        assert "shield_save" in tool_names
        assert "shield_load" in tool_names
        assert "reflect" in tool_names
        assert "skill_list" in tool_names
        assert "skill_get" in tool_names

    def test_resource_registered(self):
        # Check status resource exists
        resources = mcp._resource_manager.list_resources()
        uris = [str(r.uri) for r in resources]
        assert any("status" in u for u in uris)


class TestMemoryTools:
    @pytest.mark.asyncio
    async def test_memory_store(self):
        with patch("caveman.mcp.server.MemoryManager") as MockMgr:
            mock_mgr = AsyncMock()
            mock_mgr.store.return_value = "mem_123"
            MockMgr.return_value = mock_mgr

            result = await memory_store(
                content="Python uses 0-based indexing",
                tags="python,basics",
                source="claude-code",
            )
            assert result["stored"]
            assert result["memory_id"] == "mem_123"
            assert "python" in result["tags"]

    @pytest.mark.asyncio
    async def test_memory_search(self):
        with patch("caveman.mcp.server.MemoryManager") as MockMgr:
            mock_result = MagicMock()
            mock_result.content = "Python uses 0-based indexing"
            mock_result.memory_type = MagicMock()
            mock_result.memory_type.value = "episodic"
            mock_result.id = "mem_456"

            mock_mgr = AsyncMock()
            mock_mgr.recall_scored.return_value = [(0.85, mock_result)]
            MockMgr.return_value = mock_mgr

            result = await memory_search(query="python indexing")
            assert result["count"] == 1
            assert result["results"][0]["content"] == "Python uses 0-based indexing"


class TestShieldTools:
    @pytest.mark.asyncio
    async def test_shield_save(self, tmp_path):
        with patch("caveman.mcp.server.SESSIONS_DIR", tmp_path):
            result = await shield_save(
                session_id="test-session",
                task="Build API",
                decisions="Use FastAPI\nUse JWT auth",
                progress="Created project structure",
                todos="Add rate limiting",
            )
            assert result["saved"]
            assert result["decisions"] == 2
            assert result["progress"] == 1
            assert result["todos"] == 1
            assert (tmp_path / "test-session.yaml").exists()

    @pytest.mark.asyncio
    async def test_shield_load_not_found(self, tmp_path):
        with patch("caveman.mcp.server.SESSIONS_DIR", tmp_path):
            result = await shield_load(session_id="nonexistent")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_shield_load_most_recent(self, tmp_path):
        with patch("caveman.mcp.server.SESSIONS_DIR", tmp_path):
            # First save a session
            await shield_save(
                session_id="recent-session",
                task="Test task",
                decisions="Decision 1",
            )
            # Load most recent (no session_id)
            result = await shield_load()
            assert "error" not in result or result.get("session_id") == "recent-session"


class TestReflectTool:
    @pytest.mark.asyncio
    async def test_reflect_success(self):
        result = await reflect(
            task="Deploy API",
            outcome="success",
            what_worked="Used CI/CD pipeline\nAutomated tests",
            what_failed="Manual deployment was slow",
            lessons="Always use CI/CD",
        )
        assert result["reflected"]
        assert result["outcome"] == "success"
        assert result["patterns_recorded"] == 2
        assert result["anti_patterns_recorded"] == 1
        assert result["lessons_recorded"] == 1


class TestSkillTools:
    @pytest.mark.asyncio
    async def test_skill_list_empty(self, tmp_path):
        with patch("caveman.mcp.server.SKILLS_DIR", tmp_path):
            result = await skill_list()
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_skill_get_not_found(self, tmp_path):
        with patch("caveman.mcp.server.SKILLS_DIR", tmp_path):
            result = await skill_get(name="nonexistent")
            assert "error" in result


class TestStatusResource:
    @pytest.mark.asyncio
    async def test_status(self, tmp_path):
        import json
        with patch("caveman.mcp.server.SESSIONS_DIR", tmp_path), \
             patch("caveman.mcp.server.MEMORY_DIR", tmp_path):
            result = await get_status()
            data = json.loads(result)
            assert "engines" in data
            assert "shield" in data["engines"]
