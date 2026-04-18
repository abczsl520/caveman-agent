"""Tests for Round 18 Phase 2 — CLI wiki + mcp commands."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from caveman.cli.main import app

runner = CliRunner()


class TestWikiCLI:
    def test_wiki_status(self, tmp_path):
        with patch("caveman.wiki.WikiStore") as MockStore:
            mock_store = MagicMock()
            mock_store.stats.return_value = {"working": 5, "episodic": 3, "semantic": 1, "procedural": 0}
            MockStore.return_value = mock_store

            result = runner.invoke(app, ["wiki", "status"])
            assert result.exit_code == 0
            assert "9 entries" in result.stdout

    def test_wiki_compile(self, tmp_path):
        with patch("caveman.wiki.compiler.WikiCompiler") as MockCompiler:
            from caveman.wiki import CompilationResult
            mock_compiler = MagicMock()
            mock_compiler.compile.return_value = CompilationResult(
                entries_processed=10, entries_promoted=2, entries_expired=1, duration_ms=5.3
            )
            mock_compiler.store = MagicMock()
            MockCompiler.return_value = mock_compiler

            result = runner.invoke(app, ["wiki", "compile"])
            assert result.exit_code == 0
            assert "Compiled" in result.stdout

    def test_wiki_search_no_query(self):
        result = runner.invoke(app, ["wiki", "search"])
        assert result.exit_code == 1

    def test_wiki_search_with_query(self):
        with patch("caveman.wiki.WikiStore") as MockStore:
            from caveman.wiki import WikiEntry
            mock_store = MagicMock()
            mock_store.search.return_value = [
                WikiEntry(id="x1", tier="semantic", title="Python tip", content="Use enumerate", confidence=0.8)
            ]
            MockStore.return_value = mock_store

            result = runner.invoke(app, ["wiki", "search", "python"])
            assert result.exit_code == 0
            assert "Python tip" in result.stdout

    def test_wiki_context_empty(self):
        with patch("caveman.wiki.compiler.WikiCompiler") as MockCompiler:
            mock_compiler = MagicMock()
            mock_compiler.get_compiled_context.return_value = ""
            MockCompiler.return_value = mock_compiler

            result = runner.invoke(app, ["wiki", "context"])
            assert result.exit_code == 0
            assert "empty" in result.stdout.lower()

    def test_wiki_unknown_action(self):
        result = runner.invoke(app, ["wiki", "invalid"])
        assert result.exit_code == 1


class TestMCPCLI:
    def test_mcp_tools(self):
        result = runner.invoke(app, ["mcp", "tools"])
        assert result.exit_code == 0
        assert "MCP tools" in result.stdout
        assert "memory_store" in result.stdout

    def test_mcp_status(self):
        result = runner.invoke(app, ["mcp", "status"])
        assert result.exit_code == 0
        assert "not running" in result.stdout

    def test_mcp_unknown_action(self):
        result = runner.invoke(app, ["mcp", "invalid"])
        assert result.exit_code == 1
