"""Tests for workspace_adapter — OpenClaw → Caveman content transformation."""
import pytest
from caveman.import_.workspace_adapter import (
    adapt_workspace_content,
    validate_adapted_content,
    _strip_reporting_rules_from_soul,
)


class TestToolNameReplacement:
    def test_backtick_message_to_progress(self):
        content = "Use `message` to send updates"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`progress`" in result
        assert "`message`" not in result

    def test_backtick_exec_to_bash(self):
        content = "Run `exec` to execute commands"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`bash`" in result
        assert "`exec`" not in result

    def test_backtick_read_to_file_read(self):
        content = "Use `read` to view files"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`file_read`" in result

    def test_backtick_write_to_file_write(self):
        content = "Use `write` to create files"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`file_write`" in result

    def test_unavailable_tool_marked(self):
        content = "Use `tts` for speech"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "not available" in result

    def test_memory_search_unchanged(self):
        content = "Use `memory_search` to find things"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`memory_search`" in result

    def test_multiple_replacements(self):
        content = "First `message`, then `exec`, finally `read`"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`progress`" in result
        assert "`bash`" in result
        assert "`file_read`" in result


class TestBackgroundReplacement:
    def test_exec_background_full_line(self):
        content = "预计 >60s：`exec` + `background: true`，poll timeout 300000"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "process_start" in result
        assert "process_output" in result
        assert "background: true" not in result

    def test_exec_background_inline(self):
        content = "Use `exec` + `background: true` for long tasks"
        result = adapt_workspace_content("AGENTS.md", content)
        assert "`process_start`" in result


class TestSoulReportingStrip:
    def test_removes_reporting_section(self):
        content = """# My Agent

## 身份
I am Caveman.

## ⚠️ 实时汇报（铁律）
**禁止闷头干活！**
1. 开工必报
2. 阶段必报
3. 连续 3 次工具调用不汇报 = 违规

## 元飞轮
I can self-audit.
"""
        result = _strip_reporting_rules_from_soul(content)
        assert "实时汇报" not in result
        assert "禁止闷头干活" not in result
        assert "元飞轮" in result
        assert "身份" in result

    def test_keeps_non_reporting_content(self):
        content = "# Agent\n\n## Style\nBe friendly.\n"
        result = _strip_reporting_rules_from_soul(content)
        assert result == content


class TestValidation:
    def test_clean_content_no_warnings(self):
        content = "Use `progress` to report. Use `bash` to run."
        warnings = validate_adapted_content("AGENTS.md", content)
        assert warnings == []

    def test_remaining_openclaw_tool_warns(self):
        content = "Use `message` to send"
        warnings = validate_adapted_content("AGENTS.md", content)
        assert any("message" in w for w in warnings)

    def test_sessions_spawn_warns(self):
        content = "Use sessions_spawn to create agents"
        warnings = validate_adapted_content("AGENTS.md", content)
        assert any("session" in w.lower() for w in warnings)


class TestGatewaySection:
    def test_gateway_prohibition_adapted(self):
        content = "**禁止擅自改网关** 子区session禁止 `gateway` 工具。"
        result = adapt_workspace_content("AGENTS.md", content)
        # gateway should be replaced but prohibition kept
        assert "禁止" in result
