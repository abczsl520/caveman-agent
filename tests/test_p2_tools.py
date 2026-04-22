"""Tests for P2 tool modules — terminal, file ops, web, delegate."""
from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ── Terminal Tests ──

class TestTerminal:
    def test_guard_blocks_dangerous(self):
        from caveman.tools.builtin.terminal_v2 import _check_guards
        assert _check_guards("rm -rf /") is not None
        assert _check_guards("mkfs /dev/sda") is not None
        assert _check_guards("ls -la") is None
        assert _check_guards("echo hello") is None

    def test_truncate_output(self):
        from caveman.tools.builtin.terminal_v2 import _truncate_output
        short = "hello"
        assert _truncate_output(short) == short
        long = "x\n" * 1000
        truncated = _truncate_output(long)
        assert len(truncated) < len(long)

    def test_validate_workdir(self):
        from caveman.tools.builtin.terminal_v2 import _validate_workdir
        assert _validate_workdir("") is None
        assert _validate_workdir("/tmp") is None
        assert _validate_workdir("/nonexistent_dir_xyz") is not None

    @pytest.mark.asyncio
    async def test_execute_simple(self):
        from caveman.tools.builtin.terminal_v2 import terminal_execute
        result = await terminal_execute("echo hello")
        assert result["ok"]
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_blocked(self):
        from caveman.tools.builtin.terminal_v2 import terminal_execute
        result = await terminal_execute("rm -rf /")
        assert not result["ok"]
        assert "Blocked" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        from caveman.tools.builtin.terminal_v2 import terminal_execute
        result = await terminal_execute("sleep 10", timeout=1)
        assert not result["ok"]
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_with_workdir(self):
        from caveman.tools.builtin.terminal_v2 import terminal_execute
        result = await terminal_execute("pwd", workdir="/tmp")
        assert result["ok"]
        assert "/tmp" in result["stdout"] or "/private/tmp" in result["stdout"]


# ── File Ops Tests ──

class TestFileOps:
    def test_is_binary(self):
        from caveman.tools.builtin.file_ops_v2 import is_binary
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("hello")
            txt_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode="wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00")
            bin_path = f.name
        try:
            assert not is_binary(txt_path)
            assert is_binary(bin_path)
        finally:
            os.unlink(txt_path)
            os.unlink(bin_path)

    def test_add_line_numbers(self):
        from caveman.tools.builtin.file_ops_v2 import add_line_numbers
        result = add_line_numbers("a\nb\nc", start=1)
        assert "1 | a" in result
        assert "3 | c" in result

    @pytest.mark.asyncio
    async def test_file_read(self):
        from caveman.tools.builtin.file_ops_v2 import file_read_v2
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("\n".join(f"line {i}" for i in range(100)))
            path = f.name
        try:
            result = await file_read_v2(path, offset=10, limit=5)
            assert result["ok"]
            assert "10" in result["showing"]
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_file_read_not_found(self):
        from caveman.tools.builtin.file_ops_v2 import file_read_v2
        result = await file_read_v2("/nonexistent_file_xyz.txt")
        assert not result["ok"]

    def test_file_patch(self):
        from caveman.tools.builtin.file_tools import patch_file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("hello world")
            path = f.name
        try:
            result = patch_file(path, [{"old": "hello", "new": "goodbye"}])
            assert "error" not in result
            assert Path(path).read_text() == "goodbye world"
        finally:
            os.unlink(path)

    def test_file_patch_not_found(self):
        from caveman.tools.builtin.file_tools import patch_file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("hello")
            path = f.name
        try:
            result = patch_file(path, [{"old": "xyz", "new": "abc"}])
            assert not result.get("success", True) or result.get("errors")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_file_search_by_name(self):
        from caveman.tools.builtin.file_ops import file_search
        result = await file_search("*.py", path="caveman/gateway", target="files", limit=5)
        assert "error" not in result
        assert result.get("total_count", 0) > 0

    @pytest.mark.asyncio
    async def test_file_diff(self):
        from caveman.tools.builtin.file_ops_v2 import file_diff
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("hello\nworld")
            path = f.name
        try:
            result = await file_diff(path, content_b="hello\nuniverse")
            assert result["ok"]
            assert result["changed"]
        finally:
            os.unlink(path)


# ── Web Tools Tests ──

class TestWebTools:
    def test_ssrf_check(self):
        from caveman.tools.builtin.web_tools_v2 import _is_ssrf_safe
        assert _is_ssrf_safe("https://google.com")
        assert not _is_ssrf_safe("http://localhost")
        assert not _is_ssrf_safe("http://169.254.169.254")

    def test_clean_html(self):
        from caveman.tools.builtin.web_tools_v2 import _clean_html
        html = "<html><script>evil()</script><p>Hello <b>world</b></p></html>"
        text = _clean_html(html)
        assert "Hello" in text
        assert "world" in text
        assert "evil" not in text
        assert "<" not in text

    def test_truncate_content(self):
        from caveman.tools.builtin.web_tools_v2 import _truncate_content
        short = "hello"
        assert _truncate_content(short) == short
        long = "word " * 20000
        truncated = _truncate_content(long, max_chars=100)
        assert len(truncated) < len(long)
        assert "truncated" in truncated

    def test_cache_roundtrip(self):
        from caveman.tools.builtin.web_tools_v2 import _set_cache, _get_cached
        _set_cache("https://test.example.com/cache_test", "cached content")
        result = _get_cached("https://test.example.com/cache_test")
        assert result == "cached content"


# ── Delegate Tests ──

class TestDelegate:
    @pytest.mark.asyncio
    async def test_delegate_single(self):
        from caveman.tools.builtin.delegate_tool import DelegateManager
        mgr = DelegateManager(agent_fn=AsyncMock(return_value="result"))
        task = await mgr.delegate_single("do something")
        assert task.status == "completed"
        assert task.result == "result"
        assert task.duration_ms > 0

    @pytest.mark.asyncio
    async def test_delegate_timeout(self):
        from caveman.tools.builtin.delegate_tool import DelegateManager
        async def slow_fn(*a, **kw):
            await asyncio.sleep(10)
        import asyncio
        mgr = DelegateManager(agent_fn=slow_fn)
        task = await mgr.delegate_single("slow task", timeout=0.1)
        assert task.status == "failed"
        assert "Timeout" in task.error

    @pytest.mark.asyncio
    async def test_delegate_parallel(self):
        from caveman.tools.builtin.delegate_tool import DelegateManager
        call_count = 0
        async def counting_fn(*a, **kw):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"
        mgr = DelegateManager(agent_fn=counting_fn)
        tasks = [{"prompt": "task1"}, {"prompt": "task2"}, {"prompt": "task3"}]
        results = await mgr.delegate_parallel(tasks)
        assert len(results) == 3
        assert all(t.status == "completed" for t in results)

    @pytest.mark.asyncio
    async def test_merge_results(self):
        from caveman.tools.builtin.delegate_tool import DelegateManager, DelegateTask
        mgr = DelegateManager()
        tasks = [
            DelegateTask(status="completed", result="answer 1"),
            DelegateTask(status="failed", error="timeout"),
        ]
        merged = mgr.merge_results(tasks)
        assert "answer 1" in merged
        assert "FAILED" in merged

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        from caveman.tools.builtin.delegate_tool import DelegateManager
        import asyncio
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracking_fn(*a, **kw):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1
            return "ok"

        mgr = DelegateManager(agent_fn=tracking_fn, max_concurrent=2)
        tasks = [{"prompt": f"t{i}"} for i in range(5)]
        await mgr.delegate_parallel(tasks)
        assert max_concurrent <= 2
