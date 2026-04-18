"""Tests for tools."""
import pytest
import asyncio
import tempfile
from pathlib import Path
from caveman.tools.registry import ToolRegistry
from caveman.tools.builtin.bash import bash_exec
from caveman.tools.builtin.file_ops import file_read, file_write


def test_tool_registry():
    reg = ToolRegistry()
    reg.register("test", lambda: "ok", "test tool", {"type": "object"})
    schemas = reg.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "test"


@pytest.mark.asyncio
async def test_bash_exec():
    result = await bash_exec("echo hello")
    assert result["success"]
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_bash_dangerous():
    result = await bash_exec("rm -rf /")
    assert not result["success"]
    assert "Blocked" in result["stderr"]


@pytest.mark.asyncio
async def test_file_ops():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "test.txt")
        await file_write(path, "hello world")
        result = await file_read(path)
        d = result.to_dict() if hasattr(result, 'to_dict') else result
        assert "hello world" in d["content"]
