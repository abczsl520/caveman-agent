"""Tests for tools."""
import pytest
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


def test_hidden_alias_stays_out_of_user_and_model_visible_tool_lists():
    async def noop() -> dict:
        return {"ok": True}

    reg = ToolRegistry()
    reg.register("primary_tool", noop, "primary", {"type": "object"})
    reg.register_alias("legacy_tool", "primary_tool")

    assert "legacy_tool" not in {s["name"] for s in reg.get_schemas()}
    assert "legacy_tool" not in reg.list_tools()
    assert "legacy_tool" in reg.list_tools(include_hidden=True)
    assert reg.tool_count == 1


@pytest.mark.asyncio
async def test_bash_exec():
    result = await bash_exec("echo hello")
    assert result["success"]
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_bash_dangerous():
    result = await bash_exec("rm -rf /")
    assert not result["success"]
    assert "dangerous" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_file_ops():
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "test.txt")
        await file_write(path, "hello world")
        result = await file_read(path)
        d = result.to_dict() if hasattr(result, 'to_dict') else result
        assert "hello world" in d["content"]
