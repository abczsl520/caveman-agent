"""Integration tests for ToolResult through file_ops."""
import json

import pytest

from caveman.result import ToolResult, Ok, Err
from caveman.tools.builtin.file_ops import file_read, file_write, file_edit


@pytest.mark.asyncio
async def test_toolresult_file_ops_end_to_end(tmp_path):
    f = tmp_path / "tool.txt"
    w = await file_write(str(f), "alpha\nbeta\n")
    assert isinstance(w, ToolResult) and w.ok is True

    r = await file_read(str(f))
    assert isinstance(r, ToolResult) and r.ok is True
    assert "alpha" in r.data["content"]

    e = await file_edit(str(f), "beta", "gamma")
    assert isinstance(e, ToolResult) and e.ok is True
    reread = await file_read(str(f))
    assert "gamma" in reread.data["content"]


@pytest.mark.asyncio
async def test_toolresult_error_cases(tmp_path):
    missing = await file_read(str(tmp_path / "missing.txt"))
    assert isinstance(missing, ToolResult) and missing.ok is False

    f = tmp_path / "edit.txt"
    await file_write(str(f), "one two")
    bad = await file_edit(str(f), "nope", "yes")
    assert isinstance(bad, ToolResult) and bad.ok is False


@pytest.mark.asyncio
async def test_toolresult_serialization(tmp_path):
    f = tmp_path / "json.txt"
    await file_write(str(f), "hello")
    ok_result = await file_read(str(f))
    ok_dict = ok_result.to_dict()
    assert ok_dict["ok"] is True and "content" in ok_dict and "error" not in ok_dict

    err_result = await file_read(str(tmp_path / "no-file.txt"))
    err_dict = err_result.to_dict()
    assert err_dict == {"ok": False, "error": err_result.error}

    parsed = json.loads(ok_result.to_content())
    assert parsed == ok_dict


@pytest.mark.asyncio
async def test_ok_err_helpers_shape():
    ok = Ok(value=1)
    err = Err("boom")
    assert isinstance(ok, ToolResult) and ok.to_dict() == {"ok": True, "value": 1}
    assert isinstance(err, ToolResult) and json.loads(err.to_content()) == {"ok": False, "error": "boom"}
