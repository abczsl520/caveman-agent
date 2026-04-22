"""Tests for caveman.aio — async file I/O wrappers."""
import pytest
from pathlib import Path

from caveman.aio import (
    aio_read_text, aio_write_text, aio_read_bytes, aio_write_bytes,
    aio_exists, aio_is_file, aio_is_dir, aio_stat, aio_mkdir,
    aio_glob, aio_iterdir, aio_unlink, aio_rename,
)


@pytest.mark.asyncio
async def test_read_write_text(tmp_path):
    p = tmp_path / "hello.txt"
    await aio_write_text(p, "hello world")
    assert await aio_read_text(p) == "hello world"


@pytest.mark.asyncio
async def test_read_write_bytes(tmp_path):
    p = tmp_path / "data.bin"
    await aio_write_bytes(p, b"\x00\x01\x02")
    assert await aio_read_bytes(p) == b"\x00\x01\x02"


@pytest.mark.asyncio
async def test_exists_is_file_is_dir(tmp_path):
    p = tmp_path / "test.txt"
    assert not await aio_exists(p)
    assert not await aio_is_file(p)
    p.write_text("x")
    assert await aio_exists(p)
    assert await aio_is_file(p)
    assert not await aio_is_dir(p)
    assert await aio_is_dir(tmp_path)


@pytest.mark.asyncio
async def test_stat(tmp_path):
    p = tmp_path / "stat.txt"
    p.write_text("abc")
    st = await aio_stat(p)
    assert st.st_size == 3


@pytest.mark.asyncio
async def test_mkdir(tmp_path):
    d = tmp_path / "a" / "b" / "c"
    assert not d.exists()
    await aio_mkdir(d)
    assert d.is_dir()


@pytest.mark.asyncio
async def test_glob(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.py").write_text("c")
    result = await aio_glob(tmp_path, "*.txt")
    assert len(result) == 2
    assert all(isinstance(p, Path) for p in result)


@pytest.mark.asyncio
async def test_iterdir(tmp_path):
    (tmp_path / "x").write_text("x")
    (tmp_path / "y").write_text("y")
    result = await aio_iterdir(tmp_path)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_unlink(tmp_path):
    p = tmp_path / "del.txt"
    p.write_text("bye")
    await aio_unlink(p)
    assert not p.exists()
    # missing_ok=True by default
    await aio_unlink(p)  # Should not raise


@pytest.mark.asyncio
async def test_rename(tmp_path):
    src = tmp_path / "old.txt"
    dst = tmp_path / "new.txt"
    src.write_text("data")
    result = await aio_rename(src, dst)
    assert not src.exists()
    assert dst.read_text() == "data"
    assert result == dst
