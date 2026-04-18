"""Tests for sandbox, transcribe, image_gen, and url_safety tools (Round 97)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caveman.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg._register_builtins()
    return reg


# ── Sandbox tool tests ──


@pytest.mark.asyncio
async def test_sandbox_exec_simple(registry):
    r = await registry.dispatch("sandbox_exec", {"code": "print('hello world')"})
    assert r["ok"] is True
    assert "hello world" in r["stdout"]
    assert r["returncode"] == 0
    assert r["duration_s"] >= 0


@pytest.mark.asyncio
async def test_sandbox_exec_timeout(registry):
    r = await registry.dispatch("sandbox_exec", {"code": "import time; time.sleep(30)", "timeout": 1})
    assert r["ok"] is False
    assert "Timeout" in r["stderr"]
    assert r["returncode"] == -1


@pytest.mark.asyncio
async def test_sandbox_exec_output_limit(registry):
    # Generate output larger than 100KB
    code = "print('x' * 200000)"
    r = await registry.dispatch("sandbox_exec", {"code": code})
    assert r["ok"] is True
    # Output should be truncated to ~100KB
    assert len(r["stdout"]) <= 100 * 1024 + 100  # small margin for decode


@pytest.mark.asyncio
async def test_sandbox_eval_literal(registry):
    r = await registry.dispatch("sandbox_eval", {"expression": "[1, 2, 3]"})
    assert r["ok"] is True
    assert r["result"] == "[1, 2, 3]"


@pytest.mark.asyncio
async def test_sandbox_eval_complex(registry):
    r = await registry.dispatch("sandbox_eval", {"expression": "2 + 2"})
    assert r["ok"] is True
    assert r["result"] == "4"


# ── Transcribe tool tests ──


@pytest.mark.asyncio
async def test_transcribe_missing_file(registry):
    r = await registry.dispatch("transcribe", {"file_path": "/nonexistent/audio.mp3"})
    assert "error" in r
    assert "not found" in r["error"].lower()


@pytest.mark.asyncio
async def test_transcribe_unsupported_format(registry, tmp_path):
    f = tmp_path / "data.xyz"
    f.write_bytes(b"not audio")
    r = await registry.dispatch("transcribe", {"file_path": str(f)})
    assert "error" in r
    assert "Unsupported" in r["error"]


# ── Image generation tool tests ──


@pytest.mark.asyncio
async def test_image_generate_mock(registry, tmp_path):
    import base64

    fake_b64 = base64.b64encode(b"fake png data").decode()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"b64_json": fake_b64, "revised_prompt": "a cat sitting"}]
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("caveman.tools.builtin.image_gen_tool._OUTPUT_DIR", tmp_path), \
         patch("caveman.tools.builtin.image_gen_tool.os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("httpx.AsyncClient", return_value=mock_client):
        r = await registry.dispatch("image_generate", {"prompt": "a cat"})
        assert r["ok"] is True
        assert r["revised_prompt"] == "a cat sitting"
        assert Path(r["path"]).exists()


# ── URL safety tool tests ──


@pytest.mark.asyncio
async def test_url_check_safe(registry):
    r = await registry.dispatch("url_check", {"url": "https://example.com/page"})
    assert r["ok"] is True
    assert r["safe"] is True
    assert r["domain"] == "example.com"
    assert r["warnings"] == []


@pytest.mark.asyncio
async def test_url_check_suspicious_ip(registry):
    r = await registry.dispatch("url_check", {"url": "http://192.168.1.1/login"})
    assert r["ok"] is True
    assert r["safe"] is False
    # Should have both HTTP and IP warnings
    warning_text = " ".join(r["warnings"])
    assert "IP" in warning_text
    assert "HTTP" in warning_text


@pytest.mark.asyncio
async def test_url_check_data_uri(registry):
    r = await registry.dispatch("url_check", {"url": "data:text/html,<script>alert(1)</script>"})
    assert r["ok"] is True
    assert r["safe"] is False
    assert any("data:" in w for w in r["warnings"])
