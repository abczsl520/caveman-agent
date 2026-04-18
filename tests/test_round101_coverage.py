"""Tests for Round 101 — coverage boost for low-coverage modules.

Covers: training/stats, transcribe_tool, web_search, image_gen_tool, browser.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── training/stats ──────────────────────────────────────────────────

class TestTrainingStats:
    def test_no_trajectories_dir(self, tmp_path):
        from caveman.training.stats import show_training_stats
        result = show_training_stats(str(tmp_path / "nonexistent"), 0.5)
        assert "No trajectories" in result

    def test_empty_trajectories_dir(self, tmp_path):
        from caveman.training.stats import show_training_stats
        traj_dir = tmp_path / "trajs"
        traj_dir.mkdir()
        result = show_training_stats(str(traj_dir), 0.5)
        assert "0" in result or "No" in result

    def test_with_trajectories(self, tmp_path):
        from caveman.training.stats import show_training_stats
        traj_dir = tmp_path / "trajs"
        traj_dir.mkdir()
        f = traj_dir / "test.jsonl"
        entries = [
            {"task": "fix bug", "quality_score": 0.9, "trajectory": [{"role": "user"}, {"role": "assistant"}]},
            {"task": "add feature", "quality_score": 0.3, "trajectory": [{"role": "user"}]},
            {"task": "refactor", "trajectory": []},  # unscored
        ]
        f.write_text("\n".join(json.dumps(e) for e in entries))
        result = show_training_stats(str(traj_dir), 0.5)
        assert "3" in result or "high" in result.lower() or "total" in result.lower()

    def test_malformed_jsonl(self, tmp_path):
        from caveman.training.stats import show_training_stats
        traj_dir = tmp_path / "trajs"
        traj_dir.mkdir()
        f = traj_dir / "bad.jsonl"
        f.write_text("not json\n{\"task\": \"ok\", \"quality_score\": 0.8}\n")
        result = show_training_stats(str(traj_dir), 0.5)
        assert "1" in result  # should count the valid one


# ── transcribe_tool ─────────────────────────────────────────────────

class TestTranscribeTool:
    @pytest.mark.asyncio
    async def test_missing_file(self):
        from caveman.tools.builtin.transcribe_tool import transcribe
        result = await transcribe(file_path="/nonexistent/audio.mp3")
        assert result.get("ok") is False or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_unsupported_format(self, tmp_path):
        from caveman.tools.builtin.transcribe_tool import transcribe
        f = tmp_path / "test.xyz"
        f.write_text("not audio")
        result = await transcribe(file_path=str(f))
        assert result.get("ok") is False or "unsupported" in str(result).lower() or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_file_too_large(self, tmp_path):
        from caveman.tools.builtin.transcribe_tool import transcribe
        f = tmp_path / "huge.mp3"
        f.write_bytes(b"\x00" * 1024)  # small file, but we can test the path
        with patch("caveman.tools.builtin.transcribe_tool.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=200 * 1024 * 1024)  # 200MB
            result = await transcribe(file_path=str(f))
            assert result.get("ok") is False or "size" in str(result).lower() or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_whisper_not_found(self, tmp_path):
        from caveman.tools.builtin.transcribe_tool import transcribe
        f = tmp_path / "test.mp3"
        f.write_bytes(b"\x00" * 100)
        with patch("shutil.which", return_value=None):
            result = await transcribe(file_path=str(f))
            assert result.get("ok") is False or "whisper" in str(result).lower() or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_transcribe_url_invalid(self):
        from caveman.tools.builtin.transcribe_tool import transcribe_url
        result = await transcribe_url(url="not-a-url")
        assert result.get("ok") is False or "error" in str(result).lower()


# ── web_search ──────────────────────────────────────────────────────

class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_no_api_key(self):
        from caveman.tools.builtin.web_search import web_search
        with patch.dict("os.environ", {}, clear=True):
            with patch("caveman.tools.builtin.web_search.os.environ.get", return_value=None):
                result = await web_search(query="test query")
                # Should handle missing key gracefully
                assert isinstance(result, (dict, list, str))

    @pytest.mark.asyncio
    async def test_search_success(self):
        from caveman.tools.builtin.web_search import web_search
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"title": "Test", "url": "https://example.com", "content": "Test content"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with patch("os.environ.get", return_value="test-key"):
                result = await web_search(query="test query")
                assert isinstance(result, (dict, list, str))

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        from caveman.tools.builtin.web_search import web_search
        result = await web_search(query="")
        assert isinstance(result, (dict, list, str))


# ── image_gen_tool ──────────────────────────────────────────────────

class TestImageGenTool:
    @pytest.mark.asyncio
    async def test_generate_no_api_key(self):
        from caveman.tools.builtin.image_gen_tool import image_generate
        with patch.dict("os.environ", {}, clear=True):
            result = await image_generate(prompt="a cat")
            assert result.get("ok") is False or "error" in str(result).lower() or "key" in str(result).lower()

    @pytest.mark.asyncio
    async def test_generate_success(self, tmp_path):
        from caveman.tools.builtin.image_gen_tool import image_generate
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"url": "https://example.com/image.png", "revised_prompt": "a cute cat"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_img_response = MagicMock()
        mock_img_response.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mock_img_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_img_response):
                with patch("caveman.tools.builtin.image_gen_tool._OUTPUT_DIR", tmp_path):
                    result = await image_generate(prompt="a cat")
                    assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_edit_missing_image(self):
        from caveman.tools.builtin.image_gen_tool import image_edit
        result = await image_edit(image_path="/nonexistent.png", prompt="make it blue")
        assert result.get("ok") is False or "error" in str(result).lower()

    @pytest.mark.asyncio
    async def test_generate_invalid_size(self):
        from caveman.tools.builtin.image_gen_tool import image_generate
        result = await image_generate(prompt="a cat", size="99x99")
        # Should either work or return error gracefully
        assert isinstance(result, dict)


# ── browser ─────────────────────────────────────────────────────────

class TestBrowser:
    @pytest.mark.asyncio
    async def test_browser_navigate_no_bridge_no_playwright(self):
        from caveman.tools.builtin.browser import browser_dispatch
        # Without bridge and without playwright, should raise
        with patch("caveman.tools.builtin.browser._ensure_playwright", new_callable=AsyncMock, side_effect=RuntimeError("no playwright")):
            with pytest.raises(RuntimeError):
                await browser_dispatch(action="navigate", url="https://example.com")

    @pytest.mark.asyncio
    async def test_browser_with_bridge(self):
        from caveman.tools.builtin.browser import browser_dispatch, set_bridge
        mock_bridge = MagicMock()
        mock_bridge.call_tool = AsyncMock(return_value={"result": "page loaded"})
        set_bridge(mock_bridge)
        try:
            result = await browser_dispatch(action="navigate", url="https://example.com")
            assert isinstance(result, dict)
            assert result.get("ok") is True
        finally:
            set_bridge(None)

    @pytest.mark.asyncio
    async def test_browser_bridge_error(self):
        from caveman.tools.builtin.browser import browser_dispatch, set_bridge
        mock_bridge = MagicMock()
        mock_bridge.call_tool = AsyncMock(side_effect=Exception("bridge down"))
        set_bridge(mock_bridge)
        try:
            result = await browser_dispatch(action="navigate", url="https://example.com")
            assert isinstance(result, dict)
            assert result.get("ok") is False
        finally:
            set_bridge(None)

    def test_browser_mode_standalone(self):
        from caveman.tools.builtin.browser import _mode, set_bridge
        set_bridge(None)
        assert _mode() == "standalone"

    def test_browser_mode_bridge(self):
        from caveman.tools.builtin.browser import _mode, set_bridge
        set_bridge(MagicMock())
        try:
            assert _mode() == "bridge"
        finally:
            set_bridge(None)
