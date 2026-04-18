"""Tests for todo_tool, skill_manager_tool, and vision_tool (Round 88)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from caveman.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg._register_builtins()
    return reg


# ── Todo tool tests ──

@pytest.mark.asyncio
async def test_todo_lifecycle(registry, tmp_path):
    todo_file = tmp_path / "todos.json"
    with patch("caveman.tools.builtin.todo_tool._TODO_FILE", todo_file):
        # Add
        r = await registry.dispatch("todo_add", {"title": "Buy milk", "priority": "high"})
        assert r["ok"] is True
        tid = r["id"]

        # List pending
        items = await registry.dispatch("todo_list", {"status": "pending"})
        assert len(items) == 1
        assert items[0]["title"] == "Buy milk"
        assert items[0]["priority"] == "high"

        # Done
        r = await registry.dispatch("todo_done", {"id": tid})
        assert r["ok"] is True

        # List pending (empty now)
        items = await registry.dispatch("todo_list", {"status": "pending"})
        assert len(items) == 0

        # List done
        items = await registry.dispatch("todo_list", {"status": "done"})
        assert len(items) == 1

        # Remove
        r = await registry.dispatch("todo_remove", {"id": tid})
        assert r["ok"] is True

        # List all (empty)
        items = await registry.dispatch("todo_list", {"status": "all"})
        assert len(items) == 0


@pytest.mark.asyncio
async def test_todo_persistence(registry, tmp_path):
    todo_file = tmp_path / "todos.json"
    with patch("caveman.tools.builtin.todo_tool._TODO_FILE", todo_file):
        await registry.dispatch("todo_add", {"title": "Persist me"})
        # Verify file exists and is valid JSON
        data = json.loads(todo_file.read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persist me"


# ── Skill manager tool tests ──

@pytest.mark.asyncio
async def test_skill_list(registry, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    import yaml as _yaml; (skills_dir / "greet.yaml").write_text(_yaml.dump({
        "name": "greet", "version": 2, "description": "Say hello", "last_used": "2025-01-01",
    }))
    (skills_dir / "deploy.yaml").write_text(_yaml.dump({
        "name": "deploy", "version": 1, "description": "Deploy app",
    }))
    with patch("caveman.tools.builtin.skill_manager_tool.SKILLS_DIR", skills_dir):
        result = await registry.dispatch("skill_list", {})
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert names == {"deploy", "greet"}


@pytest.mark.asyncio
async def test_skill_show(registry, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    import yaml as _yaml; (skills_dir / "greet.yaml").write_text(_yaml.dump({
        "name": "greet", "version": 3, "description": "Say hello",
        "trigger": "hello *", "trigger_patterns": ["hello world"],
        "created_at": "2025-01-01", "updated_at": "2025-06-01",
    }))
    with patch("caveman.tools.builtin.skill_manager_tool.SKILLS_DIR", skills_dir):
        result = await registry.dispatch("skill_show", {"name": "greet"})
        assert result["name"] == "greet"
        assert result["version"] == 3
        assert result["trigger"] == "hello *"


@pytest.mark.asyncio
async def test_skill_show_not_found(registry, tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    with patch("caveman.tools.builtin.skill_manager_tool.SKILLS_DIR", skills_dir):
        result = await registry.dispatch("skill_show", {"name": "nope"})
        assert "error" in result


# ── Vision tool tests ──

@pytest.mark.asyncio
async def test_vision_describe(registry, tmp_path):
    img = tmp_path / "test.png"
    # Minimal 1x1 PNG
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    result = await registry.dispatch("vision_describe", {"image_path": str(img)})
    assert result["ok"] is True
    assert result["mime_type"] == "image/png"
    assert result["image_size"] == 108
    assert isinstance(result["base64_preview"], str)
    assert len(result["base64_preview"]) <= 100


@pytest.mark.asyncio
async def test_vision_describe_not_found(registry):
    result = await registry.dispatch("vision_describe", {"image_path": "/nonexistent/img.png"})
    assert "error" in result


@pytest.mark.asyncio
async def test_vision_describe_unsupported_format(registry, tmp_path):
    f = tmp_path / "data.xyz"
    f.write_bytes(b"not an image")
    result = await registry.dispatch("vision_describe", {"image_path": str(f)})
    assert "error" in result
    assert "Unsupported" in result["error"]
