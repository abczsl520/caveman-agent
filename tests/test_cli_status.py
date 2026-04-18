"""Tests for CLI status dashboard."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from caveman.cli.status import (
    _count_memories,
    _count_sessions,
    _count_skills,
    _count_projects,
    status_text,
)


def test_count_memories_empty(tmp_path):
    with patch("caveman.cli.status.MEMORY_DIR", tmp_path / "nonexistent"):
        assert _count_memories() == {}


def test_count_memories_with_data(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "episodic.json").write_text(json.dumps([{"id": "1"}, {"id": "2"}]))
    (mem_dir / "semantic.json").write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
    with patch("caveman.cli.status.MEMORY_DIR", mem_dir):
        counts = _count_memories()
        assert counts["episodic"] == 2
        assert counts["semantic"] == 3


def test_count_sessions_empty(tmp_path):
    with patch("caveman.cli.status.SESSIONS_DIR", tmp_path / "nonexistent"):
        assert _count_sessions() == 0


def test_count_sessions_with_data(tmp_path):
    sess_dir = tmp_path / "sessions"
    sess_dir.mkdir()
    (sess_dir / "s1.yaml").write_text("task: test")
    (sess_dir / "s2.yaml").write_text("task: test2")
    (sess_dir / "s3.txt").write_text("not yaml")
    with patch("caveman.cli.status.SESSIONS_DIR", sess_dir):
        assert _count_sessions() == 2


def test_count_skills_empty(tmp_path):
    with patch("caveman.cli.status.SKILLS_DIR", tmp_path / "nonexistent"):
        assert _count_skills() == (0, 0)


def test_count_skills_mixed(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    s1 = skills_dir / "skill1"
    s1.mkdir()
    (s1 / "skill.yaml").write_text("name: s1\nstatus: active")
    s2 = skills_dir / "skill2"
    s2.mkdir()
    (s2 / "skill.yaml").write_text("name: s2\nstatus: draft")
    with patch("caveman.cli.status.SKILLS_DIR", skills_dir):
        active, draft = _count_skills()
        assert active == 1
        assert draft == 1


def test_count_projects_empty(tmp_path):
    with patch("caveman.cli.status.PROJECTS_DIR", tmp_path / "nonexistent"):
        assert _count_projects() == 0


def test_status_text_contains_key_info():
    text = status_text()
    assert "Caveman" in text
    assert "Model:" in text
    assert "Memory:" in text
    assert "Sessions:" in text
    assert "Engines:" in text
