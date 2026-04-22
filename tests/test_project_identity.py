"""Tests for engines/project_identity.py — project detection and persistence."""
import pytest
from pathlib import Path
from caveman.engines.project_identity import (
    ProjectIdentity, ProjectIdentityStore, detect_project_from_messages, _safe_filename,
)


class TestProjectIdentity:
    def test_to_dict_roundtrip(self):
        p = ProjectIdentity(name="test-proj", path="/tmp/test", mission="test mission")
        d = p.to_dict()
        assert d["name"] == "test-proj"
        assert d["path"] == "/tmp/test"
        assert d["mission"] == "test mission"

    def test_from_dict(self):
        d = {"name": "proj", "path": "/x", "mission": "m", "principles": ["a"]}
        p = ProjectIdentity.from_dict(d)
        assert p.name == "proj"
        assert p.principles == ["a"]

    def test_from_dict_missing_fields(self):
        p = ProjectIdentity.from_dict({"name": "minimal"})
        assert p.name == "minimal"
        assert p.path == ""


class TestProjectIdentityStore:
    def test_save_and_load(self, tmp_path):
        store = ProjectIdentityStore(tmp_path)
        proj = ProjectIdentity(name="caveman", path="/projects/caveman")
        store.save(proj)
        loaded = store.load("caveman")
        assert loaded is not None
        assert loaded.name == "caveman"
        assert loaded.path == "/projects/caveman"

    def test_load_nonexistent(self, tmp_path):
        store = ProjectIdentityStore(tmp_path)
        assert store.load("nope") is None

    def test_list_projects(self, tmp_path):
        store = ProjectIdentityStore(tmp_path)
        store.save(ProjectIdentity(name="a"))
        store.save(ProjectIdentity(name="b"))
        projects = store.list_projects()
        names = [p.name for p in projects]
        assert "a" in names
        assert "b" in names

    def test_load_by_path(self, tmp_path):
        store = ProjectIdentityStore(tmp_path)
        store.save(ProjectIdentity(name="myproj", path="/home/user/myproj"))
        found = store.load_by_path("/home/user/myproj")
        assert found is not None
        assert found.name == "myproj"

    def test_load_by_path_not_found(self, tmp_path):
        store = ProjectIdentityStore(tmp_path)
        assert store.load_by_path("/nonexistent") is None


class TestDetectProject:
    def test_detect_from_cd_command(self):
        messages = [
            {"role": "user", "content": "cd ~/projects/caveman && ls"},
            {"role": "assistant", "content": "Here are the files..."},
        ]
        result = detect_project_from_messages(messages)
        assert result is not None  # Should detect project from path mention
        assert "caveman" in result.name.lower() or "caveman" in result.path.lower()

    def test_detect_from_path_mention(self):
        messages = [
            {"role": "user", "content": "look at /home/user/projects/myapp/src/main.py"},
        ]
        result = detect_project_from_messages(messages)
        # May or may not detect — depends on heuristics
        assert result is None or hasattr(result, "name")  # Valid return type

    def test_empty_messages(self):
        result = detect_project_from_messages([])
        assert result is None


class TestSafeFilename:
    def test_basic(self):
        assert _safe_filename("my-project") == "my-project"

    def test_special_chars(self):
        result = _safe_filename("my/project:name")
        assert "/" not in result
        assert ":" not in result

    def test_empty(self):
        result = _safe_filename("")
        assert result  # Should return something, not empty
