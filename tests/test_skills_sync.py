"""Tests for skills sync."""
import pytest
from pathlib import Path

from caveman.skills.sync import sync_skills, _dir_hash, _load_manifest, _save_manifest


@pytest.fixture
def bundled_dir(tmp_path):
    d = tmp_path / "bundled"
    d.mkdir()
    return d


@pytest.fixture
def user_dir(tmp_path):
    d = tmp_path / "user_skills"
    d.mkdir()
    return d


@pytest.fixture
def manifest_path(user_dir):
    return user_dir / ".bundled_manifest"


def _create_skill(parent, name, content="default"):
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\n{content}")
    return skill_dir


class TestDirHash:
    def test_deterministic(self, tmp_path):
        _create_skill(tmp_path, "test")
        h1 = _dir_hash(tmp_path / "test")
        h2 = _dir_hash(tmp_path / "test")
        assert h1 == h2

    def test_different_content(self, tmp_path):
        _create_skill(tmp_path, "a", "content1")
        _create_skill(tmp_path, "b", "content2")
        assert _dir_hash(tmp_path / "a") != _dir_hash(tmp_path / "b")


class TestManifest:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "manifest"
        data = {"skill-a": "abc123", "skill-b": "def456"}
        _save_manifest(path, data)
        loaded = _load_manifest(path)
        assert loaded == data

    def test_empty(self, tmp_path):
        assert _load_manifest(tmp_path / "nonexistent") == {}

    def test_v1_migration(self, tmp_path):
        path = tmp_path / "manifest"
        path.write_text("skill-a\nskill-b\n")
        loaded = _load_manifest(path)
        assert loaded == {"skill-a": "", "skill-b": ""}


class TestSyncSkills:
    def test_add_new_skill(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "new-skill")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "new-skill" in result["added"]
        assert (user_dir / "new-skill" / "SKILL.md").exists()

    def test_update_unmodified(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "my-skill", "v1")
        sync_skills(bundled_dir, user_dir, manifest_path)

        # Update bundled version
        (bundled_dir / "my-skill" / "SKILL.md").write_text("# my-skill\nv2")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "my-skill" in result["updated"]
        assert "v2" in (user_dir / "my-skill" / "SKILL.md").read_text()

    def test_skip_customized(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "my-skill", "v1")
        sync_skills(bundled_dir, user_dir, manifest_path)

        # User modifies
        (user_dir / "my-skill" / "SKILL.md").write_text("# my-skill\ncustom")
        # Bundled also updates
        (bundled_dir / "my-skill" / "SKILL.md").write_text("# my-skill\nv2")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "my-skill" in result["skipped_customized"]
        assert "custom" in (user_dir / "my-skill" / "SKILL.md").read_text()

    def test_respect_user_deletion(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "my-skill")
        sync_skills(bundled_dir, user_dir, manifest_path)

        # User deletes
        import shutil
        shutil.rmtree(user_dir / "my-skill")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "my-skill" in result["skipped_deleted"]
        assert not (user_dir / "my-skill").exists()

    def test_clean_removed_from_bundled(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "old-skill")
        sync_skills(bundled_dir, user_dir, manifest_path)

        # Remove from bundled
        import shutil
        shutil.rmtree(bundled_dir / "old-skill")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "old-skill" in result["removed_from_manifest"]

    def test_dry_run(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, "new-skill")
        result = sync_skills(bundled_dir, user_dir, manifest_path, dry_run=True)
        assert "new-skill" in result["added"]
        assert not (user_dir / "new-skill").exists()  # Not actually created

    def test_no_bundled_dir(self, tmp_path, user_dir, manifest_path):
        result = sync_skills(tmp_path / "nonexistent", user_dir, manifest_path)
        assert all(len(v) == 0 for v in result.values())

    def test_skip_hidden_dirs(self, bundled_dir, user_dir, manifest_path):
        _create_skill(bundled_dir, ".hidden")
        _create_skill(bundled_dir, "visible")
        result = sync_skills(bundled_dir, user_dir, manifest_path)
        assert "visible" in result["added"]
        assert ".hidden" not in result["added"]
