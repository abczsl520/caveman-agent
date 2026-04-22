"""Tests for config loader caching."""
import time
from pathlib import Path

import pytest
import yaml

from caveman.config.loader import (
    load_config,
    invalidate_config_cache,
    _cache,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure cache is clean before/after each test."""
    _cache.clear()
    yield
    _cache.clear()


@pytest.fixture
def config_dir(tmp_path):
    """Create a minimal config setup."""
    cfg = {"agent": {"model": "test-model"}, "gateway": {"platform": "test"}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg))
    return tmp_path, config_file


class TestConfigCache:
    def test_cache_hit(self, config_dir):
        """Second call returns cached result without re-reading disk."""
        _, config_file = config_dir
        c1 = load_config(config_file, validate=False)
        c2 = load_config(config_file, validate=False)
        assert c1 is c2  # Same object reference = cache hit

    def test_cache_invalidation(self, config_dir):
        _, config_file = config_dir
        c1 = load_config(config_file, validate=False)
        invalidate_config_cache()
        c2 = load_config(config_file, validate=False)
        assert c1 is not c2  # Different object = cache miss
        assert c1 == c2  # But same content

    def test_cache_detects_mtime_change(self, config_dir):
        _, config_file = config_dir
        c1 = load_config(config_file, validate=False)

        # Modify file (ensure mtime changes)
        time.sleep(0.05)
        new_cfg = {"agent": {"model": "updated-model"}, "gateway": {"platform": "test"}}
        config_file.write_text(yaml.dump(new_cfg))

        c2 = load_config(config_file, validate=False)
        assert c2["agent"]["model"] == "updated-model"
        assert c1 is not c2

    def test_different_paths_separate_cache(self, tmp_path):
        cfg_a = tmp_path / "a.yaml"
        cfg_b = tmp_path / "b.yaml"
        cfg_a.write_text(yaml.dump({"name": "a"}))
        cfg_b.write_text(yaml.dump({"name": "b"}))

        ca = load_config(cfg_a, validate=False)
        cb = load_config(cfg_b, validate=False)
        assert ca["name"] == "a"
        assert cb["name"] == "b"
