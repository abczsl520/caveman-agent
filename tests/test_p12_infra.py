"""Tests for P12: auxiliary_client, rate_limit_tracker, cronjob, tool_result_storage, model_normalize, backup."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


# ── Auxiliary Client ──

class TestAuxiliaryClient:
    def test_auto_detect_provider(self):
        from caveman.agent.auxiliary_client import _auto_detect_provider
        import os
        saved = {k: os.environ.pop(k, None) for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        try:
            os.environ["OPENAI_API_KEY"] = "test"
            assert _auto_detect_provider() == "openai"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    def test_default_model(self):
        from caveman.agent.auxiliary_client import _default_model
        assert _default_model("deepseek") == "deepseek-chat"
        assert _default_model("openai") == "gpt-4o-mini"

    def test_config_from_env(self):
        from caveman.agent.auxiliary_client import AuxiliaryConfig
        import os
        saved = os.environ.pop("CAVEMAN_AUX_MODEL", None)
        try:
            os.environ["CAVEMAN_AUX_MODEL"] = "test-model"
            config = AuxiliaryConfig.from_env()
            assert config.model == "test-model"
        finally:
            if saved:
                os.environ["CAVEMAN_AUX_MODEL"] = saved
            else:
                os.environ.pop("CAVEMAN_AUX_MODEL", None)


# ── Rate Limit Tracker ──

class TestRateLimitTracker:
    def test_initial_state(self):
        from caveman.agent.rate_limit_tracker import RateLimitTracker
        tracker = RateLimitTracker()
        state = tracker.get_state("openai:gpt-4o")
        assert not state.is_limited
        assert state.wait_seconds == 0

    def test_record_429(self):
        from caveman.agent.rate_limit_tracker import RateLimitTracker
        tracker = RateLimitTracker()
        tracker.record_429("key1", retry_after=5)
        assert tracker.should_wait("key1") > 0

    def test_record_success_resets(self):
        from caveman.agent.rate_limit_tracker import RateLimitTracker
        tracker = RateLimitTracker()
        tracker.record_429("key1")
        tracker.record_success("key1")
        state = tracker.get_state("key1")
        assert state.consecutive_429s == 0

    def test_backoff(self):
        from caveman.agent.rate_limit_tracker import RateLimitState
        state = RateLimitState(consecutive_429s=3)
        assert state.backoff_seconds() == 4.0  # 1 * 2^2

    def test_update_from_headers(self):
        from caveman.agent.rate_limit_tracker import RateLimitTracker
        tracker = RateLimitTracker()
        tracker.update_from_headers("key1", {
            "x-ratelimit-remaining-requests": "5",
            "x-ratelimit-remaining-tokens": "10000",
        })
        state = tracker.get_state("key1")
        assert state.requests_remaining == 5

    def test_parse_duration(self):
        from caveman.agent.rate_limit_tracker import _parse_duration
        assert _parse_duration("30s") == 30.0
        assert _parse_duration("5m") == 300.0
        assert _parse_duration("1h") == 3600.0
        assert _parse_duration("6m0s") == 360.0


# ── Cronjob ──

class TestCronjob:
    def test_parse_interval(self):
        from caveman.tools.builtin.cronjob import parse_interval
        assert parse_interval("5m") == 300
        assert parse_interval("1h") == 3600
        assert parse_interval("30s") == 30
        assert parse_interval("invalid") is None

    def test_add_and_list(self, tmp_path):
        from caveman.tools.builtin.cronjob import CronManager
        mgr = CronManager(persist_dir=tmp_path / "cron")
        job = mgr.add("test-job", "5m", command="echo hello")
        assert job.id
        assert job.next_run > time.time()
        jobs = mgr.list_jobs()
        assert len(jobs) == 1

    def test_remove(self, tmp_path):
        from caveman.tools.builtin.cronjob import CronManager
        mgr = CronManager(persist_dir=tmp_path / "cron")
        job = mgr.add("test", "1h")
        assert mgr.remove(job.id)
        assert not mgr.remove(job.id)

    def test_mark_run(self, tmp_path):
        from caveman.tools.builtin.cronjob import CronManager
        mgr = CronManager(persist_dir=tmp_path / "cron")
        job = mgr.add("test", "5m")
        mgr.mark_run(job.id, result="ok")
        updated = mgr.get(job.id)
        assert updated.run_count == 1
        assert updated.last_result == "ok"

    def test_persistence(self, tmp_path):
        from caveman.tools.builtin.cronjob import CronManager
        mgr1 = CronManager(persist_dir=tmp_path / "cron")
        mgr1.add("persist-test", "1h")
        mgr2 = CronManager(persist_dir=tmp_path / "cron")
        assert len(mgr2.list_jobs()) == 1


# ── Tool Result Storage ──

class TestToolResultStorage:
    def test_store_and_get(self):
        from caveman.tools.builtin.tool_result_storage import ToolResultStore
        store = ToolResultStore()
        store.store("bash", "echo hello", "hello\n")
        result = store.get("bash", "echo hello")
        assert result == "hello\n"

    def test_cache_miss(self):
        from caveman.tools.builtin.tool_result_storage import ToolResultStore
        store = ToolResultStore()
        assert store.get("bash", "unknown") is None

    def test_invalidate(self):
        from caveman.tools.builtin.tool_result_storage import ToolResultStore
        store = ToolResultStore()
        store.store("bash", "cmd", "result")
        assert store.invalidate("bash", "cmd")
        assert store.get("bash", "cmd") is None

    def test_stats(self):
        from caveman.tools.builtin.tool_result_storage import ToolResultStore
        store = ToolResultStore()
        store.store("t1", "i1", "r1")
        store.get("t1", "i1")
        stats = store.stats()
        assert stats["entries"] == 1
        assert stats["total_hits"] == 1


# ── Model Normalize ──

class TestModelNormalize:
    def test_resolve_alias(self):
        from caveman.cli.model_normalize import resolve_alias
        assert resolve_alias("opus") == "claude-opus-4-6"
        assert resolve_alias("4o") == "gpt-4o"
        assert resolve_alias("flash") == "gemini-2.5-flash"

    def test_detect_provider(self):
        from caveman.cli.model_normalize import detect_provider
        assert detect_provider("claude-opus-4-6") == "anthropic"
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gemini-2.5-pro") == "google"

    def test_normalize_model(self):
        from caveman.cli.model_normalize import normalize_model
        model, provider = normalize_model("opus")
        assert model == "claude-opus-4-6"
        assert provider == "anthropic"

    def test_normalize_with_provider(self):
        from caveman.cli.model_normalize import normalize_model
        model, provider = normalize_model("openai/gpt-4o")
        assert model == "gpt-4o"
        assert provider == "openai"

    def test_list_known_models(self):
        from caveman.cli.model_normalize import list_known_models
        models = list_known_models()
        assert "anthropic" in models
        assert "openai" in models


# ── Backup ──

class TestBackup:
    def test_create_backup(self, tmp_path):
        from caveman.cli.backup import BackupManager
        home = tmp_path / "home"
        (home / "config.json").parent.mkdir(parents=True)
        (home / "config.json").write_text('{"test": true}')
        mgr = BackupManager(home_dir=home, backup_dir=tmp_path / "backups")
        manifest = mgr.create(description="test backup", include=["config"])
        assert manifest.id
        assert "config" in manifest.includes

    def test_list_backups(self, tmp_path):
        from caveman.cli.backup import BackupManager
        home = tmp_path / "home"
        (home / "config.json").parent.mkdir(parents=True)
        (home / "config.json").write_text("{}")
        mgr = BackupManager(home_dir=home, backup_dir=tmp_path / "backups")
        mgr.create(include=["config"])
        backups = mgr.list_backups()
        assert len(backups) == 1

    def test_delete_backup(self, tmp_path):
        from caveman.cli.backup import BackupManager
        home = tmp_path / "home"
        (home / "config.json").parent.mkdir(parents=True)
        (home / "config.json").write_text("{}")
        mgr = BackupManager(home_dir=home, backup_dir=tmp_path / "backups")
        manifest = mgr.create(include=["config"])
        assert mgr.delete(manifest.id)
        assert len(mgr.list_backups()) == 0

    def test_restore(self, tmp_path):
        from caveman.cli.backup import BackupManager
        home = tmp_path / "home"
        (home / "config.json").parent.mkdir(parents=True)
        (home / "config.json").write_text('{"version": 1}')
        mgr = BackupManager(home_dir=home, backup_dir=tmp_path / "backups")
        manifest = mgr.create(include=["config"])
        # Modify original
        (home / "config.json").write_text('{"version": 2}')
        # Restore
        result = mgr.restore(manifest.id)
        assert result["success"]
        assert json.loads((home / "config.json").read_text())["version"] == 1
