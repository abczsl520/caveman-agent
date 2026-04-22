"""Tests for cron system."""
import pytest
import sqlite3
from pathlib import Path
from caveman.cron import CronStore, CronJob, CronScheduler, _next_run_time, _simple_interval_next
from datetime import datetime, timezone, timedelta


class TestCronStore:
    def test_add_and_get(self, tmp_path):
        store = CronStore(tmp_path / "test.db")
        job = CronJob(id="", name="test", schedule="*/5 * * * *", task="echo hi")
        store.add_job(job)
        assert job.id  # auto-generated

        got = store.get_job(job.id)
        assert got is not None
        assert got.name == "test"
        assert got.schedule == "*/5 * * * *"
        assert got.task == "echo hi"
        assert got.enabled is True
        store.close()

    def test_list_jobs(self, tmp_path):
        store = CronStore(tmp_path / "test.db")
        store.add_job(CronJob(id="", name="a", schedule="5m", task="task a"))
        store.add_job(CronJob(id="", name="b", schedule="1h", task="task b", enabled=False))

        all_jobs = store.list_jobs()
        assert len(all_jobs) == 2

        enabled = store.list_jobs(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].name == "a"
        store.close()

    def test_update_job(self, tmp_path):
        store = CronStore(tmp_path / "test.db")
        job = CronJob(id="", name="old", schedule="5m", task="old task")
        store.add_job(job)

        store.update_job(job.id, name="new", task="new task")
        got = store.get_job(job.id)
        assert got.name == "new"
        assert got.task == "new task"
        store.close()

    def test_delete_job(self, tmp_path):
        store = CronStore(tmp_path / "test.db")
        job = CronJob(id="", name="del", schedule="5m", task="x")
        store.add_job(job)
        assert store.delete_job(job.id)
        assert store.get_job(job.id) is None
        store.close()

    def test_record_and_get_runs(self, tmp_path):
        from caveman.cron import CronRun
        store = CronStore(tmp_path / "test.db")
        job = CronJob(id="", name="r", schedule="5m", task="x")
        store.add_job(job)

        run = CronRun(id="r1", job_id=job.id, started_at="2026-01-01T00:00:00Z")
        store.record_run(run)
        store.update_run("r1", status="success", result="done")

        runs = store.get_recent_runs(job.id)
        assert len(runs) == 1
        assert runs[0].status == "success"
        store.close()


class TestScheduleParsing:
    def test_simple_interval_minutes(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _simple_interval_next("5m", base)
        assert result == base + timedelta(minutes=5)

    def test_simple_interval_hours(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _simple_interval_next("2h", base)
        assert result == base + timedelta(hours=2)

    def test_simple_interval_seconds(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _simple_interval_next("30s", base)
        assert result == base + timedelta(seconds=30)

    def test_invalid_interval(self):
        assert _simple_interval_next("invalid") is None

    def test_next_run_time_with_croniter(self):
        """Test with croniter if available."""
        try:
            import croniter
            base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            result = _next_run_time("*/5 * * * *", base)
            assert result is not None
            assert result > base
        except ImportError:
            pytest.skip("croniter not installed")

    def test_next_run_time_fallback(self):
        """Falls back to simple interval for non-cron expressions."""
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _next_run_time("10m", base)
        assert result == base + timedelta(minutes=10)


class TestCronTool:
    @pytest.mark.asyncio
    async def test_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("caveman.paths.CAVEMAN_HOME", tmp_path)
        # Create the DB first
        CronStore(tmp_path / "sessions.db").close()

        from caveman.tools.builtin.cron_tool import cron_tool
        result = await cron_tool({"action": "list"})
        assert result["ok"]
        assert result["jobs"] == []

    @pytest.mark.asyncio
    async def test_create_and_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("caveman.paths.CAVEMAN_HOME", tmp_path)
        CronStore(tmp_path / "sessions.db").close()

        from caveman.tools.builtin.cron_tool import cron_tool
        result = await cron_tool({
            "action": "create",
            "name": "test job",
            "schedule": "5m",
            "task": "check server",
        })
        assert result["ok"]
        job_id = result["job_id"]

        result = await cron_tool({"action": "list"})
        assert len(result["jobs"]) == 1
        assert result["jobs"][0]["name"] == "test job"

        result = await cron_tool({"action": "delete", "job_id": job_id})
        assert result["ok"]

    @pytest.mark.asyncio
    async def test_create_missing_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("caveman.paths.CAVEMAN_HOME", tmp_path)
        CronStore(tmp_path / "sessions.db").close()

        from caveman.tools.builtin.cron_tool import cron_tool
        result = await cron_tool({"action": "create", "name": "x"})
        assert not result["ok"]
        assert "required" in result["error"]
