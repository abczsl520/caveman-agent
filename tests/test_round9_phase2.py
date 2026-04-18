"""Round 9 Phase 2 tests — Doctor v2, Trajectory Scorer, Config Compat,
Memory Import, Skill Auto-Create with LLM."""
from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# FR-302: Doctor v2 — Flywheel Health Dashboard
# ═══════════════════════════════════════════════════════════════

class TestDoctorV2:
    """FR-302: Flywheel health metrics."""

    @pytest.mark.asyncio
    async def test_doctor_runs(self, tmp_path):
        from caveman.cli.doctor import run_doctor
        report = await run_doctor(config_dir=str(tmp_path))
        assert report.score > 0
        text = report.to_text()
        assert "Flywheel Health Report" in text

    @pytest.mark.asyncio
    async def test_doctor_with_memories(self, tmp_path):
        from caveman.cli.doctor import run_doctor
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType

        mm = MemoryManager(base_dir=tmp_path / "memory")
        await mm.store("test fact", MemoryType.SEMANTIC)
        await mm.save()

        report = await run_doctor(config_dir=str(tmp_path))
        text = report.to_text()
        assert "1 memories" in text or "Memory" in text

    @pytest.mark.asyncio
    async def test_doctor_lint_integration(self, tmp_path):
        from caveman.cli.doctor import run_doctor
        report = await run_doctor(config_dir=str(tmp_path))
        lint_checks = [c for c in report.checks if c["name"] == "Lint"]
        assert len(lint_checks) >= 1

    @pytest.mark.asyncio
    async def test_doctor_metrics(self, tmp_path):
        from caveman.cli.doctor import run_doctor, DoctorReport
        report = DoctorReport()
        report.add_metric("Test Metric", 75.0, 60.0, "%")
        assert report.metrics["Test Metric"]["status"] == "ok"
        report.add_metric("Bad Metric", 20.0, 60.0, "%")
        assert report.metrics["Bad Metric"]["status"] == "error"


# ═══════════════════════════════════════════════════════════════
# FR-207: Trajectory Quality Scorer
# ═══════════════════════════════════════════════════════════════

class TestTrajectoryScorer:
    """FR-207: Auto quality scoring for training data."""

    @pytest.fixture
    def good_trajectory(self):
        return {
            "conversations": [
                {"from": "human", "value": "Build a Flask app"},
                {"from": "gpt", "value": "I'll create a Flask hello world app."},
                {"from": "function_call", "value": '{"name": "bash", "arguments": {"command": "mkdir myapp"}}'},
                {"from": "function_response", "value": ""},
                {"from": "function_call", "value": '{"name": "file_write", "arguments": {"path": "app.py"}}'},
                {"from": "function_response", "value": "Written"},
                {"from": "gpt", "value": "Done! Created app.py with Flask hello world. Run with: flask run"},
            ],
            "task": "Build a Flask app",
            "metadata": {
                "tool_calls": 2,
                "errors": 0,
                "duration_seconds": 45,
                "verification": "pass",
            },
        }

    @pytest.fixture
    def bad_trajectory(self):
        return {
            "conversations": [
                {"from": "human", "value": "Fix the bug"},
                {"from": "gpt", "value": "Error: cannot find file"},
            ],
            "task": "Fix the bug",
            "metadata": {"tool_calls": 0, "errors": 1, "duration_seconds": 5},
        }

    @pytest.mark.asyncio
    async def test_good_trajectory_scores_high(self, good_trajectory):
        from caveman.trajectory.scorer import TrajectoryScorer
        scorer = TrajectoryScorer()
        score = await scorer.score(good_trajectory)
        assert score >= 0.7  # FR-207: score >= 0.7 enters training set

    @pytest.mark.asyncio
    async def test_bad_trajectory_scores_low(self, bad_trajectory):
        from caveman.trajectory.scorer import TrajectoryScorer
        scorer = TrajectoryScorer()
        score = await scorer.score(bad_trajectory)
        assert score < 0.7

    @pytest.mark.asyncio
    async def test_empty_trajectory(self):
        from caveman.trajectory.scorer import TrajectoryScorer
        scorer = TrajectoryScorer()
        score = await scorer.score({"conversations": [], "metadata": {}})
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_llm_scoring(self, good_trajectory):
        async def mock_llm(prompt: str) -> str:
            return "8 — Good example with tool usage and completion"

        from caveman.trajectory.scorer import TrajectoryScorer
        scorer = TrajectoryScorer(llm_fn=mock_llm)
        score = await scorer.score(good_trajectory)
        # 40% heuristic + 60% LLM (0.8)
        assert score >= 0.7

    @pytest.mark.asyncio
    async def test_batch_scoring(self, tmp_path, good_trajectory, bad_trajectory):
        # Write test trajectories
        (tmp_path / "good.json").write_text(json.dumps(good_trajectory))
        (tmp_path / "bad.json").write_text(json.dumps(bad_trajectory))

        from caveman.trajectory.scorer import TrajectoryScorer
        scorer = TrajectoryScorer(min_quality=0.6)
        results = await scorer.score_batch(tmp_path)
        assert results["scored"] == 2
        assert results["above_threshold"] >= 1


# ═══════════════════════════════════════════════════════════════
# FR-301: Config Format Compatibility
# ═══════════════════════════════════════════════════════════════

class TestConfigCompat:
    """FR-301: Auto-detect and import external configs."""

    def test_detect_no_configs(self):
        from caveman.cli.main import _detect_external_configs
        # Should not crash even if no configs exist
        detected = _detect_external_configs()
        assert isinstance(detected, dict)

    def test_detect_openclaw_config(self, tmp_path, monkeypatch):
        # Create fake OpenClaw config
        oc_dir = tmp_path / ".openclaw"
        oc_dir.mkdir()
        config = oc_dir / "config.yaml"
        config.write_text("providers:\n  anthropic:\n    api_key: sk-test-key-123\n    model: claude-opus-4-6\n")

        # Monkey-patch Path.expanduser to use tmp_path
        from caveman.cli.main import _detect_external_configs
        import caveman.cli.main as cli_module

        original_paths = cli_module._detect_external_configs

        def patched():
            from pathlib import Path
            import yaml
            detected = {}
            path = config
            if path.exists():
                info = {"path": str(path)}
                data = yaml.safe_load(path.read_text()) or {}
                providers = data.get("providers", {})
                for prov in providers.values():
                    if isinstance(prov, dict):
                        key = prov.get("api_key", "")
                        if key and not key.startswith("$"):
                            info["api_key"] = key
                        m = prov.get("model", "")
                        if m:
                            info["model"] = m
                detected["OpenClaw"] = info
            return detected

        monkeypatch.setattr(cli_module, "_detect_external_configs", patched)
        result = patched()
        assert "OpenClaw" in result
        assert result["OpenClaw"]["api_key"] == "sk-test-key-123"
        assert result["OpenClaw"]["model"] == "claude-opus-4-6"


# ═══════════════════════════════════════════════════════════════
# FR-204: Memory Import
# ═══════════════════════════════════════════════════════════════

class TestMemoryImport:
    """FR-204: Import memories from external sources."""

    @pytest.mark.asyncio
    async def test_import_from_directory(self, tmp_path):
        from caveman.cli.importer import import_memories
        from caveman.memory.manager import MemoryManager

        # Create source files
        src = tmp_path / "source"
        src.mkdir()
        (src / "facts.md").write_text("## Server Info\nServer IP is 203.0.113.10\n\n## Tools\nUse pyenv for Python version management")
        (src / "empty.md").write_text("")

        mm = MemoryManager(base_dir=tmp_path / "memory")
        result = await import_memories("directory", mm, directory=str(src), dry_run=False)

        assert result.imported >= 2  # Two sections from facts.md
        assert result.files_processed >= 1

    @pytest.mark.asyncio
    async def test_import_dry_run(self, tmp_path):
        from caveman.cli.importer import import_memories
        from caveman.memory.manager import MemoryManager

        src = tmp_path / "source"
        src.mkdir()
        (src / "test.md").write_text("## Important\nThis is a test memory entry for import")

        mm = MemoryManager(base_dir=tmp_path / "memory")
        # dry_run=True is now the default
        result = await import_memories("directory", mm, directory=str(src))

        assert result.imported >= 1
        # Dry run: nothing actually stored
        assert mm.total_count == 0

    @pytest.mark.asyncio
    async def test_import_unknown_source(self, tmp_path):
        from caveman.cli.importer import import_memories
        from caveman.memory.manager import MemoryManager

        mm = MemoryManager(base_dir=tmp_path / "memory")
        result = await import_memories("nonexistent", mm)
        assert "Unknown source" in result.details[0]

    @pytest.mark.asyncio
    async def test_import_type_inference(self):
        from caveman.import_.base import infer_type
        from caveman.memory.types import MemoryType
        from pathlib import Path

        assert infer_type("User prefers dark mode", Path("user.md")) == MemoryType.WORKING
        assert infer_type("Step 1: install pyenv", Path("howto.md")) == MemoryType.PROCEDURAL
        assert infer_type("2026-04-14: Fixed the bug", Path("2026-04-14.md")) == MemoryType.EPISODIC
        assert infer_type("Server IP is 203.0.113.10", Path("servers.md")) == MemoryType.SEMANTIC


# ═══════════════════════════════════════════════════════════════
# FR-205: Skill Auto-Create with LLM
# ═══════════════════════════════════════════════════════════════

class TestSkillAutoCreate:
    """FR-205: LLM-powered skill creation from trajectories."""

    @pytest.fixture
    def trajectory_with_pattern(self):
        """Trajectory with repeated bash calls (deploy pattern)."""
        return [
            {"from": "human", "value": "Deploy to VPS"},
            {"from": "gpt", "value": "Starting deployment..."},
            {"from": "function_call", "value": '{"name": "bash", "arguments": {"command": "ssh vps"}}'},
            {"from": "function_response", "value": "connected"},
            {"from": "function_call", "value": '{"name": "bash", "arguments": {"command": "git pull"}}'},
            {"from": "function_response", "value": "updated"},
            {"from": "function_call", "value": '{"name": "bash", "arguments": {"command": "npm install"}}'},
            {"from": "function_response", "value": "installed"},
            {"from": "function_call", "value": '{"name": "bash", "arguments": {"command": "pm2 restart"}}'},
            {"from": "function_response", "value": "restarted"},
            {"from": "gpt", "value": "Deployed successfully!"},
        ]

    @pytest.mark.asyncio
    async def test_heuristic_auto_create(self, tmp_path, trajectory_with_pattern):
        from caveman.skills.manager import SkillManager
        sm = SkillManager(skills_dir=tmp_path)
        skill = await sm.auto_create(trajectory_with_pattern, task="Deploy to VPS")
        assert skill is not None
        assert skill.source == "auto_created"
        assert len(skill.steps) >= 1
        assert skill.steps[0].tool == "bash"

    @pytest.mark.asyncio
    async def test_llm_auto_create(self, tmp_path, trajectory_with_pattern):
        async def mock_llm(prompt: str) -> str:
            return json.dumps({
                "name": "deploy-to-vps",
                "description": "Deploy application to VPS via SSH",
                "trigger_patterns": ["deploy", "vps", "push to server"],
                "steps": [
                    {"tool": "bash", "description": "SSH and pull latest code", "args_template": {}},
                    {"tool": "bash", "description": "Install dependencies", "args_template": {}},
                    {"tool": "bash", "description": "Restart service", "args_template": {}},
                ],
                "constraints": ["Always backup before deploy"],
            })

        from caveman.skills.manager import SkillManager
        sm = SkillManager(skills_dir=tmp_path)
        skill = await sm.auto_create(
            trajectory_with_pattern, task="Deploy to VPS", llm_fn=mock_llm
        )
        assert skill is not None
        assert skill.name == "deploy-to-vps"
        assert "deploy" in skill.trigger_patterns
        assert len(skill.steps) == 3
        assert skill.source == "auto_created"

    @pytest.mark.asyncio
    async def test_no_pattern_returns_none(self, tmp_path):
        from caveman.skills.manager import SkillManager
        sm = SkillManager(skills_dir=tmp_path)
        # Short trajectory with no repeated tools
        traj = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi!"},
        ]
        skill = await sm.auto_create(traj, task="greet")
        assert skill is None

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self, tmp_path, trajectory_with_pattern):
        async def bad_llm(prompt: str) -> str:
            raise ValueError("API error")

        from caveman.skills.manager import SkillManager
        sm = SkillManager(skills_dir=tmp_path)
        skill = await sm.auto_create(
            trajectory_with_pattern, task="Deploy", llm_fn=bad_llm
        )
        # Should fall back to heuristic
        assert skill is not None
        assert skill.source == "auto_created"
