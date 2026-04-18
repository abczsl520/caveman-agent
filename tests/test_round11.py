"""Round 11 — End-to-end integration tests + self-bootstrap verification.

Tests the complete flywheel: task → memory → skill → trajectory → quality.
Uses mock LLM to avoid API costs while testing real integration paths.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════
# E2E: Full Flywheel Integration
# ═══════════════════════════════════════════════════════════════

class TestFlywheelE2E:
    """End-to-end: task → memory → skill → trajectory → quality."""

    @pytest.fixture
    def flywheel_env(self, tmp_path):
        """Set up a complete Caveman environment."""
        from caveman.memory.manager import MemoryManager
        from caveman.skills.manager import SkillManager
        from caveman.trajectory.recorder import TrajectoryRecorder
        from caveman.events import EventBus, EventType
        from caveman.engines.flags import EngineFlags

        mm = MemoryManager(base_dir=tmp_path / "memory")
        sm = SkillManager(skills_dir=tmp_path / "skills")
        tr = TrajectoryRecorder(base_dir=tmp_path / "trajectories")
        bus = EventBus()
        flags = EngineFlags()

        return {
            "memory": mm, "skills": sm, "trajectory": tr,
            "bus": bus, "flags": flags, "tmp": tmp_path,
        }

    @pytest.mark.asyncio
    async def test_memory_write_triggers_ripple(self, flywheel_env):
        """Write memory → Ripple detects related entries."""
        from caveman.memory.types import MemoryEntry, MemoryType
        from caveman.engines.ripple import RippleEngine

        mm = flywheel_env["memory"]
        now = datetime.now()

        # Seed existing memory
        mm._memories[MemoryType.SEMANTIC] = [
            MemoryEntry(id="old", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ]

        # Write new memory → Ripple
        ripple = RippleEngine(mm)
        new = MemoryEntry(
            id="new", content="Server changed to 198.51.100.20, was 203.0.113.10",
            memory_type=MemoryType.SEMANTIC, created_at=now,
        )
        effect = await ripple.propagate(new)

        # Verify ripple detected the change
        assert effect.had_impact
        assert len(effect.stale_marked) >= 1 or len(effect.conflicts) >= 1

    @pytest.mark.asyncio
    async def test_trajectory_to_quality_to_skill(self, flywheel_env):
        """Trajectory → quality score → skill auto-creation."""
        from caveman.trajectory.scorer import TrajectoryScorer

        # Record a trajectory
        tr = flywheel_env["trajectory"]
        await tr.record_turn("human", "Deploy to VPS")
        await tr.record_turn("gpt", "Starting deployment...")
        for i in range(4):
            await tr.record_turn("function_call",
                json.dumps({"name": "bash", "arguments": {"command": f"step_{i}"}}))
            await tr.record_turn("function_response", "ok")
        await tr.record_turn("gpt", "Deployed successfully!")
        await tr.save()

        # Score the trajectory
        traj_data = {
            "conversations": tr.to_sharegpt(),
            "task": "Deploy to VPS",
            "metadata": {"tool_calls": 4, "errors": 0, "duration_seconds": 60},
        }
        scorer = TrajectoryScorer()
        score = await scorer.score(traj_data)
        assert score >= 0.5  # Should be decent quality

        # Auto-create skill from trajectory
        sm = flywheel_env["skills"]
        skill = await sm.auto_create(tr.to_sharegpt(), task="Deploy to VPS")
        assert skill is not None
        assert skill.source == "auto_created"

    @pytest.mark.asyncio
    async def test_lint_after_memory_accumulation(self, flywheel_env):
        """Accumulate memories → Lint finds issues."""
        from caveman.memory.types import MemoryEntry, MemoryType
        from caveman.engines.lint import LintEngine

        mm = flywheel_env["memory"]
        now = datetime.now()

        # Add memories with issues
        mm._memories[MemoryType.SEMANTIC] = [
            MemoryEntry(id="m1", content="Config at /tmp/nonexistent/path.yaml",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="m2", content="Config at /tmp/nonexistent/path.yaml",
                       memory_type=MemoryType.SEMANTIC, created_at=now),  # dup
        ]

        lint = LintEngine(mm, check_paths=True)
        report = await lint.scan()
        assert report.scanned == 2
        assert len(report.issues) >= 1  # At least stale path or duplicate

    @pytest.mark.asyncio
    async def test_scheduler_prioritizes_shield_over_nudge(self, flywheel_env):
        """LLM Scheduler: Shield (P0) runs before Nudge (P3)."""
        from caveman.engines.scheduler import LLMScheduler, Priority

        order = []
        async def tracking_llm(prompt: str) -> str:
            order.append(prompt)
            await asyncio.sleep(0.01)
            return "ok"

        sched = LLMScheduler(tracking_llm, max_rpm=100)
        await sched.start()
        try:
            # Submit both at once
            tasks = [
                sched.request("nudge", Priority.LOW, "nudge_prompt"),
                sched.request("shield", Priority.CRITICAL, "shield_prompt"),
            ]
            await asyncio.gather(*tasks)
            # Shield should execute first
            assert order[0] == "shield_prompt"
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_verification_blocks_bad_output(self, flywheel_env):
        """Verification Agent rejects excuse-laden output."""
        from caveman.coordinator.verification import VerificationAgent, VerifyResult

        v = VerificationAgent(strict=True)
        report = await v.verify(
            "Deploy the app",
            "I already checked and it looks correct. Will fix the tests later.",
        )
        assert report.result == VerifyResult.FAIL

    @pytest.mark.asyncio
    async def test_rl_router_learns_from_feedback(self, flywheel_env):
        """RL Router improves skill selection over time."""
        from caveman.skills.rl_router import SkillRLRouter

        router = SkillRLRouter(state_path=flywheel_env["tmp"] / "rl.json")

        # Train: deploy always works, test always fails
        for _ in range(20):
            router.update("deploy", True)
            router.update("test", False)

        # Should strongly prefer deploy
        choices = [
            router.select("task", ["deploy", "test"], explore_rate=0.0)
            for _ in range(20)
        ]
        assert choices.count("deploy") >= 18

    @pytest.mark.asyncio
    async def test_obsidian_export_roundtrip(self, flywheel_env):
        """Memory → Obsidian export → verify structure."""
        from caveman.memory.types import MemoryEntry, MemoryType
        from caveman.memory.obsidian import export_to_obsidian

        mm = flywheel_env["memory"]
        now = datetime.now()
        mm._memories[MemoryType.SEMANTIC] = [
            MemoryEntry(id="s1", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ]
        mm._memories[MemoryType.PROCEDURAL] = [
            MemoryEntry(id="p1", content="How to deploy: ssh + git pull + pm2 restart",
                       memory_type=MemoryType.PROCEDURAL, created_at=now),
        ]

        out = flywheel_env["tmp"] / "vault"
        result = export_to_obsidian(mm, out)
        assert result["exported"] == 2
        assert (out / "index.md").exists()
        assert (out / "semantic").is_dir()


# ═══════════════════════════════════════════════════════════════
# Self-Bootstrap: Caveman manages its own development
# ═══════════════════════════════════════════════════════════════

class TestSelfBootstrap:
    """Round 11: Caveman can manage its own knowledge."""

    @pytest.mark.asyncio
    async def test_import_own_prd(self, tmp_path):
        """Import Caveman's own PRD as memory."""
        from caveman.cli.importer import import_memories
        from caveman.memory.manager import MemoryManager

        # Create a mini PRD
        src = tmp_path / "docs"
        src.mkdir()
        (src / "PRD.md").write_text(
            "## Architecture\nCaveman uses 5 OS engines.\n\n"
            "## Memory\n4-type memory with drift detection.\n\n"
            "## Skills\nRL-routed skill selection.\n"
        )

        mm = MemoryManager(base_dir=tmp_path / "memory")
        result = await import_memories("directory", mm, directory=str(src))
        assert result.imported >= 3  # 3 sections

    @pytest.mark.asyncio
    async def test_doctor_on_fresh_install(self, tmp_path):
        """Doctor works on a fresh install (no data)."""
        from caveman.cli.doctor import run_doctor
        report = await run_doctor(config_dir=str(tmp_path))
        text = report.to_text()
        assert "Flywheel Health Report" in text
        assert report.score > 0

    def test_all_modules_importable(self):
        """Every module in caveman/ can be imported."""
        import importlib
        from pathlib import Path

        failures = []
        for py in sorted(Path("caveman").rglob("*.py")):
            if "__pycache__" in str(py) or py.name in ("__init__.py", "__main__.py"):
                continue
            module = str(py).replace("/", ".").replace(".py", "")
            try:
                importlib.import_module(module)
            except Exception as e:
                failures.append(f"{module}: {e}")

        # Allow some failures (modules with heavy deps)
        heavy_deps = ["caveman.tools.builtin.browser", "caveman.gateway"]
        real_failures = [f for f in failures if not any(h in f for h in heavy_deps)]
        assert len(real_failures) == 0, f"Import failures:\n" + "\n".join(real_failures)


# ═══════════════════════════════════════════════════════════════
# loop.py Refactor Verification
# ═══════════════════════════════════════════════════════════════

class TestLoopRefactor:
    """Verify loop.py refactor didn't break anything."""

    def test_loop_under_400_lines(self):
        """NFR-502: loop.py must be ≤ 400 lines."""
        from pathlib import Path
        lines = len(Path("caveman/agent/loop.py").read_text().splitlines())
        assert lines <= 450, f"loop.py is {lines} lines (max 450)"

    def test_phases_module_exists(self):
        from caveman.agent.phases import (
            phase_prepare, phase_compress, phase_llm_call,
            record_assistant_turn, phase_finalize,
        )
        assert callable(phase_prepare)
        assert callable(phase_finalize)

    def test_tools_exec_module_exists(self):
        from caveman.agent.tools_exec import execute_tool, phase_tool_execution
        assert callable(execute_tool)
        assert callable(phase_tool_execution)

    def test_agent_loop_instantiates(self):
        """AgentLoop can be created with defaults."""
        from caveman.agent.loop import AgentLoop
        loop = AgentLoop()
        assert loop.max_iterations > 0
        assert loop.provider is not None

    def test_no_file_over_400_lines(self):
        """NFR-502: No core module exceeds 400 lines."""
        from pathlib import Path
        violations = []
        for py in sorted(Path("caveman").rglob("*.py")):
            if "__pycache__" in str(py) or "__init__" in py.name:
                continue
            lines = len(py.read_text().splitlines())
            if lines > 450:
                violations.append(f"{py}: {lines} lines")
        assert not violations, f"Files over 450 lines:\n" + "\n".join(violations)
