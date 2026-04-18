"""Round 9 tests — LLM Scheduler, Lint Engine, Nudge Phase 2,
Verification Agent, Skill Harness (Guide+Sensor)."""
from __future__ import annotations

import asyncio
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock


# ═══════════════════════════════════════════════════════════════
# FR-208: LLM Scheduler
# ═══════════════════════════════════════════════════════════════

class TestLLMScheduler:
    """FR-208: Priority-based LLM request scheduling."""

    @pytest.fixture
    def mock_llm(self):
        async def llm(prompt: str) -> str:
            await asyncio.sleep(0.01)
            return f"response to: {prompt[:30]}"
        return llm

    @pytest.mark.asyncio
    async def test_basic_request(self, mock_llm):
        from caveman.engines.scheduler import LLMScheduler, Priority
        sched = LLMScheduler(mock_llm, max_rpm=100)
        await sched.start()
        try:
            result = await sched.request("shield", Priority.CRITICAL, "test prompt")
            assert "response to:" in result
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self, mock_llm):
        """P0 (Shield) should execute before P3 (Nudge)."""
        from caveman.engines.scheduler import LLMScheduler, Priority
        order = []
        async def tracking_llm(prompt: str) -> str:
            order.append(prompt)
            await asyncio.sleep(0.01)
            return "ok"

        sched = LLMScheduler(tracking_llm, max_rpm=100)
        # Don't start worker yet — queue requests first
        sched._running = True
        sched._seq = 0

        loop = asyncio.get_event_loop()
        # Queue low priority first, then critical
        f1 = loop.create_future()
        f2 = loop.create_future()
        from caveman.engines.scheduler import LLMRequest
        sched._queue.put_nowait(LLMRequest(
            priority=Priority.LOW, submitted_at=time.monotonic(),
            caller="nudge", prompt="low_prio", future=f1,
            request_id=1, _seq=1,
        ))
        sched._queue.put_nowait(LLMRequest(
            priority=Priority.CRITICAL, submitted_at=time.monotonic(),
            caller="shield", prompt="critical_prio", future=f2,
            request_id=2, _seq=2,
        ))

        # Now start worker
        sched._worker_task = asyncio.create_task(sched._worker())
        await asyncio.sleep(0.1)
        await sched.stop()

        # Critical should have been processed first
        assert len(order) >= 2
        assert order[0] == "critical_prio"
        assert order[1] == "low_prio"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_llm):
        from caveman.engines.scheduler import LLMScheduler, Priority
        sched = LLMScheduler(mock_llm, max_rpm=100)
        await sched.start()
        try:
            await sched.request("shield", Priority.CRITICAL, "p1")
            await sched.request("nudge", Priority.LOW, "p2")
            stats = sched.get_stats()
            assert stats["total_calls"] == 2
            assert "shield" in stats["callers"]
            assert "nudge" in stats["callers"]
            assert stats["callers"]["shield"]["calls"] == 1
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_queue_full_raises(self, mock_llm):
        from caveman.engines.scheduler import LLMScheduler, LLMRequest, Priority
        sched = LLMScheduler(mock_llm, max_rpm=100, max_queue_size=1)
        await sched.start()
        try:
            # Fill the queue (worker might not process fast enough)
            sched._running = False  # Pause worker
            await asyncio.sleep(0.05)
            sched._queue.put_nowait(
                LLMRequest(
                    priority=0, submitted_at=0, caller="x",
                    prompt="x", future=asyncio.get_event_loop().create_future(),
                    _seq=0,
                )
            )
            with pytest.raises(RuntimeError, match="queue full"):
                await sched.request("test", Priority.LOW, "overflow", timeout=0.1)
        finally:
            sched._running = False

    @pytest.mark.asyncio
    async def test_error_handling(self):
        async def failing_llm(prompt: str) -> str:
            raise ValueError("API error")

        from caveman.engines.scheduler import LLMScheduler, Priority
        sched = LLMScheduler(failing_llm, max_rpm=100)
        await sched.start()
        try:
            with pytest.raises(ValueError, match="API error"):
                await sched.request("test", Priority.NORMAL, "fail me")
            stats = sched.get_stats("test")
            assert stats["errors"] == 1
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_auto_start(self, mock_llm):
        """Scheduler auto-starts on first request."""
        from caveman.engines.scheduler import LLMScheduler, Priority
        sched = LLMScheduler(mock_llm, max_rpm=100)
        assert not sched._running
        result = await sched.request("test", Priority.NORMAL, "auto start")
        assert sched._running
        assert "response to:" in result
        await sched.stop()


# ═══════════════════════════════════════════════════════════════
# FR-106: Lint Engine
# ═══════════════════════════════════════════════════════════════

class TestLintEngine:
    """FR-106: Knowledge audit and garbage collection."""

    @pytest.fixture
    def memory_with_entries(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryEntry, MemoryType
        mm = MemoryManager(base_dir=tmp_path)

        # Add test memories
        now = datetime.now()
        old = now - timedelta(days=120)
        entries = [
            MemoryEntry(id="m1", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="m2", content="Server IP is 198.51.100.20",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="m3", content="Config at /tmp/nonexistent/config.yaml",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="m4", content="Config at /tmp/nonexistent/config.yaml",
                       memory_type=MemoryType.SEMANTIC, created_at=now),  # duplicate
            MemoryEntry(id="m5", content="Old fact from long ago",
                       memory_type=MemoryType.EPISODIC, created_at=old),
        ]
        for e in entries:
            mm._memories.setdefault(e.memory_type, []).append(e)
        return mm

    @pytest.mark.asyncio
    async def test_scan_finds_stale_paths(self, memory_with_entries):
        from caveman.engines.lint import LintEngine, IssueType
        lint = LintEngine(memory_with_entries, check_paths=True)
        report = await lint.scan()
        stale_path_issues = [
            i for i in report.issues if i.issue_type == IssueType.STALE_PATH
        ]
        assert len(stale_path_issues) >= 1

    @pytest.mark.asyncio
    async def test_scan_finds_duplicates(self, memory_with_entries):
        from caveman.engines.lint import LintEngine, IssueType
        lint = LintEngine(memory_with_entries, check_paths=False)
        report = await lint.scan()
        dup_issues = [
            i for i in report.issues if i.issue_type == IssueType.DUPLICATE
        ]
        assert len(dup_issues) >= 1

    @pytest.mark.asyncio
    async def test_scan_finds_aged(self, memory_with_entries):
        from caveman.engines.lint import LintEngine, IssueType
        lint = LintEngine(memory_with_entries, stale_days=90, check_paths=False)
        report = await lint.scan()
        aged_issues = [
            i for i in report.issues if i.issue_type == IssueType.AGED
        ]
        assert len(aged_issues) >= 1

    @pytest.mark.asyncio
    async def test_scan_finds_contradictions(self, memory_with_entries):
        from caveman.engines.lint import LintEngine, IssueType
        from caveman.memory.types import MemoryEntry, MemoryType
        # Add two memories with same IP but different claims (contradiction)
        now = datetime.now()
        memory_with_entries._memories[MemoryType.SEMANTIC].extend([
            MemoryEntry(id="c1", content="Main server 203.0.113.10 runs Ubuntu",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="c2", content="Main server 203.0.113.10 runs Windows",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ])
        lint = LintEngine(memory_with_entries, check_paths=False)
        report = await lint.scan()
        contradiction_issues = [
            i for i in report.issues if i.issue_type == IssueType.CONTRADICTION
        ]
        assert len(contradiction_issues) >= 1

    @pytest.mark.asyncio
    async def test_report_text(self, memory_with_entries):
        from caveman.engines.lint import LintEngine
        lint = LintEngine(memory_with_entries, check_paths=False)
        report = await lint.scan()
        text = report.to_text()
        assert "Lint Report" in text
        assert "scanned" in text

    @pytest.mark.asyncio
    async def test_empty_memory(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.engines.lint import LintEngine
        mm = MemoryManager(base_dir=tmp_path)
        lint = LintEngine(mm)
        report = await lint.scan()
        assert report.scanned == 0
        assert len(report.issues) == 0
        assert report.is_healthy


# ═══════════════════════════════════════════════════════════════
# FR-103: Nudge Phase 2 (Refiner)
# ═══════════════════════════════════════════════════════════════

class TestNudgeRefiner:
    """FR-103: LLM-powered memory refinement."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        return MemoryManager(base_dir=tmp_path)

    @pytest.fixture
    def sample_memories(self):
        from caveman.memory.types import MemoryEntry, MemoryType
        now = datetime.now()
        return [
            MemoryEntry(id="r1", content="pyenv is not pre-installed on macOS",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="r2", content="pyenv is not preinstalled on macos",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="r3", content="Use brew install pyenv on macOS",
                       memory_type=MemoryType.PROCEDURAL, created_at=now),
            MemoryEntry(id="r4", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ]

    @pytest.mark.asyncio
    async def test_heuristic_dedup(self, memory_manager, sample_memories):
        from caveman.memory.refiner import NudgeRefiner
        refiner = NudgeRefiner(memory_manager, llm_fn=None)
        result = await refiner.refine(sample_memories)
        assert result.input_count == 4
        assert result.removed_duplicates >= 1
        assert result.output_count < result.input_count

    @pytest.mark.asyncio
    async def test_llm_refinement(self, memory_manager, sample_memories):
        async def mock_llm(prompt: str) -> str:
            return '''{
                "refined": [
                    {"content": "pyenv is not pre-installed on macOS; install via brew", "type": "semantic", "action": "merge"},
                    {"content": "Use brew install pyenv on macOS", "type": "procedural", "action": "keep"},
                    {"content": "Server IP is 203.0.113.10", "type": "semantic", "action": "keep"}
                ],
                "conflicts": [],
                "removed_indices": [1]
            }'''

        from caveman.memory.refiner import NudgeRefiner
        refiner = NudgeRefiner(memory_manager, llm_fn=mock_llm)
        result = await refiner.refine(sample_memories)
        assert result.input_count == 4
        assert result.output_count == 3
        assert result.removed_duplicates == 1
        assert result.merged == 1
        assert result.llm_cost_est > 0
        assert result.llm_cost_est < 0.01  # FR-103: < $0.01

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, memory_manager, sample_memories):
        async def bad_llm(prompt: str) -> str:
            raise ValueError("API down")

        from caveman.memory.refiner import NudgeRefiner
        refiner = NudgeRefiner(memory_manager, llm_fn=bad_llm)
        result = await refiner.refine(sample_memories)
        # Should fall back to heuristic
        assert result.output_count > 0

    @pytest.mark.asyncio
    async def test_empty_input(self, memory_manager):
        from caveman.memory.refiner import NudgeRefiner
        refiner = NudgeRefiner(memory_manager)
        result = await refiner.refine([])
        assert result.input_count == 0
        assert result.output_count == 0


# ═══════════════════════════════════════════════════════════════
# Verification Agent
# ═══════════════════════════════════════════════════════════════

class TestVerificationAgent:
    """Coordinator Verification — anti-rationalization checks."""

    @pytest.mark.asyncio
    async def test_detect_excuses(self):
        from caveman.coordinator.verification import VerificationAgent, VerifyResult
        v = VerificationAgent()
        report = await v.verify(
            "Deploy the app",
            "I already checked and it looks correct. Will fix the tests later.",
        )
        excuse_checks = [c for c in report.checks if "excuse" in c["name"]]
        assert len(excuse_checks) >= 2  # "already checked" + "looks correct" + "fix later"

    @pytest.mark.asyncio
    async def test_detect_empty_output(self):
        from caveman.coordinator.verification import VerificationAgent, VerifyResult
        v = VerificationAgent()
        report = await v.verify("Build something", "")
        assert report.result == VerifyResult.FAIL

    @pytest.mark.asyncio
    async def test_detect_incomplete(self):
        from caveman.coordinator.verification import VerificationAgent
        v = VerificationAgent()
        report = await v.verify(
            "Implement auth",
            "Created auth module. TODO: add rate limiting. FIXME: password hashing.",
        )
        incomplete = [c for c in report.checks if c["name"] == "incomplete_signals"]
        assert len(incomplete) == 1
        assert incomplete[0]["result"] == "warn"

    @pytest.mark.asyncio
    async def test_pass_clean_output(self):
        from caveman.coordinator.verification import VerificationAgent, VerifyResult
        v = VerificationAgent()
        report = await v.verify(
            "Create hello world",
            "Created app.py with Flask hello world endpoint. Server runs on port 5000.",
        )
        assert report.result == VerifyResult.PASS

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        from caveman.coordinator.verification import VerificationAgent, VerifyResult
        v = VerificationAgent(strict=True)
        report = await v.verify(
            "Deploy",
            "I already checked the deployment. It looks fine.",
        )
        # Strict mode: WARN → FAIL
        assert report.result == VerifyResult.FAIL

    @pytest.mark.asyncio
    async def test_coordinator_integration(self):
        """Coordinator uses VerificationAgent to reject bad output."""
        from caveman.coordinator.engine import Coordinator, TaskStatus
        from caveman.coordinator.verification import VerificationAgent

        verifier = VerificationAgent(strict=True)

        async def bad_agent(task: str, ctx: dict):
            return "I already checked and it looks correct. Will fix later."

        coord = Coordinator(verifier=verifier, verify_tasks=True)
        coord.register_agent("bad", bad_agent)
        plan = coord.plan("test", [{"id": "t1", "description": "do thing", "agent": "bad"}])
        results = await coord.execute(plan)
        assert results["tasks"]["t1"]["status"] == "failed"
        assert "Verification failed" in results["tasks"]["t1"]["error"]


# ═══════════════════════════════════════════════════════════════
# Skill Harness (Guide + Sensor)
# ═══════════════════════════════════════════════════════════════

class TestSkillHarness:
    """Skill Harness — Guide + Sensor abstraction."""

    def test_guide_to_prompt(self):
        from caveman.skills.harness import GuideConfig
        guide = GuideConfig(
            system_prompt_addition="You are a deployment expert",
            output_format="json",
            constraints=["No destructive operations", "Log all changes"],
            disallowed_tools=["bash_dangerous"],
        )
        prompt = guide.to_prompt()
        assert "deployment expert" in prompt
        assert "json" in prompt
        assert "No destructive" in prompt
        assert "bash_dangerous" in prompt

    @pytest.mark.asyncio
    async def test_sensor_non_empty(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        result = await sensor.evaluate("some output")
        assert result.passed
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_sensor_empty_fails(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        result = await sensor.evaluate("")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_sensor_json_format(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        result = await sensor.evaluate('{"key": "value"}', expected_format="json")
        json_checks = [c for c in result.checks if "json" in c.name]
        assert len(json_checks) == 1
        assert json_checks[0].passed

    @pytest.mark.asyncio
    async def test_sensor_invalid_json(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        result = await sensor.evaluate("not json {", expected_format="json")
        json_checks = [c for c in result.checks if "json" in c.name]
        assert len(json_checks) == 1
        assert not json_checks[0].passed

    @pytest.mark.asyncio
    async def test_sensor_quality_gate(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        gates = [{"name": "has_flask", "check": "output_contains", "expected": "Flask"}]
        result = await sensor.evaluate("Created Flask app", quality_gates=gates)
        gate_checks = [c for c in result.checks if c.name == "has_flask"]
        assert len(gate_checks) == 1
        assert gate_checks[0].passed

    @pytest.mark.asyncio
    async def test_harness_full_flow(self):
        from caveman.skills.harness import SkillHarness, GuideConfig, Sensor
        harness = SkillHarness(
            guide=GuideConfig(output_format="code"),
            sensor=Sensor(),
            quality_gates=[
                {"name": "has_def", "check": "output_contains", "expected": "def "},
            ],
        )
        # Guide produces prompt
        prompt = harness.guide.to_prompt()
        assert "code" in prompt

        # Sensor evaluates output
        result = await harness.evaluate("def hello():\n    return 'world'")
        assert result.passed
        assert result.overall_score > 0.5

    @pytest.mark.asyncio
    async def test_sensor_error_detection(self):
        from caveman.skills.harness import Sensor
        sensor = Sensor()
        result = await sensor.evaluate("Error: connection refused\nTraceback:")
        error_checks = [c for c in result.checks if c.name == "no_errors"]
        assert len(error_checks) == 1
        assert not error_checks[0].passed
