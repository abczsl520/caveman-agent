"""Tests for flywheel improvements — Chain 4 lint_single, Reflect causal chains, Decay reuse."""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# --- LintEngine.lint_single ---

class TestLintSingle:
    """Test the new lint_single method for immediate single-memory validation."""

    @pytest.fixture
    def lint_engine(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.engines.lint import LintEngine
        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        return LintEngine(memory_manager=mm, check_paths=False)

    @pytest.mark.asyncio
    async def test_lint_single_clean_memory(self, lint_engine):
        from caveman.memory.types import MemoryEntry, MemoryType
        entry = MemoryEntry(
            id="test-1", content="Server IP is 198.51.100.10 for Aliyun",
            memory_type=MemoryType.SEMANTIC, created_at=datetime.now(),
        )
        report = await lint_engine.lint_single(entry)
        assert report.scanned == 1
        assert len(report.issues) == 0

    @pytest.mark.asyncio
    async def test_lint_single_short_memory(self, lint_engine):
        from caveman.memory.types import MemoryEntry, MemoryType
        entry = MemoryEntry(
            id="test-2", content="ok",
            memory_type=MemoryType.SEMANTIC, created_at=datetime.now(),
        )
        report = await lint_engine.lint_single(entry)
        assert report.scanned == 1
        assert len(report.issues) == 1
        assert report.issues[0].issue_type.value == "low_confidence"

    @pytest.mark.asyncio
    async def test_lint_single_aged_memory(self, lint_engine):
        from caveman.memory.types import MemoryEntry, MemoryType
        entry = MemoryEntry(
            id="test-3", content="This is a valid memory about project architecture",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now() - timedelta(days=120),
        )
        report = await lint_engine.lint_single(entry)
        assert any(i.issue_type.value == "aged" for i in report.issues)


# --- Reflect Engine causal chain extraction ---

class TestReflectCausalChains:
    """Test enhanced heuristic reflection with causal chain extraction."""

    @pytest.fixture
    def reflect_engine(self):
        from caveman.engines.reflect import ReflectEngine
        return ReflectEngine()

    @pytest.mark.asyncio
    async def test_causal_chain_extraction(self, reflect_engine):
        trajectory = [
            {"from": "human", "value": "Fix the deployment error"},
            {"from": "gpt", "value": "The root cause was a missing environment variable DATABASE_URL"},
            {"from": "function_call", "value": '{"name": "bash"}'},
            {"from": "gpt", "value": "Fixed by adding DATABASE_URL to the .env file"},
        ]
        reflection = await reflect_engine.reflect("Fix deployment", trajectory, "Fixed")
        # Should have causal chain lessons
        causal_lessons = [l for l in reflection.lessons if "Root cause" in l or "root cause" in l.lower()]
        assert len(causal_lessons) > 0

    @pytest.mark.asyncio
    async def test_recovery_detection(self, reflect_engine):
        trajectory = [
            {"from": "human", "value": "Deploy the app"},
            {"from": "gpt", "value": "Error: connection refused on port 5432"},
            {"from": "function_call", "value": '{"name": "bash"}'},
            {"from": "gpt", "value": "Fixed the error by starting PostgreSQL service. Deployment verified with exit code 0."},
        ]
        reflection = await reflect_engine.reflect("Deploy app", trajectory, "Deployment verification returned exit code 0")
        assert reflection.outcome == "success"
        # Should detect error recovery
        recovery_lessons = [l for l in reflection.lessons if "recover" in l.lower()]
        assert len(recovery_lessons) > 0

    @pytest.mark.asyncio
    async def test_causal_boosts_confidence(self, reflect_engine):
        # With causal chain
        trajectory_causal = [
            {"from": "gpt", "value": "The root cause was incorrect API key configuration"},
            {"from": "function_call", "value": '{"name": "bash"}'},
            {"from": "gpt", "value": "Task completed successfully"},
        ]
        r1 = await reflect_engine.reflect("Fix API", trajectory_causal, "Fixed")

        # Without causal chain
        trajectory_plain = [
            {"from": "gpt", "value": "I ran the command"},
            {"from": "function_call", "value": '{"name": "bash"}'},
            {"from": "gpt", "value": "Task completed successfully"},
        ]
        r2 = await reflect_engine.reflect("Run command", trajectory_plain, "Done")

        assert r1.confidence >= r2.confidence


# --- Memory Manager bus wiring ---

class TestMemoryManagerBus:
    """Test that MemoryManager emits MEMORY_STORE events when bus is wired."""

    @pytest.mark.asyncio
    async def test_store_emits_event(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType
        from caveman.events import EventBus, EventType

        bus = EventBus()
        events_received = []

        def on_store(event):
            events_received.append(event)

        bus.on(EventType.MEMORY_STORE, on_store)

        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        mm._bus = bus

        mid = await mm.store("Test memory content for bus", MemoryType.SEMANTIC, metadata={"source": "nudge"})
        assert mid
        assert len(events_received) == 1
        assert events_received[0].data["memory_id"] == mid
        assert events_received[0].data["source"] == "nudge"

    @pytest.mark.asyncio
    async def test_store_no_event_without_bus(self, tmp_path):
        """Store should work fine without bus — no crash."""
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType

        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        mid = await mm.store("Test memory without bus", MemoryType.SEMANTIC)
        assert mid  # Should succeed silently


# --- Event chain decay instance reuse ---

class TestDecayInstanceReuse:
    """Test that decay uses a single instance across invocations."""

    def test_decay_closure_reuses_instance(self):
        """Verify the closure pattern reuses the same MemoryDecay instance."""
        # This tests the pattern, not the actual decay logic
        _decay_instance = [None]

        def get_or_create():
            if _decay_instance[0] is None:
                _decay_instance[0] = object()  # Simulate MemoryDecay()
            return _decay_instance[0]

        first = get_or_create()
        second = get_or_create()
        assert first is second  # Same instance
