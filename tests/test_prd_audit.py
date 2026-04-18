"""Tests for PRD deep audit fixes — event chain, memory types, file limits."""
from __future__ import annotations

import asyncio
import pytest

from caveman.events import EventBus, EventType
from caveman.engines.event_chain import wire_inner_flywheel, unwire_inner_flywheel
from caveman.engines.manager import EngineSet
from caveman.memory.types import MemoryType, MemorySource, SOURCE_TO_TYPE


class TestEventChain:
    """PRD §5.3: Inner flywheel event chain must be wired."""

    def test_shield_update_triggers_nudge(self):
        """SHIELD_UPDATE event should trigger Nudge extraction."""
        bus = EventBus()
        nudge_called = []

        class MockNudge:
            def should_nudge(self, turn_count):
                return True

            async def run(self, turns, task=""):
                nudge_called.append(task)
                return []

        engines = EngineSet(nudge=MockNudge())
        turns_data = [{"role": "user", "content": "test"}]
        handlers = wire_inner_flywheel(
            bus, engines,
            get_turns=lambda: turns_data,
            get_task=lambda: "test task",
        )

        # Emit SHIELD_UPDATE
        asyncio.run(bus.emit(EventType.SHIELD_UPDATE, {
            "session_id": "test", "turn_count": 5,
        }, source="shield"))

        assert len(nudge_called) == 1
        assert nudge_called[0] == "test task"

        # Cleanup
        unwire_inner_flywheel(bus, handlers)

    def test_shield_update_respects_should_nudge(self):
        """SHIELD_UPDATE should NOT trigger Nudge if should_nudge returns False."""
        bus = EventBus()
        nudge_called = []

        class MockNudge:
            def should_nudge(self, turn_count):
                return False  # Not time yet

            async def run(self, turns, task=""):
                nudge_called.append(task)
                return []

        engines = EngineSet(nudge=MockNudge())
        handlers = wire_inner_flywheel(
            bus, engines,
            get_turns=lambda: [{"role": "user", "content": "test"}],
            get_task=lambda: "test",
        )

        asyncio.run(bus.emit(EventType.SHIELD_UPDATE, {
            "session_id": "test", "turn_count": 1,
        }, source="shield"))

        assert len(nudge_called) == 0
        unwire_inner_flywheel(bus, handlers)

    def test_tool_error_triggers_nudge(self):
        """TOOL_ERROR should trigger immediate Nudge extraction."""
        bus = EventBus()
        nudge_called = []

        class MockNudge:
            def should_nudge(self, turn_count):
                return True

            async def run(self, turns, task=""):
                nudge_called.append("error_nudge")
                return []

        engines = EngineSet(nudge=MockNudge())
        handlers = wire_inner_flywheel(
            bus, engines,
            get_turns=lambda: [{"role": "user", "content": "test"}],
            get_task=lambda: "fix bug",
        )

        asyncio.run(bus.emit(EventType.TOOL_ERROR, {
            "error": "command failed",
        }, source="tool"))

        assert len(nudge_called) == 1
        unwire_inner_flywheel(bus, handlers)

    def test_unwire_removes_handlers(self):
        """unwire_inner_flywheel should remove all handlers."""
        bus = EventBus()
        nudge_called = []

        class MockNudge:
            def should_nudge(self, turn_count):
                return True

            async def run(self, turns, task=""):
                nudge_called.append(1)
                return []

        engines = EngineSet(nudge=MockNudge())
        handlers = wire_inner_flywheel(
            bus, engines,
            get_turns=lambda: [{"role": "user", "content": "test"}],
            get_task=lambda: "test",
        )

        unwire_inner_flywheel(bus, handlers)

        # After unwire, events should not trigger nudge
        asyncio.run(bus.emit(EventType.SHIELD_UPDATE, {
            "session_id": "test", "turn_count": 5,
        }, source="shield"))

        assert len(nudge_called) == 0


class TestMemoryTypes:
    """PRD §5.2: Memory types should include user/feedback/project/reference."""

    def test_memory_source_enum(self):
        """MemorySource should have all 4 PRD types."""
        assert MemorySource.USER.value == "user"
        assert MemorySource.FEEDBACK.value == "feedback"
        assert MemorySource.PROJECT.value == "project"
        assert MemorySource.REFERENCE.value == "reference"

    def test_source_to_type_mapping(self):
        """PRD source types should map to cognitive types."""
        assert SOURCE_TO_TYPE[MemorySource.USER] == MemoryType.SEMANTIC
        assert SOURCE_TO_TYPE[MemorySource.FEEDBACK] == MemoryType.EPISODIC
        assert SOURCE_TO_TYPE[MemorySource.PROJECT] == MemoryType.SEMANTIC
        assert SOURCE_TO_TYPE[MemorySource.REFERENCE] == MemoryType.PROCEDURAL

    def test_nudge_type_map_includes_prd_types(self):
        """TYPE_MAP should accept both PRD and cognitive types."""
        from caveman.memory.types import TYPE_MAP

        # PRD types
        assert "user" in TYPE_MAP
        assert "feedback" in TYPE_MAP
        assert "project" in TYPE_MAP
        assert "reference" in TYPE_MAP

        # Cognitive types (backward compat)
        assert "episodic" in TYPE_MAP
        assert "semantic" in TYPE_MAP
        assert "procedural" in TYPE_MAP
        assert "working" in TYPE_MAP

    def test_nudge_heuristic_uses_prd_types(self):
        """Heuristic extraction should use PRD source types."""
        from caveman.memory.nudge import MemoryNudge

        class MockMM:
            async def store(self, *a, **kw):
                return "id"
            def recent(self, limit=20):
                return []
            async def update_metadata(self, *a, **kw):
                return True

        nudge = MemoryNudge(MockMM())
        turns = [
            {"role": "user", "content": "fix the login bug"},
            {"role": "assistant", "content": "I found the issue was in /src/auth/login.py. Fixed by adding null check."},
        ]
        candidates = nudge._extract_heuristic(turns, "fix login bug")

        # Should use PRD types, not cognitive types
        types_used = {c["type"] for c in candidates}
        prd_types = {"user", "feedback", "project", "reference"}
        cognitive_types = {"episodic", "semantic", "procedural", "working"}

        # At least one candidate should use a PRD type
        assert types_used & prd_types, f"Expected PRD types, got {types_used}"


class TestFileLimits:
    """NFR-502: No file should exceed 400 lines."""

    def test_core_files_under_400_lines(self):
        from pathlib import Path
        import os

        core_files = [
            "caveman/agent/loop.py",
            "caveman/memory/sqlite_store.py",
            "caveman/memory/manager.py",
            "caveman/memory/nudge.py",
            "caveman/engines/manager.py",
            "caveman/engines/reflect.py",
            "caveman/engines/shield.py",
            "caveman/engines/lint.py",
            "caveman/engines/ripple.py",
        ]

        root = Path(__file__).parent.parent
        violations = []
        for f in core_files:
            path = root / f
            if path.exists():
                lines = len(path.read_text().splitlines())
                if lines > 400:
                    violations.append(f"{f}: {lines} lines")

        assert not violations, f"Files over 400 lines: {violations}"
