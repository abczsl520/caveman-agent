"""Tests for engines/event_chain.py — inner flywheel wiring."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from caveman.events import EventBus, EventType
from caveman.engines.event_chain import wire_inner_flywheel, unwire_inner_flywheel


class MockEngineSet:
    def __init__(self, nudge=None, ripple=None, lint=None, recall=None, outcome=None, reflect=None):
        self.nudge = nudge
        self.ripple = ripple
        self.lint = lint
        self.recall = recall
        self.shield = None
        self.outcome = outcome
        self.reflect = reflect


class TestWireInnerFlywheel:
    def test_no_engines_returns_infra_only(self):
        """Wiki auto-trigger and memory decay register even with no engines."""
        bus = EventBus()
        engines = MockEngineSet()
        handlers = wire_inner_flywheel(bus, engines)
        event_types = [et for et, _ in handlers]
        # Only infrastructure handlers (wiki, decay) — no engine-specific ones
        for et in event_types:
            assert et in (EventType.NUDGE_EXTRACT, EventType.LOOP_END)

    def test_nudge_wires_shield_update(self):
        bus = EventBus()
        nudge = MagicMock()
        nudge.run = AsyncMock(return_value=[])
        engines = MockEngineSet(nudge=nudge)
        handlers = wire_inner_flywheel(bus, engines)
        # Should have SHIELD_UPDATE and TOOL_ERROR handlers
        event_types = [h[0] for h in handlers]
        assert EventType.SHIELD_UPDATE in event_types
        assert EventType.TOOL_ERROR in event_types

    def test_ripple_wires_nudge_extract(self):
        bus = EventBus()
        nudge = MagicMock()
        nudge.run = AsyncMock(return_value=[])
        ripple = MagicMock()
        engines = MockEngineSet(nudge=nudge, ripple=ripple)
        handlers = wire_inner_flywheel(bus, engines)
        event_types = [h[0] for h in handlers]
        assert EventType.NUDGE_EXTRACT in event_types

    def test_lint_wires_memory_store(self):
        bus = EventBus()
        lint = MagicMock()
        engines = MockEngineSet(lint=lint)
        handlers = wire_inner_flywheel(bus, engines)
        event_types = [h[0] for h in handlers]
        assert EventType.MEMORY_STORE in event_types

    @pytest.mark.asyncio
    async def test_shield_update_triggers_nudge(self):
        bus = EventBus()
        memory_entry = MagicMock()
        memory_entry.memory_type = MagicMock()
        memory_entry.memory_type.value = "semantic"
        nudge = MagicMock()
        nudge.run = AsyncMock(return_value=[memory_entry])
        engines = MockEngineSet(nudge=nudge)
        turns = [{"role": "user", "content": "test"}]
        wire_inner_flywheel(bus, engines, get_turns=lambda: turns, get_task=lambda: "test")
        # Emit with turn_count >= 3 to pass throttle
        await bus.emit(EventType.SHIELD_UPDATE, {"turn_count": 5}, source="shield")
        nudge.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_shield_update_throttled(self):
        bus = EventBus()
        nudge = MagicMock()
        nudge.run = AsyncMock(return_value=[])
        engines = MockEngineSet(nudge=nudge)
        wire_inner_flywheel(bus, engines, get_turns=lambda: [{"role": "user", "content": "x"}])
        # turn_count 1 — should be throttled (< 3 gap from 0)
        await bus.emit(EventType.SHIELD_UPDATE, {"turn_count": 1}, source="shield")
        nudge.run.assert_not_called()


class TestUnwire:
    @pytest.mark.asyncio
    async def test_unwire_removes_handlers(self):
        bus = EventBus()
        nudge = MagicMock()
        nudge.run = AsyncMock(return_value=[])
        engines = MockEngineSet(nudge=nudge)
        handlers = wire_inner_flywheel(bus, engines)
        assert len(handlers) > 0
        unwire_inner_flywheel(bus, handlers)
        # After unwire, emitting should not trigger nudge
        await bus.emit(EventType.SHIELD_UPDATE, {"turn_count": 5}, source="shield")
        nudge.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_end_decay_logs_quarantine_counts(self):
        """Decay integration should surface quarantines, not only decays/prunes."""
        bus = EventBus()
        engines = MockEngineSet()
        decay_result = MagicMock(
            memories_decayed=0,
            memories_pruned=0,
            memories_quarantined=3,
        )

        with patch("caveman.memory.decay.MemoryDecay") as decay_cls, patch(
            "caveman.engines.event_chain.logger"
        ) as logger:
            decay_cls.return_value.run.return_value = decay_result
            wire_inner_flywheel(bus, engines)

            for _ in range(10):
                await bus.emit(EventType.LOOP_END, {"task": "t", "result": "ok"}, source="test")

        decay_cls.return_value.run.assert_called_once()
        logger.info.assert_any_call(
            "Memory decay: %d decayed, %d pruned, %d quarantined",
            0,
            0,
            3,
        )
