"""PRD #25 acceptance: 10-turn Shield→Nudge→Ripple→Lint→Recall flywheel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from caveman.engines.event_chain import wire_inner_flywheel
from caveman.engines.recall import RecallEngine
from caveman.events import EventBus, EventType
from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryEntry, MemoryType


@dataclass
class Engines:
    nudge: object
    ripple: object
    lint: object
    recall: object
    shield: object | None = None
    outcome: object | None = None
    reflect: object | None = None


class AcceptanceNudge:
    """Deterministic nudge that extracts one durable memory from current turns."""

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.calls = 0

    async def run(self, turns, task: str = ""):
        self.calls += 1
        content = f"Caveman PRD acceptance milestone {self.calls}: Shield Nudge Ripple Lint Recall chain processed turn_count={len(turns)}."
        mid = await self.memory.store(
            content,
            MemoryType.SEMANTIC,
            metadata={"source": "nudge", "trust_score": 0.7},
        )
        if not mid:
            return []
        return [MemoryEntry(mid, content, MemoryType.SEMANTIC, datetime.now(), {"source": "nudge"})]


class AcceptanceRipple:
    def __init__(self) -> None:
        self.entries: list[str] = []

    async def propagate(self, entry: MemoryEntry):
        self.entries.append(entry.id)
        return object()


class AcceptanceLint:
    def __init__(self) -> None:
        self.entries: list[str] = []

    async def lint_single(self, entry: MemoryEntry):
        self.entries.append(entry.id)

        class Report:
            issues: list = []

        return Report()


@pytest.mark.asyncio
async def test_10_turn_inner_flywheel_acceptance(tmp_path):
    bus = EventBus()
    observed: list[str] = []
    bus.on_all(lambda event: observed.append(str(event.type)))

    memory = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
    memory._bus = bus
    ripple = AcceptanceRipple()
    memory.set_ripple(ripple)
    nudge = AcceptanceNudge(memory)
    lint = AcceptanceLint()
    recall = RecallEngine(sessions_dir=tmp_path / "sessions", memory_manager=memory, max_memories=5)

    turns: list[dict[str, str]] = []
    engines = Engines(nudge=nudge, ripple=ripple, lint=lint, recall=recall)
    wire_inner_flywheel(
        bus,
        engines,
        get_turns=lambda: turns,
        get_task=lambda: "verify Shield Nudge Ripple Lint Recall acceptance milestone",
        memory_manager=memory,
    )

    # Ten conversation turns. Shield emits on every turn; event-chain throttle should
    # trigger Nudge at turns 3/6/9, and each Nudge store then ripples and lints.
    for i in range(10):
        turns.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}: continue acceptance milestone"})
        await bus.emit(EventType.SHIELD_UPDATE, {"turn_count": i + 1}, source="shield")

    restored = await recall.restore_structured("acceptance milestone Shield Nudge Ripple Lint Recall")

    assert nudge.calls == 3
    assert memory.total_count == 3
    assert len(ripple.entries) == 3
    assert len(lint.entries) == 3
    assert restored.memories_loaded >= 1
    assert "acceptance milestone" in restored.memory_text
    assert observed.count(EventType.SHIELD_UPDATE.value) == 10
    assert observed.count(EventType.NUDGE_EXTRACT.value) == 3
    assert observed.count(EventType.MEMORY_STORE.value) == 3
