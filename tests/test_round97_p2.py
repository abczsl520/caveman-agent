"""Tests for Round 97 P2 fixes — metrics p95, prompt CJK tokens, engine injection."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ── P2 #2: MetricsCollector p95 off-by-one fix ──


class TestMetricsP95:
    def test_p95_single_value(self):
        """p95 with 1 value should not IndexError."""
        from caveman.events import MetricsCollector, Event, EventType
        mc = MetricsCollector()
        # Simulate a tool call + result pair
        mc.handle(Event(type=EventType.TOOL_CALL.value, data={"call_id": "a"}))
        import time
        mc._start_times["tool:a"] = time.time() - 0.5
        mc.handle(Event(type=EventType.TOOL_RESULT.value, data={"call_id": "a"}))
        snap = mc.snapshot()
        assert "tool_duration_p95" in snap
        assert snap["tool_duration_count"] == 1

    def test_p95_many_values(self):
        """p95 with 100 values should pick the 95th percentile."""
        from caveman.events import MetricsCollector
        mc = MetricsCollector()
        mc._timings["test"] = list(range(100))
        snap = mc.snapshot()
        assert snap["test_p95"] == 95  # index 95 of [0..99]

    def test_p95_boundary(self):
        """p95 index should never exceed len-1."""
        from caveman.events import MetricsCollector
        mc = MetricsCollector()
        mc._timings["test"] = [1.0, 2.0]
        snap = mc.snapshot()
        # int(2 * 0.95) = 1, min(1, 1) = 1 → value 2.0
        assert snap["test_p95"] == 2.0


# ── P2 #4: Prompt token estimation CJK consistency ──


class TestPromptCJKTokens:
    def test_cjk_layer_estimate(self):
        """CJK content should estimate ~1 token per character."""
        from caveman.agent.prompt import PromptLayer
        cjk_text = "这是一个中文测试内容" * 10  # 100 CJK chars
        layer = PromptLayer(name="cjk", content=cjk_text, priority=10, budget=5000)
        # CJK: 100 chars → ~100 tokens + 3 overhead = 103
        assert layer.token_estimate == 103

    def test_mixed_layer_estimate(self):
        """Mixed CJK + ASCII should blend estimates."""
        from caveman.agent.prompt import PromptLayer
        mixed = "Hello 你好 World 世界" * 5
        layer = PromptLayer(name="mixed", content=mixed, priority=10, budget=5000)
        est = layer.token_estimate
        # Should be between pure ASCII (len/4) and pure CJK (len)
        assert est > 0
        assert est < len(mixed)  # Not 1:1 for mixed

    def test_truncate_cjk_aware(self):
        """_truncate should respect CJK token density."""
        from caveman.agent.prompt import _truncate
        cjk_text = "中" * 200  # 200 CJK chars ≈ 203 tokens
        truncated, was_truncated = _truncate(cjk_text, 100)
        assert was_truncated
        assert "truncated" in truncated
        # Truncated text should be shorter than original
        assert len(truncated) < len(cjk_text)


# ── P2 #5: Engine injection (no double-creation) ──


class TestEngineInjection:
    def test_injected_shield_used(self):
        """AgentLoop should use injected shield instead of creating new one."""
        from caveman.agent.loop import AgentLoop
        from caveman.engines.shield import CompactionShield
        from caveman.memory.manager import MemoryManager

        mock_shield = MagicMock(spec=CompactionShield)
        mock_shield.essence = MagicMock()
        mock_shield.essence.session_id = "test-123"
        mock_shield.essence.turn_count = 0
        mock_shield.essence.task = None

        loop = AgentLoop(shield=mock_shield)
        assert loop._shield is mock_shield

    def test_injected_recall_used(self):
        """AgentLoop should use injected recall engine."""
        from caveman.agent.loop import AgentLoop
        from caveman.engines.recall import RecallEngine

        mock_recall = MagicMock(spec=RecallEngine)
        loop = AgentLoop(recall_engine=mock_recall)
        assert loop._recall is mock_recall

    def test_injected_nudge_used(self):
        """AgentLoop should use injected nudge engine."""
        from caveman.agent.loop import AgentLoop
        from caveman.memory.nudge import MemoryNudge

        mock_nudge = MagicMock(spec=MemoryNudge)
        mock_nudge.should_nudge = MagicMock(return_value=False)
        loop = AgentLoop(nudge_engine=mock_nudge)
        assert loop._nudge is mock_nudge

    def test_injected_reflect_used(self):
        """AgentLoop should use injected reflect engine."""
        from caveman.agent.loop import AgentLoop
        from caveman.engines.reflect import ReflectEngine

        mock_reflect = MagicMock(spec=ReflectEngine)
        mock_reflect.reflections = []
        loop = AgentLoop(reflect_engine=mock_reflect)
        assert loop._reflect is mock_reflect

    def test_default_engines_created_when_not_injected(self):
        """AgentLoop should create default engines when none injected."""
        from caveman.agent.loop import AgentLoop
        from caveman.engines.shield import CompactionShield
        from caveman.engines.recall import RecallEngine
        from caveman.memory.nudge import MemoryNudge
        from caveman.engines.reflect import ReflectEngine

        loop = AgentLoop()
        assert isinstance(loop._shield, CompactionShield)
        assert isinstance(loop._recall, RecallEngine)
        assert isinstance(loop._nudge, MemoryNudge)
        assert isinstance(loop._reflect, ReflectEngine)
