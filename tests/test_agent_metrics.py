"""Tests for agent/metrics.py — AgentMetrics tracking."""
import time
from caveman.agent.metrics import AgentMetrics


class TestAgentMetrics:
    def test_increment(self):
        m = AgentMetrics()
        m.increment("turns")
        m.increment("turns")
        m.increment("turns", 3)
        assert m._counters["turns"] == 5

    def test_record_timing(self):
        m = AgentMetrics()
        m.record_timing("llm_call", 1.5)
        m.record_timing("llm_call", 2.5)
        assert len(m._timings["llm_call"]) == 2

    def test_timer_context_manager(self):
        m = AgentMetrics()
        with m.timer("test_op"):
            time.sleep(0.01)
        assert len(m._timings["test_op"]) == 1
        assert m._timings["test_op"][0] >= 0.01

    def test_summary_empty(self):
        m = AgentMetrics()
        s = m.summary()
        assert s["counters"] == {}
        assert s["timings"] == {}

    def test_summary_with_data(self):
        m = AgentMetrics()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.record_timing("op", v)
        m.increment("calls", 10)
        s = m.summary()
        assert s["counters"]["calls"] == 10
        assert s["timings"]["op"]["count"] == 5
        assert s["timings"]["op"]["avg"] == 3.0
        assert s["timings"]["op"]["min"] == 1.0
        assert s["timings"]["op"]["max"] == 5.0

    def test_reset(self):
        m = AgentMetrics()
        m.increment("x")
        m.record_timing("y", 1.0)
        m.reset()
        assert m._counters == {}
        assert m._timings == {}

    def test_flywheel_health_empty(self):
        m = AgentMetrics()
        h = m.flywheel_health()
        assert h["recall_hit_rate"] == 0.0
        assert h["skill_match_rate"] == 0.0
        assert h["task_success_rate"] == 0.0

    def test_flywheel_health_with_data(self):
        m = AgentMetrics()
        m.increment("recall_attempts", 10)
        m.increment("recall_hits", 7)
        m.increment("skill_match_attempts", 5)
        m.increment("skill_match_hits", 3)
        m.increment("turns_completed", 20)
        m.increment("task_successes", 15)
        h = m.flywheel_health()
        assert h["recall_hit_rate"] == 0.7
        assert h["skill_match_rate"] == 0.6
        assert h["task_success_rate"] == 0.75
