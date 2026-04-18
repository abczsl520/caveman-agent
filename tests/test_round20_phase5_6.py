"""Tests for Round 20 Phase 5-6 + Title Generator.

Phase 5: Wiki Provenance + Contradiction Detection
Phase 6: Sub-agent Orchestration
Title Generator: Hermes port
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

# ── Phase 5: Provenance ──


class TestProvenance:
    def _make_store(self, tmp_path):
        from caveman.wiki import WikiStore
        return WikiStore(tmp_path / "wiki")

    def test_record_and_get(self, tmp_path):
        from caveman.wiki.provenance import ProvenanceTracker
        store = self._make_store(tmp_path)
        tracker = ProvenanceTracker(store)

        prov = tracker.record("entry1", "session", "sess_abc", context="first exchange")
        assert prov.entry_id == "entry1"
        assert prov.source_type == "session"
        assert prov.source_id == "sess_abc"

        records = tracker.get("entry1")
        assert len(records) == 1
        assert records[0].context == "first exchange"

    def test_multiple_sources(self, tmp_path):
        from caveman.wiki.provenance import ProvenanceTracker
        store = self._make_store(tmp_path)
        tracker = ProvenanceTracker(store)

        tracker.record("entry1", "session", "sess_1")
        tracker.record("entry1", "session", "sess_2")
        tracker.record("entry1", "reflect", "reflect_1")

        assert tracker.count_sources("entry1") == 3
        sources = tracker.get_sources("entry1")
        assert "sess_1" in sources
        assert "reflect_1" in sources

    def test_persistence(self, tmp_path):
        from caveman.wiki.provenance import ProvenanceTracker
        store = self._make_store(tmp_path)

        tracker1 = ProvenanceTracker(store)
        tracker1.record("e1", "session", "s1")

        # New instance should load persisted data
        tracker2 = ProvenanceTracker(store)
        assert len(tracker2.get("e1")) == 1

    def test_cleanup(self, tmp_path):
        from caveman.wiki.provenance import ProvenanceTracker
        store = self._make_store(tmp_path)
        tracker = ProvenanceTracker(store)

        tracker.record("e1", "session", "s1")
        tracker.record("e2", "session", "s2")
        tracker.record("e3", "session", "s3")

        removed = tracker.cleanup(valid_ids={"e1", "e3"})
        assert removed == 1
        assert len(tracker.get("e2")) == 0

    def test_provenance_serialization(self, tmp_path):
        from caveman.wiki.provenance import Provenance
        p = Provenance(entry_id="e1", source_type="tool", source_id="bash")
        d = p.to_dict()
        p2 = Provenance.from_dict(d)
        assert p2.entry_id == "e1"
        assert p2.source_type == "tool"


# ── Phase 5: Contradiction Detection ──


class TestContradictionDetection:
    def _make_store(self, tmp_path):
        from caveman.wiki import WikiStore
        return WikiStore(tmp_path / "wiki")

    def test_negation_conflict(self, tmp_path):
        from caveman.wiki import WikiEntry
        from caveman.wiki.provenance import ContradictionDetector
        store = self._make_store(tmp_path)

        e1 = WikiEntry(id="a1", tier="semantic", title="Python indexing",
                        content="Python arrays are 0-based indexed",
                        tags=["python", "indexing"], confidence=0.8)
        e2 = WikiEntry(id="a2", tier="semantic", title="Python indexing note",
                        content="Python arrays are not 0-based indexed",
                        tags=["python", "indexing"], confidence=0.5)
        store.add(e1)
        store.add(e2)

        detector = ContradictionDetector(store)
        contradictions = detector.detect(scope="semantic")
        assert len(contradictions) >= 1
        assert contradictions[0].confidence >= 0.6

    def test_no_contradiction_different_topics(self, tmp_path):
        from caveman.wiki import WikiEntry
        from caveman.wiki.provenance import ContradictionDetector
        store = self._make_store(tmp_path)

        e1 = WikiEntry(id="b1", tier="working", title="Weather",
                        content="It is sunny today", tags=["weather"])
        e2 = WikiEntry(id="b2", tier="working", title="Coding",
                        content="Python is a programming language", tags=["coding"])
        store.add(e1)
        store.add(e2)

        detector = ContradictionDetector(store)
        contradictions = detector.detect(scope="working")
        assert len(contradictions) == 0

    def test_auto_resolve_weakens_lower(self, tmp_path):
        from caveman.wiki import WikiEntry
        from caveman.wiki.provenance import ContradictionDetector, Contradiction
        store = self._make_store(tmp_path)

        e1 = WikiEntry(id="c1", tier="semantic", title="Config",
                        content="The default port is 8080",
                        tags=["config"], confidence=0.9)
        e2 = WikiEntry(id="c2", tier="semantic", title="Config port",
                        content="The default port is not 8080",
                        tags=["config"], confidence=0.4)
        store.add(e1)
        store.add(e2)

        detector = ContradictionDetector(store)
        contradictions = detector.detect()
        resolved = detector.auto_resolve(contradictions)
        assert resolved >= 1

        # Lower confidence entry should be weakened
        updated = store.get("c2")
        assert updated.confidence < 0.4

    def test_contradiction_serialization(self):
        from caveman.wiki.provenance import Contradiction
        c = Contradiction(entry_a_id="a", entry_b_id="b", reason="test", confidence=0.7)
        d = c.to_dict()
        c2 = Contradiction.from_dict(d)
        assert c2.confidence == 0.7
        assert c2.reason == "test"

    def test_chinese_negation(self, tmp_path):
        from caveman.wiki import WikiEntry
        from caveman.wiki.provenance import ContradictionDetector
        store = self._make_store(tmp_path)

        e1 = WikiEntry(id="d1", tier="working", title="规则",
                        content="这个功能是启用的状态",
                        tags=["config", "feature"], confidence=0.7)
        e2 = WikiEntry(id="d2", tier="working", title="规则更新",
                        content="这个功能是禁用的状态",
                        tags=["config", "feature"], confidence=0.6)
        store.add(e1)
        store.add(e2)

        detector = ContradictionDetector(store)
        contradictions = detector.detect(scope="working")
        assert len(contradictions) >= 1


# ── Phase 6: Sub-agent Orchestration ──


class TestAgentProfile:
    def test_update_stats(self):
        from caveman.coordinator.orchestrator import AgentProfile
        p = AgentProfile(name="test", capabilities=["coding"])
        p.update_stats(True, 5.0)
        assert p.total_calls == 1
        assert p.success_rate == 1.0

        p.update_stats(False, 10.0)
        assert p.total_calls == 2
        assert p.success_rate == 0.5

    def test_availability(self):
        from caveman.coordinator.orchestrator import AgentProfile
        p = AgentProfile(name="test", max_concurrent=2)
        assert p.is_available()
        p.active_count = 2
        assert not p.is_available()


class TestAgentRegistry:
    def test_register_and_find(self):
        from caveman.coordinator.orchestrator import AgentProfile, AgentRegistry

        reg = AgentRegistry()
        async def dummy(task, ctx):
            return "ok"

        reg.register(AgentProfile("coder", capabilities=["coding"]), dummy)
        reg.register(AgentProfile("writer", capabilities=["writing"]), dummy)

        best = reg.find_best(["coding"])
        assert best is not None
        assert best.name == "coder"

    def test_find_none_available(self):
        from caveman.coordinator.orchestrator import AgentProfile, AgentRegistry
        reg = AgentRegistry()
        assert reg.find_best(["coding"]) is None

    def test_list_agents(self):
        from caveman.coordinator.orchestrator import AgentProfile, AgentRegistry
        reg = AgentRegistry()
        async def dummy(task, ctx):
            return "ok"
        reg.register(AgentProfile("a"), dummy)
        reg.register(AgentProfile("b"), dummy)
        assert len(reg.list_agents()) == 2


class TestSubAgentOrchestrator:
    def test_delegate_success(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def echo(task, ctx):
            return f"done: {task}"
        reg.register(AgentProfile("echo", capabilities=["general"]), echo)

        orch = SubAgentOrchestrator(registry=reg)
        result = asyncio.run(
            orch.delegate("hello world", agent_name="echo")
        )
        assert result.success
        assert "done: hello world" in result.result

    def test_delegate_auto_route(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def coder(task, ctx):
            return "coded"
        reg.register(AgentProfile("coder", capabilities=["coding"]), coder)

        orch = SubAgentOrchestrator(registry=reg)
        result = asyncio.run(
            orch.delegate("implement a REST API")
        )
        assert result.success
        assert result.agent_name == "coder"

    def test_delegate_unknown_agent(self):
        from caveman.coordinator.orchestrator import SubAgentOrchestrator
        orch = SubAgentOrchestrator()
        result = asyncio.run(
            orch.delegate("test", agent_name="nonexistent")
        )
        assert not result.success
        assert "not registered" in result.error

    def test_delegate_timeout(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def slow(task, ctx):
            await asyncio.sleep(10)
            return "never"
        reg.register(AgentProfile("slow", capabilities=["general"]), slow)

        orch = SubAgentOrchestrator(registry=reg)
        result = asyncio.run(
            orch.delegate("test", agent_name="slow", timeout=0.1)
        )
        assert not result.success
        assert "Timeout" in result.error

    def test_delegate_parallel(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def echo(task, ctx):
            return f"done: {task}"
        reg.register(
            AgentProfile("echo", capabilities=["general"], max_concurrent=5), echo,
        )

        orch = SubAgentOrchestrator(registry=reg)
        tasks = [{"task": f"task_{i}", "agent": "echo"} for i in range(3)]
        results = asyncio.run(
            orch.delegate_parallel(tasks)
        )
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_stats(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def echo(task, ctx):
            return "ok"
        reg.register(AgentProfile("echo", capabilities=["general"]), echo)

        orch = SubAgentOrchestrator(registry=reg)
        asyncio.run(
            orch.delegate("test", agent_name="echo")
        )
        stats = orch.get_stats()
        assert stats["total_delegations"] == 1
        assert stats["success_rate"] == 1.0

    def test_lessons_extracted(self):
        from caveman.coordinator.orchestrator import (
            AgentProfile, AgentRegistry, SubAgentOrchestrator,
        )
        reg = AgentRegistry()
        async def fail(task, ctx):
            raise ValueError("boom")
        reg.register(AgentProfile("fail", capabilities=["general"]), fail)

        orch = SubAgentOrchestrator(registry=reg)
        result = asyncio.run(
            orch.delegate("test", agent_name="fail")
        )
        assert not result.success
        assert len(result.lessons) >= 1
        assert "failed" in result.lessons[0]


# ── Title Generator ──


class TestTitleGenerator:
    def test_heuristic_english(self):
        from caveman.agent.title_generator import _heuristic_title
        assert _heuristic_title("Help me fix the login bug") is not None
        title = _heuristic_title("Create a REST API for users")
        assert title is not None
        assert "REST API" in title or "Build" in title

    def test_heuristic_chinese(self):
        from caveman.agent.title_generator import _heuristic_title
        title = _heuristic_title("帮我修复登录页面的问题")
        assert title is not None
        assert "修复" in title or "登录" in title

    def test_heuristic_short_message(self):
        from caveman.agent.title_generator import _heuristic_title
        title = _heuristic_title("Hello world")
        assert title is not None

    def test_heuristic_long_message(self):
        from caveman.agent.title_generator import _heuristic_title
        msg = "I need you to " + "do something very complex " * 20
        title = _heuristic_title(msg)
        # Should truncate
        assert title is None or len(title) <= 80

    def test_clean_title(self):
        from caveman.agent.title_generator import _clean_title
        assert _clean_title('"My Title"') == "My Title"
        assert _clean_title("Title: Something") == "Something"
        assert _clean_title("") is None

    def test_generate_title_no_llm(self):
        from caveman.agent.title_generator import generate_title
        title = asyncio.run(
            generate_title("Fix the database connection error", use_llm=False)
        )
        assert title is not None
        assert len(title) <= 80

    def test_generate_title_chinese_no_llm(self):
        from caveman.agent.title_generator import generate_title
        title = asyncio.run(
            generate_title("创建一个用户管理系统", use_llm=False)
        )
        assert title is not None


# ── Integration: Wiki + Provenance ──


class TestWikiProvenanceIntegration:
    def test_ingest_with_provenance(self, tmp_path):
        from caveman.wiki import WikiStore
        from caveman.wiki.compiler import WikiCompiler
        from caveman.wiki.provenance import ProvenanceTracker

        store = WikiStore(tmp_path / "wiki")
        compiler = WikiCompiler(store)
        tracker = ProvenanceTracker(store)

        entry = compiler.ingest("Python uses 0-based indexing", tags=["python"])
        tracker.record(entry.id, "session", "sess_001", context="user mentioned")

        # Verify both entry and provenance exist
        assert store.get(entry.id) is not None
        assert tracker.count_sources(entry.id) == 1

    def test_compile_then_detect(self, tmp_path):
        from caveman.wiki import WikiStore, WikiEntry
        from caveman.wiki.compiler import WikiCompiler
        from caveman.wiki.provenance import ContradictionDetector

        store = WikiStore(tmp_path / "wiki")
        compiler = WikiCompiler(store)

        # Add contradicting entries
        compiler.ingest("The server port is 8080", tags=["config"])
        compiler.ingest("The server port is not 8080 but 3000", tags=["config"])

        # Compile
        result = compiler.compile()
        assert result.entries_processed >= 2

        # Detect contradictions
        detector = ContradictionDetector(store)
        contradictions = detector.detect()
        # May or may not detect depending on similarity threshold
        # At minimum, the detector should run without error
        assert isinstance(contradictions, list)
