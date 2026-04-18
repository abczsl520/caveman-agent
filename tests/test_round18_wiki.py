"""Tests for Round 18 Phase 1 — Wiki Compiler Engine."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from caveman.wiki import WikiEntry, WikiStore, CompilationResult, TIERS, TIER_CONFIG
from caveman.wiki.compiler import WikiCompiler, _entry_id, _hours_since


class TestWikiEntry:
    def test_create_entry(self):
        e = WikiEntry(id="test1", tier="working", title="Test", content="Hello")
        assert e.tier == "working"
        assert e.confidence == 0.5
        assert e.created_at != ""

    def test_invalid_tier(self):
        with pytest.raises(Exception):
            WikiEntry(id="x", tier="invalid", title="T", content="C")

    def test_to_dict_roundtrip(self):
        e = WikiEntry(id="t1", tier="semantic", title="Fact", content="X=1", tags=["math"])
        d = e.to_dict()
        e2 = WikiEntry.from_dict(d)
        assert e2.id == "t1"
        assert e2.tier == "semantic"
        assert e2.tags == ["math"]

    def test_to_markdown(self):
        e = WikiEntry(id="m1", tier="procedural", title="Pattern", content="Always test first",
                      tags=["testing"], links=["m2"])
        md = e.to_markdown()
        assert "# Pattern" in md
        assert "tier: procedural" in md
        assert "[[m2]]" in md
        assert "testing" in md


class TestWikiStore:
    def test_add_and_load(self, tmp_path):
        store = WikiStore(tmp_path)
        e = WikiEntry(id="s1", tier="working", title="Obs", content="Saw X")
        store.add(e)
        loaded = store.load_tier("working")
        assert len(loaded) == 1
        assert loaded[0].id == "s1"

    def test_update_existing(self, tmp_path):
        store = WikiStore(tmp_path)
        e = WikiEntry(id="s1", tier="working", title="V1", content="Old")
        store.add(e)
        e.content = "New"
        store.add(e)
        loaded = store.load_tier("working")
        assert len(loaded) == 1
        assert loaded[0].content == "New"

    def test_remove(self, tmp_path):
        store = WikiStore(tmp_path)
        store.add(WikiEntry(id="r1", tier="episodic", title="T", content="C"))
        assert store.remove("episodic", "r1")
        assert len(store.load_tier("episodic")) == 0

    def test_get_across_tiers(self, tmp_path):
        store = WikiStore(tmp_path)
        store.add(WikiEntry(id="g1", tier="semantic", title="T", content="C"))
        assert store.get("g1") is not None
        assert store.get("nonexistent") is None

    def test_search(self, tmp_path):
        store = WikiStore(tmp_path)
        store.add(WikiEntry(id="x1", tier="working", title="Python tips", content="Use list comprehensions"))
        store.add(WikiEntry(id="x2", tier="semantic", title="Rust tips", content="Use pattern matching"))
        results = store.search("python")
        assert len(results) >= 1
        assert results[0].id == "x1"

    def test_stats(self, tmp_path):
        store = WikiStore(tmp_path)
        store.add(WikiEntry(id="a", tier="working", title="T", content="C"))
        store.add(WikiEntry(id="b", tier="working", title="T", content="C"))
        store.add(WikiEntry(id="c", tier="semantic", title="T", content="C"))
        stats = store.stats()
        assert stats["working"] == 2
        assert stats["semantic"] == 1

    def test_empty_tier(self, tmp_path):
        store = WikiStore(tmp_path)
        assert store.load_tier("procedural") == []

    def test_corrupted_json(self, tmp_path):
        store = WikiStore(tmp_path)
        (tmp_path / "working.json").write_text("not json")
        assert store.load_tier("working") == []


class TestWikiCompiler:
    def test_ingest(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        entry = compiler.ingest("Python uses 0-based indexing", tags=["python"])
        assert entry.tier == "working"
        assert entry.confidence == 0.5

    def test_ingest_dedup(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        e1 = compiler.ingest("Same content")
        e2 = compiler.ingest("Same content")
        assert e1.id == e2.id
        assert e2.reinforcement_count == 1
        assert e2.confidence > 0.5

    def test_ingest_session(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        entry = compiler.ingest_session(
            session_id="sess-1",
            task="Build API",
            decisions=["Use FastAPI"],
            progress=["Created routes"],
            todos=["Add auth"],
        )
        assert entry.tier == "episodic"
        assert "FastAPI" in entry.content

    def test_compile_basic(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        compiler.ingest("Fact A")
        compiler.ingest("Fact B")
        result = compiler.compile()
        assert result.entries_processed >= 2
        assert result.duration_ms > 0

    def test_compile_promotion(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        # Ingest and reinforce enough times to promote
        entry = compiler.ingest("Important fact")
        for _ in range(3):
            compiler.reinforce(entry.id)

        result = compiler.compile()
        assert result.entries_promoted >= 1

        # Should now be in episodic
        episodic = compiler.store.load_tier("episodic")
        assert any(e.id == entry.id for e in episodic)

    def test_compile_max_entries(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        # Add more than max working entries
        for i in range(105):
            compiler.ingest(f"Observation {i}", confidence=i / 200)
        result = compiler.compile()
        working = compiler.store.load_tier("working")
        assert len(working) <= 100

    def test_get_compiled_context(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        compiler.ingest("Python tip: use enumerate", tags=["python"])
        compiler.ingest("Rust tip: use match", tags=["rust"])
        ctx = compiler.get_compiled_context(max_tokens=4000)
        assert "Compiled Wiki" in ctx
        assert "Working Knowledge" in ctx

    def test_get_compiled_context_empty(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        ctx = compiler.get_compiled_context()
        assert ctx == ""

    def test_reinforce(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        entry = compiler.ingest("Useful fact")
        assert compiler.reinforce(entry.id)
        updated = compiler.store.get(entry.id)
        assert updated.access_count == 1
        assert updated.confidence > 0.5

    def test_reinforce_nonexistent(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        assert not compiler.reinforce("nonexistent")

    def test_weaken(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        entry = compiler.ingest("Dubious claim", confidence=0.8)
        compiler.weaken(entry.id, amount=0.3)
        updated = compiler.store.get(entry.id)
        assert updated.confidence == pytest.approx(0.5, abs=0.01)

    def test_export_markdown(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        compiler.ingest("Export test", tags=["test"])
        count = compiler.export_markdown(tmp_path / "export")
        assert count == 1
        md_files = list((tmp_path / "export" / "working").glob("*.md"))
        assert len(md_files) == 1

    def test_generate_index(self, tmp_path):
        compiler = WikiCompiler(WikiStore(tmp_path))
        compiler.ingest("Fact 1", tags=["a"])
        compiler.ingest("Fact 2", tags=["b"])
        index = compiler.generate_index()
        assert "Wiki Index" in index
        assert "Working" in index


class TestHelpers:
    def test_entry_id_deterministic(self):
        assert _entry_id("hello") == _entry_id("hello")
        assert _entry_id("hello") != _entry_id("world")

    def test_hours_since_recent(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        assert _hours_since(now) < 0.01

    def test_hours_since_invalid(self):
        assert _hours_since("not-a-date") == float("inf")

    def test_tiers_order(self):
        assert TIERS == ("working", "episodic", "semantic", "procedural")

    def test_tier_config_complete(self):
        for tier in TIERS:
            assert tier in TIER_CONFIG
            assert "max_age_hours" in TIER_CONFIG[tier]
            assert "min_confidence" in TIER_CONFIG[tier]
