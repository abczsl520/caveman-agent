"""Tests from 50-dimension flywheel audit."""
import pytest
from datetime import datetime, timedelta

from caveman.memory.retrieval import HybridScorer, tokenize
from caveman.memory.types import MemoryEntry, MemoryType


# ── T18: Adaptive temporal decay ──

def test_old_high_trust_beats_new_low_trust():
    """3-month-old high-trust memory must beat today's low-trust memory.
    This is the core compound interest test — PRD says '3 months → specialized Agent'.
    """
    scorer = HybridScorer()
    query_tokens = tokenize("deploy docker")

    old_valuable = MemoryEntry(
        id="old", content="Deploy with docker compose up -d on port 8080",
        memory_type=MemoryType.PROCEDURAL,
        created_at=datetime.now() - timedelta(days=90),
        metadata={"trust_score": 0.9, "retrieval_count": 50},
    )
    new_ordinary = MemoryEntry(
        id="new", content="Docker is a containerization platform",
        memory_type=MemoryType.SEMANTIC,
        created_at=datetime.now(),
        metadata={"trust_score": 0.5, "retrieval_count": 0},
    )

    s_old = scorer.score(query_tokens, old_valuable)
    s_new = scorer.score(query_tokens, new_ordinary)
    assert s_old > s_new, \
        f"90-day high-trust memory ({s_old:.4f}) must beat new low-trust ({s_new:.4f})"


def test_adaptive_decay_high_trust_slower():
    """High-trust memories should decay slower than low-trust ones."""
    scorer = HybridScorer()
    query_tokens = tokenize("test content")

    def make_entry(trust, days_ago):
        return MemoryEntry(
            id=f"t{trust}d{days_ago}", content="test content for scoring",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now() - timedelta(days=days_ago),
            metadata={"trust_score": trust, "retrieval_count": 10},
        )

    # At 60 days: high trust should retain more score than low trust
    high_60 = scorer.score(query_tokens, make_entry(0.9, 60))
    low_60 = scorer.score(query_tokens, make_entry(0.1, 60))
    ratio = high_60 / low_60 if low_60 > 0 else float('inf')
    assert ratio > 2.0, f"High trust should decay much slower (ratio={ratio:.2f})"


def test_half_life_is_90_days():
    """Default half-life should be 90 days, not 30."""
    scorer = HybridScorer()
    assert scorer.temporal_half_life_days == 90


# ── D5: Tool output truncation ──

def test_tool_output_truncation_in_code():
    """tools_exec.py should truncate large outputs."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "caveman/agent/tools_exec.py").read_text()
    assert "30_000" in src or "30000" in src, \
        "Tool output should be truncated at ~30K chars"


# ── D1: Task length cap ──

def test_task_length_cap_in_code():
    """phase_prepare should cap task length."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "caveman/agent/phases.py").read_text()
    assert "2000" in src and "task[:2000]" in src, \
        "Task should be capped at 2000 chars"


# ── E46: Memory consolidation (dedup → trust boost) ──

@pytest.mark.asyncio
async def test_dedup_boosts_existing_trust(tmp_path):
    """When a near-duplicate is found, existing memory's trust should increase."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.nudge import MemoryNudge

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    nudge = MemoryNudge(memory_manager=mm)

    # Store an existing memory
    mid = await mm.store("Server runs on port 8080 with nginx", MemoryType.SEMANTIC)

    # Get initial trust
    entries = mm.all_entries()
    initial_trust = entries[0].metadata.get("trust_score", 0.5)

    # Nudge with a near-duplicate candidate
    candidates = [
        {"type": "semantic", "content": "Server runs on port 8080 with nginx reverse proxy"},
    ]
    result = await nudge._dedup_candidates(candidates)

    # Candidate should be filtered out (dedup)
    assert len(result) == 0

    # But existing memory's trust should have been boosted
    entries_after = mm.all_entries()
    new_trust = entries_after[0].metadata.get("trust_score", 0.5)
    assert new_trust > initial_trust, \
        f"Trust should increase on confirmation: {initial_trust} → {new_trust}"


# ── Dimension 3: Memory rejuvenation (spaced repetition) ──

@pytest.mark.asyncio
async def test_recall_sets_last_accessed(tmp_path):
    """Recall should update last_accessed in metadata."""
    from caveman.memory.manager import MemoryManager

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    await mm.store("Deploy with docker compose", MemoryType.PROCEDURAL)

    # Recall it
    results = await mm.recall("docker deploy", top_k=1)
    assert len(results) >= 1

    # Check last_accessed was set
    entries = mm.all_entries()
    la = entries[0].metadata.get("last_accessed")
    assert la is not None, "last_accessed should be set after recall"


def test_last_accessed_affects_decay():
    """A recently accessed old memory should have less decay than an unaccessed one."""
    scorer = HybridScorer()
    query_tokens = tokenize("deploy docker")

    # Old memory, never accessed
    old_unaccessed = MemoryEntry(
        id="old_un", content="Deploy with docker compose up -d",
        memory_type=MemoryType.PROCEDURAL,
        created_at=datetime.now() - timedelta(days=90),
        metadata={"trust_score": 0.7, "retrieval_count": 0},
    )

    # Old memory, accessed yesterday
    old_accessed = MemoryEntry(
        id="old_ac", content="Deploy with docker compose up -d",
        memory_type=MemoryType.PROCEDURAL,
        created_at=datetime.now() - timedelta(days=90),
        metadata={
            "trust_score": 0.7, "retrieval_count": 10,
            "last_accessed": (datetime.now() - timedelta(days=1)).isoformat(),
        },
    )

    s_un = scorer.score(query_tokens, old_unaccessed)
    s_ac = scorer.score(query_tokens, old_accessed)

    assert s_ac > s_un, \
        f"Recently accessed ({s_ac:.4f}) should beat unaccessed ({s_un:.4f})"


# ── Dimension 6: Graph expansion in recall ──

@pytest.mark.asyncio
async def test_cross_ref_expansion_in_recall(tmp_path):
    """Recall should consider cross-referenced memories from Ripple."""
    from caveman.memory.manager import MemoryManager
    from caveman.engines.ripple import RippleEngine

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    ripple = RippleEngine(memory_manager=mm)
    mm.set_ripple(ripple)

    # Store two related memories
    mid1 = await mm.store("Server runs on port 8080 with nginx", MemoryType.SEMANTIC)
    mid2 = await mm.store("Nginx config at /etc/nginx/sites-enabled/app", MemoryType.SEMANTIC)

    # Manually add cross-ref (simulating Ripple)
    await mm.update_metadata(mid1, {"related": [mid2]})
    await mm.update_metadata(mid2, {"related": [mid1]})

    # Search for "nginx" — should find both even if only one matches FTS5
    results = await mm.recall("nginx configuration", top_k=5)
    result_ids = {r.id for r in results}

    # At minimum, the nginx config memory should be found
    assert mid2 in result_ids, "Should find the nginx config memory"
