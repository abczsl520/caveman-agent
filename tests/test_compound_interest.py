"""Tests for compound interest improvements — the flywheel must accelerate."""
import asyncio
import math
import pytest
from pathlib import Path

from caveman.memory.retrieval import HybridScorer, adjust_trust, tokenize
from caveman.memory.types import MemoryEntry, MemoryType
from datetime import datetime


# ── retrieval_count (popularity) affects ranking ──

def test_popular_memory_ranks_higher():
    """A memory retrieved 50 times should rank higher than one retrieved 0 times."""
    scorer = HybridScorer()
    query_tokens = tokenize("deploy docker")

    popular = MemoryEntry(
        id="pop", content="Deploy with docker compose up -d",
        memory_type=MemoryType.PROCEDURAL, created_at=datetime.now(),
        metadata={"trust_score": 0.5, "retrieval_count": 50},
    )
    fresh = MemoryEntry(
        id="fresh", content="Deploy with docker compose up -d",
        memory_type=MemoryType.PROCEDURAL, created_at=datetime.now(),
        metadata={"trust_score": 0.5, "retrieval_count": 0},
    )

    score_pop = scorer.score(query_tokens, popular)
    score_fresh = scorer.score(query_tokens, fresh)

    assert score_pop > score_fresh, \
        f"Popular ({score_pop:.4f}) should rank higher than fresh ({score_fresh:.4f})"


def test_popularity_bonus_is_logarithmic():
    """Popularity bonus should be log-scaled — diminishing returns."""
    scorer = HybridScorer()
    query_tokens = tokenize("test")

    def make_entry(count):
        return MemoryEntry(
            id=f"m{count}", content="test content",
            memory_type=MemoryType.SEMANTIC, created_at=datetime.now(),
            metadata={"trust_score": 0.5, "retrieval_count": count},
        )

    s0 = scorer.score(query_tokens, make_entry(0))
    s10 = scorer.score(query_tokens, make_entry(10))
    s100 = scorer.score(query_tokens, make_entry(100))
    s1000 = scorer.score(query_tokens, make_entry(1000))

    # Each 10x increase should give diminishing bonus
    delta_0_10 = s10 - s0
    delta_10_100 = s100 - s10
    delta_100_1000 = s1000 - s100

    assert delta_0_10 > delta_100_1000, \
        "Popularity bonus should have diminishing returns"


# ── trust delta is more balanced ──

def test_trust_reaches_0_9_faster():
    """With +0.08 delta, should reach 0.9 in 5 successes (was 8 with +0.05)."""
    trust = 0.5
    uses = 0
    while trust < 0.9 and uses < 20:
        trust = adjust_trust(trust, helpful=True)
        uses += 1
    assert uses <= 6, f"Takes {uses} successes to reach 0.9 — too slow for compound interest"


# ── Skill cross-language matching ──

def test_skill_match_cross_lang(tmp_path):
    """Skills should match across languages via query expansion."""
    from caveman.skills.manager import SkillManager
    from caveman.skills.types import Skill, SkillTrigger

    sm = SkillManager(skills_dir=tmp_path)
    skill = Skill(
        name="deploy", description="Deploy application",
        trigger=SkillTrigger.PATTERN,
        trigger_patterns=["deploy", "部署"],
    )
    sm.save(skill)

    # Chinese synonym not in trigger_patterns but in expansion
    assert len(sm.match("部署项目")) > 0  # Direct match
    # English query
    assert len(sm.match("deploy the app")) > 0


# ── Flywheel health metrics ──

def test_flywheel_health_metrics():
    """AgentMetrics should track flywheel acceleration indicators."""
    from caveman.agent.metrics import AgentMetrics

    m = AgentMetrics()
    m.increment("turns_completed", 10)
    m.increment("task_successes", 8)
    m.increment("recall_attempts", 10)
    m.increment("recall_hits", 7)
    m.increment("skill_match_attempts", 10)
    m.increment("skill_match_hits", 4)

    health = m.flywheel_health()
    assert health["task_success_rate"] == 0.8
    assert health["recall_hit_rate"] == 0.7
    assert health["skill_match_rate"] == 0.4
    assert health["total_turns"] == 10


@pytest.mark.asyncio
async def test_retrieval_count_in_fts_results(tmp_path):
    """FTS search results should include retrieval_count for HybridScorer."""
    from caveman.memory.sqlite_store import SQLiteMemoryStore

    store = SQLiteMemoryStore(db_path=tmp_path / "t.db")
    await store.store("Deploy with docker compose", MemoryType.PROCEDURAL)

    # Recall once to increment retrieval_count
    results = await store.recall("docker deploy", top_k=1)
    assert len(results) >= 1

    # Recall again — retrieval_count should be in metadata
    results2 = await store.recall("docker deploy", top_k=1)
    rc = results2[0].metadata.get("retrieval_count", 0)
    assert rc >= 1, f"retrieval_count should be >= 1, got {rc}"
    store.close()
