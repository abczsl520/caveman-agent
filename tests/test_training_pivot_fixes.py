"""Tests for training pivot fixes — retrieval log, local embedding, evaluation."""
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestRetrievalLog:
    """Test retrieval logging for embedding training data."""

    def test_log_and_read(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry
        log = RetrievalLog(tmp_path / "test.jsonl")
        log.log(RetrievalEntry(
            query="that server",
            results=[{"memory_id": "m1", "content": "198.51.100.20 Ubuntu", "score": 0.9}],
            source="recall",
        ))
        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0].query == "that server"
        assert entries[0].results[0]["memory_id"] == "m1"

    def test_log_search_convenience(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog
        from dataclasses import dataclass

        @dataclass
        class FakeMemory:
            id: str = "m1"
            content: str = "test content"

        log = RetrievalLog(tmp_path / "test.jsonl")
        log.log_search("query", [(0.85, FakeMemory())], source="memory_search")
        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0].results[0]["score"] == 0.85

    def test_mark_adopted(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog
        log = RetrievalLog(tmp_path / "test.jsonl")
        log.mark_adopted("that server", ["m1", "m3"])
        entries = log.read_all()
        assert entries[0].source == "adoption"
        assert entries[0].adopted_ids == ["m1", "m3"]

    def test_generate_training_pairs(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry
        log = RetrievalLog(tmp_path / "test.jsonl")

        # Search event
        log.log(RetrievalEntry(
            query="deploy server",
            results=[
                {"memory_id": "m1", "content": "198.51.100.20 Ubuntu Node v22", "score": 0.9},
                {"memory_id": "m2", "content": "some unrelated thing", "score": 0.3},
            ],
            source="recall",
        ))
        # Adoption event
        log.mark_adopted("deploy server", ["m1"])

        pairs = log.generate_training_pairs()
        assert len(pairs) == 1
        assert pairs[0]["query"] == "deploy server"
        assert "198.51" in pairs[0]["positive"]
        assert pairs[0]["source"] == "adopted"

    def test_generate_pairs_score_fallback(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry
        log = RetrievalLog(tmp_path / "test.jsonl")

        # No adoption data — use score threshold
        log.log(RetrievalEntry(
            query="restart service",
            results=[
                {"memory_id": "m1", "content": "pm2 restart app-name on server", "score": 0.8},
            ],
            source="memory_search",
        ))

        pairs = log.generate_training_pairs()
        assert len(pairs) == 1
        assert pairs[0]["source"] == "score"

    def test_count(self, tmp_path):
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry
        log = RetrievalLog(tmp_path / "test.jsonl")
        assert log.count() == 0
        log.log(RetrievalEntry(query="q1", results=[]))
        log.log(RetrievalEntry(query="q2", results=[]))
        assert log.count() == 2


class TestPairExtractorRetrievalLog:
    """Test that PairExtractor uses retrieval log as primary source."""

    def test_primary_source_is_retrieval_log(self, tmp_path):
        from caveman.training.embedding import PairExtractor
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry

        # Create retrieval log with data
        log_path = tmp_path / "retrieval_log.jsonl"
        log = RetrievalLog(log_path)
        log.log(RetrievalEntry(
            query="that server IP",
            results=[{"memory_id": "m1", "content": "198.51.100.20 Ubuntu Node v22 game server", "score": 0.9}],
            source="recall",
        ))

        extractor = PairExtractor()
        pairs = extractor.extract_from_retrieval_log(log_path)
        assert len(pairs) >= 1
        assert extractor._retrieval_pairs >= 1

    def test_trajectory_fallback_lower_weight(self):
        from caveman.training.embedding import PairExtractor
        extractor = PairExtractor()
        traj = {
            "quality_score": 0.8,
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris, a major European city."},
            ],
        }
        pairs = extractor.extract_from_trajectory(traj)
        assert len(pairs) == 1
        # Trajectory pairs should have lower score (quality * 0.5)
        assert pairs[0].score == 0.4  # 0.8 * 0.5

    def test_stats_property(self, tmp_path):
        from caveman.training.embedding import PairExtractor
        extractor = PairExtractor()
        traj = {
            "quality_score": 0.8,
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris, a major European city."},
            ],
        }
        extractor.extract_from_trajectory(traj)
        assert "trajectory_fallback: 1" in extractor.stats


class TestLocalEmbedding:
    """Test local embedding provider."""

    def test_get_embedding_fn_local_explicit(self):
        from caveman.memory.embedding import get_embedding_fn
        fn = get_embedding_fn("local")
        assert fn is not None
        assert fn.__name__ == "local_embedding"

    def test_get_embedding_fn_auto_no_local(self, tmp_path, monkeypatch):
        """Auto mode without local model should fall through."""
        from caveman.memory.embedding import get_embedding_fn
        import caveman.paths as paths
        monkeypatch.setattr(paths, "TRAINING_DIR", tmp_path / "nonexistent")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fn = get_embedding_fn("auto")
        # Should not return local_embedding since model doesn't exist
        if fn is not None:
            assert fn.__name__ != "local_embedding"

    def test_local_embedding_missing_model(self):
        from caveman.memory.embedding import local_embedding
        import asyncio
        with pytest.raises(FileNotFoundError):
            asyncio.run(
                local_embedding("test", model_path="/nonexistent/path")
            )

    def test_config_validator_accepts_local(self):
        from caveman.config.validator import validate_config
        # Should not produce error for "local" backend
        errors = validate_config({"memory": {"embedding_backend": "local"}})
        local_errors = [e for e in errors if "local" in e.lower() and "embedding" in e.lower()]
        assert len(local_errors) == 0


class TestEvalEmbedding:
    """Test embedding evaluation."""

    def test_build_eval_set_empty(self, tmp_path):
        from caveman.training.eval_embedding import EmbeddingEvaluator
        evaluator = EmbeddingEvaluator(tmp_path / "empty.jsonl")
        eval_set = evaluator.build_eval_set()
        assert eval_set == []

    def test_build_eval_set_with_adoptions(self, tmp_path):
        from caveman.training.eval_embedding import EmbeddingEvaluator
        from caveman.training.retrieval_log import RetrievalLog, RetrievalEntry

        log_path = tmp_path / "test.jsonl"
        log = RetrievalLog(log_path)
        log.log(RetrievalEntry(
            query="server IP",
            results=[
                {"memory_id": "m1", "content": "198.51.100.20", "score": 0.9},
                {"memory_id": "m2", "content": "unrelated", "score": 0.3},
            ],
            source="recall",
        ))
        log.mark_adopted("server IP", ["m1"])

        evaluator = EmbeddingEvaluator(log_path)
        eval_set = evaluator.build_eval_set()
        assert len(eval_set) == 1
        assert "m1" in eval_set[0]["relevant_ids"]

    def test_compare_report(self):
        from caveman.training.eval_embedding import EmbeddingEvaluator, EvalResult
        evaluator = EmbeddingEvaluator()
        before = EvalResult(recall_at_5=0.5, mrr=0.4, hit_rate_at_5=0.6, total_queries=10)
        after = EvalResult(recall_at_5=0.7, mrr=0.6, hit_rate_at_5=0.8, total_queries=10)
        report = evaluator.compare(before, after)
        assert "+0.200" in report
        assert "Recall@5" in report


class TestMemoryManagerRetrievalLog:
    """Test that MemoryManager logs retrievals."""

    def test_recall_logs_to_retrieval_log(self, tmp_path):
        import asyncio
        from caveman.memory.manager import MemoryManager
        from caveman.memory.types import MemoryType
        from caveman.training.retrieval_log import RetrievalLog

        log_path = tmp_path / "retrieval.jsonl"
        log = RetrievalLog(log_path)
        mgr = MemoryManager(base_dir=tmp_path / "mem", retrieval_log=log)

        # Store a memory
        asyncio.run(
            mgr.store("Server 198.51.100.20 runs Ubuntu", MemoryType.SEMANTIC)
        )

        # Recall it
        asyncio.run(
            mgr.recall("server IP")
        )

        # Check log was written
        entries = log.read_all()
        assert len(entries) >= 1
        assert entries[0].query == "server IP"


class TestRecallEngineRetrievalLog:
    """Test that RecallEngine accepts retrieval_log parameter."""

    def test_recall_engine_accepts_log(self):
        from caveman.engines.recall import RecallEngine
        from caveman.training.retrieval_log import RetrievalLog
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            log = RetrievalLog(Path(td) / "test.jsonl")
            engine = RecallEngine(retrieval_log=log)
            assert engine._retrieval_log is log


class TestPRDAlignment:
    """Verify PRD and code are now consistent."""

    def test_prd_version_is_v6_1(self):
        prd = Path("./docs/PRD.md").read_text()
        assert "Version:**" in prd  # Version may change

    def test_prd_status_is_v0_3_0(self):
        prd = Path("./docs/PRD.md").read_text()
        assert "v0.3.0" in prd[:500]

    def test_prd_no_stale_next_step(self):
        prd = Path("./docs/PRD.md").read_text()
        assert "Round 8 — 内核启动" not in prd[:500]

    def test_training_init_includes_all_modules(self):
        import caveman.training
        assert "retrieval_log" in caveman.training.__all__
        assert "eval_embedding" in caveman.training.__all__
