"""Acceptance tests for embedding eval-only and auto-selection gates (#28/#29)."""
from __future__ import annotations

import json
import inspect


def test_eval_result_quality_score_and_dict():
    from caveman.training.eval_embedding import EvalResult

    result = EvalResult(
        recall_at_5=0.5,
        recall_at_10=0.7,
        mrr=0.4,
        hit_rate_at_5=0.8,
        total_queries=3,
        model_path="m",
    )
    assert result.quality_score > 0
    data = result.to_dict()
    assert data["quality_score"] == result.quality_score
    assert data["model_path"] == "m"


def test_evaluate_logged_results_and_improvement_gate(tmp_path):
    from caveman.training.eval_embedding import EmbeddingEvaluator, EvalResult
    from caveman.training.retrieval_log import RetrievalEntry, RetrievalLog

    log_path = tmp_path / "retrieval.sqlite"
    log = RetrievalLog(log_path)
    log.log(RetrievalEntry(
        query="server IP",
        results=[
            {"memory_id": "m1", "content": "198.51.100.20 Ubuntu", "score": 0.9},
            {"memory_id": "m2", "content": "unrelated", "score": 0.2},
        ],
        source="recall",
    ))
    log.mark_adopted("server IP", ["m1"])

    evaluator = EmbeddingEvaluator(log_path)
    logged = evaluator.evaluate_logged_results()
    assert logged.total_queries == 1
    assert logged.recall_at_5 == 1.0
    assert logged.mrr == 1.0

    before = EvalResult(recall_at_5=0.2, recall_at_10=0.2, mrr=0.2, hit_rate_at_5=0.0, total_queries=2)
    after = EvalResult(recall_at_5=0.6, recall_at_10=0.6, mrr=0.5, hit_rate_at_5=1.0, total_queries=2)
    selected, reason = evaluator.improvement_decision(before, after, min_delta=0.01)
    assert selected is True
    assert "improved" in reason

    out = tmp_path / "selected_embedding.json"
    assert evaluator.write_selection("/model/path", before, after, out, min_delta=0.01) is True
    data = json.loads(out.read_text())
    assert data["selected_model_path"] == "/model/path"
    assert data["after"]["quality_score"] > data["before"]["quality_score"]


def test_eval_only_returns_metrics_without_training(monkeypatch):
    from caveman.training.cli_handler import run_train
    from caveman.training.eval_embedding import EmbeddingEvaluator, EvalResult

    monkeypatch.setattr(
        EmbeddingEvaluator,
        "evaluate_logged_results",
        lambda self, model_path="logged-baseline": EvalResult(
            recall_at_5=0.4, recall_at_10=0.5, mrr=0.3,
            hit_rate_at_5=0.6, total_queries=5, model_path=model_path,
        ),
    )

    result = run_train(
        target="embedding", model="", trajectory_dir=None, output_dir=None,
        min_quality=0.5, epochs=1, format="sharegpt", dry_run=False,
        eval_only=True,
    )
    assert "eval-only" in result
    assert "Recall@5=0.400" in result


def test_train_cli_exposes_eval_only_and_auto_select():
    import caveman.cli.main as cli

    sig = inspect.signature(cli.train)
    assert "eval_only" in sig.parameters
    assert "auto_select" in sig.parameters
    assert "min_eval_delta" in sig.parameters


def test_pyproject_embedding_train_extra_declared():
    text = open("pyproject.toml", encoding="utf-8").read()
    assert "embedding-train" in text
    assert "sentence-transformers" in text
    assert "torch" in text
