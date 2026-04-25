"""Embedding evaluation — measure retrieval quality before/after training.

Metrics:
  - Recall@K: what fraction of relevant memories appear in top-K results
  - MRR (Mean Reciprocal Rank): average 1/rank of first relevant result
  - Hit Rate: fraction of queries where at least one relevant result is in top-K

Usage:
  caveman train --target embedding --eval-only
  or: python -m caveman.training.eval_embedding
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation metrics for an embedding model."""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    hit_rate_at_5: float = 0.0
    total_queries: int = 0
    model_path: str = ""

    @property
    def quality_score(self) -> float:
        """Single gate score used for A/B model selection."""
        if self.total_queries <= 0:
            return 0.0
        return (
            self.recall_at_5 * 0.35
            + self.recall_at_10 * 0.25
            + self.mrr * 0.25
            + self.hit_rate_at_5 * 0.15
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "hit_rate_at_5": self.hit_rate_at_5,
            "total_queries": self.total_queries,
            "model_path": self.model_path,
            "quality_score": self.quality_score,
        }

    def __str__(self) -> str:
        return (
            f"Recall@5={self.recall_at_5:.3f} Recall@10={self.recall_at_10:.3f} "
            f"MRR={self.mrr:.3f} HitRate@5={self.hit_rate_at_5:.3f} "
            f"Quality={self.quality_score:.3f} ({self.total_queries} queries)"
        )


class EmbeddingEvaluator:
    """Evaluate embedding model quality using retrieval log as ground truth."""

    def __init__(self, retrieval_log_path: Path | None = None) -> None:
        from caveman.training.retrieval_log import RetrievalLog
        self._log = RetrievalLog(retrieval_log_path) if retrieval_log_path else RetrievalLog()

    def build_eval_set(self) -> list[dict]:
        """Build evaluation set from retrieval log.

        Uses entries with adoption data as ground truth.
        Falls back to high-score results if no adoption data.
        """
        entries = self._log.read_all()
        eval_set = []

        # Index adoptions
        adoptions: dict[str, list[str]] = {}
        for e in entries:
            if e.source == "adoption" and e.adopted_ids:
                adoptions[e.query] = e.adopted_ids

        for entry in entries:
            if entry.source == "adoption" or not entry.results:
                continue

            relevant_ids = adoptions.get(entry.query, [])
            if not relevant_ids:
                # Fallback: treat top-scored results as relevant
                relevant_ids = [
                    r["memory_id"] for r in entry.results
                    if r.get("score", 0) >= 0.7 and r.get("memory_id")
                ]

            if relevant_ids:
                eval_set.append({
                    "query": entry.query,
                    "relevant_ids": relevant_ids,
                    "all_results": entry.results,
                })

        return eval_set

    def evaluate_logged_results(self, model_path: str = "logged-baseline") -> EvalResult:
        """Evaluate currently logged retrieval rankings without rerunning search.

        This gives `caveman train --target embedding --eval-only` a deterministic
        benchmark that works on any machine: adoption/high-score retrieval log
        entries become ground truth, and their stored ranking is scored with the
        same Recall@K/MRR/HitRate metrics used by live evaluation.
        """
        eval_set = self.build_eval_set()
        if not eval_set:
            return EvalResult(total_queries=0, model_path=model_path)

        recall5: list[float] = []
        recall10: list[float] = []
        mrrs: list[float] = []
        hits5: list[float] = []

        for item in eval_set:
            relevant = set(item["relevant_ids"])
            ranked_ids = [r.get("memory_id", "") for r in item.get("all_results", [])]
            if not relevant or not ranked_ids:
                continue

            top5 = set(ranked_ids[:5])
            top10 = set(ranked_ids[:10])
            recall5.append(len(relevant & top5) / len(relevant))
            recall10.append(len(relevant & top10) / len(relevant))
            hits5.append(1.0 if relevant & top5 else 0.0)

            rr = 0.0
            for rank, rid in enumerate(ranked_ids, 1):
                if rid in relevant:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

        n = len(mrrs)
        return EvalResult(
            recall_at_5=sum(recall5) / max(n, 1),
            recall_at_10=sum(recall10) / max(n, 1),
            mrr=sum(mrrs) / max(n, 1),
            hit_rate_at_5=sum(hits5) / max(n, 1),
            total_queries=n,
            model_path=model_path,
        )

    async def evaluate(
        self,
        embedding_fn: Any,
        memory_manager: Any,
        k_values: tuple[int, ...] = (5, 10),
    ) -> EvalResult:
        """Run evaluation: for each query, search with the embedding fn and measure metrics."""
        eval_set = self.build_eval_set()
        if not eval_set:
            logger.warning("No evaluation data available (need retrieval log with results)")
            return EvalResult(total_queries=0)

        recalls = {k: [] for k in k_values}
        mrrs: list[float] = []
        hits_at_5: list[float] = []

        for item in eval_set:
            query = item["query"]
            relevant = set(item["relevant_ids"])

            try:
                results = await memory_manager.recall(query, top_k=max(k_values))
                result_ids = [getattr(r, "id", "") for r in results]
            except Exception as e:
                logger.warning("Eval search failed for '%s': %s", query[:50], e)
                continue

            # Recall@K
            for k in k_values:
                top_k_ids = set(result_ids[:k])
                recall = len(relevant & top_k_ids) / len(relevant) if relevant else 0
                recalls[k].append(recall)

            # MRR
            rr = 0.0
            for rank, rid in enumerate(result_ids, 1):
                if rid in relevant:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            # Hit Rate@5
            top_5 = set(result_ids[:5])
            hits_at_5.append(1.0 if relevant & top_5 else 0.0)

        n = len(mrrs)
        return EvalResult(
            recall_at_5=sum(recalls.get(5, [0])) / max(n, 1),
            recall_at_10=sum(recalls.get(10, [0])) / max(n, 1),
            mrr=sum(mrrs) / max(n, 1),
            hit_rate_at_5=sum(hits_at_5) / max(n, 1),
            total_queries=n,
        )

    def improvement_decision(
        self,
        before: EvalResult,
        after: EvalResult,
        min_delta: float = 0.01,
        min_queries: int = 1,
    ) -> tuple[bool, str]:
        """Return whether `after` should replace `before`.

        The gate deliberately uses objective metrics only. A model is selected
        only when both sides have enough queries and the weighted quality score
        improves by at least `min_delta`.
        """
        if before.total_queries < min_queries or after.total_queries < min_queries:
            return False, f"insufficient eval queries: before={before.total_queries}, after={after.total_queries}, required={min_queries}"
        delta = after.quality_score - before.quality_score
        if delta >= min_delta:
            return True, f"quality improved by {delta:.3f} >= {min_delta:.3f}"
        return False, f"quality delta {delta:.3f} < {min_delta:.3f}"

    def write_selection(
        self,
        model_path: str | Path,
        before: EvalResult,
        after: EvalResult,
        output_path: Path,
        min_delta: float = 0.01,
    ) -> bool:
        """Persist selected embedding model only when the A/B gate passes."""
        import json
        from datetime import datetime

        selected, reason = self.improvement_decision(before, after, min_delta=min_delta)
        if not selected:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "selected_model_path": str(model_path),
                    "selected_at": datetime.now().isoformat(),
                    "reason": reason,
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                    "min_delta": min_delta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return True

    def compare(self, before: EvalResult, after: EvalResult) -> str:
        """Generate a comparison report between two eval results."""
        lines = ["## Embedding Training Evaluation"]
        lines.append(f"Queries: {before.total_queries} → {after.total_queries}")
        lines.append("")
        lines.append("| Metric | Before | After | Δ |")
        lines.append("|--------|--------|-------|---|")

        for name, b, a in [
            ("Recall@5", before.recall_at_5, after.recall_at_5),
            ("Recall@10", before.recall_at_10, after.recall_at_10),
            ("MRR", before.mrr, after.mrr),
            ("HitRate@5", before.hit_rate_at_5, after.hit_rate_at_5),
            ("Quality", before.quality_score, after.quality_score),
        ]:
            delta = a - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {b:.3f} | {a:.3f} | {sign}{delta:.3f} |")

        return "\n".join(lines)
