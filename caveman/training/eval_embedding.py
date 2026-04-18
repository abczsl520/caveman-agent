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

    def __str__(self) -> str:
        return (
            f"Recall@5={self.recall_at_5:.3f} Recall@10={self.recall_at_10:.3f} "
            f"MRR={self.mrr:.3f} HitRate@5={self.hit_rate_at_5:.3f} "
            f"({self.total_queries} queries)"
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
        ]:
            delta = a - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {b:.3f} | {a:.3f} | {sign}{delta:.3f} |")

        return "\n".join(lines)
