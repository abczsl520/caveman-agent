"""Embedding model training — fine-tune local embedding for better memory retrieval.

Core training target for Caveman: make memory search understand user context.
Example: user says "that server" → embedding maps to the correct IP.

Pipeline:
  Trajectories → Extract query-memory pairs → Fine-tune embedding model

Supports:
  - sentence-transformers (nomic-embed-text, bge-small, etc.)
  - Ollama-compatible output
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTrainConfig:
    """Configuration for embedding fine-tuning."""

    base_model: str = "nomic-ai/nomic-embed-text-v1.5"
    output_dir: str | None = None
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    min_quality: float = 0.5  # lower threshold — embedding pairs are simpler
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.output_dir is None:
            from caveman.paths import TRAINING_DIR
            self.output_dir = str(TRAINING_DIR / "embedding_output")


@dataclass
class QueryMemoryPair:
    """A query-memory pair for embedding training."""

    query: str
    positive: str  # relevant memory content
    negative: str = ""  # irrelevant memory (hard negative)
    score: float = 1.0


class PairExtractor:
    """Extract query-memory pairs for embedding training.

    Primary source: retrieval log (actual search queries → actual memory results).
    Fallback source: conversation trajectories (less accurate but available earlier).
    """

    def __init__(self, min_quality: float = 0.5) -> None:
        self.min_quality = min_quality
        self._retrieval_pairs = 0
        self._trajectory_pairs = 0

    @property
    def stats(self) -> str:
        return f"retrieval_log: {self._retrieval_pairs}, trajectory_fallback: {self._trajectory_pairs}"

    def extract_from_retrieval_log(self, log_path: Path | None = None) -> list[QueryMemoryPair]:
        """Primary source: extract from retrieval log (query → actual memory match)."""
        from caveman.training.retrieval_log import RetrievalLog
        log = RetrievalLog(log_path) if log_path else RetrievalLog()
        raw_pairs = log.generate_training_pairs()
        pairs = []
        for p in raw_pairs:
            query = p.get("query", "")
            positive = p.get("positive", "")
            if len(query) > 5 and len(positive) > 10:
                pairs.append(QueryMemoryPair(
                    query=query[:512], positive=positive[:512],
                    score=1.0 if p.get("source") == "adopted" else 0.7,
                ))
        self._retrieval_pairs = len(pairs)
        return pairs

    def extract_from_trajectory(self, trajectory: dict) -> list[QueryMemoryPair]:
        """Fallback source: extract from conversation Q&A pairs.

        Less accurate than retrieval log — user Q&A ≠ query-memory pairs.
        Only used when retrieval log has insufficient data.
        """
        pairs: list[QueryMemoryPair] = []
        quality = trajectory.get("quality_score", 0)
        if quality < self.min_quality:
            return pairs

        conversations = trajectory.get("conversations", [])
        if not conversations:
            return pairs

        for i, turn in enumerate(conversations):
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role in ("human", "user") and content:
                if i + 1 < len(conversations):
                    next_turn = conversations[i + 1]
                    next_role = next_turn.get("from", next_turn.get("role", ""))
                    next_content = next_turn.get("value", next_turn.get("content", ""))

                    if next_role in ("gpt", "assistant") and next_content:
                        if len(content) > 10 and len(next_content) > 20:
                            pairs.append(QueryMemoryPair(
                                query=content[:512],
                                positive=next_content[:512],
                                score=quality * 0.5,  # lower weight — less reliable source
                            ))
        self._trajectory_pairs += len(pairs)
        return pairs

    def extract_from_directory(
        self, trajectory_dir: Path, limit: int = 0
    ) -> list[QueryMemoryPair]:
        """Extract pairs: retrieval log first, trajectory fallback second."""
        # Primary: retrieval log
        all_pairs = self.extract_from_retrieval_log()

        # Fallback: trajectories (only if retrieval log has < 50 pairs)
        if len(all_pairs) < 50:
            count = 0
            for fmt in ("*.json", "*.jsonl"):
                for f in sorted(trajectory_dir.rglob(fmt)):
                    if limit and count >= limit:
                        break
                    try:
                        if f.suffix == ".jsonl":
                            text = f.read_text(encoding="utf-8")
                            for line in text.splitlines():
                                if line.strip():
                                    traj = json.loads(line)
                                    all_pairs.extend(self.extract_from_trajectory(traj))
                        else:
                            text = f.read_text(encoding="utf-8")
                            traj = json.loads(text)
                            all_pairs.extend(self.extract_from_trajectory(traj))
                        count += 1
                    except (OSError, UnicodeDecodeError) as e:
                        logger.warning("Skip unreadable %s: %s", f, e)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skip %s: %s", f, e)
                if limit and count >= limit:
                    break

        logger.info(
            "Extracted %d pairs (%s)", len(all_pairs), self.stats
        )
        return all_pairs

    def build_dataset(self, pairs: list[QueryMemoryPair], output_path: Path) -> Path:
        """Write pairs as JSONL for sentence-transformers training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "query": pair.query,
                    "positive": pair.positive,
                }
                if pair.negative:
                    record["negative"] = pair.negative
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote %d pairs to %s", len(pairs), output_path)
        return output_path


class EmbeddingTrainer:
    """Fine-tune embedding model using sentence-transformers."""

    def __init__(self, config: EmbeddingTrainConfig | None = None) -> None:
        self.config = config or EmbeddingTrainConfig()

    def train(self, dataset_path: Path) -> dict[str, Any]:
        """Train embedding model. Requires sentence-transformers installed."""
        try:
            from sentence_transformers import SentenceTransformer, InputExample
            from sentence_transformers.losses import MultipleNegativesRankingLoss
            from torch.utils.data import DataLoader
        except ImportError:
            return {
                "status": "skip",
                "reason": "sentence-transformers not installed. Run: pip install sentence-transformers",
                "dataset": str(dataset_path),
            }

        # Load dataset
        examples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        examples.append(InputExample(
                            texts=[record["query"], record["positive"]]
                        ))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed JSONL line in %s: %s", dataset_path, e)
                        continue

        if not examples:
            return {"status": "error", "reason": "No training examples found"}

        model = SentenceTransformer(self.config.base_model)
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=self.config.batch_size)
        train_loss = MultipleNegativesRankingLoss(model)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=int(len(train_dataloader) * self.config.warmup_ratio),
            output_path=str(output_dir),
        )

        return {
            "status": "success",
            "model_path": str(output_dir),
            "examples": len(examples),
            "epochs": self.config.epochs,
            "base_model": self.config.base_model,
        }
