"""Tests for training direction pivot — embedding training + data export."""
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestEmbeddingPairExtractor:
    """Test query-memory pair extraction from trajectories."""

    def make_trajectory(self, quality: float = 0.7) -> dict:
        return {
            "quality_score": quality,
            "conversations": [
                {"from": "human", "value": "What server should I deploy to?"},
                {"from": "gpt", "value": "Based on your setup, deploy to 198.51.100.20 (Ubuntu, Node v22)."},
                {"from": "human", "value": "How do I restart the service?"},
                {"from": "gpt", "value": "SSH in and run: pm2 restart app-name"},
            ],
        }

    def test_extract_pairs_from_trajectory(self):
        from caveman.training.embedding import PairExtractor
        extractor = PairExtractor(min_quality=0.5)
        pairs = extractor.extract_from_trajectory(self.make_trajectory())
        assert len(pairs) == 2
        assert "server" in pairs[0].query.lower()
        assert "198.51.100.20" in pairs[0].positive

    def test_skip_low_quality(self):
        from caveman.training.embedding import PairExtractor
        extractor = PairExtractor(min_quality=0.8)
        pairs = extractor.extract_from_trajectory(self.make_trajectory(quality=0.5))
        assert len(pairs) == 0

    def test_build_dataset(self, tmp_path):
        from caveman.training.embedding import PairExtractor, QueryMemoryPair
        extractor = PairExtractor()
        pairs = [
            QueryMemoryPair(query="deploy where", positive="198.51.100.20"),
            QueryMemoryPair(query="restart how", positive="pm2 restart"),
        ]
        output = tmp_path / "pairs.jsonl"
        extractor.build_dataset(pairs, output)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "query" in record
        assert "positive" in record

    def test_extract_from_directory(self, tmp_path):
        from caveman.training.embedding import PairExtractor
        traj = self.make_trajectory()
        (tmp_path / "t1.json").write_text(json.dumps(traj))
        extractor = PairExtractor(min_quality=0.5)
        pairs = extractor.extract_from_directory(tmp_path)
        assert len(pairs) >= 1


class TestEmbeddingTrainConfig:
    def test_default_config(self):
        from caveman.training.embedding import EmbeddingTrainConfig
        config = EmbeddingTrainConfig()
        assert "nomic" in config.base_model
        assert config.epochs == 3
        assert config.min_quality == 0.5

    def test_custom_config(self):
        from caveman.training.embedding import EmbeddingTrainConfig
        config = EmbeddingTrainConfig(
            base_model="BAAI/bge-small-en-v1.5",
            epochs=5,
            output_dir="/tmp/test_emb",
        )
        assert "bge" in config.base_model
        assert config.output_dir == "/tmp/test_emb"


class TestEmbeddingTrainer:
    def test_train_without_sentence_transformers(self, tmp_path):
        """Training gracefully skips when sentence-transformers not installed."""
        from caveman.training.embedding import EmbeddingTrainer, EmbeddingTrainConfig
        config = EmbeddingTrainConfig(output_dir=str(tmp_path))
        trainer = EmbeddingTrainer(config)
        # Create a minimal dataset
        dataset = tmp_path / "pairs.jsonl"
        dataset.write_text(json.dumps({"query": "test", "positive": "answer"}) + "\n")
        result = trainer.train(dataset)
        # Either succeeds (if installed) or returns skip status
        assert result["status"] in ("success", "skip")


class TestCLIHandler:
    def test_unknown_target(self):
        from caveman.training.cli_handler import run_train
        result = run_train(
            target="unknown", model="", trajectory_dir=None,
            output_dir=None, min_quality=0.5, epochs=1,
            format="sharegpt", dry_run=False,
        )
        assert "Unknown target" in result

    def test_embedding_no_data(self, tmp_path):
        from caveman.training.cli_handler import run_train
        result = run_train(
            target="embedding", model="",
            trajectory_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            min_quality=0.5, epochs=1,
            format="sharegpt", dry_run=False,
        )
        assert "No pairs" in result

    def test_embedding_dry_run(self, tmp_path):
        from caveman.training.cli_handler import run_train
        # Create a trajectory
        traj = {
            "quality_score": 0.8,
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris, a major European city."},
            ],
        }
        (tmp_path / "t.json").write_text(json.dumps(traj))
        result = run_train(
            target="embedding", model="",
            trajectory_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            min_quality=0.5, epochs=1,
            format="sharegpt", dry_run=True,
        )
        assert "Dry run" in result

    def test_sft_dry_run(self, tmp_path):
        from caveman.training.cli_handler import run_train
        result = run_train(
            target="sft", model="",
            trajectory_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            min_quality=0.5, epochs=1,
            format="sharegpt", dry_run=True,
        )
        # Should complete (empty dataset is ok for dry run)
        assert isinstance(result, str)


class TestPRDAlignment:
    """Verify code aligns with updated PRD direction."""

    def test_sft_docstring_mentions_export(self):
        import caveman.training.sft as sft_mod
        assert "export" in sft_mod.__doc__.lower()

    def test_rl_docstring_mentions_export(self):
        import caveman.training.rl as rl_mod
        assert "export" in rl_mod.__doc__.lower()

    def test_embedding_module_exists(self):
        from caveman.training.embedding import EmbeddingTrainConfig, PairExtractor, EmbeddingTrainer
        assert EmbeddingTrainConfig is not None
        assert PairExtractor is not None
        assert EmbeddingTrainer is not None

    def test_cli_default_target_is_embedding(self):
        """CLI train command should default to embedding, not sft."""
        import caveman.cli.main as cli
        # Check the train function's default
        import inspect
        sig = inspect.signature(cli.train)
        target_param = sig.parameters.get("target")
        assert target_param is not None
        assert target_param.default.default == "embedding"

    def test_training_init_includes_embedding(self):
        import caveman.training
        assert "embedding" in caveman.training.__all__
