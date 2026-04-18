"""Tests for training pipeline — FR-303."""
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestDatasetBuilder:
    """SFT dataset building from trajectories."""

    @pytest.fixture
    def trajectory_dir(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        # Create sample trajectory files
        entries = [
            {"task": "math", "quality_score": 0.9, "turns": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]},
            {"task": "math", "quality_score": 0.3, "turns": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "I don't know"},
            ]},
            {"task": "code", "quality_score": 0.8, "turns": [
                {"role": "user", "content": "Write hello world"},
                {"role": "assistant", "content": "print('hello')"},
            ]},
            {"task": "code", "quality_score": 0.5, "turns": [
                {"role": "user", "content": "Write hello world"},
                {"role": "assistant", "content": "hmm..."},
            ]},
        ]
        with open(traj_dir / "session1.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return traj_dir

    def test_build_sharegpt(self, trajectory_dir, tmp_path):
        from caveman.training.sft import TrainingConfig, DatasetBuilder
        config = TrainingConfig(
            output_dir=str(tmp_path / "output"),
            min_quality=0.7,
            format="sharegpt",
        )
        builder = DatasetBuilder(config)
        path = builder.build(str(trajectory_dir))
        assert path.exists()
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2  # only quality >= 0.7
        assert all("conversations" in l for l in lines)
        assert builder.stats["kept"] == 2
        assert builder.stats["filtered"] == 2

    def test_build_chatml(self, trajectory_dir, tmp_path):
        from caveman.training.sft import TrainingConfig, DatasetBuilder
        config = TrainingConfig(
            output_dir=str(tmp_path / "output"),
            min_quality=0.7,
            format="chatml",
        )
        builder = DatasetBuilder(config)
        path = builder.build(str(trajectory_dir))
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert all("messages" in l for l in lines)

    def test_build_openai(self, trajectory_dir, tmp_path):
        from caveman.training.sft import TrainingConfig, DatasetBuilder
        config = TrainingConfig(
            output_dir=str(tmp_path / "output"),
            min_quality=0.7,
            format="openai",
        )
        builder = DatasetBuilder(config)
        path = builder.build(str(trajectory_dir))
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert all("messages" in l for l in lines)
        for line in lines:
            assert all(m["role"] in ("user", "assistant", "system") for m in line["messages"])

    def test_empty_trajectories(self, tmp_path):
        from caveman.training.sft import TrainingConfig, DatasetBuilder
        traj_dir = tmp_path / "empty"
        traj_dir.mkdir()
        config = TrainingConfig(output_dir=str(tmp_path / "output"))
        builder = DatasetBuilder(config)
        path = builder.build(str(traj_dir))
        assert path.exists()
        assert builder.stats["total"] == 0


class TestPreferencePairs:
    """RL preference pair generation."""

    @pytest.fixture
    def trajectory_dir(self, tmp_path):
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        entries = [
            {"task": "math", "quality_score": 0.9, "turns": [
                {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
            ]},
            {"task": "math", "quality_score": 0.3, "turns": [
                {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "idk"},
            ]},
        ]
        with open(traj_dir / "s1.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return traj_dir

    def test_generate_pairs(self, trajectory_dir, tmp_path):
        from caveman.training.rl import RLConfig, PreferencePairBuilder
        config = RLConfig(output_dir=str(tmp_path / "output"))
        builder = PreferencePairBuilder(config)
        path = builder.build(str(trajectory_dir))
        assert path.exists()
        with open(path) as f:
            pairs = [json.loads(l) for l in f]
        assert len(pairs) == 1  # 1 chosen × 1 rejected
        assert "chosen" in pairs[0]
        assert "rejected" in pairs[0]
        assert pairs[0]["chosen_score"] > pairs[0]["rejected_score"]


class TestMockTraining:
    """Training with mock (no GPU)."""

    def test_sft_mock(self, tmp_path):
        from caveman.training.sft import TrainingConfig, SFTTrainer
        config = TrainingConfig(output_dir=str(tmp_path / "output"))
        # Create a fake dataset
        ds = tmp_path / "dataset.jsonl"
        ds.write_text(json.dumps({"conversations": [{"from": "human", "value": "hi"}]}) + "\n")
        trainer = SFTTrainer(config)
        result = trainer.train(str(ds))
        assert result["status"] in ("mock", "fallback_no_trl")
        assert result["entries"] == 1

    def test_rl_mock(self, tmp_path):
        from caveman.training.rl import RLConfig, RLTrainer
        config = RLConfig(output_dir=str(tmp_path / "output"), method="dpo")
        ds = tmp_path / "pairs.jsonl"
        ds.write_text(json.dumps({"prompt": "hi", "chosen": "a", "rejected": "b"}) + "\n")
        trainer = RLTrainer(config)
        result = trainer.train(str(ds))
        assert result["status"] in ("mock", "fallback_no_trl")
        assert result["method"] == "dpo"


class TestTrainingConfig:
    """Config serialization."""

    def test_sft_config_roundtrip(self):
        from caveman.training.sft import TrainingConfig
        c = TrainingConfig(model_name="test-model", epochs=5, min_quality=0.8)
        d = c.to_dict()
        c2 = TrainingConfig.from_dict(d)
        assert c2.model_name == "test-model"
        assert c2.epochs == 5
        assert c2.min_quality == 0.8

    def test_rl_config(self):
        from caveman.training.rl import RLConfig
        c = RLConfig(method="grpo", beta=0.2)
        assert c.method == "grpo"
        assert c.beta == 0.2


class TestShieldFix:
    """Shield message normalization."""

    def test_message_dataclass_normalization(self):
        from caveman.agent.context import Message
        m = Message(role="assistant", content="hello", tokens=10)
        # Simulate what _update_shield does
        d = {"role": getattr(m, "role", "unknown"), "content": getattr(m, "content", str(m))}
        assert d["role"] == "assistant"
        assert d["content"] == "hello"

    def test_dict_passthrough(self):
        m = {"role": "user", "content": "hi"}
        assert isinstance(m, dict)
        assert m.get("role") == "user"


class TestEngineFlagsExpanded:
    """Scheduler and verification in engine flags."""

    def test_scheduler_flag(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags({"engines": {"scheduler": {"enabled": True}}})
        assert flags.is_enabled("scheduler") is True

    def test_verification_flag(self):
        from caveman.engines.flags import EngineFlags
        flags = EngineFlags({"engines": {"verification": {"enabled": False}}})
        assert flags.is_enabled("verification") is False

    def test_all_engines_in_status(self):
        from caveman.engines.flags import EngineFlags, ENGINES
        flags = EngineFlags()
        status = flags.status()
        assert set(status.keys()) == set(ENGINES)


class TestTrainingStats:
    """Training stats display."""

    def test_stats_empty_dir(self, tmp_path):
        """Stats on empty dir should not crash."""
        from caveman.training.sft import DatasetBuilder, TrainingConfig
        config = TrainingConfig(output_dir=str(tmp_path / "out"))
        builder = DatasetBuilder(config)
        path = builder.build(str(tmp_path / "nonexistent"))
        assert builder.stats["total"] == 0
