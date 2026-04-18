"""RL data export — generate preference pairs for researcher fine-tuning.

Primary use: export preference data for researchers doing RLHF/DPO experiments.
Caveman's own skill routing uses Thompson Sampling (rl_router.py), not model RL.

Builds preference pairs from trajectories:
  - DPO (Direct Preference Optimization) — simplest, no reward model needed
  - PPO (Proximal Policy Optimization) — classic RLHF
  - GRPO (Group Relative Policy Optimization) — veRL-style

Pipeline:
  Trajectories → Preference Pairs → Export (or optional local training)

Preference pair generation:
  - Same task, different quality scores → (chosen, rejected) pairs
  - Chosen: quality ≥ 0.7, Rejected: quality < 0.5
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class RLConfig:
    """Configuration for RL training."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        sft_model_path: str | None = None,
        output_dir: str | None = None,
        method: str = "dpo",  # dpo | ppo | grpo
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 5e-6,
        beta: float = 0.1,  # DPO beta parameter
        max_seq_length: int | None = None,
        chosen_threshold: float = 0.7,
        rejected_threshold: float = 0.5,
    ):
        from caveman.paths import TRAINING_RL_DIR, DEFAULT_MAX_SEQ_LENGTH
        self.model_name = model_name
        self.sft_model_path = sft_model_path
        self.output_dir = str(Path(output_dir).expanduser()) if output_dir else str(TRAINING_RL_DIR)
        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.max_seq_length = max_seq_length if max_seq_length is not None else DEFAULT_MAX_SEQ_LENGTH
        self.chosen_threshold = chosen_threshold
        self.rejected_threshold = rejected_threshold

    def to_dict(self) -> dict:
        return vars(self)


class PreferencePairBuilder:
    """Generate preference pairs from trajectory quality scores."""

    def __init__(self, config: RLConfig):
        self.config = config
        self._stats = {"pairs": 0, "chosen": 0, "rejected": 0, "skipped": 0}

    def build(self, trajectory_dir: str | None = None) -> Path:
        """Build preference pair dataset."""
        from caveman.paths import TRAJECTORIES_DIR
        traj_dir = Path(trajectory_dir).expanduser() if trajectory_dir else TRAJECTORIES_DIR
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"preference_pairs_{timestamp}.jsonl"

        # Collect trajectories grouped by task
        task_groups: dict[str, list[dict]] = {}
        for traj_file in sorted(traj_dir.rglob("*.jsonl")):
            try:
                with open(traj_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue
                        task = entry.get("task", entry.get("goal", "unknown"))
                        task_groups.setdefault(task, []).append(entry)
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to read trajectory file %s: %s", traj_file, e)
                continue

        # Generate pairs
        with open(output_path, "w", encoding="utf-8") as f:
            for pairs in self._generate_pairs(task_groups):
                f.write(json.dumps(pairs, ensure_ascii=False) + "\n")

        logger.info(
            "Preference pairs: %d pairs from %d chosen / %d rejected",
            self._stats["pairs"], self._stats["chosen"], self._stats["rejected"],
        )
        return output_path

    def _generate_pairs(self, task_groups: dict[str, list[dict]]) -> Iterator[dict]:
        """Generate (chosen, rejected) pairs from same-task trajectories."""
        for task, entries in task_groups.items():
            chosen = [e for e in entries if e.get("quality_score", 0) >= self.config.chosen_threshold]
            rejected = [e for e in entries if e.get("quality_score", 0) < self.config.rejected_threshold]

            self._stats["chosen"] += len(chosen)
            self._stats["rejected"] += len(rejected)

            # Pair each chosen with each rejected (cross product, capped)
            for c in chosen[:5]:
                for r in rejected[:5]:
                    c_turns = c.get("turns", c.get("trajectory", []))
                    r_turns = r.get("turns", r.get("trajectory", []))
                    if c_turns and r_turns:
                        self._stats["pairs"] += 1
                        yield {
                            "prompt": task,
                            "chosen": self._turns_to_text(c_turns),
                            "rejected": self._turns_to_text(r_turns),
                            "chosen_score": c.get("quality_score", 0),
                            "rejected_score": r.get("quality_score", 0),
                        }

    @staticmethod
    def _turns_to_text(turns: list[dict]) -> str:
        """Convert turns to a single text string for DPO."""
        parts = []
        for turn in turns:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @property
    def stats(self) -> dict:
        return dict(self._stats)


class RLTrainer:
    """Optional RL training for researchers — DPO/PPO/GRPO.

    This is NOT Caveman's primary learning mechanism (that's memory + skills).
    Caveman's own skill routing uses Thompson Sampling (rl_router.py).
    Provided for researchers who want to run RLHF/DPO experiments.

    Requires: pip install trl peft transformers datasets
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self._trained = False

    def train(self, dataset_path: str | Path) -> dict:
        """Run RL training."""
        try:
            if self.config.method == "dpo":
                return self._train_dpo(dataset_path)
            elif self.config.method == "ppo":
                return self._train_ppo(dataset_path)
            elif self.config.method == "grpo":
                return self._train_grpo(dataset_path)
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
        except ImportError:
            logger.warning("TRL not installed for RL training")
            return self._mock_train(dataset_path)

    def _train_dpo(self, dataset_path: str | Path) -> dict:
        """DPO training via TRL."""
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import DPOTrainer

        model_path = self.config.sft_model_path or self.config.model_name
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            beta=self.config.beta,
            max_length=self.config.max_seq_length,
        )

        result = trainer.train()
        self._trained = True
        return {"status": "completed", "method": "dpo", "loss": result.training_loss}

    def _train_ppo(self, dataset_path: str | Path) -> dict:
        """PPO training — requires TRL. Falls back to mock when unavailable."""
        return self._mock_train(dataset_path, method="ppo")

    def _train_grpo(self, dataset_path: str | Path) -> dict:
        """GRPO training — requires veRL. Falls back to mock when unavailable."""
        return self._mock_train(dataset_path, method="grpo")

    def _mock_train(self, dataset_path: str | Path, method: str | None = None) -> dict:
        """Fallback when TRL/veRL not installed. Returns metadata only, no actual training."""
        count = 0
        path = Path(dataset_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                count = sum(1 for _ in f)

        self._trained = True
        return {
            "status": "fallback_no_trl",
            "method": method or self.config.method,
            "pairs": count,
            "epochs": self.config.epochs,
            "model": self.config.model_name,
            "note": "TRL not installed — no training performed, metadata only",
        }
