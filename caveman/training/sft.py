"""SFT data export — convert trajectories to standard training formats.

Primary use: export data for researchers who want to fine-tune their own models.
Caveman's own "learning" happens through memory + skills, not model training.

Converts high-quality Caveman trajectories into training data for:
  - HuggingFace TRL (SFTTrainer)
  - OpenAI fine-tuning API
  - Axolotl / LLaMA-Factory

Pipeline:
  Trajectories → Filter (quality ≥ 0.7) → Transform (ShareGPT/ChatML) → Export/Train

Supports:
  - ShareGPT format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
  - ChatML format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
  - OpenAI JSONL: {"messages": [...]}
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for SFT training."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        output_dir: str | None = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        max_seq_length: int | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        min_quality: float = 0.7,
        format: str = "sharegpt",  # sharegpt | chatml | openai
    ):
        self.model_name = model_name
        from caveman.paths import TRAINING_SFT_DIR, DEFAULT_MAX_SEQ_LENGTH
        self.output_dir = str(Path(output_dir).expanduser()) if output_dir else str(TRAINING_SFT_DIR)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length if max_seq_length is not None else DEFAULT_MAX_SEQ_LENGTH
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.min_quality = min_quality
        self.format = format

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        import inspect
        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        return cls(**{k: v for k, v in d.items() if k in valid_params})


class DatasetBuilder:
    """Build training datasets from trajectory files."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._stats = {"total": 0, "filtered": 0, "kept": 0}

    def build(self, trajectory_dir: str | None = None) -> Path:
        """Build dataset from trajectory files.

        Returns path to the output JSONL file.
        """
        from caveman.paths import TRAJECTORIES_DIR
        traj_dir = Path(trajectory_dir).expanduser() if trajectory_dir else TRAJECTORIES_DIR
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"sft_dataset_{timestamp}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in self._iter_trajectories(traj_dir):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "Dataset built: %d kept / %d total (%d filtered) → %s",
            self._stats["kept"], self._stats["total"],
            self._stats["filtered"], output_path,
        )
        return output_path

    def _iter_trajectories(self, traj_dir: Path) -> Iterator[dict]:
        """Iterate trajectory files, filter by quality, transform format."""
        for traj_file in sorted(traj_dir.rglob("*.jsonl")):
            try:
                with open(traj_file, encoding="utf-8") as f:
                    for line in f:
                        self._stats["total"] += 1
                        try:
                            entry = json.loads(line.strip())
                        except json.JSONDecodeError:
                            self._stats["filtered"] += 1
                            continue

                        # Quality filter
                        quality = entry.get("quality_score", 0)
                        if quality < self.config.min_quality:
                            self._stats["filtered"] += 1
                            continue

                        # Transform to target format
                        turns = entry.get("turns", entry.get("trajectory", []))
                        if not turns:
                            self._stats["filtered"] += 1
                            continue

                        transformed = self._transform(turns)
                        if transformed:
                            self._stats["kept"] += 1
                            yield transformed
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to read trajectory file %s: %s", traj_file, e)
                continue

    def _transform(self, turns: list[dict]) -> dict | None:
        """Transform turns to target format."""
        if self.config.format == "sharegpt":
            return self._to_sharegpt(turns)
        elif self.config.format == "chatml":
            return self._to_chatml(turns)
        elif self.config.format == "openai":
            return self._to_openai(turns)
        return None

    @staticmethod
    def _to_sharegpt(turns: list[dict]) -> dict:
        """Convert to ShareGPT format."""
        conversations = []
        for turn in turns:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            if role in ("user", "human"):
                conversations.append({"from": "human", "value": content})
            elif role in ("assistant", "gpt"):
                conversations.append({"from": "gpt", "value": content})
            elif role == "system":
                conversations.append({"from": "system", "value": content})
            elif role in ("tool", "function_response"):
                conversations.append({"from": "function_response", "value": content})
        return {"conversations": conversations} if conversations else None

    @staticmethod
    def _to_chatml(turns: list[dict]) -> dict:
        """Convert to ChatML format."""
        messages = []
        for turn in turns:
            role = turn.get("role", "")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant"):
                role = "assistant"
            content = turn.get("content", turn.get("value", ""))
            messages.append({"role": role, "content": content})
        return {"messages": messages} if messages else None

    @staticmethod
    def _to_openai(turns: list[dict]) -> dict:
        """Convert to OpenAI fine-tuning JSONL format."""
        messages = []
        for turn in turns:
            role = turn.get("role", "")
            if role in ("human",):
                role = "user"
            elif role in ("gpt",):
                role = "assistant"
            content = turn.get("content", turn.get("value", ""))
            messages.append({"role": role, "content": content})
        return {"messages": messages} if messages else None

    @property
    def stats(self) -> dict:
        return dict(self._stats)


class SFTTrainer:
    """Optional SFT training for researchers — wraps HuggingFace TRL SFTTrainer.

    This is NOT Caveman's primary learning mechanism (that's memory + skills).
    Provided for researchers who want to fine-tune models on exported trajectories.

    Requires: pip install trl peft transformers datasets
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._trained = False

    def train(self, dataset_path: str | Path) -> dict:
        """Run SFT training.

        Returns training metrics.
        """
        try:
            return self._train_trl(dataset_path)
        except ImportError:
            logger.warning("TRL not installed. Run: pip install trl peft transformers datasets")
            return self._mock_train(dataset_path)

    def _train_trl(self, dataset_path: str | Path) -> dict:
        """Real TRL training."""
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig
        from trl import SFTTrainer as TRLSFTTrainer

        # Load dataset
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        # Load model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype="auto", device_map="auto"
        )

        # LoRA config
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training args
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
        )

        # Train
        trainer = TRLSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_seq_length=self.config.max_seq_length,
        )

        result = trainer.train()
        self._trained = True

        return {
            "status": "completed",
            "loss": result.training_loss,
            "epochs": self.config.epochs,
            "model_path": self.config.output_dir,
        }

    def _mock_train(self, dataset_path: str | Path) -> dict:
        """Fallback when TRL not installed. Returns metadata only, no actual training."""
        # Count entries
        count = 0
        path = Path(dataset_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                count = sum(1 for _ in f)

        self._trained = True
        return {
            "status": "fallback_no_trl",
            "entries": count,
            "epochs": self.config.epochs,
            "model": self.config.model_name,
            "note": "TRL not installed — no training performed, metadata only",
        }
