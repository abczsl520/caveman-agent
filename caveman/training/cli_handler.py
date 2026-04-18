"""CLI handler for `caveman train` — dispatches to embedding/sft/rl targets."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_train(
    target: str,
    model: str,
    trajectory_dir: str | None,
    output_dir: str | None,
    min_quality: float,
    epochs: int,
    format: str,
    dry_run: bool,
) -> str:
    """Execute training target. Returns status message."""
    if target == "embedding":
        return _run_embedding(model, trajectory_dir, output_dir, min_quality, epochs, dry_run)
    elif target == "sft":
        return _run_sft(model, trajectory_dir, output_dir, min_quality, epochs, format, dry_run)
    elif target in ("dpo", "ppo", "grpo"):
        return _run_rl(target, model, trajectory_dir, output_dir, epochs, dry_run)
    else:
        return f"❌ Unknown target: {target}. Use: embedding, sft, dpo, ppo, grpo"


def _run_embedding(
    model: str, trajectory_dir: str | None, output_dir: str | None,
    min_quality: float, epochs: int, dry_run: bool,
) -> str:
    from caveman.training.embedding import EmbeddingTrainConfig, PairExtractor, EmbeddingTrainer
    from caveman.paths import TRAJECTORIES_DIR

    traj_dir = Path(trajectory_dir).expanduser() if trajectory_dir else TRAJECTORIES_DIR
    emb_model = model or "nomic-ai/nomic-embed-text-v1.5"
    config = EmbeddingTrainConfig(
        base_model=emb_model, output_dir=output_dir,
        epochs=epochs, min_quality=min_quality,
    )

    extractor = PairExtractor(min_quality=min_quality)
    pairs = extractor.extract_from_directory(traj_dir)

    if not pairs:
        return "❌ No pairs found. Run more tasks to generate trajectories first."

    dataset_path = Path(config.output_dir) / "embedding_pairs.jsonl"
    extractor.build_dataset(pairs, dataset_path)

    if dry_run:
        return f"🏁 Dry run — {len(pairs)} pairs at {dataset_path}"

    trainer = EmbeddingTrainer(config)
    result = trainer.train(dataset_path)
    return f"✅ {result}"


def _run_sft(
    model: str, trajectory_dir: str | None, output_dir: str | None,
    min_quality: float, epochs: int, format: str, dry_run: bool,
) -> str:
    from caveman.training.sft import TrainingConfig, DatasetBuilder, SFTTrainer

    sft_model = model or "meta-llama/Llama-3.1-8B-Instruct"
    config = TrainingConfig(
        model_name=sft_model, output_dir=output_dir,
        epochs=epochs, min_quality=min_quality, format=format,
    )
    builder = DatasetBuilder(config)
    dataset_path = builder.build(trajectory_dir)

    if dry_run:
        return f"🏁 Dry run — {builder.stats} at {dataset_path}"

    trainer = SFTTrainer(config)
    result = trainer.train(dataset_path)
    return f"✅ {result}"


def _run_rl(
    method: str, model: str, trajectory_dir: str | None,
    output_dir: str | None, epochs: int, dry_run: bool,
) -> str:
    from caveman.training.rl import RLConfig, PreferencePairBuilder, RLTrainer

    rl_model = model or "meta-llama/Llama-3.1-8B-Instruct"
    config = RLConfig(
        model_name=rl_model, output_dir=output_dir,
        method=method, epochs=epochs,
    )
    builder = PreferencePairBuilder(config)
    dataset_path = builder.build(trajectory_dir)

    if dry_run:
        return f"🏁 Dry run — {builder.stats} at {dataset_path}"

    trainer = RLTrainer(config)
    result = trainer.train(dataset_path)
    return f"✅ {result}"
