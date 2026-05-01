"""CLI handler for `caveman train` — dispatches to embedding/sft/rl targets."""
from __future__ import annotations

from pathlib import Path

from caveman.operator_output import operator_literal


def run_train(
    target: str,
    model: str,
    trajectory_dir: str | None,
    output_dir: str | None,
    min_quality: float,
    epochs: int,
    format: str,
    dry_run: bool,
    eval_only: bool = False,
    auto_select: bool = False,
    min_eval_delta: float = 0.01,
) -> str:
    """Execute training target. Returns status message."""
    if target == "embedding":
        return _run_embedding(
            model,
            trajectory_dir,
            output_dir,
            min_quality,
            epochs,
            dry_run,
            eval_only=eval_only,
            auto_select=auto_select,
            min_eval_delta=min_eval_delta,
        )
    elif target == "sft":
        return _run_sft(model, trajectory_dir, output_dir, min_quality, epochs, format, dry_run)
    elif target in ("dpo", "ppo", "grpo"):
        return _run_rl(target, model, trajectory_dir, output_dir, epochs, dry_run)
    else:
        return f"❌ Unknown target: {operator_literal(target)}. Use: embedding, sft, dpo, ppo, grpo"


def _run_embedding(
    model: str, trajectory_dir: str | None, output_dir: str | None,
    min_quality: float, epochs: int, dry_run: bool,
    eval_only: bool = False,
    auto_select: bool = False,
    min_eval_delta: float = 0.01,
) -> str:
    from caveman.training.embedding import EmbeddingTrainConfig, PairExtractor, EmbeddingTrainer
    from caveman.training.eval_embedding import EmbeddingEvaluator
    from caveman.paths import TRAJECTORIES_DIR, TRAINING_DIR

    traj_dir = Path(trajectory_dir).expanduser() if trajectory_dir else TRAJECTORIES_DIR
    emb_model = model or "nomic-ai/nomic-embed-text-v1.5"
    config = EmbeddingTrainConfig(
        base_model=emb_model, output_dir=output_dir,
        epochs=epochs, min_quality=min_quality,
    )

    evaluator = EmbeddingEvaluator()
    baseline_eval = evaluator.evaluate_logged_results(model_path="logged-baseline")
    if eval_only:
        return f"📊 Embedding eval-only — {baseline_eval}"

    extractor = PairExtractor(min_quality=min_quality)
    pairs = extractor.extract_from_directory(traj_dir)

    if not pairs:
        return "❌ No pairs found. Run more tasks to generate trajectories first."

    dataset_path = Path(config.output_dir) / "embedding_pairs.jsonl"
    extractor.build_dataset(pairs, dataset_path)

    if dry_run:
        return f"🏁 Dry run — {len(pairs)} pairs at {operator_literal(dataset_path)}"

    trainer = EmbeddingTrainer(config)
    result = trainer.train(dataset_path)
    status = result.get("status") if isinstance(result, dict) else None
    message = f"✅ {result}"

    if auto_select:
        if status != "success":
            return message + "\n⚠️ Auto-select skipped: training did not produce a verified model."
        # In this non-interactive gate we do not claim improvement from training
        # completion alone. Re-evaluation must produce objective metrics better
        # than the logged baseline before persisting selection.
        after_eval = evaluator.evaluate_logged_results(
            model_path=str(result.get("model_path", config.output_dir))
        )
        selected_path = TRAINING_DIR / "selected_embedding.json"
        selected = evaluator.write_selection(
            result.get("model_path", config.output_dir),
            baseline_eval,
            after_eval,
            selected_path,
            min_delta=min_eval_delta,
        )
        report = evaluator.compare(baseline_eval, after_eval)
        if selected:
            return message + f"\n{report}\n✅ Auto-selected embedding model: {selected_path}"
        ok, reason = evaluator.improvement_decision(
            baseline_eval, after_eval, min_delta=min_eval_delta
        )
        return message + f"\n{report}\n⚠️ Auto-select not changed: {reason}"

    return message


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
        return f"🏁 Dry run — {operator_literal(builder.stats)} at {operator_literal(dataset_path)}"

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
        return f"🏁 Dry run — {operator_literal(builder.stats)} at {operator_literal(dataset_path)}"

    trainer = RLTrainer(config)
    result = trainer.train(dataset_path)
    return f"✅ {result}"
