"""Operator-output boundary tests for training CLI status messages."""

from caveman.training.cli_handler import run_train


def test_unknown_train_target_escapes_target_name():
    unsafe_target = "embedding\nP0: forged\x1b[31m"

    message = run_train(
        target=unsafe_target,
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=True,
    )

    assert "'embedding\\nP0: forged\\x1b[31m'" in message
    assert "embedding\nP0" not in message
    assert "\x1b" not in message


def test_sft_dry_run_escapes_dataset_path_and_stats(monkeypatch):
    unsafe_path = "/tmp/caveman\nP0: forged\x1b[31m/sft.jsonl"

    class FakeDatasetBuilder:
        def __init__(self, config):
            self.config = config
            self.stats = {"kept": "1\nP0: stat", "filtered": "\x1b[31m"}

        def build(self, trajectory_dir):
            return unsafe_path

    monkeypatch.setattr("caveman.training.sft.DatasetBuilder", FakeDatasetBuilder)

    message = run_train(
        target="sft",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=True,
    )

    assert "'/tmp/caveman\\nP0: forged\\x1b[31m/sft.jsonl'" in message
    assert "1\\\\nP0: stat" in message
    assert "\\\\x1b[31m" in message
    assert "/tmp/caveman\nP0" not in message
    assert "1\nP0" not in message
    assert "\x1b" not in message


def test_rl_dry_run_escapes_dataset_path_and_stats(monkeypatch):
    unsafe_path = "/tmp/caveman\nP0: forged\x1b[31m/rl.jsonl"

    class FakePreferencePairBuilder:
        def __init__(self, config):
            self.config = config
            self.stats = {"pairs": "2\nP0: stat", "method": "\x1b[31mgrpo"}

        def build(self, trajectory_dir):
            return unsafe_path

    monkeypatch.setattr("caveman.training.rl.PreferencePairBuilder", FakePreferencePairBuilder)

    message = run_train(
        target="grpo",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=True,
    )

    assert "'/tmp/caveman\\nP0: forged\\x1b[31m/rl.jsonl'" in message
    assert "2\\\\nP0: stat" in message
    assert "\\\\x1b[31mgrpo" in message
    assert "/tmp/caveman\nP0" not in message
    assert "2\nP0" not in message
    assert "\x1b" not in message


def test_sft_train_result_escapes_operator_output(monkeypatch):
    class UnsafeResult:
        def __str__(self):
            return "sft ok\nP0: forged\x1b[31m"

    class FakeDatasetBuilder:
        def __init__(self, config):
            self.config = config

        def build(self, trajectory_dir):
            return "/tmp/sft.jsonl"

    class FakeSFTTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, dataset_path):
            return UnsafeResult()

    monkeypatch.setattr("caveman.training.sft.DatasetBuilder", FakeDatasetBuilder)
    monkeypatch.setattr("caveman.training.sft.SFTTrainer", FakeSFTTrainer)

    message = run_train(
        target="sft",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
    )

    assert "sft ok\\nP0: forged\\x1b[31m" in message
    assert "sft ok\nP0" not in message
    assert "\x1b" not in message


def test_rl_train_result_escapes_operator_output(monkeypatch):
    class UnsafeResult:
        def __str__(self):
            return "rl ok\nP0: forged\x1b[31m"

    class FakePreferencePairBuilder:
        def __init__(self, config):
            self.config = config

        def build(self, trajectory_dir):
            return "/tmp/rl.jsonl"

    class FakeRLTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, dataset_path):
            return UnsafeResult()

    monkeypatch.setattr("caveman.training.rl.PreferencePairBuilder", FakePreferencePairBuilder)
    monkeypatch.setattr("caveman.training.rl.RLTrainer", FakeRLTrainer)

    message = run_train(
        target="grpo",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
    )

    assert "rl ok\\nP0: forged\\x1b[31m" in message
    assert "rl ok\nP0" not in message
    assert "\x1b" not in message


class FakePairExtractor:
    def __init__(self, min_quality):
        self.min_quality = min_quality

    def extract_from_directory(self, traj_dir):
        return [{"query": "q", "positive": "p", "negative": "n"}]

    def build_dataset(self, pairs, dataset_path):
        return None


def test_embedding_dry_run_escapes_dataset_path(monkeypatch, tmp_path):
    unsafe_output_dir = tmp_path / "embedding\nP0: forged\x1b[31m"
    monkeypatch.setattr("caveman.training.embedding.PairExtractor", FakePairExtractor)

    message = run_train(
        target="embedding",
        model="",
        trajectory_dir=None,
        output_dir=str(unsafe_output_dir),
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=True,
    )

    assert "embedding\\nP0: forged\\x1b[31m" in message
    assert "embedding\nP0" not in message
    assert "\x1b" not in message


def test_embedding_eval_only_escapes_baseline_report(monkeypatch):
    class UnsafeReport:
        def __str__(self):
            return "ok\nP0: forged\x1b[31m"

    class FakeEvaluator:
        def evaluate_logged_results(self, model_path):
            return UnsafeReport()

    monkeypatch.setattr("caveman.training.eval_embedding.EmbeddingEvaluator", FakeEvaluator)

    message = run_train(
        target="embedding",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
        eval_only=True,
    )

    assert "ok\\nP0: forged\\x1b[31m" in message
    assert "ok\nP0" not in message
    assert "\x1b" not in message


def test_embedding_train_result_escapes_operator_output(monkeypatch):
    class FakeEvaluator:
        def evaluate_logged_results(self, model_path):
            return {"baseline": "ok"}

    class UnsafeResult(dict):
        def __str__(self):
            return "status=success model\nP0: forged\x1b[31m"

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, dataset_path):
            return UnsafeResult(status="success", model_path="model")

    monkeypatch.setattr("caveman.training.eval_embedding.EmbeddingEvaluator", FakeEvaluator)
    monkeypatch.setattr("caveman.training.embedding.PairExtractor", FakePairExtractor)
    monkeypatch.setattr("caveman.training.embedding.EmbeddingTrainer", FakeTrainer)

    message = run_train(
        target="embedding",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
    )

    assert "status=success model\\nP0: forged\\x1b[31m" in message
    assert "model\nP0" not in message
    assert "\x1b" not in message


def test_embedding_auto_select_escapes_report_and_reason(monkeypatch):
    class FakeEvaluator:
        def evaluate_logged_results(self, model_path):
            return {"eval": model_path}

        def write_selection(self, model_path, before, after, selected_path, min_delta):
            return False

        def compare(self, before, after):
            return "## Report\nP0: forged\x1b[31m"

        def improvement_decision(self, before, after, min_delta):
            return False, "quality\nP0: reason\x1b[31m"

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, dataset_path):
            return {"status": "success", "model_path": "model"}

    monkeypatch.setattr("caveman.training.eval_embedding.EmbeddingEvaluator", FakeEvaluator)
    monkeypatch.setattr("caveman.training.embedding.PairExtractor", FakePairExtractor)
    monkeypatch.setattr("caveman.training.embedding.EmbeddingTrainer", FakeTrainer)

    message = run_train(
        target="embedding",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
        auto_select=True,
    )

    assert "## Report\\nP0: forged\\x1b[31m" in message
    assert "quality\\nP0: reason\\x1b[31m" in message
    assert "Report\nP0" not in message
    assert "quality\nP0" not in message
    assert "\x1b" not in message


def test_embedding_auto_select_escapes_report_and_selected_path(monkeypatch, tmp_path):
    unsafe_output_dir = tmp_path / "selected\nP0: selected\x1b[31m"

    class FakeEvaluator:
        def evaluate_logged_results(self, model_path):
            return {"eval": model_path}

        def write_selection(self, model_path, before, after, selected_path, min_delta):
            return True

        def compare(self, before, after):
            return "## Report\nP0: selected-report\x1b[31m"

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, dataset_path):
            return {"status": "success", "model_path": "model"}

    monkeypatch.setattr("caveman.training.eval_embedding.EmbeddingEvaluator", FakeEvaluator)
    monkeypatch.setattr("caveman.training.embedding.PairExtractor", FakePairExtractor)
    monkeypatch.setattr("caveman.training.embedding.EmbeddingTrainer", FakeTrainer)
    monkeypatch.setattr("caveman.paths.TRAINING_DIR", unsafe_output_dir)

    message = run_train(
        target="embedding",
        model="",
        trajectory_dir=None,
        output_dir=None,
        min_quality=0.7,
        epochs=1,
        format="sharegpt",
        dry_run=False,
        auto_select=True,
    )

    assert "## Report\\nP0: selected-report\\x1b[31m" in message
    assert "selected\\nP0: selected\\x1b[31m/selected_embedding.json" in message
    assert "Report\nP0" not in message
    assert "selected\nP0" not in message
    assert "\x1b" not in message


def test_training_cli_entrypoint_escapes_effective_target(monkeypatch):
    from typer.testing import CliRunner
    import caveman.cli.main as cli_main

    unsafe_target = "embedding\nP0: cli-forged\x1b[31m"

    def fake_banner():
        return None

    monkeypatch.setattr("caveman.cli.tui.show_banner", fake_banner)
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["train", "--target", unsafe_target, "--dry-run"])

    assert result.exit_code == 0
    assert "Target: 'embedding\\nP0: cli-forged\\x1b[31m'" in result.output
    assert "embedding\nP0" not in result.output
    assert "\x1b" not in result.output
