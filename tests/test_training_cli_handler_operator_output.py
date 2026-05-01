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
