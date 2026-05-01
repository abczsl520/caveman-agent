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
