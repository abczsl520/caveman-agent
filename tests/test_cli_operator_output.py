from caveman.cli import main as cli_main
import caveman.paths as caveman_paths


def test_setup_escapes_detected_config_source_path_and_model(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    home_path = tmp_path / "home"
    model = "safe-model\nSPOOF_MODEL\x1b[31m"
    source_path = "/tmp/safe-config\nSPOOF_PATH\x1b[32m"

    monkeypatch.setattr(caveman_paths, "CONFIG_PATH", config_path)
    monkeypatch.setattr(caveman_paths, "CAVEMAN_HOME", home_path)
    monkeypatch.setattr(
        cli_main,
        "_detect_external_configs",
        lambda: {
            "Hermes\nSPOOF_SOURCE\x1b[33m": {
                "path": source_path,
                "api_key": "sk-redacted-test",
                "model": model,
            }
        },
    )
    answers = iter([False])
    monkeypatch.setattr(cli_main.typer, "confirm", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(cli_main.typer, "prompt", lambda *args, default="", **kwargs: default)

    cli_main.setup()

    output = capsys.readouterr().out
    assert "Hermes\\nSPOOF_SOURCE\\x1b[33m" in output
    assert source_path.encode("unicode_escape").decode() in output
    assert model.encode("unicode_escape").decode() in output
    assert str(config_path) in output
    assert "\nSPOOF_SOURCE" not in output
    assert "\nSPOOF_PATH" not in output
    assert "\nSPOOF_MODEL" not in output
    assert "\x1b" not in output


def test_setup_redacts_detected_api_key(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    home_path = tmp_path / "home"
    imported_key = "sk-tes...alue"

    monkeypatch.setattr(caveman_paths, "CONFIG_PATH", config_path)
    monkeypatch.setattr(caveman_paths, "CAVEMAN_HOME", home_path)
    monkeypatch.setattr(
        cli_main,
        "_detect_external_configs",
        lambda: {"Hermes": {"path": "/tmp/config.yaml", "api_key": imported_key}},
    )
    answers = iter([False])
    monkeypatch.setattr(cli_main.typer, "confirm", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(cli_main.typer, "prompt", lambda *args, default="", **kwargs: default)

    cli_main.setup()

    output = capsys.readouterr().out
    assert "API key: [REDACTED]" in output
    assert imported_key not in output
    assert imported_key[:8] not in output
    assert imported_key[-4:] not in output
