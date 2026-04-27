"""Regression tests for gateway logging setup idempotency."""
from __future__ import annotations

import logging


def _reset_logger(root: logging.Logger):
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_setup_logging_is_idempotent_for_same_file(tmp_path):
    from caveman.logging_config import setup_logging

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    for handler in old_handlers:
        root.removeHandler(handler)

    log_file = tmp_path / "gateway.log"
    try:
        setup_logging(level="INFO", log_file=str(log_file))
        setup_logging(level="INFO", log_file=str(log_file))

        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_file)
        ]
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]

        assert len(file_handlers) == 1
        assert len(stream_handlers) == 1
    finally:
        _reset_logger(root)
        for handler in old_handlers:
            root.addHandler(handler)


def test_setup_logging_can_disable_console_when_file_is_already_stdout_target(tmp_path):
    from caveman.logging_config import setup_logging

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    for handler in old_handlers:
        root.removeHandler(handler)

    try:
        setup_logging(level="INFO", log_file=str(tmp_path / "gateway.log"), console=False)

        assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
        assert not any(type(h) is logging.StreamHandler for h in root.handlers)
    finally:
        _reset_logger(root)
        for handler in old_handlers:
            root.addHandler(handler)
