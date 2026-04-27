"""Structured JSON log formatter for production deployments.

Usage:
  In config.yaml:
    logging:
      format: json  # or "text" (default)

  Or programmatically:
    from caveman.logging_config import setup_logging
    setup_logging(format="json")
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
            entry["error_type"] = type(record.exc_info[1]).__name__
        if hasattr(record, "session_id"):
            entry["session_id"] = record.session_id
        if hasattr(record, "tool_name"):
            entry["tool_name"] = record.tool_name
        if hasattr(record, "duration_ms"):
            entry["duration_ms"] = record.duration_ms
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    format: str = "text",
    log_file: str | None = None,
    console: bool = True,
) -> None:
    """Configure logging for the gateway.

    When a daemon supervisor already redirects stderr/stdout to the same log
    file, adding both a StreamHandler and FileHandler produces duplicate lines.
    Gateway service startup should pass ``console=False`` with ``log_file``.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    if format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
