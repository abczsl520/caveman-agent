"""Safe stdio wrapper — prevents encoding errors from crashing the agent.

Ported from Hermes _SafeWriter (MIT, Nous Research).

When running under nohup, launchd, or systemd, stdout/stderr may be
redirected to files with restricted encodings. A single non-encodable
character in a log message would crash the entire agent.
"""
from __future__ import annotations
import sys
from typing import Any, TextIO, cast


class _SafeWriter:
    """Wraps a file-like object to swallow write/flush errors."""

    def __init__(self, inner: TextIO) -> None:
        object.__setattr__(self, "_inner_stream", inner)

    @property
    def _inner(self) -> TextIO:
        return cast(TextIO, object.__getattribute__(self, "_inner_stream"))

    def write(self, data: str) -> int:
        try:
            return self._inner.write(data)
        except (OSError, ValueError, UnicodeEncodeError):
            return len(data) if isinstance(data, str) else 0

    def flush(self) -> None:
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass  # intentional: OSError/ValueError suppressed

    def fileno(self) -> int:
        return self._inner.fileno()

    def isatty(self) -> bool:
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_safe_stdio() -> None:
    """Wrap stdout/stderr so console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))
