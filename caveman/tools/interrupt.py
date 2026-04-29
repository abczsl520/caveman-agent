"""Per-thread interrupt signaling for concurrent tool execution.

Thread-scoped so interrupting one gateway session doesn't kill tools
in other sessions running in the same process.
"""
from __future__ import annotations

import threading

__all__ = [
    "set_interrupt",
    "is_interrupted",
    "clear_all",
]


_interrupted_threads: set[int] = set()
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None) -> None:
    """Set or clear interrupt for a specific thread."""
    tid = thread_id if thread_id is not None else threading.get_ident()
    with _lock:
        if active:
            _interrupted_threads.add(tid)
        else:
            _interrupted_threads.discard(tid)


def is_interrupted() -> bool:
    """Check if the current thread has been interrupted."""
    tid = threading.get_ident()
    with _lock:
        return tid in _interrupted_threads


def clear_all() -> None:
    """Clear all interrupt signals (for testing)."""
    with _lock:
        _interrupted_threads.clear()


# Legacy compat
_interrupt_event = threading.Event()
