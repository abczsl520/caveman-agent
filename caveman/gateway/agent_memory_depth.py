"""Agent Memory Depth — transcript estimation, memory flush, compaction.

Supplements agent_memory.py with transcript token estimation,
memory flush to disk, and compaction triggers. Extracted from
OpenClaw agent-runner-memory.ts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "estimate_tokens_for_model",
    "estimate_transcript_tokens",
    "MODEL_CONTEXT_WINDOWS",
    "get_context_window",
    "should_compact",
    "MemoryFlushConfig",
    "flush_transcript",
    "load_transcript",
    "CompactionResult",
    "prepare_compaction",
]


logger = logging.getLogger("caveman.gateway.agent_memory_depth")


# ── Token Estimation ──

# Average chars per token by model family (canonical copy in caveman.utils)
_CHARS_PER_TOKEN = {
    "claude": 3.5,
    "gpt": 4.0,
    "gemini": 3.8,
    "deepseek": 3.2,
    "default": 3.7,
}


def estimate_tokens_for_model(text: str, model: str = "") -> int:
    """Estimate token count for text based on model family.

    Delegates to caveman.utils.estimate_tokens_for_model.
    """
    from caveman.utils import estimate_tokens_for_model as _canonical
    return _canonical(text, model)


def estimate_transcript_tokens(messages: List[Dict[str, Any]], model: str = "") -> int:
    """Estimate total tokens in a message transcript.

    Delegates to compression.utils.estimate_tokens for consistency.
    """
    from caveman.compression.utils import estimate_tokens as _est_msgs
    return _est_msgs(messages)


# ── Context Window Management ──

MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o4-mini": 200_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "deepseek-chat": 64_000,
}


def get_context_window(model: str) -> int:
    """Get context window size for a model."""
    for prefix, window in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return window
    return 128_000  # Safe default


def should_compact(
    messages: List[Dict[str, Any]],
    model: str = "",
    threshold: float = 0.75,
) -> bool:
    """Check if transcript should be compacted."""
    window = get_context_window(model)
    tokens = estimate_transcript_tokens(messages, model)
    return tokens > window * threshold


# ── Memory Flush ──

@dataclass
class MemoryFlushConfig:
    """Configuration for memory flush to disk."""
    base_dir: Path = field(default_factory=lambda: Path.home() / ".caveman" / "memory_flush")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    flush_interval: float = 300  # 5 minutes
    compress: bool = False


def flush_transcript(
    session_key: str,
    messages: List[Dict[str, Any]],
    config: Optional[MemoryFlushConfig] = None,
) -> Optional[Path]:
    """Flush transcript to disk for persistence."""
    config = config or MemoryFlushConfig()
    config.base_dir.mkdir(parents=True, exist_ok=True)

    safe_key = session_key.replace(":", "_").replace("/", "_")
    path = config.base_dir / f"{safe_key}.jsonl"

    try:
        with open(path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        if path.stat().st_size > config.max_file_size:
            logger.warning(
                "Transcript flush for %s exceeds max size (%d bytes)",
                session_key, path.stat().st_size,
            )

        return path
    except Exception as e:
        logger.error("Failed to flush transcript for %s: %s", session_key, e)
        return None


def load_transcript(
    session_key: str,
    config: Optional[MemoryFlushConfig] = None,
) -> List[Dict[str, Any]]:
    """Load transcript from disk."""
    config = config or MemoryFlushConfig()
    safe_key = session_key.replace(":", "_").replace("/", "_")
    path = config.base_dir / f"{safe_key}.jsonl"

    if not path.exists():
        return []

    messages = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
    except Exception as e:
        logger.error("Failed to load transcript for %s: %s", session_key, e)

    return messages


# ── Compaction ──

@dataclass
class CompactionResult:
    """Result of a transcript compaction."""
    original_messages: int
    compacted_messages: int
    original_tokens: int
    compacted_tokens: int
    summary: str = ""
    preserved_recent: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compacted_tokens / self.original_tokens


def prepare_compaction(
    messages: List[Dict[str, Any]],
    model: str = "",
    preserve_recent: int = 10,
) -> Dict[str, Any]:
    """Prepare messages for compaction by splitting into summary-able and preserved."""
    if len(messages) <= preserve_recent:
        return {
            "needs_compaction": False,
            "to_summarize": [],
            "to_preserve": messages,
        }

    to_summarize = messages[:-preserve_recent]
    to_preserve = messages[-preserve_recent:]

    return {
        "needs_compaction": True,
        "to_summarize": to_summarize,
        "to_preserve": to_preserve,
        "summary_tokens": estimate_transcript_tokens(to_summarize, model),
        "preserve_tokens": estimate_transcript_tokens(to_preserve, model),
    }
