"""Status Panel Depth — token stats, model info, session list, cost tracking.

Supplements status_panel.py with detailed token/cost tracking and
session listing. Extracted from OpenClaw status.ts (930 lines).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

__all__ = [
    "TokenStats",
    "ModelInfo",
    "get_model_info",
    "SessionListEntry",
    "format_session_list",
    "format_token_stats",
    "format_model_info",
]



@dataclass
class TokenStats:
    """Detailed token statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0
    api_calls: int = 0
    avg_latency_ms: float = 0
    compaction_count: int = 0
    compaction_tokens_saved: int = 0

    def update(self, prompt: int, completion: int, latency_ms: float = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.api_calls += 1
        if latency_ms > 0:
            # Running average
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.api_calls - 1) + latency_ms) / self.api_calls
            )

    def estimate_cost(self, model: str = "") -> float:
        """Estimate cost based on model pricing."""
        PRICING = {
            "claude-opus-4-6": (15.0, 75.0),
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-3-5-haiku-20241022": (0.80, 4.0),
            "gpt-4o": (2.50, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "o4-mini": (1.10, 4.40),
            "gemini-2.5-pro": (1.25, 10.0),
            "gemini-2.5-flash": (0.15, 0.60),
            "deepseek-chat": (0.14, 0.28),
        }
        for prefix, (input_price, output_price) in PRICING.items():
            if model.startswith(prefix):
                self.cost_usd = (
                    (self.prompt_tokens / 1_000_000) * input_price
                    + (self.completion_tokens / 1_000_000) * output_price
                )
                return self.cost_usd
        return 0


@dataclass
class ModelInfo:
    """Information about the current model."""
    provider: str = ""
    model: str = ""
    context_window: int = 0
    max_output: int = 0
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True


MODEL_INFO_DB: Dict[str, ModelInfo] = {
    "claude-opus-4-6": ModelInfo("anthropic", "claude-opus-4-6", 200000, 32000, True),
    "claude-sonnet-4-20250514": ModelInfo("anthropic", "claude-sonnet-4-20250514", 200000, 64000, True),
    "gpt-4o": ModelInfo("openai", "gpt-4o", 128000, 16384, True),
    "gpt-4o-mini": ModelInfo("openai", "gpt-4o-mini", 128000, 16384, True),
    "o4-mini": ModelInfo("openai", "o4-mini", 200000, 100000, True),
    "gemini-2.5-pro": ModelInfo("google", "gemini-2.5-pro", 1000000, 65536, True),
    "deepseek-chat": ModelInfo("deepseek", "deepseek-chat", 64000, 8192, False),
}


def get_model_info(model: str) -> ModelInfo:
    """Get model info by name (prefix match)."""
    for prefix, info in MODEL_INFO_DB.items():
        if model.startswith(prefix):
            return info
    return ModelInfo(model=model)


@dataclass
class SessionListEntry:
    """Entry in the session list."""
    session_key: str
    model: str = ""
    total_tokens: int = 0
    messages: int = 0
    last_activity: float = 0
    is_active: bool = False
    cost_usd: float = 0

    @property
    def idle_seconds(self) -> float:
        if not self.last_activity:
            return 0
        return time.time() - self.last_activity


def format_session_list(sessions: List[SessionListEntry], surface: str = "cli") -> str:
    """Format session list for display."""
    if not sessions:
        return "No active sessions."

    lines = [f"Sessions ({len(sessions)}):"]
    for s in sorted(sessions, key=lambda x: -x.last_activity):
        active = "●" if s.is_active else "○"
        idle = _format_idle(s.idle_seconds)
        cost = f"${s.cost_usd:.3f}" if s.cost_usd > 0 else ""
        model_short = s.model.split("/")[-1][:20] if s.model else "default"
        lines.append(
            f"  {active} {s.session_key[:30]:30s} | {model_short:20s} | "
            f"{s.total_tokens:>8,} tok | {s.messages:>3} msg | {idle} {cost}"
        )
    return "\n".join(lines)


def format_token_stats(stats: TokenStats, model: str = "", surface: str = "cli") -> str:
    """Format token statistics for display."""
    lines = [
        "Token Usage:",
        f"  Prompt:     {stats.prompt_tokens:>10,}",
        f"  Completion: {stats.completion_tokens:>10,}",
        f"  Total:      {stats.total_tokens:>10,}",
    ]
    if stats.cached_tokens:
        lines.append(f"  Cached:     {stats.cached_tokens:>10,}")
    lines.append(f"  API calls:  {stats.api_calls:>10,}")
    if stats.avg_latency_ms > 0:
        lines.append(f"  Avg latency: {stats.avg_latency_ms:>8.0f}ms")
    if stats.compaction_count:
        lines.append(f"  Compactions: {stats.compaction_count:>9,}")
    if stats.cost_usd > 0 or model:
        cost = stats.cost_usd or stats.estimate_cost(model)
        lines.append(f"  Est. cost:  ${cost:>9.4f}")
    return "\n".join(lines)


def format_model_info(info: ModelInfo) -> str:
    """Format model info for display."""
    features = []
    if info.supports_vision:
        features.append("vision")
    if info.supports_tools:
        features.append("tools")
    if info.supports_streaming:
        features.append("streaming")

    return (
        f"Model: {info.provider}/{info.model}\n"
        f"Context: {info.context_window:,} tokens\n"
        f"Max output: {info.max_output:,} tokens\n"
        f"Features: {', '.join(features) or 'none'}"
    )


def _format_idle(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.0f}d ago"
