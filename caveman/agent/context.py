"""Context manager — manages agent context window and compression."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from caveman.paths import DEFAULT_CONTEXT_WINDOW, DEFAULT_COMPRESSION_THRESHOLD


@dataclass
class Message:
    """A single message in the conversation context."""
    role: str  # user | assistant | system | tool
    content: str | list[Any]
    tokens: int = 0
    ephemeral: bool = False  # If True, sent to API but not persisted to transcript


@dataclass
class AgentContext:
    """Manages the active context window for an agent session."""
    messages: list[Message] = field(default_factory=list)
    max_tokens: int = DEFAULT_CONTEXT_WINDOW
    compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 0
        return self.total_tokens / self.max_tokens

    def should_compress(self) -> bool:
        return self.utilization >= self.compression_threshold

    def add_message(self, role: str, content: str | list, tokens: int = 0,
                     ephemeral: bool = False) -> None:
        """Add a message to context.

        Args:
            ephemeral: If True, message is sent to API but excluded from
                       transcript persistence and session snapshots.
                       Use for format reminders, style hints, etc.
        """
        if tokens == 0:
            tokens = self._estimate_tokens(content)
        self.messages.append(Message(role=role, content=content, tokens=tokens,
                                     ephemeral=ephemeral))

    @staticmethod
    def _estimate_tokens(content: str | list) -> int:
        """Estimate token count from content.

        Uses a more accurate heuristic than simple char/4:
        - English: ~4 chars/token (GPT-family)
        - CJK: ~1.5 chars/token (each character is often 1 token)
        - Code: ~3.5 chars/token (shorter identifiers, symbols)
        - Mixed: weighted blend
        """
        if isinstance(content, str):
            return _estimate_str_tokens(content)
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        total += _estimate_str_tokens(str(text))
                    else:
                        total += len(str(block)) // 4
                    total += 5  # per-block overhead
                elif isinstance(block, str):
                    total += _estimate_str_tokens(block)
            return total + 3
        return 10

    def clear(self) -> None:
        """Clear all messages and reset context."""
        self.messages.clear()

    def to_api_format(self) -> list[dict]:
        """Convert to LLM API message format (includes ephemeral messages)."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def persistable_messages(self) -> list[Message]:
        """Return only non-ephemeral messages (for transcript persistence)."""
        return [m for m in self.messages if not m.ephemeral]

    def fork(self, label: str = "") -> "AgentContext":
        """Create a deep copy of this context for branching.

        The fork shares no mutable state with the original.
        Use for exploring alternative conversation paths.
        """
        import copy
        forked = AgentContext(
            messages=[copy.deepcopy(m) for m in self.messages],
            max_tokens=self.max_tokens,
            compression_threshold=self.compression_threshold,
        )
        return forked


def _estimate_str_tokens(text: str) -> int:
    """Estimate tokens for a string with CJK awareness.

    Delegates to utils.estimate_tokens — single source of truth.
    """
    from caveman.utils import estimate_tokens
    return estimate_tokens(text)
