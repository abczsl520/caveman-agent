"""Auxiliary LLM client — cheap model for side tasks.

Ported from Hermes agent/auxiliary_client.py (2613 lines → ~180 lines).
Uses a cheap/fast model for auxiliary tasks like:
  - Session title generation
  - Quick classification
  - Summary generation
  - Tag extraction

The heavy lifting (2613 lines) in Hermes was mostly provider routing
and caching. We keep it simple: one function, one model.
"""
from __future__ import annotations

__all__ = ["AuxiliaryClient", "generate_title", "classify_intent", "extract_tags"]

import logging
import re
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Default cheap model for auxiliary tasks
DEFAULT_AUX_MODEL = "gpt-4o-mini"


class AuxiliaryClient:
    """Lightweight LLM client for side tasks.

    Usage:
        aux = AuxiliaryClient(llm_fn=my_cheap_llm)
        title = await aux.generate_title(messages)
        intent = await aux.classify_intent("Deploy the app")
        tags = await aux.extract_tags("Built a FastAPI REST API with JWT auth")
    """

    def __init__(
        self,
        llm_fn: Callable[..., Awaitable[str]] | None = None,
        model: str = DEFAULT_AUX_MODEL,
    ) -> None:
        self._llm_fn = llm_fn
        self._model = model

    async def _call(self, prompt: str, max_tokens: int = 100) -> str:
        """Call the LLM. Falls back to heuristic if no LLM available."""
        if not self._llm_fn:
            raise RuntimeError("No LLM function configured")
        return await self._llm_fn(prompt)

    async def generate_title(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a short title for a conversation."""
        # Heuristic fallback
        if not self._llm_fn:
            return _heuristic_title(messages)

        first_user = next(
            (m["content"][:200] for m in messages if m.get("role") == "user"),
            "New conversation",
        )
        try:
            prompt = (
                f"Generate a short title (max 6 words) for this conversation. "
                f"Reply with ONLY the title, no quotes.\n\n"
                f"First message: {first_user}"
            )
            title = await self._call(prompt, max_tokens=20)
            return title.strip().strip('"').strip("'")[:60]
        except Exception:
            return _heuristic_title(messages)

    async def classify_intent(self, text: str) -> str:
        """Classify user intent: 'code', 'question', 'task', 'chat'."""
        if not self._llm_fn:
            return _heuristic_intent(text)

        try:
            prompt = (
                f"Classify this message into exactly one category: "
                f"code, question, task, chat.\n"
                f"Reply with ONLY the category.\n\n"
                f"Message: {text[:300]}"
            )
            result = await self._call(prompt, max_tokens=10)
            category = result.strip().lower()
            if category in ("code", "question", "task", "chat"):
                return category
            return _heuristic_intent(text)
        except Exception:
            return _heuristic_intent(text)

    async def extract_tags(self, text: str, max_tags: int = 5) -> list[str]:
        """Extract topic tags from text."""
        if not self._llm_fn:
            return _heuristic_tags(text, max_tags)

        try:
            prompt = (
                f"Extract up to {max_tags} topic tags from this text. "
                f"Reply with comma-separated tags only.\n\n"
                f"Text: {text[:500]}"
            )
            result = await self._call(prompt, max_tokens=50)
            tags = [t.strip().lower() for t in result.split(",")]
            return [t for t in tags if t and len(t) < 30][:max_tags]
        except Exception:
            return _heuristic_tags(text, max_tags)


# --- Heuristic fallbacks (no LLM needed) ---

def _heuristic_title(messages: list[dict[str, str]]) -> str:
    """Generate title from first user message."""
    first = next(
        (m["content"] for m in messages if m.get("role") == "user"),
        "New conversation",
    )
    # Take first sentence or first 50 chars
    title = first.split("\n")[0].split(".")[0][:50]
    return title.strip() or "New conversation"


_CODE_KEYWORDS = re.compile(
    r"\b(build|create|fix|debug|deploy|refactor|implement|test|code|write|function|class|api)\b",
    re.IGNORECASE,
)
_QUESTION_KEYWORDS = re.compile(
    r"\b(what|how|why|when|where|who|which|can|does|is|are|explain|tell me)\b",
    re.IGNORECASE,
)
_TASK_KEYWORDS = re.compile(
    r"\b(do|run|execute|install|setup|configure|update|delete|move|copy|send)\b",
    re.IGNORECASE,
)


def _heuristic_intent(text: str) -> str:
    """Classify intent without LLM."""
    if _CODE_KEYWORDS.search(text):
        return "code"
    if text.strip().endswith("?") or _QUESTION_KEYWORDS.search(text):
        return "question"
    if _TASK_KEYWORDS.search(text):
        return "task"
    return "chat"


def _heuristic_tags(text: str, max_tags: int = 5) -> list[str]:
    """Extract tags without LLM — simple keyword extraction."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    # Count frequency, skip common words
    _STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "been", "will", "are", "was", "not"}
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOP:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=freq.get, reverse=True)  # type: ignore
    return sorted_words[:max_tags]


# Convenience functions
async def generate_title(
    messages: list[dict[str, str]],
    llm_fn: Callable[..., Awaitable[str]] | None = None,
) -> str:
    return await AuxiliaryClient(llm_fn=llm_fn).generate_title(messages)


async def classify_intent(
    text: str,
    llm_fn: Callable[..., Awaitable[str]] | None = None,
) -> str:
    return await AuxiliaryClient(llm_fn=llm_fn).classify_intent(text)


async def extract_tags(
    text: str,
    max_tags: int = 5,
    llm_fn: Callable[..., Awaitable[str]] | None = None,
) -> list[str]:
    return await AuxiliaryClient(llm_fn=llm_fn).extract_tags(text, max_tags)
