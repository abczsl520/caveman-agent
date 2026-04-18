"""Smart model routing — cheap model for simple turns, strong for complex.

Ported from Hermes smart_model_routing.py (MIT, Nous Research).
Conservative: if any sign of complexity, keep the primary model.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_COMPLEX_KEYWORDS = frozenset({
    "debug", "debugging", "implement", "implementation", "refactor",
    "patch", "traceback", "stacktrace", "exception", "error",
    "analyze", "analysis", "investigate", "architecture", "design",
    "compare", "benchmark", "optimize", "review", "terminal", "shell",
    "tool", "tools", "pytest", "test", "tests", "plan", "planning",
    "delegate", "subagent", "cron", "docker", "kubernetes",
    # Chinese equivalents
    "调试", "实现", "重构", "分析", "架构", "设计", "优化", "审查",
    "部署", "测试", "规划", "错误", "异常",
})

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_MARKERS = {"```", "`", "def ", "class ", "import ", "from "}


def is_simple_turn(
    message: str,
    max_chars: int = 160,
    max_words: int = 28,
) -> bool:
    """Check if a message is simple enough for a cheap model.

    Conservative: returns False if any complexity signal is detected.
    """
    text = (message or "").strip()
    if not text:
        return False

    # Length checks
    if len(text) > max_chars:
        return False
    if len(text.split()) > max_words:
        return False

    # Multi-line = complex
    if text.count("\n") > 1:
        return False

    # Code markers
    for marker in _CODE_MARKERS:
        if marker in text:
            return False

    # URLs
    if _URL_RE.search(text):
        return False

    # Complex keywords (check both word-split and substring for CJK)
    lowered = text.lower()
    words = {token.strip(".,;:!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return False

    # CJK substring check (Chinese keywords don't split on spaces)
    for kw in _COMPLEX_KEYWORDS:
        if len(kw) > 1 and kw in lowered:
            return False

    return True


def choose_model(
    message: str,
    primary_model: str,
    cheap_model: str | None = None,
    routing_enabled: bool = False,
) -> tuple[str, str | None]:
    """Choose model for a turn. Returns (model_name, routing_reason).

    routing_reason is None if primary model is used.
    """
    if not routing_enabled or not cheap_model:
        return primary_model, None

    if is_simple_turn(message):
        return cheap_model, "simple_turn"

    return primary_model, None
