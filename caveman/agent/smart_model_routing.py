"""Smart Model Routing — cheap vs strong model selection per turn.

Routes simple messages to a cheaper model while keeping complex
tasks on the primary model. Extracted from Hermes
agent/smart_model_routing.py (195 lines).
"""
from __future__ import annotations

import re
from typing import Optional

_COMPLEX_KEYWORDS = {
    "debug", "debugging", "implement", "implementation", "refactor",
    "patch", "traceback", "stacktrace", "exception", "error",
    "analyze", "analysis", "investigate", "architecture", "design",
    "compare", "benchmark", "optimize", "review", "terminal",
    "shell", "tool", "tools", "pytest", "test", "tests", "plan",
    "planning", "delegate", "subagent", "cron", "docker", "kubernetes",
    "deploy", "migration", "security", "vulnerability", "performance",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_INDICATORS = {"```", "`", "def ", "class ", "import ", "function ", "const ", "let ", "var "}

from dataclasses import dataclass

__all__ = [
    "RoutingConfig",
    "RouteDecision",
    "classify_message_complexity",
    "choose_route",
]



@dataclass
class RoutingConfig:
    """Configuration for smart model routing."""
    enabled: bool = False
    cheap_model: str = ""
    cheap_provider: str = ""
    max_simple_chars: int = 160
    max_simple_words: int = 28
    max_newlines: int = 1


@dataclass
class RouteDecision:
    """Result of a routing decision."""
    model: str
    provider: str = ""
    is_cheap: bool = False
    reason: str = ""




def classify_message_complexity(text: str, config: Optional[RoutingConfig] = None) -> str:
    """Classify a message as 'simple' or 'complex'.

    Conservative: if in doubt, return 'complex'.
    """
    config = config or RoutingConfig()
    text = (text or "").strip()

    if not text:
        return "simple"

    # Length checks
    if len(text) > config.max_simple_chars:
        return "complex"
    if len(text.split()) > config.max_simple_words:
        return "complex"
    if text.count("\n") > config.max_newlines:
        return "complex"

    # Code indicators
    for indicator in _CODE_INDICATORS:
        if indicator in text:
            return "complex"

    # URL
    if _URL_RE.search(text):
        return "complex"

    # Complex keywords
    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return "complex"

    return "simple"


def choose_route(
    user_message: str,
    primary_model: str,
    config: Optional[RoutingConfig] = None,
) -> RouteDecision:
    """Choose the model route for a message."""
    config = config or RoutingConfig()

    if not config.enabled or not config.cheap_model:
        return RouteDecision(model=primary_model, reason="routing_disabled")

    complexity = classify_message_complexity(user_message, config)

    if complexity == "simple":
        return RouteDecision(
            model=config.cheap_model,
            provider=config.cheap_provider,
            is_cheap=True,
            reason="simple_turn",
        )

    return RouteDecision(model=primary_model, reason=f"complex:{complexity}")
