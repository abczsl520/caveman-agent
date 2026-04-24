"""Output format validator — enforces closing format on final responses.

When conversation_lifecycle rules request ✅---本轮已完成---✅ as closing,
this validator ensures the LLM output actually uses it instead of bare ✅.
"""
from __future__ import annotations
import re, logging

logger = logging.getLogger(__name__)

# The canonical closing line — read from behavior_rules for single source of truth
from caveman.agent.behavior_rules import get_rule as _get_rule
from caveman.agent.conversation_lifecycle import ConversationComplexity, ConversationState
CLOSING_LINE = _get_rule("CLOSING_FORMAT") or "✅---本轮已完成---✅"

# Patterns that indicate LLM tried to close but used wrong format
_BARE_CHECKMARK = re.compile(
    r'(?:^|\n)\s*✅\s*$',  # bare ✅ on its own line at end
    re.MULTILINE,
)
_WRONG_CLOSING = re.compile(
    r'✅[^-\n]*(?:完成|结束|done|complete|finished)[^-\n]*✅',
    re.IGNORECASE,
)
_QUESTION_ENDING = re.compile(r'[?？]\s*(?:[)）】\]"”’。.!！…]*)\s*$')


def final_sentence_is_question(text: str) -> bool:
    """Return True when the visible final sentence is a question.

    Closing markers are terminal completion signals. If the assistant ends by
    asking the user something, the turn is intentionally open and must not be
    auto-closed.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if CLOSING_LINE in stripped:
        stripped = stripped.replace(CLOSING_LINE, "").strip()
    if not stripped:
        return False
    last_non_empty = next((line.strip() for line in reversed(stripped.splitlines()) if line.strip()), "")
    return bool(_QUESTION_ENDING.search(last_non_empty))


def should_use_closing_marker(
    *,
    state: ConversationState,
    final_text: str = "",
    surface: str = "cli",
) -> bool:
    """Decide whether a final response should carry the canonical closing marker.

    Long-term policy:
    - Questions keep the conversation open, so never auto-close.
    - Simple answers should feel natural; no ceremony.
    - Complex work needs a structural terminal signal.
    - Medium work closes only when it has meaningful execution weight.
    """
    if surface == "agent":
        return state.tool_call_count >= 1
    if not final_text or final_sentence_is_question(final_text):
        return False
    if "HEARTBEAT" in final_text:
        return False

    complexity = state.complexity
    if complexity == ConversationComplexity.SIMPLE:
        return False
    if complexity == ConversationComplexity.COMPLEX:
        return True

    # Medium: avoid marker spam after tiny one-tool lookups, but close real work.
    return (
        state.has_progress_calls
        or state.tool_call_count >= 3
        or state.turn_count >= 3
        or state.iteration_count >= 2
    )


def enforce_closing_format(text: str, should_close: bool, surface: str = "cli") -> str:
    """Fix closing format if LLM used bare ✅ instead of the canonical format.

    Args:
        text: The LLM's final response text.
        should_close: Whether the conversation lifecycle expects a closing line.
        surface: The surface type (cli, gateway, agent). Agent uses ✅ DONE.

    Returns:
        Text with corrected closing format, or unchanged if no fix needed.
    """
    if not should_close or not text:
        return text

    # Skip enforcement for heartbeat responses and open-ended questions
    if "HEARTBEAT" in text or final_sentence_is_question(text):
        return text

    # Agent surface uses different closing marker
    if surface == "agent":
        _agent_marker = "✅ DONE"
        if _agent_marker in text:
            return text
        return text.rstrip() + "\n\n" + _agent_marker

    # Already has correct closing — no-op
    if CLOSING_LINE in text:
        return text

    # Case 1: bare ✅ at end of text
    stripped = text.rstrip()
    if stripped.endswith("✅") and "---" not in stripped[-30:]:
        fixed = stripped[:-1].rstrip() + "\n\n" + CLOSING_LINE
        logger.info("Closing format fixed: bare ✅ → canonical format")
        return fixed

    # Case 2: wrong closing pattern (e.g. ✅完成✅, ✅ done ✅)
    if _WRONG_CLOSING.search(text):
        fixed = _WRONG_CLOSING.sub(CLOSING_LINE, text)
        logger.info("Closing format fixed: wrong pattern → canonical format")
        return fixed

    # Case 3: no closing at all but should have one — append it
    # A closing attempt = ✅ at start/end of one of the last 3 lines
    last_lines = text.rstrip().split("\n")[-3:]
    has_closing_attempt = any(
        line.strip().startswith("✅") or line.strip().endswith("✅")
        for line in last_lines
    )
    if should_close and not has_closing_attempt:
        fixed = text.rstrip() + "\n\n" + CLOSING_LINE
        logger.info("Closing format added: missing → appended")
        return fixed

    return text
