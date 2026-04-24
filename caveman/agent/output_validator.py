"""Output format validator — enforces closing format on final responses.

When conversation_lifecycle rules request ✅---本轮已完成---✅ as closing,
this validator ensures the LLM output actually uses it instead of bare ✅.
"""
from __future__ import annotations
import re, logging

logger = logging.getLogger(__name__)

# The canonical closing line — read from behavior_rules for single source of truth
from caveman.agent.behavior_rules import get_rule as _get_rule
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

    # Skip enforcement for heartbeat responses
    if "HEARTBEAT" in text:
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
