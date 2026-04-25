"""Output format validator — suppresses premature terminal markers.

The canonical ✅---本轮已完成---✅ marker is retained only as a legacy token for
recognition/stripping. It must not be synthesized or preserved by default:
completion is a semantic state verified by work/results, not an emoji line.
"""
from __future__ import annotations
import re, logging

logger = logging.getLogger(__name__)

from caveman.agent.conversation_lifecycle import ConversationComplexity, ConversationState

# Legacy terminal markers are recognized only so they can be stripped. Do not
# import them from behavior_rules: they are not part of current output policy.
LEGACY_CLOSING_LINE = "✅---本轮已完成---✅"
LEGACY_AGENT_CLOSING_LINE = "✅ DONE"
CLOSING_LINE = LEGACY_CLOSING_LINE

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

# Final answers that end in an obviously incomplete fragment must not be
# treated as completed work. This catches provider/gateway truncation cases
# where stop_reason is not reliably reported as max_tokens (observed: final
# response persisted ending with the half word "- PR").
_INCOMPLETE_LINE_START = re.compile(r'^\s*(?:[-*•]|\d+[.)、]|#{1,6}\s+)\s*\S{0,40}\s*$')
_INCOMPLETE_CODE_FENCE = re.compile(r'```')
_SENTENCE_ENDING = tuple('。.!！？?…」』”’）)]}`')


def final_text_looks_truncated(text: str) -> bool:
    """Heuristically detect a final response that is visibly cut off.

    This is intentionally conservative: it only flags dangling bullets/headings,
    unmatched code fences/brackets, or obvious mid-word fragments at the end of
    a substantial final response. It is a continuation guard, not a success
    detector.
    """
    if not text or not text.strip():
        return False
    stripped = strip_closing_markers(text).strip()

    # Unclosed fenced code block almost always means the final answer was cut,
    # even if the snippet is short.
    if len(_INCOMPLETE_CODE_FENCE.findall(stripped)) % 2 == 1:
        return True

    if len(stripped) < 120:
        return False

    # Cheap unmatched bracket guard for prose/code snippets.
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('（', '）'), ('【', '】')]
    for left, right in pairs:
        if stripped.count(left) > stripped.count(right) + 1:
            return True

    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1].strip()

    # Dangling list/header fragments, e.g. the observed failure ending in
    # "  - PR" after a long PRD audit summary.
    if _INCOMPLETE_LINE_START.match(last) and not last.endswith(_SENTENCE_ENDING):
        return True

    # Markdown link/code span cut halfway.
    if last.count('`') % 2 == 1 or last.count('[') > last.count(']'):
        return True

    # English/identifier fragment after a long answer; avoid flagging normal
    # Chinese prose that often omits punctuation.
    if re.search(r'(?:^|\s)[A-Za-z_/#.-]{2,18}$', last) and not last.endswith(_SENTENCE_ENDING):
        return True

    return False


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
    """Terminal completion markers are disabled by default.

    The function is kept for call-site compatibility, but a structural marker
    must not be inferred from complexity/tool counts. Natural summaries are OK;
    terminal emoji markers are not.
    """
    return False

def strip_closing_markers(text: str, surface: str = "cli") -> str:
    """Remove terminal closing markers when policy says the turn should stay open.

    This is deliberately conservative: it removes only canonical terminal
    markers and a bare trailing checkmark line, not ordinary checkmarks used
    inside content.
    """
    if not text:
        return text
    marker = LEGACY_AGENT_CLOSING_LINE if surface == "agent" else CLOSING_LINE
    cleaned = text.replace(marker, "")
    cleaned = _WRONG_CLOSING.sub("", cleaned)
    cleaned = _BARE_CHECKMARK.sub("", cleaned).rstrip()
    return cleaned

def enforce_closing_format(text: str, should_close: bool, surface: str = "cli") -> str:
    """Suppress terminal completion markers.

    ``should_close`` is intentionally ignored while completion markers are
    disabled. This strips both canonical markers and malformed trailing attempts
    so accidental prompt/model inertia cannot signal done to the gateway/flywheel.
    """
    if not text:
        return text
    return strip_closing_markers(text, surface=surface)
