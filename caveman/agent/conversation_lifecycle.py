"""Conversation lifecycle awareness — dynamic format rules based on dialog state.

Instead of static format rules that never change, this module provides
context-aware guidance that adapts to:
  - Conversation complexity (simple / medium / complex)
  - Current phase (opening / working)
  - Accumulated tool calls and turns

The guidance is intentionally positive-only: it describes the current working
state and evidence requirements without quoting legacy terminal wording.
Negative prompts that quote the bad token tend to prime the model to repeat it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ConversationComplexity",
    "ConversationPhase",
    "ConversationState",
    "get_phase_rules",
    "get_section_markers",
]


class ConversationComplexity(Enum):
    """How complex is this conversation?"""
    SIMPLE = "simple"      # Single turn, no tools, quick Q&A
    MEDIUM = "medium"      # 2-5 turns, some tool use
    COMPLEX = "complex"    # 5+ turns, heavy tool use, multi-stage


class ConversationPhase(Enum):
    """Where are we in the conversation lifecycle?"""
    OPENING = "opening"    # First turn — set expectations
    WORKING = "working"    # Middle turns — doing the work
    CLOSING = "closing"    # Reserved for future structural state, not prompt text


@dataclass
class ConversationState:
    """Snapshot of current conversation state for phase inference."""
    turn_count: int = 0
    tool_call_count: int = 0
    has_progress_calls: bool = False  # Did we send progress updates?
    iteration_count: int = 0  # LLM iterations within current turn

    @property
    def complexity(self) -> ConversationComplexity:
        """Infer complexity from accumulated signals."""
        if self.turn_count <= 1 and self.tool_call_count <= 2:
            return ConversationComplexity.SIMPLE
        if self.turn_count <= 5 and self.tool_call_count <= 20:
            return ConversationComplexity.MEDIUM
        return ConversationComplexity.COMPLEX

    @property
    def phase(self) -> ConversationPhase:
        """Infer current phase."""
        if self.turn_count <= 1 and self.iteration_count == 0:
            return ConversationPhase.OPENING
        return ConversationPhase.WORKING


# ---------------------------------------------------------------------------
# Phase-aware guidance
# ---------------------------------------------------------------------------

# Section markers — deliberately excludes verdict/completion-like symbols.
_SECTION_MARKERS = "📌 🔍 💡 ⚡ 🎯 🔧"

# These strings previously triggered false terminal behavior and must stay out
# of runtime guidance. Keep this list in tests too.
_PRIMING_TERMS = (
    "Done.",
    "done",
    "✅",
    "❌",
    "本轮已完成",
    "FINAL response",
    "terminal completion marker",
    "completion marker",
)

# Base rules that always apply (surface-specific base is in response_style.py).
# Use positive wording only. Do not quote forbidden legacy markers here.
_PHASE_RULES: dict[str, dict[str, str]] = {
    "discord": {
        "opening": (
            "This is the start of the conversation.\n"
            "- Answer directly without preamble\n"
            "- For small questions, give the answer and stop naturally\n"
            f"- Use these section markers when helpful: {_SECTION_MARKERS}"
        ),
        "working": (
            "This task is still in active execution.\n"
            "- Report progress concisely with evidence\n"
            f"- Use these section markers when helpful: {_SECTION_MARKERS}\n"
            "- If work remains, state the remaining items and continue the next concrete action"
        ),
        "working_complex": (
            "This is a complex task under active verification.\n"
            "- Report progress concisely with evidence, remaining risks, and next action\n"
            f"- Use these section markers when helpful: {_SECTION_MARKERS}\n"
            "- Keep acting while verified work remains\n"
            "- Give a natural summary only after checks prove the requested outcome"
        ),
    },
    "telegram": {
        "opening": (
            "Start of conversation.\n"
            "- Answer directly\n"
            "- Small questions can stop naturally\n"
            f"- Section markers when helpful: {_SECTION_MARKERS}"
        ),
        "working": (
            "Task is in active execution.\n"
            f"- Section markers when helpful: {_SECTION_MARKERS}\n"
            "- State evidence, remaining items, and next action"
        ),
        "working_complex": (
            "Complex task under active verification.\n"
            f"- Section markers when helpful: {_SECTION_MARKERS}\n"
            "- Continue concrete actions while verified work remains\n"
            "- Summarize naturally only after checks prove the requested outcome"
        ),
    },
    "cli": {
        "opening": "",
        "working": "",
        "working_complex": (
            "Complex task under active verification. Continue concrete actions while verified work remains; "
            "summarize naturally only after checks prove the requested outcome."
        ),
    },
}


def _assert_no_priming_terms(text: str) -> None:
    """Fail fast if runtime lifecycle text reintroduces false-terminal priming."""
    lower = text.lower()
    for term in _PRIMING_TERMS:
        needle = term.lower()
        if needle and needle in lower:
            raise AssertionError(f"Lifecycle guidance contains priming term: {term!r}")


def get_phase_rules(surface: str, state: ConversationState) -> str:
    """Get phase-appropriate guidance for the current conversation state.

    Called by the prompt builder to inject dynamic guidance. Returns empty string
    if no special rules are needed.
    """
    rules = _PHASE_RULES.get(surface, _PHASE_RULES.get("cli", {}))
    if not rules:
        return ""

    phase = state.phase
    if phase == ConversationPhase.WORKING and state.complexity == ConversationComplexity.COMPLEX:
        text = rules.get("working_complex", rules.get("working", ""))
    else:
        text = rules.get(phase.value, "")
    _assert_no_priming_terms(text)
    return text


def get_section_markers() -> str:
    """Get the approved section marker emoji list."""
    return _SECTION_MARKERS
