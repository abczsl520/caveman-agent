"""Conversation lifecycle awareness — dynamic format rules based on dialog state.

Instead of static format rules that never change, this module provides
context-aware formatting guidance that adapts to:
  - Conversation complexity (simple / medium / complex)
  - Current phase (opening / working / closing)
  - Accumulated tool calls and turns

This is a Harness-layer capability (PRD §8.2), not a prompt patch.
The prompt builder injects phase-appropriate rules, so the LLM naturally
produces the right format without guessing.

Design principles:
  - Complexity is inferred, not declared — from turn count + tool calls
  - Phase transitions are automatic — no manual signaling needed
  - Rules are additive — base style always applies, phase rules layer on top
  - Closing signal is structural — not emoji-based guessing
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
    CLOSING = "closing"    # Final answer — wrap up


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
# Phase-aware format rules
# ---------------------------------------------------------------------------

# Section markers — deliberately excludes ✅ and ❌ which have
# "conclusion/judgment" semantics. Those are reserved for closing only.
_SECTION_MARKERS = "📌 🔍 💡 ⚡ 🎯 🔧"

# Base rules that always apply (surface-specific base is in response_style.py)
_PHASE_RULES: dict[str, dict[str, str]] = {
    "discord": {
        # Opening: first impression, set expectations
        "opening": (
            "This is the start of the conversation.\n"
            "- Jump straight into the answer — no preamble\n"
            "- If the task is simple, answer directly and stop — no closing ceremony\n"
            f"- Use these emoji as section markers: {_SECTION_MARKERS}\n"
            "- Do NOT use ✅ or ❌ anywhere in your response"
        ),
        # Working: middle of a multi-turn task
        "working": (
            "You are in the middle of a multi-turn task.\n"
            "- Report progress concisely\n"
            f"- Use these emoji as section markers: {_SECTION_MARKERS}\n"
            "- Do NOT use ✅ or ❌ anywhere — you're not done yet\n"
            "- Do NOT add closing statements — you're still working"
        ),
        # Working but complex enough to warrant closing format when done
        "working_complex": (
            "You are working on a complex multi-turn task.\n"
            "- Report progress concisely\n"
            f"- Use these emoji as section markers: {_SECTION_MARKERS}\n"
            "- Do NOT use ✅ or ❌ as section markers\n"
            "- If you are making MORE tool calls after this response, do NOT add any closing\n"
            "- If this is your FINAL response (no more tool calls needed), end with:\n"
            "  ✅ --- 本轮已结束 --- ✅\n"
            "  (brief 2-3 sentence summary before the closing line)"
        ),
    },
    "telegram": {
        "opening": (
            "Start of conversation.\n"
            "- Answer directly, no preamble\n"
            "- Simple questions: answer and stop\n"
            f"- Section markers: {_SECTION_MARKERS}\n"
            "- No ✅/❌ anywhere"
        ),
        "working": (
            "Mid-task.\n"
            f"- Section markers: {_SECTION_MARKERS}\n"
            "- No ✅/❌ — not done yet\n"
            "- No premature closing"
        ),
        "working_complex": (
            "Complex multi-turn task in progress.\n"
            f"- Section markers: {_SECTION_MARKERS}\n"
            "- No ✅/❌ as markers\n"
            "- If this is your FINAL response: end with ✅ --- 本轮已结束 --- ✅"
        ),
    },
    "cli": {
        "opening": "",
        "working": "",
        "working_complex": "",
    },
}


def get_phase_rules(surface: str, state: ConversationState) -> str:
    """Get phase-appropriate format rules for the current conversation state.

    Called by the prompt builder to inject dynamic formatting guidance.
    Returns empty string if no special rules needed (e.g., CLI).
    """
    rules = _PHASE_RULES.get(surface, _PHASE_RULES.get("cli", {}))
    if not rules:
        return ""

    phase = state.phase
    if phase == ConversationPhase.WORKING:
        # Complex conversations get conditional closing instructions
        if state.complexity == ConversationComplexity.COMPLEX:
            return rules.get("working_complex", rules.get("working", ""))
    return rules.get(phase.value, "")


def get_section_markers() -> str:
    """Get the approved section marker emoji list."""
    return _SECTION_MARKERS
