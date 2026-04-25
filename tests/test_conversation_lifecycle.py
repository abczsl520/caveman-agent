"""Tests for conversation lifecycle awareness."""

from caveman.agent.conversation_lifecycle import (
    ConversationComplexity,
    ConversationPhase,
    ConversationState,
    get_phase_rules,
    get_section_markers,
)

FALSE_TERMINAL_PRIMING_TERMS = (
    "Done.",
    "done",
    "✅",
    "❌",
    "本轮已完成",
    "FINAL response",
    "terminal completion marker",
    "completion marker",
)


def assert_no_false_terminal_priming(text: str) -> None:
    lower = text.lower()
    for term in FALSE_TERMINAL_PRIMING_TERMS:
        assert term.lower() not in lower, f"runtime prompt contains false-terminal priming term: {term!r}\n{text}"


class TestConversationState:
    def test_simple_complexity(self):
        state = ConversationState(turn_count=1, tool_call_count=0)
        assert state.complexity == ConversationComplexity.SIMPLE

    def test_medium_complexity(self):
        state = ConversationState(turn_count=3, tool_call_count=10)
        assert state.complexity == ConversationComplexity.MEDIUM

    def test_complex_complexity(self):
        state = ConversationState(turn_count=8, tool_call_count=30)
        assert state.complexity == ConversationComplexity.COMPLEX

    def test_opening_phase(self):
        state = ConversationState(turn_count=1, iteration_count=0)
        assert state.phase == ConversationPhase.OPENING

    def test_working_phase(self):
        state = ConversationState(turn_count=3, iteration_count=2)
        assert state.phase == ConversationPhase.WORKING

    def test_boundary_simple_medium(self):
        assert ConversationState(turn_count=1, tool_call_count=2).complexity == ConversationComplexity.SIMPLE
        assert ConversationState(turn_count=1, tool_call_count=3).complexity == ConversationComplexity.MEDIUM

    def test_boundary_medium_complex(self):
        assert ConversationState(turn_count=5, tool_call_count=20).complexity == ConversationComplexity.MEDIUM
        assert ConversationState(turn_count=6, tool_call_count=20).complexity == ConversationComplexity.COMPLEX


class TestPhaseRules:
    def test_discord_opening(self):
        state = ConversationState(turn_count=1, iteration_count=0)
        rules = get_phase_rules("discord", state)
        assert "start of the conversation" in rules.lower()
        assert "answer directly" in rules.lower()
        assert_no_false_terminal_priming(rules)

    def test_discord_working_medium(self):
        """Medium complexity working phase uses positive active-work guidance."""
        state = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        rules = get_phase_rules("discord", state)
        assert "active execution" in rules
        assert "remaining items" in rules
        assert_no_false_terminal_priming(rules)

    def test_discord_working_complex(self):
        """Complex working phase uses verification guidance without terminal priming."""
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        rules = get_phase_rules("discord", state)
        assert "complex" in rules.lower()
        assert "active verification" in rules.lower()
        assert "evidence" in rules.lower()
        assert_no_false_terminal_priming(rules)

    def test_cli_simple_returns_empty(self):
        state = ConversationState(turn_count=1)
        rules = get_phase_rules("cli", state)
        assert rules == ""

    def test_cli_complex_has_no_false_terminal_priming(self):
        state = ConversationState(turn_count=6, tool_call_count=25, iteration_count=5)
        rules = get_phase_rules("cli", state)
        assert "active verification" in rules.lower()
        assert_no_false_terminal_priming(rules)

    def test_unknown_surface_returns_empty(self):
        state = ConversationState(turn_count=1)
        rules = get_phase_rules("unknown_surface", state)
        assert rules == ""

    def test_all_runtime_lifecycle_rules_avoid_false_terminal_priming(self):
        states = [
            ConversationState(turn_count=1, tool_call_count=0, iteration_count=0),
            ConversationState(turn_count=3, tool_call_count=10, iteration_count=2),
            ConversationState(turn_count=8, tool_call_count=30, iteration_count=5),
        ]
        for surface in ("discord", "telegram", "cli"):
            for state in states:
                assert_no_false_terminal_priming(get_phase_rules(surface, state))


class TestSectionMarkers:
    def test_no_verdict_markers(self):
        markers = get_section_markers()
        assert "✅" not in markers
        assert "❌" not in markers

    def test_has_useful_markers(self):
        markers = get_section_markers()
        assert "📌" in markers
        assert "🔍" in markers
        assert "🔧" in markers


class TestResponseStyleIntegration:
    """Verify response_style.py no longer includes verdict symbols as section markers."""

    def test_discord_style_no_checkmark_marker(self):
        from caveman.agent.response_style import get_response_style
        style = get_response_style("discord")
        lines = style.split("\n")
        for line in lines:
            if "section markers" in line.lower() and "NOT" not in line:
                assert "✅" not in line, f"checkmark found as section marker in: {line}"


class TestPromptBuilderIntegration:
    """Lifecycle rules are injected dynamically per-iteration, not in cached prompt."""

    def test_no_lifecycle_in_cached_prompt(self):
        """Lifecycle rules should NOT be in the cached system prompt."""
        from caveman.agent.prompt import build_system_prompt
        state = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        prompt = build_system_prompt(surface="discord", conversation_state=state)
        assert "Conversation Phase" not in prompt
        assert "Active Work State" not in prompt

    def test_no_lifecycle_without_state(self):
        from caveman.agent.prompt import build_system_prompt
        prompt = build_system_prompt(surface="discord")
        assert "Conversation Phase" not in prompt
        assert "Active Work State" not in prompt

    def test_dynamic_lifecycle_escalation(self):
        """Lifecycle rules change as conversation progresses without false-terminal priming."""
        simple = ConversationState(turn_count=1, iteration_count=0)
        rules_simple = get_phase_rules("discord", simple)
        assert "start of the conversation" in rules_simple.lower()
        assert_no_false_terminal_priming(rules_simple)

        medium = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        rules_medium = get_phase_rules("discord", medium)
        assert "active execution" in rules_medium
        assert_no_false_terminal_priming(rules_medium)

        complex_ = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        rules_complex = get_phase_rules("discord", complex_)
        assert "active verification" in rules_complex
        assert_no_false_terminal_priming(rules_complex)


class TestAgentLoopIntegration:
    """Verify AgentLoop correctly tracks conversation state."""

    def test_reset_session_no_crash(self):
        """reset_session must not crash (was broken by property shadowing)."""
        from unittest.mock import MagicMock
        from caveman.agent.loop import AgentLoop

        provider = MagicMock()
        provider.model = "test"
        loop = AgentLoop(model="test", provider=provider)
        loop.reset_session()  # Should not raise AttributeError
        state = loop._conversation_state
        assert state.turn_count == 0
        assert state.tool_call_count == 0
        assert state.iteration_count == 0

    def test_iteration_count_tracked(self):
        """_iteration_count should be passed to ConversationState."""
        from unittest.mock import MagicMock
        from caveman.agent.loop import AgentLoop

        provider = MagicMock()
        provider.model = "test"
        loop = AgentLoop(model="test", provider=provider)
        loop._iteration_count = 7
        assert loop._conversation_state.iteration_count == 7
