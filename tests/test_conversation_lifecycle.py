"""Tests for conversation lifecycle awareness."""
import pytest

from caveman.agent.conversation_lifecycle import (
    ConversationComplexity,
    ConversationPhase,
    ConversationState,
    get_phase_rules,
    get_section_markers,
)


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
        assert "✅" in rules  # mentioned as "do NOT use"

    def test_discord_working_medium(self):
        """Medium complexity working phase — has conditional closing."""
        state = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        rules = get_phase_rules("discord", state)
        assert "complex" in rules.lower()
        assert "✅---本轮已完成---✅" in rules
        assert "FINAL response" in rules

    def test_discord_working_complex(self):
        """Complex working phase — conditional closing instructions."""
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        rules = get_phase_rules("discord", state)
        assert "complex" in rules.lower()
        assert "✅---本轮已完成---✅" in rules
        assert "FINAL response" in rules

    def test_discord_working_complex_conditional(self):
        """Complex phase tells LLM to only close if no more tool calls."""
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        rules = get_phase_rules("discord", state)
        assert "MORE tool calls" in rules
        assert "FINAL response" in rules

    def test_cli_simple_returns_empty(self):
        state = ConversationState(turn_count=1)
        rules = get_phase_rules("cli", state)
        assert rules == ""

    def test_cli_complex_has_closing(self):
        """CLI complex conversations also get closing format."""
        state = ConversationState(turn_count=6, tool_call_count=25, iteration_count=5)
        rules = get_phase_rules("cli", state)
        assert "✅---本轮已完成---✅" in rules

    def test_unknown_surface_returns_empty(self):
        state = ConversationState(turn_count=1)
        rules = get_phase_rules("unknown_surface", state)
        assert rules == ""


class TestSectionMarkers:
    def test_no_checkmark_in_markers(self):
        markers = get_section_markers()
        assert "✅" not in markers
        assert "❌" not in markers

    def test_has_useful_markers(self):
        markers = get_section_markers()
        assert "📌" in markers
        assert "🔍" in markers
        assert "🔧" in markers


class TestResponseStyleIntegration:
    """Verify response_style.py no longer includes ✅/❌ as section markers."""

    def test_discord_style_no_checkmark_marker(self):
        from caveman.agent.response_style import get_response_style
        style = get_response_style("discord")
        lines = style.split("\n")
        for line in lines:
            if "section markers" in line.lower() and "NOT" not in line:
                assert "✅" not in line, f"✅ found as section marker in: {line}"


class TestPromptBuilderIntegration:
    """Lifecycle rules are now injected dynamically per-iteration, not in cached prompt."""

    def test_no_lifecycle_in_cached_prompt(self):
        """Lifecycle rules should NOT be in the cached system prompt."""
        from caveman.agent.prompt import build_system_prompt
        state = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        prompt = build_system_prompt(surface="discord", conversation_state=state)
        # Lifecycle is now injected per-iteration, not in cached prompt
        assert "Conversation Phase" not in prompt

    def test_no_lifecycle_without_state(self):
        from caveman.agent.prompt import build_system_prompt
        prompt = build_system_prompt(surface="discord")
        assert "Conversation Phase" not in prompt

    def test_dynamic_lifecycle_escalation(self):
        """Lifecycle rules change as conversation progresses."""
        # Simple conversation — opening
        simple = ConversationState(turn_count=1, iteration_count=0)
        rules_simple = get_phase_rules("discord", simple)
        assert "start of the conversation" in rules_simple.lower()

        # Medium conversation — now also has conditional closing
        medium = ConversationState(turn_count=3, tool_call_count=10, iteration_count=2)
        rules_medium = get_phase_rules("discord", medium)
        assert "✅---本轮已完成---✅" in rules_medium

        # Complex conversation — working_complex (conditional closing hint)
        complex_ = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        rules_complex = get_phase_rules("discord", complex_)
        assert "✅---本轮已完成---✅" in rules_complex


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
