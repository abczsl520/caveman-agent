"""P5: Prompt contract CI tests — prevent regression of contradicting instructions."""
import pytest
from pathlib import Path


class TestPromptContract:
    """Validate that all prompt layers respect their contracts."""

    def test_response_style_contract(self):
        from caveman.agent.prompt_contract import validate_layer
        from caveman.agent.response_style import get_response_style
        for surface in ["discord", "telegram", "cli"]:
            style = get_response_style(surface)
            if style:
                violations = validate_layer("response_style", style)
                assert not violations, f"{surface}: {[v.detail for v in violations]}"

    def test_lifecycle_contract(self):
        from caveman.agent.prompt_contract import validate_layer
        from caveman.agent.conversation_lifecycle import ConversationState, get_phase_rules
        for surface in ["discord", "telegram"]:
            for tc, it in [(1, 0), (3, 2), (8, 5)]:
                state = ConversationState(turn_count=tc, tool_call_count=tc*5, iteration_count=it)
                rules = get_phase_rules(surface, state)
                if rules:
                    violations = validate_layer("lifecycle", rules)
                    assert not violations, f"{surface} tc={tc}: {[v.detail for v in violations]}"

    def test_tool_descriptions_no_format_rules(self):
        """Tool descriptions must not contain format/emoji instructions."""
        from caveman.agent.prompt_contract import validate_layer
        from caveman.tools.registry import ToolRegistry
        registry = ToolRegistry()
        registry.auto_discover()
        for schema in registry.get_schemas():
            desc = schema.get("description", "")
            violations = validate_layer("tools", desc)
            assert not violations, f"Tool '{schema['name']}': {[v.detail for v in violations]}"

    def test_no_checkmark_in_tool_descriptions(self):
        """No ✅/❌ in any tool description."""
        from caveman.tools.registry import ToolRegistry
        registry = ToolRegistry()
        registry.auto_discover()
        for schema in registry.get_schemas():
            desc = schema.get("description", "")
            assert "✅" not in desc, f"Tool '{schema['name']}' has ✅ in description"
            assert "❌" not in desc, f"Tool '{schema['name']}' has ❌ in description"

    def test_cross_layer_no_contradictions(self):
        """Full cross-layer validation with all standard layers."""
        from caveman.agent.prompt_contract import validate_all_layers
        from caveman.agent.response_style import get_response_style
        from caveman.agent.conversation_lifecycle import ConversationState, get_phase_rules

        layers = []
        layers.append(("response_style", get_response_style("discord")))
        state = ConversationState(turn_count=8, tool_call_count=35)
        rules = get_phase_rules("discord", state)
        if rules:
            layers.append(("lifecycle", rules))

        violations = validate_all_layers(layers)
        # Filter out workspace violations (user-controlled)
        code_violations = [v for v in violations if v.layer_name != "workspace"]
        assert not code_violations, f"Code layer violations: {[v.detail for v in code_violations]}"

    def test_system_prompt_checkmark_consistency(self):
        """All ✅ mentions in system prompt must be either prohibitions or closing format."""
        from caveman.agent.prompt import build_system_prompt
        prompt = build_system_prompt(surface="discord")
        for i, line in enumerate(prompt.split("\n")):
            if "✅" in line:
                # Must be a "do NOT use" rule or the closing format
                is_prohibition = any(w in line.lower() for w in ["not", "don", "禁", "unless"])
                is_closing = "本轮已结束" in line
                is_list_marker = line.strip().startswith("✅")
                assert is_prohibition or is_closing or is_list_marker, \
                    f"Line {i} has ambiguous ✅ usage: {line.strip()[:80]}"


class TestPromptCacheStability:
    """Verify that system prompt is stable within a complexity level."""

    def test_same_complexity_same_lifecycle(self):
        """Within the same complexity level, lifecycle rules don't change."""
        from caveman.agent.conversation_lifecycle import ConversationState, get_phase_rules
        # All COMPLEX states should produce the same rules
        rules_a = get_phase_rules("discord", ConversationState(turn_count=8, tool_call_count=30))
        rules_b = get_phase_rules("discord", ConversationState(turn_count=10, tool_call_count=50))
        rules_c = get_phase_rules("discord", ConversationState(turn_count=20, tool_call_count=100))
        assert rules_a == rules_b == rules_c, "COMPLEX lifecycle rules should be stable"

    def test_lifecycle_not_in_cached_prompt(self):
        """Lifecycle rules must NOT be in the cached system prompt."""
        from caveman.agent.prompt import build_system_prompt
        from caveman.agent.conversation_lifecycle import ConversationState
        state = ConversationState(turn_count=8, tool_call_count=35)
        prompt = build_system_prompt(surface="discord", conversation_state=state)
        assert "Conversation Phase" not in prompt, "Lifecycle should be ephemeral, not cached"
