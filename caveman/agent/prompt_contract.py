"""Prompt Contract — defines layer boundaries to prevent conflicting instructions.

Each layer has explicit allowed/forbidden topics. The PromptBuilder validates
content against these contracts at build time, catching contradictions early.

Design principle: Single Authority Source — each topic has exactly one layer
that owns it. Other layers may reference it but not redefine it.

Inspired by OpenClaw's ProviderSystemPromptSectionId and Hermes's
tool-aware guidance injection.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "PromptTopic",
    "LayerContract",
    "detect_topics",
    "ContractViolation",
    "validate_layer",
    "validate_all_layers",
]


logger = logging.getLogger(__name__)


class PromptTopic(Enum):
    """Topics that can appear in prompt layers."""
    IDENTITY = "identity"
    SAFETY = "safety"
    FORMAT_LAYOUT = "format_layout"
    EMOJI_CHOICE = "emoji_choice"
    CLOSING_RULES = "closing_rules"
    PHASE_AWARENESS = "phase_aware"
    TOOL_BEHAVIOR = "tool_behavior"
    WORKFLOW = "workflow"
    CONTEXT = "context"
    META = "meta"


# Topic detection patterns
TOPIC_PATTERNS: dict[PromptTopic, list[str]] = {
    PromptTopic.EMOJI_CHOICE: [
        # Match emoji choice instructions, not emoji used as list markers
        r'(?:use|start with|section marker).*(?:✅|❌|emoji)',
        r'(?:✅|❌).*(?:section marker|emoji)',
    ],
    PromptTopic.CLOSING_RULES: [
        r'收尾|closing|本轮已结束|end.*response|closing ceremony',
    ],
    PromptTopic.FORMAT_LAYOUT: [
        r'(?:不用|no|禁止|avoid).*(?:markdown|表格|table|header)',
        r'(?:use|用).*(?:bold|italic|code block|段落|paragraph)',
    ],
}


@dataclass
class LayerContract:
    """Contract for a prompt layer."""
    name: str
    allowed: set[PromptTopic] = field(default_factory=set)
    forbidden: set[PromptTopic] = field(default_factory=set)
    authority_for: set[PromptTopic] = field(default_factory=set)


LAYER_CONTRACTS: dict[str, LayerContract] = {
    "identity": LayerContract(
        name="identity",
        allowed={PromptTopic.IDENTITY},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES, PromptTopic.TOOL_BEHAVIOR},
        authority_for={PromptTopic.IDENTITY},
    ),
    "safety": LayerContract(
        name="safety",
        allowed={PromptTopic.SAFETY, PromptTopic.TOOL_BEHAVIOR},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES, PromptTopic.IDENTITY},
        authority_for={PromptTopic.SAFETY},
    ),
    "response_style": LayerContract(
        name="response_style",
        allowed={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE},
        forbidden={PromptTopic.CLOSING_RULES, PromptTopic.PHASE_AWARENESS,
                   PromptTopic.TOOL_BEHAVIOR, PromptTopic.IDENTITY},
        authority_for={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE},
    ),
    "lifecycle": LayerContract(
        name="lifecycle",
        allowed={PromptTopic.CLOSING_RULES, PromptTopic.PHASE_AWARENESS,
                 PromptTopic.EMOJI_CHOICE},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.IDENTITY,
                   PromptTopic.TOOL_BEHAVIOR},
        authority_for={PromptTopic.CLOSING_RULES, PromptTopic.PHASE_AWARENESS},
    ),
    "workspace": LayerContract(
        name="workspace",
        allowed={PromptTopic.WORKFLOW, PromptTopic.IDENTITY, PromptTopic.SAFETY},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES},
        authority_for={PromptTopic.WORKFLOW},
    ),
    "tools": LayerContract(
        name="tools",
        allowed={PromptTopic.TOOL_BEHAVIOR},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES},
        authority_for={PromptTopic.TOOL_BEHAVIOR},
    ),
    "recall": LayerContract(
        name="recall",
        allowed={PromptTopic.CONTEXT},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES, PromptTopic.TOOL_BEHAVIOR},
        authority_for=set(),
    ),
    "memory": LayerContract(
        name="memory",
        allowed={PromptTopic.CONTEXT},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES},
        authority_for=set(),
    ),
    "meta": LayerContract(
        name="meta",
        allowed={PromptTopic.META},
        forbidden={PromptTopic.FORMAT_LAYOUT, PromptTopic.EMOJI_CHOICE,
                   PromptTopic.CLOSING_RULES, PromptTopic.TOOL_BEHAVIOR},
        authority_for={PromptTopic.META},
    ),
}


def detect_topics(content: str) -> set[PromptTopic]:
    """Detect which topics are present in content."""
    found = set()
    for topic, patterns in TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found.add(topic)
                break
    return found


@dataclass
class ContractViolation:
    """A contract violation found during prompt building."""
    layer_name: str
    topic: PromptTopic
    violation_type: str  # "forbidden_topic" | "authority_conflict"
    detail: str


def validate_layer(name: str, content: str) -> list[ContractViolation]:
    """Validate a layer's content against its contract."""
    contract = LAYER_CONTRACTS.get(name)
    if not contract:
        return []
    violations = []
    for topic in detect_topics(content):
        if topic in contract.forbidden:
            violations.append(ContractViolation(
                layer_name=name, topic=topic,
                violation_type="forbidden_topic",
                detail=f"Layer '{name}' contains {topic.value} content (forbidden)",
            ))
    return violations


def validate_all_layers(layers: list[tuple[str, str]]) -> list[ContractViolation]:
    """Validate all layers for contract violations and authority conflicts."""
    violations = []
    for name, content in layers:
        violations.extend(validate_layer(name, content))

    topic_owners: dict[PromptTopic, list[str]] = {}
    for name, content in layers:
        for topic in detect_topics(content):
            topic_owners.setdefault(topic, []).append(name)

    for topic, owners in topic_owners.items():
        authorities = [
            n for n in owners
            if topic in LAYER_CONTRACTS.get(n, LayerContract(name="")).authority_for
        ]
        if len(owners) > 1 and authorities:
            for na in owners:
                if na not in authorities:
                    # Skip if the layer explicitly allows this topic
                    contract = LAYER_CONTRACTS.get(na)
                    if contract and topic in contract.allowed:
                        continue
                    violations.append(ContractViolation(
                        layer_name=na, topic=topic,
                        violation_type="authority_conflict",
                        detail=f"'{na}' touches {topic.value}, authority is '{authorities[0]}'",
                    ))
    return violations
