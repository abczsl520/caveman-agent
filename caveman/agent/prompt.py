"""System prompt builder — layered assembly with token budgets.

Upgraded with insights from Hermes prompt_builder.py (MIT, Nous Research).
Each layer has a token budget; layers are assembled in priority order.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from caveman.skills.types import Skill
from caveman.agent.workspace import WorkspaceLoader

logger = logging.getLogger(__name__)

# Default token budgets per layer (total ~80K tokens for a 200K context)
DEFAULT_BUDGETS = {
    "identity": 2_000,      # SOUL.md / base persona
    "safety": 500,           # Safety rules (always included)
    "workspace": 5_000,      # AGENTS.md, USER.md, etc.
    "recall": 8_000,         # Recalled memories from prior sessions
    "memory": 4_000,         # Relevant memories for current task
    "skills": 3_000,         # Matched skill instructions
    "tools": 2_000,          # Tool schemas
    "context_refs": 10_000,  # @file/@url expanded content
    "extra": 5_000,          # Extra instructions (Recall, etc.)
    "meta": 200,             # Timestamp, context info
}

# Prompt injection patterns (from Hermes)
_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)(\s+\w+)*\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception"),
    (r'system\s+prompt\s+override', "override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules)', "disregard"),
]

_INVISIBLE_CHARS = frozenset({
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
})

SAFETY_RULES = """## Safety Rules
- Never execute dangerous commands without explicit confirmation
- Never expose secrets, API keys, or credentials in output
- Always verify file changes before overwriting
- Respect permission boundaries

## Gateway Communication (MANDATORY when running via Discord/Telegram)
When you have the `progress` tool available, you MUST use it:
1. FIRST action: `progress("开始做X…")` — announce what you're about to do
2. Every 2-3 tool calls: `progress("进度更新")` — report what you found/did
3. LAST action: `progress("完成摘要")` — summarize results
4. If stuck: `progress("遇到问题：…")` — report immediately
VIOLATION: 3+ consecutive tool calls without a progress update is a failure.
The user CANNOT see your text responses — ONLY progress tool messages reach them."""

BASE_PERSONA = """You are Caveman, an AI agent that learns, executes, and evolves.

## Core Capabilities
- Execute tasks using tools (bash, file operations, web search, and more)
- Learn from experience and store useful knowledge in memory
- Build reusable skills from repeated patterns
- Verify work quality through built-in quality gates

## Tool Selection Rules
- **File reading/editing/writing**: ALWAYS use file_read, file_edit, file_write, file_search
- **bash**: ONLY for running commands (build, test, git, system inspection), NOT for editing files
- Never use echo/sed/perl/python -c to edit files — use file_edit or file_write instead
- When file_edit fails (old_str not found), use file_read to check the current content first

## Operating Principles
1. **Explore** — Understand the task and gather context
2. **Plan** — Break down into concrete steps
3. **Execute** — Use tools to complete each step
4. **Verify** — Check results against expectations"""

# Backward compat alias
BASE_SYSTEM_PROMPT = BASE_PERSONA


@dataclass
class PromptLayer:
    """A single layer in the prompt assembly."""
    name: str
    content: str
    priority: int  # Lower = higher priority (included first)
    budget: int  # Max tokens for this layer
    required: bool = False  # If True, always included even if over budget

    @property
    def token_estimate(self) -> int:
        from caveman.agent.context import _estimate_str_tokens
        return _estimate_str_tokens(self.content)


@dataclass
class PromptBuildResult:
    """Result of prompt assembly."""
    prompt: str = ""
    layers_included: list[str] = field(default_factory=list)
    layers_truncated: list[str] = field(default_factory=list)
    layers_dropped: list[str] = field(default_factory=list)
    total_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def scan_content(content: str, filename: str = "") -> tuple[str, list[str]]:
    """Scan content for prompt injection. Returns (sanitized, findings)."""
    findings = []

    for char in _INVISIBLE_CHARS:
        if char in content:
            findings.append(f"invisible_unicode_U+{ord(char):04X}")
            content = content.replace(char, "")

    for pattern, pid in _THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            findings.append(pid)

    if findings:
        logger.warning("Content %s flagged: %s", filename, ", ".join(findings))

    return content, findings


def _truncate(content: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate content to fit token budget (CJK-aware)."""
    from caveman.agent.context import _estimate_str_tokens
    if _estimate_str_tokens(content) <= max_tokens:
        return content, False
    # Binary search for the right cut point
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _estimate_str_tokens(content[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return content[:lo] + "\n... (truncated)", True


class PromptBuilder:
    """Layered prompt builder with token budgets."""

    def __init__(
        self,
        total_budget: int = 80_000,
        layer_budgets: Optional[dict[str, int]] = None,
    ):
        self.total_budget = total_budget
        self.budgets = {**DEFAULT_BUDGETS, **(layer_budgets or {})}
        self._layers: list[PromptLayer] = []

    def add_layer(
        self, name: str, content: str,
        priority: int = 50, required: bool = False,
        budget: Optional[int] = None,
    ) -> None:
        """Add a prompt layer."""
        if not content or not content.strip():
            return
        layer_budget = budget or self.budgets.get(name, 5_000)
        self._layers.append(PromptLayer(
            name=name, content=content.strip(),
            priority=priority, budget=layer_budget,
            required=required,
        ))

    def build(self) -> PromptBuildResult:
        """Assemble all layers within token budget."""
        result = PromptBuildResult()
        sorted_layers = sorted(self._layers, key=lambda l: l.priority)
        tokens_used = 0

        for layer in sorted_layers:
            content = layer.content
            est = layer.token_estimate

            # Layer-level budget
            if est > layer.budget:
                content, _ = _truncate(content, layer.budget)
                from caveman.agent.context import _estimate_str_tokens
                est = _estimate_str_tokens(content)
                result.layers_truncated.append(layer.name)

            # Total budget check
            if tokens_used + est > self.total_budget and not layer.required:
                result.layers_dropped.append(layer.name)
                continue

            result.layers_included.append(layer.name)
            tokens_used += est

            # Add with section separator
            if result.prompt:
                result.prompt += "\n\n"
            result.prompt = (result.prompt or "") + content

        result.total_tokens = tokens_used
        return result


def _format_memories(memories: list[dict]) -> str:
    """Format memory entries for system prompt."""
    lines = []
    for m in memories[:10]:
        if isinstance(m, str):
            lines.append(f"- {m[:200]}")
        else:
            c = m.get("content", str(m))[:200]
            mtype = m.get("type", "")
            age = m.get("age_days", 0)
            prefix = f"[{mtype}]" if mtype else ""
            suffix = f" ({age}d ago)" if age > 7 else ""
            lines.append(f"- {prefix} {c}{suffix}")
    return "## Relevant Memories\n" + "\n".join(lines) if lines else ""


def _format_skills(skills) -> str:
    """Format skill entries for system prompt."""
    lines = []
    for s in skills[:5]:
        desc = f"**{s.name}** (v{s.version}): {s.description}"
        if s.system_prompt:
            desc += f"\n  Instructions: {s.system_prompt[:200]}"
        lines.append(f"- {desc}")
    return "## Available Skills\n" + "\n".join(lines) if lines else ""


def _build_meta_section() -> str:
    """Build meta context section with time and flywheel stats."""
    lines = [
        f"Current time: {datetime.now().isoformat()}",
        "Response language: Match the user's language. "
        "If the user writes in Chinese, respond entirely in Chinese. "
        "Do NOT start with English preamble like 'Now I...', 'Let me...', 'OK,'. "
        "Jump straight into the content.",
    ]
    try:
        from caveman.cli.flywheel import FlywheelStats
        fs = FlywheelStats().summary()
        if fs.get("total_rounds", 0) > 0:
            lines.append(
                f"Flywheel: {fs['total_rounds']} rounds, "
                f"{fs['total_p0_found']} P0 found, {fs['total_fixed']} fixed"
            )
    except Exception as e:
        logger.debug("Suppressed in prompt: %s", e)
    return "## Context\n" + "\n".join(lines)


def build_system_prompt(
    memories: list[dict] | None = None,
    skills: list[Skill] | None = None,
    tool_schemas: list[dict] | None = None,
    extra_instructions: str = "",
    workspace_loader: WorkspaceLoader | None = None,
    recall_context: str = "",
    wiki_context: str = "",
    total_budget: int = 80_000,
    surface: str = "cli",
) -> str:
    """Build the complete system prompt with dynamic context.

    Backward-compatible API — wraps PromptBuilder internally.
    Args:
        surface: Output surface ("discord", "telegram", "cli", "gateway").
                 Controls response formatting rules.
    """
    from caveman.agent.response_style import get_response_style

    builder = PromptBuilder(total_budget=total_budget)

    # 1. Workspace layers (highest priority)
    loader = workspace_loader or WorkspaceLoader()
    workspace_content = loader.build_prompt_layers()
    workspace_files = loader.load()

    if workspace_content:
        builder.add_layer("workspace", workspace_content, priority=5)

    # 2. Identity
    if "SOUL.md" not in workspace_files:
        builder.add_layer("identity", BASE_PERSONA, priority=10)

    # 3. Safety (always included)
    builder.add_layer("safety", SAFETY_RULES, priority=15, required=True)

    # 3b. Response style (surface-aware formatting)
    style = get_response_style(surface)
    if style:
        builder.add_layer("response_style", style, priority=12)

    # 4. Recall context (from prior sessions)
    if recall_context:
        builder.add_layer(
            "recall",
            f"## Recalled Context\n{recall_context}",
            priority=20,
        )

    # 4b. Wiki compiled context
    if wiki_context:
        builder.add_layer(
            "wiki",
            wiki_context,
            priority=25,
        )

    # 5. Memories
    if memories:
        mem_text = _format_memories(memories)
        if mem_text:
            builder.add_layer("memory", mem_text, priority=30)

    # 6. Skills
    if skills:
        skill_text = _format_skills(skills)
        if skill_text:
            builder.add_layer("skills", skill_text, priority=40)

    # 7. Tools
    if tool_schemas:
        lines = [f"- **{t['name']}**: {t['description']}" for t in tool_schemas]
        builder.add_layer(
            "tools",
            "## Available Tools\n" + "\n".join(lines),
            priority=50,
        )

    # 8. Extra instructions
    if extra_instructions:
        builder.add_layer(
            "extra",
            f"## Additional Instructions\n{extra_instructions}",
            priority=60,
        )

    # 9. Meta
    builder.add_layer("meta", _build_meta_section(), priority=99)

    result = builder.build()

    if result.layers_dropped:
        logger.warning(
            "Prompt layers dropped (budget): %s",
            ", ".join(result.layers_dropped),
        )

    return result.prompt or BASE_PERSONA
