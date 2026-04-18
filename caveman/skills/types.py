"""Skill definition types."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SkillTrigger(Enum):
    """How a skill gets activated."""
    MANUAL = "manual"       # User explicitly invokes
    AUTO = "auto"           # Agent decides based on task match
    PATTERN = "pattern"     # Regex/keyword pattern match
    ALWAYS = "always"       # Always injected into system prompt


@dataclass
class SkillStep:
    """A single step in a skill's execution plan."""
    tool: str               # Tool name to call
    args_template: dict[str, Any] = field(default_factory=dict)  # Args with {{var}} placeholders
    description: str = ""
    condition: str | None = None  # Optional condition expression


@dataclass
class QualityGate:
    """Quality check after skill execution."""
    name: str
    check: str              # What to verify (e.g., "output_contains", "file_exists", "no_errors")
    expected: str = ""      # Expected value
    severity: str = "warn"  # "warn" | "block"


@dataclass
class Skill:
    """A reusable, evolvable agent skill."""
    name: str
    description: str
    version: int = 1
    trigger: SkillTrigger = SkillTrigger.AUTO
    trigger_patterns: list[str] = field(default_factory=list)  # For PATTERN trigger

    # Execution
    system_prompt: str = ""          # Injected when skill activates
    steps: list[SkillStep] = field(default_factory=list)  # Structured execution plan
    content: str = ""                # Free-form instruction (fallback)

    # Quality
    quality_gates: list[QualityGate] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    success_count: int = 0
    fail_count: int = 0
    source: str = "manual"          # "manual" | "auto_created" | "imported" | "hermes"
    metadata: dict = field(default_factory=dict)  # Evolution tracking, etc.

    @property
    def total_uses(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        total = self.total_uses
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "trigger": self.trigger.value,
            "trigger_patterns": self.trigger_patterns,
            "system_prompt": self.system_prompt,
            "steps": [
                {"tool": s.tool, "args_template": s.args_template,
                 "description": s.description, "condition": s.condition}
                for s in self.steps
            ],
            "content": self.content,
            "quality_gates": [
                {"name": g.name, "check": g.check, "expected": g.expected, "severity": g.severity}
                for g in self.quality_gates
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", 1),
            trigger=SkillTrigger(data.get("trigger", "auto")),
            trigger_patterns=data.get("trigger_patterns", []),
            system_prompt=data.get("system_prompt", ""),
            steps=[
                SkillStep(
                    tool=s["tool"], args_template=s.get("args_template", {}),
                    description=s.get("description", ""), condition=s.get("condition"),
                )
                for s in data.get("steps", [])
            ],
            content=data.get("content", ""),
            quality_gates=[
                QualityGate(
                    name=g["name"], check=g["check"],
                    expected=g.get("expected", ""), severity=g.get("severity", "warn"),
                )
                for g in data.get("quality_gates", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            source=data.get("source", "manual"),
            metadata=data.get("metadata", {}),
        )
