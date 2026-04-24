"""Tests for Round 109 — skill evolution + reflect auto-evolve."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from caveman.skills.types import Skill, SkillTrigger, SkillStep, QualityGate
from caveman.skills.manager import SkillManager
from caveman.engines.reflect import ReflectEngine, Reflection


# ── Skill metadata and total_uses ──

def test_skill_has_metadata():
    skill = Skill(name="test", description="test skill")
    assert skill.metadata == {}
    skill.metadata["needs_evolution"] = True
    assert skill.metadata["needs_evolution"] is True


def test_skill_total_uses():
    skill = Skill(name="test", description="test", success_count=3, fail_count=2)
    assert skill.total_uses == 5
    assert skill.success_rate == 0.6


def test_skill_metadata_serialization():
    skill = Skill(name="test", description="test", metadata={"key": "value"})
    d = skill.to_dict()
    assert d["metadata"] == {"key": "value"}
    restored = Skill.from_dict(d)
    assert restored.metadata == {"key": "value"}


# ── record_outcome flags evolution ──

def test_record_outcome_flags_degraded_skill(tmp_path):
    mgr = SkillManager(skills_dir=tmp_path)
    skill = Skill(name="deploy", description="Deploy to prod", success_count=0, fail_count=2)
    mgr.save(skill)

    # Third failure should flag for evolution (success_rate < 40%)
    mgr.record_outcome("deploy", success=False)
    updated = mgr.get("deploy")
    assert updated.fail_count == 3
    assert updated.metadata.get("needs_evolution") is True


def test_record_outcome_no_flag_when_healthy(tmp_path):
    mgr = SkillManager(skills_dir=tmp_path)
    skill = Skill(name="test", description="test", success_count=5, fail_count=0)
    mgr.save(skill)

    mgr.record_outcome("test", success=True)
    updated = mgr.get("test")
    assert not updated.metadata.get("needs_evolution")


# ── evolve with LLM ──

@pytest.mark.asyncio
async def test_evolve_with_llm(tmp_path):
    mgr = SkillManager(skills_dir=tmp_path)
    skill = Skill(
        name="deploy", description="Deploy to prod",
        steps=[SkillStep(tool="bash", description="run deploy script")],
        metadata={"needs_evolution": True},
    )
    mgr.save(skill)

    mock_llm = AsyncMock(return_value='{"description": "Deploy to prod with rollback", "constraints": ["Always check health endpoint after deploy"]}')

    result = await mgr.evolve(
        "deploy", feedback="Deploy failed without rollback",
        trajectory=[{"from": "gpt", "value": "deploying..."}],
        llm_fn=mock_llm,
    )
    assert result is not None
    assert result.version == 2
    assert "rollback" in result.description.lower()
    assert not result.metadata.get("needs_evolution")  # Flag cleared


@pytest.mark.asyncio
async def test_evolve_without_llm(tmp_path):
    mgr = SkillManager(skills_dir=tmp_path)
    skill = Skill(name="test", description="test", metadata={"needs_evolution": True})
    mgr.save(skill)

    result = await mgr.evolve("test")
    assert result.version == 2
    assert not result.metadata.get("needs_evolution")


# ── Reflect auto-evolve on failure ──

@pytest.mark.asyncio
async def test_reflect_auto_evolves_degraded_skills(tmp_path):
    mgr = SkillManager(skills_dir=tmp_path)
    skill = Skill(
        name="deploy", description="Deploy",
        success_count=1, fail_count=5,
        metadata={"needs_evolution": True},
    )
    mgr.save(skill)

    engine = ReflectEngine(skill_manager=mgr, llm_fn=None)
    trajectory = [
        {"from": "human", "value": "deploy to prod"},
        {"from": "gpt", "value": "deploying... failed with error"},
        {"from": "gpt", "value": "error: connection refused"},
    ]

    reflection = await engine.reflect("deploy to prod", trajectory, "failed")
    assert reflection.outcome == "failure"

    # Skill should have been evolved (version bumped)
    updated = mgr.get("deploy")
    assert updated.version >= 2
    assert not updated.metadata.get("needs_evolution")


# ── File size checks ──

def test_skills_manager_under_400_lines():
    """NFR-502: skills/manager.py within code-health policy."""
    from caveman.cli.code_health import check_code_health
    result = check_code_health()
    issues = [i for i in result["file_size"] if "skills/manager.py" in i]
    assert not issues, f"skills/manager.py exceeds policy: {issues}"


def test_reflect_under_400_lines():
    """NFR-502: engines/reflect.py within code-health policy."""
    from caveman.cli.code_health import check_code_health
    result = check_code_health()
    issues = [i for i in result["file_size"] if "engines/reflect.py" in i]
    assert not issues, f"engines/reflect.py exceeds policy: {issues}"
