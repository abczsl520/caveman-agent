"""Tests for skill framework v2."""
import pytest
import tempfile
from caveman.skills.types import Skill, SkillTrigger, SkillStep, QualityGate
from caveman.skills.manager import SkillManager
from caveman.skills.executor import SkillExecutor, SkillResult


def test_skill_serialization():
    """Test Skill to_dict / from_dict roundtrip."""
    skill = Skill(
        name="test_skill",
        description="A test skill",
        trigger=SkillTrigger.PATTERN,
        trigger_patterns=[r"deploy\s+\w+", "release"],
        steps=[SkillStep(tool="bash", args_template={"command": "echo {{target}}"}, description="Echo target")],
        quality_gates=[QualityGate(name="check_output", check="output_not_empty", severity="block")],
        source="manual",
    )
    d = skill.to_dict()
    restored = Skill.from_dict(d)
    assert restored.name == "test_skill"
    assert restored.trigger == SkillTrigger.PATTERN
    assert len(restored.steps) == 1
    assert len(restored.quality_gates) == 1
    assert restored.steps[0].tool == "bash"
    assert restored.quality_gates[0].check == "output_not_empty"


def test_skill_success_rate():
    skill = Skill(name="t", description="", success_count=8, fail_count=2)
    assert abs(skill.success_rate - 0.8) < 0.01

    empty = Skill(name="e", description="")
    assert empty.success_rate == 0.0


def test_skill_match_pattern():
    with tempfile.TemporaryDirectory() as td:
        mgr = SkillManager(skills_dir=td)
        deploy_skill = Skill(
            name="deploy", description="Deploy applications",
            trigger=SkillTrigger.PATTERN, trigger_patterns=[r"deploy\s+\w+"],
        )
        mgr._skills["deploy"] = deploy_skill

        matches = mgr.match("deploy production")
        assert len(matches) == 1
        assert matches[0].name == "deploy"


def test_skill_match_keyword():
    with tempfile.TemporaryDirectory() as td:
        mgr = SkillManager(skills_dir=td)
        mgr._skills["search"] = Skill(
            name="search", description="Search the web for information",
            trigger=SkillTrigger.AUTO,
        )
        mgr._skills["code"] = Skill(
            name="code", description="Write Python code",
            trigger=SkillTrigger.AUTO,
        )

        matches = mgr.match("search for AI news")
        assert len(matches) >= 1
        assert matches[0].name == "search"


def test_skill_match_always():
    with tempfile.TemporaryDirectory() as td:
        mgr = SkillManager(skills_dir=td)
        mgr._skills["safety"] = Skill(
            name="safety", description="Safety checks",
            trigger=SkillTrigger.ALWAYS,
        )
        # Should match any task
        matches = mgr.match("literally anything")
        assert any(s.name == "safety" for s in matches)


def test_skill_persistence():
    with tempfile.TemporaryDirectory() as td:
        mgr1 = SkillManager(skills_dir=td)
        skill = Skill(name="persist_test", description="Test persistence", version=2)
        mgr1.save(skill)

        mgr2 = SkillManager(skills_dir=td)
        mgr2.load_all()
        loaded = mgr2.get("persist_test")
        assert loaded is not None
        assert loaded.version == 2


def test_skill_record_outcome():
    with tempfile.TemporaryDirectory() as td:
        mgr = SkillManager(skills_dir=td)
        skill = Skill(name="outcome", description="Test outcomes")
        mgr.save(skill)

        mgr.record_outcome("outcome", True)
        mgr.record_outcome("outcome", True)
        mgr.record_outcome("outcome", False)
        assert mgr.get("outcome").success_count == 2
        assert mgr.get("outcome").fail_count == 1


@pytest.mark.asyncio
async def test_skill_auto_create():
    with tempfile.TemporaryDirectory() as td:
        mgr = SkillManager(skills_dir=td)
        # Simulate trajectory with repeated tool calls
        import json
        trajectory = [
            {"from": "human", "value": "test task"},
            {"from": "function_call", "value": json.dumps({"name": "bash", "arguments": {"command": "echo 1"}})},
            {"from": "function_call", "value": json.dumps({"name": "bash", "arguments": {"command": "echo 2"}})},
            {"from": "function_call", "value": json.dumps({"name": "bash", "arguments": {"command": "echo 3"}})},
            {"from": "gpt", "value": "Done"},
        ]
        skill = await mgr.auto_create(trajectory, task="echo commands")
        assert skill is not None
        assert skill.source == "auto_created"


@pytest.mark.asyncio
async def test_executor_quality_gates():
    """Test quality gate enforcement."""
    async def mock_dispatch(tool, args):
        return "success output"

    executor = SkillExecutor(tool_dispatch_fn=mock_dispatch)
    skill = Skill(
        name="gated",
        description="Skill with gates",
        steps=[SkillStep(tool="bash", args_template={"command": "test"})],
        quality_gates=[
            QualityGate(name="not_empty", check="output_not_empty", severity="block"),
            QualityGate(name="has_success", check="output_contains", expected="success", severity="warn"),
        ],
    )

    result = await executor.execute(skill)
    assert result.success is True
    assert all(g["passed"] for g in result.gate_results)


@pytest.mark.asyncio
async def test_executor_gate_blocks():
    """Test that a blocking gate stops execution."""
    async def mock_dispatch(tool, args):
        return ""  # Empty output

    executor = SkillExecutor(tool_dispatch_fn=mock_dispatch)
    skill = Skill(
        name="will_fail",
        description="This will fail",
        steps=[SkillStep(tool="bash", args_template={})],
        quality_gates=[
            QualityGate(name="must_have_output", check="output_not_empty", severity="block"),
        ],
    )

    result = await executor.execute(skill)
    assert result.success is False
    assert result.blocked_by == "must_have_output"
