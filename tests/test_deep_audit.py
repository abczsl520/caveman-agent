"""Tests for deep flywheel audit fixes."""
import pytest
from datetime import datetime, timedelta


# ── #1: Multi-turn recall ──

def test_multi_turn_has_recall_code():
    """Loop.py should re-recall memories in multi-turn conversations."""
    from pathlib import Path
    loop_src = (Path(__file__).parent.parent / "caveman/agent/loop.py").read_text()
    assert "re-recall" in loop_src.lower() or \
           "memory_manager.recall" in loop_src, \
        "Multi-turn should re-recall memories for each new task"


# ── #2: No double trust boost ──

def test_no_double_trust_boost():
    """Reflect should NOT boost trust — finalize already does it."""
    from pathlib import Path
    loop_src = (Path(__file__).parent.parent / "caveman/agent/loop.py").read_text()
    # Should not have _boost_trust in the reflect section
    reflect_section = loop_src[loop_src.find("Phase 9"):]
    assert "_boost_trust" not in reflect_section, \
        "Reflect should not double-boost trust (finalize already does it)"


# ── #3: Memory injection includes type and age ──

def test_memory_injection_has_type():
    """System prompt should include memory type for LLM context."""
    from caveman.agent.prompt import build_system_prompt
    memories = [
        {"content": "Server runs on port 8080", "type": "semantic", "age_days": 5},
        {"content": "Deploy with docker compose", "type": "procedural", "age_days": 30},
    ]
    prompt = build_system_prompt(memories=memories)
    assert "[semantic]" in prompt
    assert "[procedural]" in prompt
    assert "30d ago" in prompt  # Old memory should show age
    assert "5d ago" not in prompt  # Recent memory (< 7d) should not show age


# ── #4: Skill auto_create threshold ──

def test_skill_autocreate_needs_tools():
    """Skill auto-creation should require tool usage, not just conversation length."""
    from pathlib import Path
    phases_src = (Path(__file__).parent.parent / "caveman/agent/phases.py").read_text()
    assert "tool_turns" in phases_src or "function_call" in phases_src, \
        "Skill auto-create should check for tool usage"
    assert ">= 6" in phases_src or ">= 8" in phases_src, \
        "Skill auto-create threshold should be >= 6 turns"


# ── #5: Nudge extracts root causes and solutions ──

def test_nudge_extracts_solutions():
    """Nudge heuristic should extract 'fixed by X' patterns."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from caveman.memory.manager import MemoryManager
    from caveman.memory.nudge import MemoryNudge

    mm = MemoryManager(base_dir="/tmp/test_nudge_sol")
    nudge = MemoryNudge(memory_manager=mm)
    turns = [
        {"role": "assistant", "content": "I added cross-language query expansion to fix the search issue."},
        {"role": "assistant", "content": "The root cause was that FTS5 only does lexical matching."},
    ]
    candidates = nudge._extract_heuristic(turns, "fix search")
    contents = " ".join(c["content"] for c in candidates)
    assert "added" in contents.lower() or "root cause" in contents.lower(), \
        f"Should extract solutions/root causes, got: {contents}"


# ── #6: Skill load_all caching ──

def test_skill_load_all_cached(tmp_path):
    """load_all() should only read disk once."""
    from caveman.skills.manager import SkillManager
    from caveman.skills.types import Skill, SkillTrigger

    sm = SkillManager(skills_dir=tmp_path)
    skill = Skill(name="test", description="test skill")
    sm.save(skill)

    # Reset loaded flag to simulate fresh start
    sm._loaded = False
    sm._skills.clear()

    sm.load_all()
    assert len(sm._skills) == 1
    assert sm._loaded is True

    # Second call should be a no-op (cached)
    sm._skills["injected"] = Skill(name="injected", description="injected")
    sm.load_all()  # Should NOT reload from disk
    assert "injected" in sm._skills  # Still there = cache worked
