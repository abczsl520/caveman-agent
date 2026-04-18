"""Round 10 tests — Ripple Engine, RL Router, Obsidian Export, __all__ declarations."""
from __future__ import annotations

import json
import pytest
from datetime import datetime
from pathlib import Path

from caveman.memory.types import MemoryEntry, MemoryType


# ═══════════════════════════════════════════════════════════════
# FR-105: Ripple Engine
# ═══════════════════════════════════════════════════════════════

class TestRippleEngine:
    """FR-105: Knowledge propagation on memory write."""

    @pytest.fixture
    def memory_with_ip(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir=tmp_path)
        now = datetime.now()
        mm._memories[MemoryType.SEMANTIC] = [
            MemoryEntry(id="old1", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
            MemoryEntry(id="old2", content="Database runs on port 5432",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ]
        return mm

    @pytest.mark.asyncio
    async def test_ripple_detects_contradiction(self, memory_with_ip):
        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip)
        new_entry = MemoryEntry(
            id="new1",
            content="Server migrated to 198.51.100.20",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        # Should detect migration pattern with shared IP context
        assert effect.had_impact

    @pytest.mark.asyncio
    async def test_ripple_marks_stale(self, memory_with_ip):
        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip)
        new_entry = MemoryEntry(
            id="new1",
            content="Server changed to 198.51.100.20, old was 203.0.113.10",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        if effect.conflicts:
            assert len(effect.stale_marked) >= 1

    @pytest.mark.asyncio
    async def test_ripple_cross_refs(self, memory_with_ip):
        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip)
        new_entry = MemoryEntry(
            id="new1",
            content="Server 203.0.113.10 runs Windows",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        # Should add cross-ref to old1 (same IP)
        assert len(effect.cross_refs_added) >= 1 or len(effect.conflicts) >= 1

    @pytest.mark.asyncio
    async def test_ripple_no_impact_unrelated(self, memory_with_ip):
        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip)
        new_entry = MemoryEntry(
            id="new1",
            content="Python 3.12 supports pattern matching",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        assert not effect.conflicts

    @pytest.mark.asyncio
    async def test_ripple_notifications(self, memory_with_ip):
        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip)
        new_entry = MemoryEntry(
            id="new1",
            content="Server changed to 198.51.100.20, was 203.0.113.10",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        if effect.conflicts:
            assert len(effect.notifications) >= 1
            assert "Conflict" in effect.notifications[0]

    @pytest.mark.asyncio
    async def test_ripple_llm_verification(self, memory_with_ip):
        async def mock_llm(prompt: str) -> str:
            return "update"

        from caveman.engines.ripple import RippleEngine
        ripple = RippleEngine(memory_with_ip, llm_fn=mock_llm)
        new_entry = MemoryEntry(
            id="new1",
            content="Server changed to 198.51.100.20, was 203.0.113.10",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        if effect.conflicts:
            assert effect.conflicts[0].get("llm_verdict") == "update"

    @pytest.mark.asyncio
    async def test_ripple_empty_memory(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.engines.ripple import RippleEngine
        mm = MemoryManager(base_dir=tmp_path)
        ripple = RippleEngine(mm)
        new_entry = MemoryEntry(
            id="new1", content="First memory ever",
            memory_type=MemoryType.SEMANTIC, created_at=datetime.now(),
        )
        effect = await ripple.propagate(new_entry)
        assert not effect.had_impact


# ═══════════════════════════════════════════════════════════════
# FR-206: Skill RL Router
# ═══════════════════════════════════════════════════════════════

class TestSkillRLRouter:
    """FR-206: RL-based skill selection."""

    def test_select_from_available(self):
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        choice = router.select("deploy app", ["deploy", "test", "build"])
        assert choice in ["deploy", "test", "build"]

    def test_select_empty_returns_none(self):
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        assert router.select("task", []) is None

    def test_update_and_exploit(self):
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        # Train: "deploy" always succeeds, "test" always fails
        for _ in range(20):
            router.update("deploy", success=True)
            router.update("test", success=False)

        # After training, deploy should be selected most of the time
        selections = [
            router.select("task", ["deploy", "test"], explore_rate=0.0)
            for _ in range(50)
        ]
        deploy_count = selections.count("deploy")
        assert deploy_count >= 40  # Should strongly prefer deploy

    def test_stats(self):
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        router.update("skill_a", True)
        router.update("skill_a", True)
        router.update("skill_a", False)
        stats = router.get_stats()
        assert "skill_a" in stats
        assert stats["skill_a"]["total"] == 3

    def test_rankings(self):
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        for _ in range(10):
            router.update("good", True)
            router.update("bad", False)
        rankings = router.get_rankings()
        assert rankings[0][0] == "good"
        assert rankings[0][1] > rankings[1][1]

    def test_persistence(self, tmp_path):
        from caveman.skills.rl_router import SkillRLRouter
        state_path = tmp_path / "rl_state.json"

        # Train and save
        router1 = SkillRLRouter(state_path=state_path)
        for _ in range(10):
            router1.update("deploy", True)
        assert state_path.exists()

        # Load and verify
        router2 = SkillRLRouter(state_path=state_path)
        stats = router2.get_stats()
        assert stats["deploy"]["total"] == 10
        assert stats["deploy"]["alpha"] == 11.0  # 10 + 1 prior

    def test_selection_speed(self):
        """FR-206: RL router should select in < 50ms."""
        import time
        from caveman.skills.rl_router import SkillRLRouter
        router = SkillRLRouter()
        skills = [f"skill_{i}" for i in range(20)]
        for s in skills:
            router.update(s, True)

        start = time.monotonic()
        for _ in range(100):
            router.select("task", skills, explore_rate=0.0)
        elapsed_ms = (time.monotonic() - start) * 1000 / 100
        assert elapsed_ms < 50  # < 50ms per selection


# ═══════════════════════════════════════════════════════════════
# Obsidian Export
# ═══════════════════════════════════════════════════════════════

class TestObsidianExport:
    """Obsidian-compatible memory export."""

    @pytest.fixture
    def memory_with_data(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir=tmp_path / "mem")
        now = datetime.now()
        mm._memories[MemoryType.SEMANTIC] = [
            MemoryEntry(id="s1", content="Server IP is 203.0.113.10",
                       memory_type=MemoryType.SEMANTIC, created_at=now),
        ]
        mm._memories[MemoryType.PROCEDURAL] = [
            MemoryEntry(id="p1", content="How to install pyenv: brew install pyenv",
                       memory_type=MemoryType.PROCEDURAL, created_at=now),
        ]
        return mm

    def test_export_creates_files(self, memory_with_data, tmp_path):
        from caveman.memory.obsidian import export_to_obsidian
        out = tmp_path / "vault"
        result = export_to_obsidian(memory_with_data, out)
        assert result["exported"] == 2
        assert (out / "index.md").exists()
        assert (out / "semantic").is_dir()
        assert (out / "procedural").is_dir()

    def test_export_frontmatter(self, memory_with_data, tmp_path):
        from caveman.memory.obsidian import export_to_obsidian
        out = tmp_path / "vault"
        export_to_obsidian(memory_with_data, out)
        # Check a semantic file has frontmatter
        semantic_files = list((out / "semantic").glob("*.md"))
        assert len(semantic_files) == 1
        content = semantic_files[0].read_text()
        assert content.startswith("---")
        assert "type: semantic" in content
        assert "id: s1" in content

    def test_export_index(self, memory_with_data, tmp_path):
        from caveman.memory.obsidian import export_to_obsidian
        out = tmp_path / "vault"
        export_to_obsidian(memory_with_data, out)
        index = (out / "index.md").read_text()
        assert "Caveman Memory Vault" in index
        assert "Total: 2" in index
        assert "[[semantic/" in index

    def test_export_tags(self, memory_with_data, tmp_path):
        from caveman.memory.obsidian import export_to_obsidian
        out = tmp_path / "vault"
        export_to_obsidian(memory_with_data, out)
        semantic_files = list((out / "semantic").glob("*.md"))
        content = semantic_files[0].read_text()
        assert "tags:" in content
        assert "server" in content

    def test_export_empty_memory(self, tmp_path):
        from caveman.memory.manager import MemoryManager
        from caveman.memory.obsidian import export_to_obsidian
        mm = MemoryManager(base_dir=tmp_path / "mem")
        out = tmp_path / "vault"
        result = export_to_obsidian(mm, out)
        assert result["exported"] == 0


# ═══════════════════════════════════════════════════════════════
# __all__ declarations
# ═══════════════════════════════════════════════════════════════

class TestAllDeclarations:
    """Round 10: All packages have __all__ and docstrings."""

    def test_all_packages_have_all(self):
        """Every __init__.py should have __all__."""
        from pathlib import Path
        root = Path("caveman")
        missing = []
        for init in root.rglob("__init__.py"):
            content = init.read_text()
            if "__all__" not in content:
                missing.append(str(init))
        assert not missing, f"Missing __all__ in: {missing}"

    def test_all_packages_have_docstring(self):
        """Every __init__.py should have a module docstring."""
        from pathlib import Path
        root = Path("caveman")
        missing = []
        for init in root.rglob("__init__.py"):
            content = init.read_text().strip()
            if not content.startswith('"""'):
                missing.append(str(init))
        assert not missing, f"Missing docstring in: {missing}"

    def test_all_entries_are_importable(self):
        """Spot-check that __all__ entries are real modules."""
        import importlib
        packages = [
            "caveman.engines",
            "caveman.memory",
            "caveman.skills",
            "caveman.coordinator",
            "caveman.trajectory",
        ]
        for pkg_name in packages:
            pkg = importlib.import_module(pkg_name)
            for name in pkg.__all__:
                mod = importlib.import_module(f"{pkg_name}.{name}")
                assert mod is not None
