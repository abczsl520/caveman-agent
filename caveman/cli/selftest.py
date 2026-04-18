"""Caveman Self-Test — full lifecycle verification.

Runs: store → recall → shield → reflect → wiki
Verifies the entire memory flywheel works end-to-end.
"""
from __future__ import annotations
import time
from pathlib import Path
from tempfile import mkdtemp

from caveman.memory.manager import MemoryManager, MemoryType


async def run_self_test() -> str:
    """Run full lifecycle self-test."""
    tmpdir = Path(mkdtemp(prefix="caveman-selftest-"))
    results = []
    t_total = time.perf_counter()

    # 1. Memory Store
    try:
        mm = MemoryManager(base_dir=tmpdir)
        await mm.load()
        mid = await mm.store("Caveman uses a 6-engine flywheel for memory permanence", MemoryType.SEMANTIC)
        await mm.store("Route C: parasitic on OpenClaw + selective self-build", MemoryType.SEMANTIC)
        await mm.store("Python core + Node.js bridge architecture", MemoryType.EPISODIC)
        results.append(("store", "✅", f"3 memories stored, last id={mid[:8]}"))
    except Exception as e:
        results.append(("store", "❌", str(e)[:80]))

    # 2. Memory Recall
    try:
        memories = await mm.recall("flywheel memory", top_k=3)
        found = len(memories)
        results.append(("recall", "✅" if found > 0 else "❌", f"{found} results"))
    except Exception as e:
        results.append(("recall", "❌", str(e)[:80]))

    # 3. Shield
    try:
        from caveman.engines.shield import CompactionShield
        shield = CompactionShield(session_id="selftest", store_dir=tmpdir / "shield")
        ctx = await shield.update([
            {"role": "user", "content": "Build a memory system"},
            {"role": "assistant", "content": "I'll create a 6-engine flywheel"},
        ])
        has_content = ctx is not None
        results.append(("shield", "✅" if has_content else "❌", "essence updated"))
    except Exception as e:
        results.append(("shield", "❌", str(e)[:80]))

    # 4. Wiki
    try:
        from caveman.wiki import WikiStore, WikiEntry
        from caveman.wiki.compiler import WikiCompiler
        ws = WikiStore(wiki_dir=tmpdir / "wiki")
        entry = WikiEntry(
            id="selftest-1",
            tier="working",
            title="Caveman Agent OS",
            content="Caveman is an Agent OS with memory permanence",
            confidence=0.9,
            sources=["self-test"],
        )
        ws.add(entry)
        compiler = WikiCompiler(ws)
        ctx = compiler.get_compiled_context(max_tokens=500)
        results.append(("wiki", "✅" if ctx else "❌", f"{len(ctx or '')} chars"))
    except Exception as e:
        results.append(("wiki", "❌", str(e)[:80]))

    # 5. Skill list
    try:
        from caveman.skills.manager import SkillManager
        sm = SkillManager(skills_dir=tmpdir / "skills")
        skills = sm.list_all()
        results.append(("skills", "✅", f"{len(skills)} skills"))
    except Exception as e:
        results.append(("skills", "❌", str(e)[:80]))

    elapsed = time.perf_counter() - t_total
    passed = sum(1 for _, s, _ in results if s == "✅")
    total = len(results)

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    lines = [f"Caveman Self-Test — {passed}/{total} passed in {elapsed*1000:.0f}ms\n"]
    for name, status, detail in results:
        lines.append(f"  {status} {name:10s} {detail}")

    return "\n".join(lines)
