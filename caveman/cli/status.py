"""CLI Status Dashboard — quick overview of Caveman's state."""
from __future__ import annotations

import json
import logging
from typing import Any

from caveman.paths import CAVEMAN_HOME, MEMORY_DIR, SESSIONS_DIR, SKILLS_DIR, PROJECTS_DIR

logger = logging.getLogger(__name__)


def _count_memories() -> dict[str, int]:
    """Count memory entries by type from JSON files."""
    counts: dict[str, int] = {}
    mem_dir = MEMORY_DIR
    if not mem_dir.exists():
        return counts
    for f in mem_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                counts[f.stem] = len(data)
            elif isinstance(data, dict):
                counts[f.stem] = len(data)
        except Exception as e:
            logger.debug("Suppressed in status: %s", e)
    return counts


def _count_sessions() -> int:
    """Count session essence YAML files."""
    if not SESSIONS_DIR.exists():
        return 0
    return len(list(SESSIONS_DIR.glob("*.yaml")))


def _count_skills() -> tuple[int, int]:
    """Count skills (active, draft). Returns (active, draft)."""
    if not SKILLS_DIR.exists():
        return 0, 0
    active = draft = 0
    for d in SKILLS_DIR.iterdir():
        if d.is_dir() and (d / "skill.yaml").exists():
            try:
                text = (d / "skill.yaml").read_text(encoding="utf-8")
                if "status: draft" in text:
                    draft += 1
                else:
                    active += 1
            except Exception:
                active += 1
    return active, draft


def _count_projects() -> int:
    """Count tracked projects."""
    if not PROJECTS_DIR.exists():
        return 0
    return len(list(PROJECTS_DIR.glob("*.json")))


def _check_engines() -> dict[str, bool]:
    """Check engine flags from config."""
    from caveman.engines.flags import EngineFlags
    try:
        from caveman.config.loader import load_config
        config = load_config()
    except Exception:
        config = {}
    flags = EngineFlags(config)
    engines = ["shield", "nudge", "recall", "ripple", "lint", "reflect"]
    return {e: flags.is_enabled(e) for e in engines}


def _get_model_info() -> tuple[str, str]:
    """Get configured model and provider."""
    try:
        from caveman.config.loader import load_config
        config = load_config()
        model = config.get("agent", {}).get("default_model", "unknown")
        if "claude" in model or "anthropic" in model:
            provider = "Anthropic"
        elif "gpt" in model or "openai" in model:
            provider = "OpenAI"
        elif "ollama" in model:
            provider = "Ollama"
        else:
            provider = "Unknown"
        return model, provider
    except Exception:
        return "unknown", "Unknown"


def status_text() -> str:
    """Generate status dashboard text."""
    model, provider = _get_model_info()
    mem_counts = _count_memories()
    total_mem = sum(mem_counts.values())
    sessions = _count_sessions()
    active_skills, draft_skills = _count_skills()
    projects = _count_projects()
    engines = _check_engines()

    engine_str = " ".join(
        f"{name.capitalize()} {'✅' if on else '❌'}" for name, on in engines.items()
    )

    mem_detail = ", ".join(f"{k}: {v}" for k, v in sorted(mem_counts.items()) if v > 0)

    lines = [
        "🦴 Caveman v0.3.0-dev",
        f"  Model: {model} ({provider})",
        f"  Memory: {total_mem:,} entries" + (f" ({mem_detail})" if mem_detail else ""),
        f"  Sessions: {sessions} essences",
        f"  Skills: {active_skills} active, {draft_skills} draft",
        f"  Projects: {projects} tracked",
        f"  Engines: {engine_str}",
        f"  Home: {CAVEMAN_HOME}",
    ]
    return "\n".join(lines)


def main():
    """Print status dashboard."""
    print(status_text())


if __name__ == "__main__":
    main()
