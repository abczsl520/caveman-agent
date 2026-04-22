"""Skill manager tool — create, list, show, edit, delete learned skills.

Integrates with Skill Guard for security scanning on create/edit.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from caveman.paths import SKILLS_DIR
from caveman.tools.registry import tool
from caveman.aio import aio_exists, aio_glob, aio_mkdir, aio_read_text, aio_unlink, aio_write_text

__all__ = [
    "skill_create",
    "skill_edit",
    "skill_list",
    "skill_show",
    "skill_delete",
]


logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


def _validate_name(name: str) -> str | None:
    """Validate skill name. Returns error message or None."""
    if "/" in name or "\\" in name or ".." in name:
        return f"Invalid skill name: {name}"
    path = SKILLS_DIR / f"{name}.yaml"
    if not path.resolve().parent == SKILLS_DIR.resolve():
        return f"Invalid skill name: {name}"
    return None


@tool(
    name="skill_create",
    description="Create a new skill from a successful approach",
    params={
        "name": {"type": "string", "description": "Skill name (alphanumeric, hyphens)"},
        "description": {"type": "string", "description": "What this skill does"},
        "trigger": {"type": "string", "description": "When to use this skill"},
        "trigger_patterns": {"type": "array", "description": "Regex patterns that trigger this skill"},
        "steps": {"type": "array", "description": "List of step descriptions"},
    },
    required=["name", "description", "trigger", "steps"],
)
async def skill_create(
    name: str, description: str, trigger: str,
    steps: list = None, trigger_patterns: list = None, **_kw,
) -> dict:
    """Create a new skill file with security scanning."""
    if yaml is None:
        return {"error": "pyyaml not installed"}
    err = _validate_name(name)
    if err:
        return {"error": err}
    path = SKILLS_DIR / f"{name}.yaml"
    if await aio_exists(path):
        return {"error": f"Skill '{name}' already exists. Use skill_edit to update."}

    await aio_mkdir(SKILLS_DIR, parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    skill_data = {
        "name": name,
        "version": 1,
        "description": description,
        "trigger": trigger,
        "trigger_patterns": trigger_patterns or [],
        "steps": steps or [],
        "created_at": now,
        "updated_at": now,
    }
    await aio_write_text(path, yaml.dump(skill_data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    logger.info("Created skill: %s", name)
    return {"ok": True, "name": name, "path": str(path)}


@tool(
    name="skill_edit",
    description="Update an existing skill",
    params={
        "name": {"type": "string", "description": "Skill name"},
        "description": {"type": "string", "description": "Updated description"},
        "trigger": {"type": "string", "description": "Updated trigger"},
        "trigger_patterns": {"type": "array", "description": "Updated trigger patterns"},
        "steps": {"type": "array", "description": "Updated steps"},
    },
    required=["name"],
)
async def skill_edit(
    name: str, description: str = "", trigger: str = "",
    steps: list = None, trigger_patterns: list = None, **_kw,
) -> dict:
    """Update an existing skill."""
    if yaml is None:
        return {"error": "pyyaml not installed"}
    err = _validate_name(name)
    if err:
        return {"error": err}
    path = SKILLS_DIR / f"{name}.yaml"
    if not await aio_exists(path):
        return {"error": f"Skill '{name}' not found"}

    data = yaml.safe_load(await aio_read_text(path, encoding="utf-8")) or {}
    if description:
        data["description"] = description
    if trigger:
        data["trigger"] = trigger
    if trigger_patterns is not None:
        data["trigger_patterns"] = trigger_patterns
    if steps is not None:
        data["steps"] = steps
    data["version"] = data.get("version", 1) + 1
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    await aio_write_text(path, yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    logger.info("Updated skill: %s (v%d)", name, data["version"])
    return {"ok": True, "name": name, "version": data["version"]}


@tool(
    name="skill_list",
    description="List all learned skills",
    params={},
    required=[],
)
async def skill_list() -> list[dict]:
    """List all skill files in the skills directory."""
    if not await aio_exists(SKILLS_DIR) or yaml is None:
        return []
    results = []
    for f in sorted(await aio_glob(SKILLS_DIR, "*.yaml")):
        try:
            data = yaml.safe_load(await aio_read_text(f, encoding="utf-8"))
            if not data:
                continue
            results.append({
                "name": data.get("name", f.stem),
                "version": data.get("version", 1),
                "description": data.get("description", ""),
                "last_used": data.get("updated_at", ""),
            })
        except (yaml.YAMLError, OSError):
            logger.warning("Skipping corrupt skill file: %s", f)
    return results


@tool(
    name="skill_show",
    description="Show a skill's details",
    params={
        "name": {"type": "string", "description": "Skill name"},
    },
    required=["name"],
)
async def skill_show(name: str) -> dict:
    """Read and return a skill's full details."""
    err = _validate_name(name)
    if err:
        return {"error": err}
    path = SKILLS_DIR / f"{name}.yaml"
    if not await aio_exists(path):
        return {"error": f"Skill '{name}' not found"}
    if yaml is None:
        return {"error": "pyyaml not installed"}
    try:
        data = yaml.safe_load(await aio_read_text(path, encoding="utf-8"))
        if not data:
            return {"error": f"Skill '{name}' is empty"}
        return {
            "name": data.get("name", name),
            "version": data.get("version", 1),
            "description": data.get("description", ""),
            "trigger": data.get("trigger", ""),
            "trigger_patterns": data.get("trigger_patterns", []),
            "steps": data.get("steps", []),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        }
    except (yaml.YAMLError, OSError) as e:
        return {"error": f"Failed to read skill '{name}': {e}"}


@tool(
    name="skill_delete",
    description="Delete a skill",
    params={
        "name": {"type": "string", "description": "Skill name"},
    },
    required=["name"],
)
async def skill_delete(name: str) -> dict:
    """Delete a skill file."""
    err = _validate_name(name)
    if err:
        return {"error": err}
    path = SKILLS_DIR / f"{name}.yaml"
    if not await aio_exists(path):
        return {"error": f"Skill '{name}' not found"}
    await aio_unlink(path)
    return {"ok": True}
