"""Skill manager tool — list, show, delete learned skills."""
from __future__ import annotations

import logging

from caveman.paths import SKILLS_DIR
from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


@tool(
    name="skill_list",
    description="List all learned skills",
    params={},
    required=[],
)
async def skill_list() -> list[dict]:
    """List all skill files in the skills directory."""
    if not SKILLS_DIR.exists() or yaml is None:
        return []
    results = []
    for f in sorted(SKILLS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
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
    if "/" in name or "\\" in name or ".." in name:
        return {"error": f"Invalid skill name: {name}"}
    path = SKILLS_DIR / f"{name}.yaml"
    if not path.resolve().parent == SKILLS_DIR.resolve():
        return {"error": f"Invalid skill name: {name}"}
    if not path.exists():
        return {"error": f"Skill '{name}' not found"}
    if yaml is None:
        return {"error": "pyyaml not installed"}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    if "/" in name or "\\" in name or ".." in name:
        return {"error": f"Invalid skill name: {name}"}
    path = SKILLS_DIR / f"{name}.yaml"
    if not path.resolve().parent == SKILLS_DIR.resolve():
        return {"error": f"Invalid skill name: {name}"}
    if not path.exists():
        return {"error": f"Skill '{name}' not found"}
    path.unlink()
    return {"ok": True}
