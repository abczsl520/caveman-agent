"""Skills sync — manifest-based seeding and updating of bundled skills.

Copies bundled skills from the repo's skills/ directory into ~/.caveman/skills/
and uses a manifest to track which skills have been synced and their origin hash.

Update logic:
  - NEW skills (not in manifest): copied to user dir, origin hash recorded.
  - EXISTING (user copy matches origin hash): safe to update from bundled.
  - EXISTING (user copy differs from origin hash): user customized → SKIP.
  - DELETED by user (in manifest, absent from user dir): respected, not re-added.
  - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

_USER_SKILLS_DIR = CAVEMAN_HOME / "skills"
_MANIFEST_FILE = _USER_SKILLS_DIR / ".bundled_manifest"


def _dir_hash(path: Path) -> str:
    """Compute MD5 hash of all files in a directory (sorted, deterministic)."""
    h = hashlib.md5()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load manifest: {skill_name: origin_hash}."""
    if not manifest_path.exists():
        return {}
    result = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            name, hash_val = line.split(":", 1)
            result[name.strip()] = hash_val.strip()
        else:
            # v1 format: plain name without hash
            result[line] = ""
    return result


def _save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    """Save manifest to disk."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}:{hash_val}" for name, hash_val in sorted(manifest.items())]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_skills(
    bundled_dir: Path,
    user_dir: Path | None = None,
    manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync bundled skills to user directory.

    Args:
        bundled_dir: Path to bundled skills (e.g., repo/skills/)
        user_dir: Target directory (default: ~/.caveman/skills/)
        manifest_path: Manifest file path (default: user_dir/.bundled_manifest)
        dry_run: If True, report what would happen without making changes.

    Returns dict with keys: added, updated, skipped_customized, skipped_deleted, removed.
    """
    udir = user_dir or _USER_SKILLS_DIR
    mpath = manifest_path or (udir / ".bundled_manifest")

    manifest = _load_manifest(mpath)
    result: dict[str, list[str]] = {
        "added": [],
        "updated": [],
        "skipped_customized": [],
        "skipped_deleted": [],
        "removed_from_manifest": [],
    }

    if not bundled_dir.exists():
        logger.warning("Bundled skills dir not found: %s", bundled_dir)
        return result

    # Scan bundled skills
    bundled_names = set()
    for skill_dir in sorted(bundled_dir.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        bundled_names.add(skill_dir.name)

        target = udir / skill_dir.name
        bundled_hash = _dir_hash(skill_dir)

        if skill_dir.name not in manifest:
            # NEW skill
            if target.exists():
                # User already has it (manual install) — record but don't overwrite
                result["skipped_customized"].append(skill_dir.name)
                manifest[skill_dir.name] = _dir_hash(target)
            else:
                if not dry_run:
                    udir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(skill_dir, target)
                manifest[skill_dir.name] = bundled_hash
                result["added"].append(skill_dir.name)
                logger.info("Skills sync: added '%s'", skill_dir.name)
        else:
            origin_hash = manifest[skill_dir.name]

            if not target.exists():
                # User deleted it — respect that
                result["skipped_deleted"].append(skill_dir.name)
                continue

            current_hash = _dir_hash(target)

            if current_hash == origin_hash:
                # User hasn't modified — safe to update
                if bundled_hash != origin_hash:
                    if not dry_run:
                        shutil.rmtree(target)
                        shutil.copytree(skill_dir, target)
                    manifest[skill_dir.name] = bundled_hash
                    result["updated"].append(skill_dir.name)
                    logger.info("Skills sync: updated '%s'", skill_dir.name)
            else:
                # User customized — skip
                result["skipped_customized"].append(skill_dir.name)
                logger.debug("Skills sync: skipped '%s' (user customized)", skill_dir.name)

    # Clean manifest entries for skills removed from bundled
    for name in list(manifest.keys()):
        if name not in bundled_names:
            del manifest[name]
            result["removed_from_manifest"].append(name)

    if not dry_run:
        _save_manifest(mpath, manifest)

    return result
