"""Skills sync — discover, checksum, diff, and sync skills from remote/bundled.

Full port from Hermes tools/skills_sync.py:
- Bundled skills discovery (shipped with the package)
- Manifest read/write with checksums
- Directory hashing for change detection
- Skill name extraction from SKILL.md
- Relative path computation for nested skills
- Full sync with conflict detection
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

__all__ = [
    "SKILLS_DIR",
    "BUNDLED_DIR",
    "MANIFEST_FILE",
    "SYNC_STATE_FILE",
    "SkillManifest",
    "SyncState",
    "SyncDiff",
    "compute_skill_checksum",
    "scan_local_skills",
    "load_sync_state",
    "save_sync_state",
    "diff_skills",
    "sync_skill_from_remote",
    "sync_all_bundled",
    "get_sync_status",
]


logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

SKILLS_DIR = Path.home() / ".caveman" / "skills"
BUNDLED_DIR = Path(__file__).parent.parent.parent / "skills" / "bundled"
MANIFEST_FILE = SKILLS_DIR / ".manifest.json"
SYNC_STATE_FILE = SKILLS_DIR / ".sync_state.json"


# ── Data classes ───────────────────────────────────────────────────────────

class _SyncAllResult(TypedDict):
    synced: int
    unchanged: int
    conflicts: int
    errors: int
    details: list[dict[str, Any]]


@dataclass
class SkillManifest:
    """Manifest entry for a single skill."""
    name: str
    path: str
    checksum: str
    version: str = ""
    source: str = "local"
    last_synced: float = 0.0


@dataclass
class SyncState:
    """Global sync state."""
    last_sync: float = 0.0
    synced_skills: Dict[str, str] = field(default_factory=dict)  # name → checksum
    errors: List[str] = field(default_factory=list)


@dataclass
class SyncDiff:
    """Diff between local and remote skills."""
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


# ── Checksumming ───────────────────────────────────────────────────────────

def compute_skill_checksum(skill_dir: Path) -> str:
    """Compute a stable checksum for a skill directory.

    Hashes all files sorted by relative path for deterministic output.
    """
    h = hashlib.sha256()
    if not skill_dir.exists():
        return ""
    files = sorted(
        f for f in skill_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    )
    for f in files:
        rel = str(f.relative_to(skill_dir))
        h.update(rel.encode("utf-8"))
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()[:16]


def _dir_hash(directory: Path) -> str:
    """Compute hash of a directory (alias for compute_skill_checksum)."""
    return compute_skill_checksum(directory)


# ── Skill name extraction ──────────────────────────────────────────────────

def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Extract skill name from SKILL.md header, or use fallback."""
    try:
        text = skill_md.read_text(encoding="utf-8")
        for line in text.splitlines()[:10]:
            line = line.strip()
            if line.startswith("# "):
                name = line[2:].strip()
                if name:
                    return name
    except OSError:
        pass  # intentional: OSError suppressed
    return fallback


# ── Discovery ──────────────────────────────────────────────────────────────

def _discover_bundled_skills(bundled_dir: Path) -> List[Tuple[str, Path]]:
    """Discover all bundled skills (shipped with the package).

    Returns list of (name, path) tuples.
    """
    if not bundled_dir.exists():
        return []
    results: List[Tuple[str, Path]] = []
    for entry in sorted(bundled_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if skill_md.exists():
            name = _read_skill_name(skill_md, entry.name)
            results.append((name, entry))
        else:
            # Check subdirectories (nested skills)
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "SKILL.md").exists():
                    name = _read_skill_name(sub / "SKILL.md", sub.name)
                    results.append((name, sub))
    return results


def scan_local_skills(skills_dir: Optional[Path] = None) -> List[SkillManifest]:
    """Scan local skills directory and return manifests."""
    sdir = skills_dir or SKILLS_DIR
    if not sdir.exists():
        return []
    manifests: List[SkillManifest] = []
    for entry in sorted(sdir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            continue
        name = _read_skill_name(skill_md, entry.name)
        checksum = compute_skill_checksum(entry)
        manifests.append(SkillManifest(
            name=name, path=str(entry), checksum=checksum, source="local",
        ))
    return manifests


def _compute_relative_dest(skill_dir: Path, bundled_dir: Path) -> Path:
    """Compute the relative destination path for a bundled skill."""
    try:
        rel = skill_dir.relative_to(bundled_dir)
        return SKILLS_DIR / rel
    except ValueError:
        return SKILLS_DIR / skill_dir.name


# ── Manifest I/O ──────────────────────────────────────────────────────────

def _read_manifest() -> Dict[str, str]:
    """Read the manifest file (name → checksum mapping)."""
    if not MANIFEST_FILE.exists():
        return {}
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass  # intentional: OSError suppressed
    return {}


def _write_manifest(entries: Dict[str, str]) -> None:
    """Write the manifest file."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Sync state I/O ────────────────────────────────────────────────────────

def load_sync_state(state_file: Optional[Path] = None) -> SyncState:
    """Load sync state from disk."""
    sf = state_file or SYNC_STATE_FILE
    if not sf.exists():
        return SyncState()
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
        return SyncState(
            last_sync=data.get("last_sync", 0.0),
            synced_skills=data.get("synced_skills", {}),
            errors=data.get("errors", []),
        )
    except (json.JSONDecodeError, OSError):
        return SyncState()


def save_sync_state(state: SyncState, state_file: Optional[Path] = None) -> None:
    """Save sync state to disk."""
    sf = state_file or SYNC_STATE_FILE
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps({
        "last_sync": state.last_sync,
        "synced_skills": state.synced_skills,
        "errors": state.errors[-20:],  # Keep last 20 errors
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Diff ───────────────────────────────────────────────────────────────────

def diff_skills(
    local_manifests: List[SkillManifest],
    remote_manifests: List[SkillManifest],
) -> SyncDiff:
    """Compute diff between local and remote skill sets."""
    local_map = {m.name: m for m in local_manifests}
    remote_map = {m.name: m for m in remote_manifests}

    diff = SyncDiff()
    for name, remote in remote_map.items():
        if name not in local_map:
            diff.added.append(name)
        elif local_map[name].checksum != remote.checksum:
            diff.updated.append(name)
        else:
            diff.unchanged.append(name)

    for name in local_map:
        if name not in remote_map:
            diff.removed.append(name)

    return diff


# ── Sync operations ────────────────────────────────────────────────────────

def sync_skill_from_remote(
    name: str,
    source_dir: Path,
    *,
    force: bool = False,
    skills_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Sync a single skill from a remote/bundled source.

    Returns dict with success status and details.
    """
    sdir = skills_dir or SKILLS_DIR
    dest = sdir / name

    if dest.exists() and not force:
        local_checksum = compute_skill_checksum(dest)
        remote_checksum = compute_skill_checksum(source_dir)
        if local_checksum == remote_checksum:
            return {"status": "unchanged", "name": name}
        # Check if local has modifications not in remote
        manifest = _read_manifest()
        if name in manifest and manifest[name] != local_checksum:
            return {
                "status": "conflict", "name": name,
                "message": "Local skill has been modified. Use force=True to overwrite.",
            }

    # Perform the copy
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
    except OSError as e:
        return {"status": "error", "name": name, "error": str(e)}

    # Update manifest
    new_checksum = compute_skill_checksum(dest)
    manifest = _read_manifest()
    manifest[name] = new_checksum
    _write_manifest(manifest)

    return {"status": "synced", "name": name, "checksum": new_checksum}


def sync_all_bundled(*, force: bool = False, quiet: bool = False) -> Dict[str, Any]:
    """Sync all bundled skills to the local skills directory.

    Returns summary with counts and any errors.
    """
    bundled = _discover_bundled_skills(BUNDLED_DIR)
    if not bundled:
        return {
            "synced": 0,
            "unchanged": 0,
            "conflicts": 0,
            "errors": 0,
            "details": [],
            "message": "No bundled skills found.",
        }

    results: _SyncAllResult = {"synced": 0, "unchanged": 0, "conflicts": 0, "errors": 0, "details": []}
    state = load_sync_state()

    for name, source_path in bundled:
        result = sync_skill_from_remote(name, source_path, force=force)
        status = result.get("status", "error")
        if status == "synced":
            results["synced"] += 1
            state.synced_skills[name] = result.get("checksum", "")
            if not quiet:
                logger.info("Synced skill: %s", name)
        elif status == "unchanged":
            results["unchanged"] += 1
        elif status == "conflict":
            results["conflicts"] += 1
            if not quiet:
                logger.warning("Conflict: %s — %s", name, result.get("message", ""))
        else:
            results["errors"] += 1
            state.errors.append(f"{name}: {result.get('error', 'unknown')}")
            if not quiet:
                logger.error("Error syncing %s: %s", name, result.get("error", ""))
        results["details"].append(result)

    state.last_sync = time.time()
    save_sync_state(state)

    if not quiet:
        logger.info(
            "Sync complete: %d synced, %d unchanged, %d conflicts, %d errors",
            results["synced"], results["unchanged"], results["conflicts"], results["errors"],
        )
    return dict(results)


def get_sync_status() -> Dict[str, Any]:
    """Get comprehensive sync status."""
    state = load_sync_state()
    local = scan_local_skills()
    bundled = _discover_bundled_skills(BUNDLED_DIR)

    return {
        "local_count": len(local),
        "bundled_count": len(bundled),
        "last_sync": state.last_sync,
        "last_sync_ago": time.time() - state.last_sync if state.last_sync else None,
        "synced_skills": list(state.synced_skills.keys()),
        "recent_errors": state.errors[-5:],
    }
