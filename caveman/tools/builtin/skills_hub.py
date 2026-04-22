"""Skills Hub — skill discovery, installation, and management.

Extracted from Hermes skills_hub.py (2775 lines).
Key patterns: GitHub source, local source, search, install, quarantine, audit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("caveman.tools.skills_hub")

# Paths
SKILLS_DIR = Path.home() / ".caveman" / "skills"
HUB_CACHE_DIR = Path.home() / ".caveman" / "hub_cache"
QUARANTINE_DIR = Path.home() / ".caveman" / "quarantine"
LOCK_FILE = Path.home() / ".caveman" / "skills.lock"
TAPS_FILE = Path.home() / ".caveman" / "taps.json"
AUDIT_LOG = Path.home() / ".caveman" / "skills_audit.jsonl"


@dataclass
class SkillMeta:
    """Metadata for a skill."""
    name: str
    description: str = ""
    version: str = "0.0.0"
    author: str = ""
    source: str = ""  # github:repo/path | local:/path | hub:name
    category: str = ""
    tags: List[str] = field(default_factory=list)
    trust_level: str = "untrusted"  # untrusted | community | verified | official
    installed_at: float = 0
    updated_at: float = 0


@dataclass
class SkillBundle:
    """A skill package ready for installation."""
    meta: SkillMeta
    files: Dict[str, str] = field(default_factory=dict)  # relative_path → content
    content_hash: str = ""

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        for path in sorted(self.files.keys()):
            h.update(path.encode())
            h.update(self.files[path].encode())
        self.content_hash = h.hexdigest()[:16]
        return self.content_hash


class SkillSource(ABC):
    """Abstract skill source."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        ...

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        ...

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        ...

    @property
    @abstractmethod
    def source_id(self) -> str:
        ...


class LocalSource(SkillSource):
    """Search and fetch skills from local directories."""

    def __init__(self, directories: Optional[List[str]] = None):
        self._dirs = [Path(d) for d in (directories or [str(SKILLS_DIR)])]

    @property
    def source_id(self) -> str:
        return "local"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        results = []
        query_lower = query.lower()
        for d in self._dirs:
            if not d.exists():
                continue
            for skill_dir in d.iterdir():
                if not skill_dir.is_dir():
                    continue
                meta = self._read_meta(skill_dir)
                if meta and (
                    query_lower in meta.name.lower()
                    or query_lower in meta.description.lower()
                    or any(query_lower in t.lower() for t in meta.tags)
                ):
                    results.append(meta)
                    if len(results) >= limit:
                        break
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        for d in self._dirs:
            skill_dir = d / identifier
            if skill_dir.exists():
                return self._read_bundle(skill_dir)
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        for d in self._dirs:
            skill_dir = d / identifier
            if skill_dir.exists():
                return self._read_meta(skill_dir)
        return None

    def _read_meta(self, skill_dir: Path) -> Optional[SkillMeta]:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        name = skill_dir.name
        desc = ""
        for line in content.split("\n"):
            if line.strip() and not line.startswith("#"):
                desc = line.strip()
                break
        return SkillMeta(name=name, description=desc, source=f"local:{skill_dir}")

    def _read_bundle(self, skill_dir: Path) -> SkillBundle:
        meta = self._read_meta(skill_dir) or SkillMeta(name=skill_dir.name)
        files = {}
        for f in skill_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                rel = str(f.relative_to(skill_dir))
                try:
                    files[rel] = f.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # Skip binary files
        bundle = SkillBundle(meta=meta, files=files)
        bundle.compute_hash()
        return bundle


class GitHubSource(SkillSource):
    """Search and fetch skills from GitHub repositories."""

    def __init__(self, token: str = "", extra_taps: Optional[List[Dict]] = None):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._taps = extra_taps or []
        self._default_repos = [
            {"repo": "openclaw/openclaw", "path": "skills/"},
        ]

    @property
    def source_id(self) -> str:
        return "github"

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            h["Authorization"] = f"token {self._token}"
        return h

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        # Search across all configured repos
        results = []
        for tap in self._default_repos + self._taps:
            try:
                skills = self._list_skills_in_repo(tap["repo"], tap.get("path", "skills/"))
                for meta in skills:
                    if query.lower() in meta.name.lower() or query.lower() in meta.description.lower():
                        results.append(meta)
            except Exception as e:
                logger.warning("GitHub search failed for %s: %s", tap["repo"], e)
        return results[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # identifier format: repo/path or just skill_name
        if "/" in identifier:
            parts = identifier.split("/", 1)
            repo = parts[0]
            path = parts[1] if len(parts) > 1 else ""
        else:
            # Search all repos for this skill name
            for tap in self._default_repos + self._taps:
                path = f"{tap.get('path', 'skills/')}{identifier}"
                bundle = self._fetch_from_repo(tap["repo"], path)
                if bundle:
                    return bundle
            return None
        return self._fetch_from_repo(repo, path)

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        bundle = self.fetch(identifier)
        return bundle.meta if bundle else None

    def _list_skills_in_repo(self, repo: str, path: str) -> List[SkillMeta]:
        """List skills in a GitHub repo directory."""
        import urllib.request
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                items = json.loads(resp.read())
            return [
                SkillMeta(name=item["name"], source=f"github:{repo}/{item['path']}")
                for item in items
                if item["type"] == "dir"
            ]
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return []

    def _fetch_from_repo(self, repo: str, path: str) -> Optional[SkillBundle]:
        """Fetch a skill bundle from GitHub."""
        import urllib.request
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                items = json.loads(resp.read())

            files = {}
            name = path.rstrip("/").split("/")[-1]
            for item in items:
                if item["type"] == "file":
                    file_req = urllib.request.Request(
                        item["download_url"], headers=self._headers(),
                    )
                    with urllib.request.urlopen(file_req, timeout=10) as file_resp:
                        files[item["name"]] = file_resp.read().decode("utf-8", errors="replace")

            meta = SkillMeta(name=name, source=f"github:{repo}/{path}")
            bundle = SkillBundle(meta=meta, files=files)
            bundle.compute_hash()
            return bundle
        except Exception as e:
            logger.warning("GitHub fetch failed: %s", e)
            return None


# ── Installation ──

class HubLockFile:
    """Track installed skills."""

    def __init__(self, path: Path = LOCK_FILE):
        self._path = path

    def load(self) -> Dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"installed": {}}

    def save(self, data: Dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_install(self, meta: SkillMeta, content_hash: str) -> None:
        data = self.load()
        data["installed"][meta.name] = {
            "version": meta.version,
            "source": meta.source,
            "hash": content_hash,
            "installed_at": time.time(),
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[Dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> List[Dict]:
        return [
            {"name": k, **v}
            for k, v in self.load()["installed"].items()
        ]


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Place a bundle in quarantine for review."""
    q_dir = QUARANTINE_DIR / bundle.meta.name
    q_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in bundle.files.items():
        file_path = q_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    return q_dir


def install_from_quarantine(name: str, force: bool = False) -> Tuple[bool, str]:
    """Install a skill from quarantine to the skills directory."""
    q_dir = QUARANTINE_DIR / name
    if not q_dir.exists():
        return False, f"Skill '{name}' not in quarantine"

    target = SKILLS_DIR / name
    if target.exists() and not force:
        return False, f"Skill '{name}' already installed (use force=True to overwrite)"

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(q_dir, target, dirs_exist_ok=True)
    shutil.rmtree(q_dir, ignore_errors=True)

    _append_audit_log("install", name, f"quarantine:{q_dir}")
    return True, f"Installed '{name}' to {target}"


def install_bundle(bundle: SkillBundle, force: bool = False) -> Tuple[bool, str]:
    """Install a skill bundle directly."""
    target = SKILLS_DIR / bundle.meta.name
    if target.exists() and not force:
        return False, f"Skill '{bundle.meta.name}' already installed"

    # Security audit before install
    files_to_audit = {}
    for rel_path, file_content in bundle.files.items():
        if any(rel_path.endswith(ext) for ext in (".py", ".sh", ".js", ".ts")):
            files_to_audit[rel_path] = file_content
    if files_to_audit:
        findings = audit_bundle_security(files_to_audit)
        if any(f.get("severity") == "high" for f in findings):
            return False, f"Security audit failed: {len(findings)} issue(s) found"

    target.mkdir(parents=True, exist_ok=True)
    for rel_path, content in bundle.files.items():
        file_path = target / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    lock = HubLockFile()
    lock.record_install(bundle.meta, bundle.content_hash)
    _append_audit_log("install", bundle.meta.name, bundle.meta.source)
    return True, f"Installed '{bundle.meta.name}'"


def uninstall_skill(name: str) -> Tuple[bool, str]:
    """Uninstall a skill."""
    target = SKILLS_DIR / name
    if not target.exists():
        return False, f"Skill '{name}' not installed"

    shutil.rmtree(target, ignore_errors=True)
    lock = HubLockFile()
    lock.record_uninstall(name)
    _append_audit_log("uninstall", name, "")
    return True, f"Uninstalled '{name}'"


def _append_audit_log(action: str, skill_name: str, source: str) -> None:
    """Append to audit log."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = json.dumps({
        "action": action,
        "skill": skill_name,
        "source": source,
        "timestamp": time.time(),
    })
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


# ── Search ──

def create_source_router(
    github_token: str = "",
    extra_dirs: Optional[List[str]] = None,
    extra_taps: Optional[List[Dict]] = None,
) -> List[SkillSource]:
    """Create a list of skill sources for searching."""
    sources: List[SkillSource] = [LocalSource(extra_dirs)]
    if github_token or os.environ.get("GITHUB_TOKEN"):
        sources.append(GitHubSource(github_token, extra_taps))
    return sources


def unified_search(
    query: str,
    sources: Optional[List[SkillSource]] = None,
    limit: int = 20,
) -> List[SkillMeta]:
    """Search across all sources."""
    if not sources:
        sources = create_source_router()

    all_results = []
    seen = set()
    for source in sources:
        try:
            results = source.search(query, limit)
            for meta in results:
                if meta.name not in seen:
                    seen.add(meta.name)
                    all_results.append(meta)
        except Exception as e:
            logger.warning("Search failed for %s: %s", source.source_id, e)

    return all_results[:limit]

from caveman.tools.builtin.skills_hub_depth import (  # noqa: F401  # depth wiring
    validate_skill_name, validate_bundle_path, GitHubAuth,
    download_directory_via_tree, find_skill_in_repo, parse_frontmatter,
    check_for_updates, parallel_search, SUSPICIOUS_PATTERNS,
    audit_bundle_security,
)

__all__ = [
    "SKILLS_DIR", "HUB_CACHE_DIR", "QUARANTINE_DIR", "LOCK_FILE", "TAPS_FILE", "AUDIT_LOG",
    "SkillMeta", "SkillBundle", "SkillSource", "LocalSource", "GitHubSource", "HubLockFile",
    "quarantine_bundle", "install_from_quarantine", "install_bundle", "uninstall_skill",
    "create_source_router", "unified_search",
    # depth re-exports
    "validate_skill_name", "validate_bundle_path", "GitHubAuth",
    "download_directory_via_tree", "find_skill_in_repo", "parse_frontmatter",
    "check_for_updates", "parallel_search", "SUSPICIOUS_PATTERNS",
    "audit_bundle_security",
]

