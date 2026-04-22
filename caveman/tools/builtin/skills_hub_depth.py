"""Skills Hub Depth — GitHub tree API, validation, update checking, parallel search.

Supplements skills_hub.py with deeper GitHub integration and safety features.
Extracted from Hermes skills_hub.py (2775 lines) — the parts we were missing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "validate_skill_name",
    "validate_bundle_path",
    "GitHubAuth",
    "download_directory_via_tree",
    "find_skill_in_repo",
    "parse_frontmatter",
    "check_for_updates",
    "parallel_search",
    "SUSPICIOUS_PATTERNS",
    "audit_bundle_security",
]


logger = logging.getLogger("caveman.tools.skills_hub_depth")

# ── Validation ──

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_UNSAFE_PATH_RE = re.compile(r"\.\.|//|\\|[\x00-\x1f]")


def validate_skill_name(name: str) -> str:
    """Validate and normalize a skill name."""
    name = name.strip().lower()
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid skill name '{name}': must be 1-64 chars, "
            "lowercase alphanumeric with hyphens/underscores"
        )
    return name


def validate_bundle_path(path: str, allow_nested: bool = False) -> str:
    """Validate a relative path within a skill bundle."""
    if _UNSAFE_PATH_RE.search(path):
        raise ValueError(f"Unsafe path: {path}")
    if not allow_nested and "/" in path:
        raise ValueError(f"Nested paths not allowed: {path}")
    return path


# ── GitHub Auth ──

class GitHubAuth:
    """Multi-strategy GitHub authentication."""

    def __init__(self):
        self._token: Optional[str] = None
        self._method: str = "none"
        self._resolve()

    def _resolve(self) -> None:
        # Strategy 1: Environment variable
        for var in ("GITHUB_TOKEN", "GH_TOKEN"):
            token = os.environ.get(var)
            if token:
                self._token = token
                self._method = f"env:{var}"
                return

        # Strategy 2: gh CLI
        token = self._try_gh_cli()
        if token:
            self._token = token
            self._method = "gh-cli"
            return

    def _try_gh_cli(self) -> Optional[str]:
        """Try to get token from gh CLI."""
        import subprocess
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # intentional: FileNotFoundError suppressed
        return None

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

    @property
    def auth_method(self) -> str:
        return self._method

    def get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        return headers


# ── GitHub Tree API ──

def download_directory_via_tree(
    repo: str, path: str, auth: GitHubAuth,
) -> Optional[Dict[str, str]]:
    """Download an entire directory using Git Trees API (single request).

    Much more efficient than Contents API for large directories.
    Falls back to None if tree is truncated.
    """
    import urllib.request
    path = path.rstrip("/")
    headers = auth.get_headers()

    # Resolve default branch
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            default_branch = json.loads(resp.read()).get("default_branch", "main")
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None

    # Fetch recursive tree
    try:
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
        req = urllib.request.Request(tree_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            tree_data = json.loads(resp.read())
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None

    if tree_data.get("truncated"):
        return None  # Too large, caller should use Contents API

    # Filter blobs under target path
    prefix = f"{path}/"
    files: Dict[str, str] = {}
    for item in tree_data.get("tree", []):
        if item.get("type") != "blob":
            continue
        item_path = item.get("path", "")
        if not item_path.startswith(prefix):
            continue
        rel_path = item_path[len(prefix):]
        content = _fetch_file_content(repo, item_path, headers)
        if content is not None:
            files[rel_path] = content

    return files if files else None


def find_skill_in_repo(repo: str, skill_name: str, auth: GitHubAuth) -> Optional[str]:
    """Find a skill directory anywhere in a repo using Trees API."""
    import urllib.request
    headers = auth.get_headers()

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            default_branch = json.loads(resp.read()).get("default_branch", "main")

        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
        req = urllib.request.Request(tree_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            tree_data = json.loads(resp.read())

        # Look for SKILL.md in a directory matching the skill name
        target = f"{skill_name}/SKILL.md"
        for item in tree_data.get("tree", []):
            if item.get("type") == "blob" and item.get("path", "").endswith(target):
                # Extract the parent directory path
                full_path = item["path"]
                skill_dir = full_path[: -len("/SKILL.md")]
                return f"{repo}/{skill_dir}"

    except Exception as e:
        logger.debug("Tree search failed for %s in %s: %s", skill_name, repo, e)

    return None


def _fetch_file_content(repo: str, path: str, headers: Dict[str, str]) -> Optional[str]:
    """Fetch a single file's content from GitHub."""
    import urllib.request
    url = f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None


# ── Frontmatter Parsing ──

def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Quick frontmatter parser for SKILL.md files."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end < 0:
        return {}
    fm_text = content[3:end].strip()
    result: Dict[str, Any] = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                # Simple list parsing
                value = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
            result[key] = value
    return result


# ── Update Checking ──

def check_for_updates(
    installed: List[Dict[str, Any]],
    sources: List[Any],
    auth: Optional[GitHubAuth] = None,
) -> List[Dict[str, Any]]:
    """Check installed skills for available updates."""
    updates = []
    for skill in installed:
        name = skill.get("name", "")
        current_hash = skill.get("hash", "")
        source_str = skill.get("source", "")

        if not source_str.startswith("github:"):
            continue

        # Fetch current version from source
        for source in sources:
            try:
                bundle = source.fetch(name)
                if bundle:
                    new_hash = bundle.compute_hash()
                    if new_hash != current_hash:
                        updates.append({
                            "name": name,
                            "current_hash": current_hash,
                            "new_hash": new_hash,
                            "source": source_str,
                        })
                    break
            except Exception as e:
                logger.debug("suppressed: %s", e)
                continue

    return updates


# ── Parallel Search ──

async def parallel_search(
    query: str,
    sources: List[Any],
    limit: int = 20,
    timeout: float = 10.0,
) -> List[Any]:
    """Search across multiple sources in parallel."""
    async def _search_one(source: Any) -> List[Any]:
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, source.search, query, limit),
                timeout=timeout,
            )
        except Exception as e:
            logger.debug("Parallel search failed for %s: %s", source.source_id, e)
            return []

    tasks = [_search_one(s) for s in sources]
    results_lists = await asyncio.gather(*tasks)

    # Deduplicate by name
    seen: Set[str] = set()
    merged = []
    for results in results_lists:
        for meta in results:
            if meta.name not in seen:
                seen.add(meta.name)
                merged.append(meta)

    return merged[:limit]


# ── Security Audit ──

SUSPICIOUS_PATTERNS = [
    re.compile(r"subprocess\.(run|call|Popen|check_output)", re.IGNORECASE),
    re.compile(r"os\.system\s*\(", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"exec\s*\(", re.IGNORECASE),
    re.compile(r"__import__\s*\(", re.IGNORECASE),
    re.compile(r"shutil\.rmtree", re.IGNORECASE),
    re.compile(r"requests?\.(get|post|put|delete)\s*\(", re.IGNORECASE),
]


def audit_bundle_security(files: Dict[str, str]) -> List[Dict[str, Any]]:
    """Audit a skill bundle for suspicious patterns."""
    findings = []
    for path, content in files.items():
        for i, line in enumerate(content.split("\n"), 1):
            for pattern in SUSPICIOUS_PATTERNS:
                if pattern.search(line):
                    findings.append({
                        "file": path,
                        "line": i,
                        "pattern": pattern.pattern,
                        "content": line.strip()[:200],
                    })
    return findings
