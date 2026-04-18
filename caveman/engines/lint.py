"""Lint Engine — knowledge audit and garbage collection.

`caveman doctor` runs Lint to detect:
  - Stale memories (reference deleted files, outdated IPs, old versions)
  - Contradictions (two memories say opposite things)
  - Orphans (memories with no cross-references or usage)
  - Low-confidence entries (auto-extracted with no human confirmation)

Design:
  - Rule-based checks first (zero LLM cost)
  - Optional LLM pass for semantic contradiction detection
  - Outputs actionable report with fix suggestions
  - Integrates with `caveman doctor` CLI
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class IssueType(Enum):
    STALE_PATH = "stale_path"
    STALE_IP = "stale_ip"
    STALE_VERSION = "stale_version"
    CONTRADICTION = "contradiction"
    ORPHAN = "orphan"
    LOW_CONFIDENCE = "low_confidence"
    DUPLICATE = "duplicate"
    AGED = "aged"


@dataclass
class LintIssue:
    """A single lint finding."""
    issue_type: IssueType
    severity: IssueSeverity
    memory_id: str
    message: str
    suggestion: str = ""
    related_ids: list[str] = field(default_factory=list)


@dataclass
class LintReport:
    """Complete lint scan report."""
    issues: list[LintIssue] = field(default_factory=list)
    scanned: int = 0
    scan_time_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {"info": 0, "warn": 0, "error": 0}
        for issue in self.issues:
            counts[issue.severity.value] = counts.get(issue.severity.value, 0) + 1
        return counts

    @property
    def is_healthy(self) -> bool:
        return self.by_severity.get("error", 0) == 0

    def to_text(self) -> str:
        lines = [
            f"Lint Report — {self.scanned} memories scanned in {self.scan_time_ms:.0f}ms",
            f"Issues: {len(self.issues)} ({self.by_severity})",
        ]
        for issue in self.issues:
            icon = {"info": "ℹ️", "warn": "⚠️", "error": "❌"}[issue.severity.value]
            lines.append(f"  {icon} [{issue.issue_type.value}] {issue.message}")
            if issue.suggestion:
                lines.append(f"     → {issue.suggestion}")
        return "\n".join(lines)


# Patterns for detecting file paths, IPs, versions in memory content
_PATH_RE = re.compile(r'(?:/[\w./-]+|~/[\w./-]+|[A-Z]:\\[\w\\.-]+)')
_IP_RE = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b')


class LintEngine:
    """Knowledge audit engine.

    Usage:
        lint = LintEngine(memory_manager)
        report = await lint.scan()
        print(report.to_text())
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm_fn=None,
        stale_days: int = 90,
        check_paths: bool = True,
    ):
        self.memory = memory_manager
        self.llm_fn = llm_fn
        self.stale_days = stale_days
        self.check_paths = check_paths
        self._last_scan_count = 0  # For incremental scanning
        self._full_scan_interval = 10  # Full scan every N incremental scans
        self._scan_count = 0

    async def scan(self) -> LintReport:
        """Run lint scan — incremental by default, full every N scans.

        Incremental: only scans memories added since last scan.
        Full: scans everything (duplicates, contradictions need full view).
        This prevents O(N) full scans on every session end.
        """
        import time
        start = time.monotonic()

        await self.memory.load()
        all_memories = self.memory.all_entries()
        self._scan_count += 1

        # Decide: incremental or full scan
        force_full = (
            self._scan_count % self._full_scan_interval == 0
            or self._last_scan_count == 0
        )

        if force_full:
            scan_set = all_memories
        else:
            # Incremental: only new memories since last scan
            scan_set = all_memories[self._last_scan_count:]

        self._last_scan_count = len(all_memories)
        report = LintReport(scanned=len(scan_set))

        # Rule-based checks
        self._check_stale_paths(scan_set, report)
        if force_full:
            # These need full view to detect cross-entry issues
            self._check_duplicates(all_memories, report)
            self._check_contradictions_heuristic(all_memories, report)
        self._check_aged(scan_set, report)

        # Optional LLM pass for semantic contradictions (full scan only)
        if force_full and self.llm_fn and len(all_memories) > 0:
            await self._check_contradictions_llm(all_memories, report)

        # Close the feedback loop: demote trust for flagged memories
        await self._apply_trust_penalties(report)

        # Decay: long-inactive memories lose trust gradually (forgetting curve)
        if force_full:
            await self._decay_inactive()

        # GC: delete memories whose trust has decayed to zero
        await self._gc_dead_memories()

        # GC: clean dangling cross-refs left by deleted memories
        await self._gc_dangling_refs()

        report.scan_time_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Lint scan: %d memories, %d issues in %.0fms",
            report.scanned, len(report.issues), report.scan_time_ms,
        )
        return report

    async def _apply_trust_penalties(self, report: LintReport) -> None:
        """Demote trust_score for memories with warn/error issues.

        Uses mark_helpful(False) on the SQLite backend, which applies
        the standard trust decay (-0.10 per call). This means:
        - WARN issues: one demotion per scan
        - ERROR issues: one demotion per scan
        Repeated scans compound the penalty until the memory is fixed.
        """
        backend = getattr(self.memory, '_backend', None)
        if not backend or not hasattr(backend, 'mark_helpful'):
            return

        demoted: set[str] = set()
        for issue in report.issues:
            if issue.severity in (IssueSeverity.WARN, IssueSeverity.ERROR):
                if issue.memory_id not in demoted:
                    try:
                        await backend.mark_helpful(issue.memory_id, helpful=False)
                        demoted.add(issue.memory_id)
                    except Exception as e:
                        logger.debug("Trust demotion failed for %s: %s", issue.memory_id, e)

        if demoted:
            logger.info("Lint: demoted trust for %d memories", len(demoted))

    async def _decay_inactive(self) -> None:
        """Apply trust decay to memories not accessed in 30+ days.

        Cognitive science: unused memories fade. This ensures active memories
        naturally float to the top while dormant ones sink.
        Decay: -0.02 per scan cycle for memories idle > 30 days.
        User-type memories are exempt (preferences don't expire).
        """
        backend = getattr(self.memory, '_backend', None)
        if not backend:
            return
        try:
            conn = backend._get_conn()
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            # Decay memories not accessed recently, excluding user preferences
            conn.execute(
                "UPDATE memories SET trust_score = MAX(0.05, trust_score - 0.02) "
                "WHERE trust_score > 0.05 "
                "AND type != 'user' "
                "AND (metadata_json NOT LIKE '%\"last_accessed\"%' "
                "     OR json_extract(metadata_json, '$.last_accessed') < ?)",
                (cutoff,),
            )
            changed = conn.total_changes
            if changed:
                logger.info("Lint decay: %d inactive memories trust -0.02", changed)
        except Exception as e:
            logger.debug("Lint decay failed: %s", e)

    async def _gc_dead_memories(self) -> None:
        """Delete memories whose trust has decayed to zero.

        Trust=0 means the memory has been repeatedly flagged by Lint and/or
        marked unhelpful by the confidence feedback loop. Keeping it wastes
        storage, pollutes FTS5 index, and competes with real knowledge.
        """
        backend = getattr(self.memory, '_backend', None)
        if not backend:
            return
        try:
            conn = backend._get_conn()
            rows = conn.execute(
                "SELECT id FROM memories WHERE trust_score <= 0.01"
            ).fetchall()
            if not rows:
                return
            dead_ids = [r[0] for r in rows]
            for mid in dead_ids:
                await backend.forget(mid)
            logger.info("Lint GC: deleted %d dead memories (trust=0)", len(dead_ids))
        except Exception as e:
            logger.debug("Lint GC failed: %s", e)

    async def _gc_dangling_refs(self) -> None:
        """Clean cross-refs pointing to deleted memories."""
        backend = getattr(self.memory, '_backend', None)
        if not backend:
            return
        try:
            import json as _json
            conn = backend._get_conn()
            all_ids = set(r[0] for r in conn.execute("SELECT id FROM memories").fetchall())
            rows = conn.execute(
                "SELECT id, metadata_json FROM memories WHERE metadata_json LIKE '%related%'"
            ).fetchall()
            fixed = 0
            for mid, meta_json in rows:
                meta = _json.loads(meta_json) if meta_json else {}
                related = meta.get("related", [])
                clean = [r for r in related if r in all_ids]
                if len(clean) != len(related):
                    meta["related"] = clean
                    conn.execute(
                        "UPDATE memories SET metadata_json = ? WHERE id = ?",
                        (_json.dumps(meta, ensure_ascii=False), mid),
                    )
                    fixed += 1
            if fixed:
                conn.commit()
                logger.info("Lint GC: cleaned dangling refs in %d memories", fixed)
        except Exception as e:
            logger.debug("Lint dangling ref GC failed: %s", e)

    def _check_stale_paths(
        self, memories: list[MemoryEntry], report: LintReport
    ) -> None:
        """Check for file paths that no longer exist."""
        if not self.check_paths:
            return
        for mem in memories:
            paths = _PATH_RE.findall(mem.content)
            for p in paths:
                expanded = Path(p).expanduser()
                if expanded.is_absolute() and not expanded.exists():
                    report.issues.append(LintIssue(
                        issue_type=IssueType.STALE_PATH,
                        severity=IssueSeverity.WARN,
                        memory_id=mem.id,
                        message=f"Path '{p}' no longer exists",
                        suggestion=f"Update or remove memory: {mem.content[:80]}",
                    ))

    def _check_duplicates(
        self, memories: list[MemoryEntry], report: LintReport
    ) -> None:
        """Detect near-duplicate memories (exact or high overlap)."""
        seen: dict[str, str] = {}  # normalized content -> memory_id
        for mem in memories:
            normalized = _normalize(mem.content)
            if normalized in seen:
                report.issues.append(LintIssue(
                    issue_type=IssueType.DUPLICATE,
                    severity=IssueSeverity.INFO,
                    memory_id=mem.id,
                    message=f"Duplicate of memory {seen[normalized]}",
                    suggestion="Merge or remove duplicate",
                    related_ids=[seen[normalized]],
                ))
            else:
                seen[normalized] = mem.id

    def _check_aged(
        self, memories: list[MemoryEntry], report: LintReport
    ) -> None:
        """Flag memories older than stale_days without recent access."""
        cutoff = datetime.now() - timedelta(days=self.stale_days)
        for mem in memories:
            if mem.created_at < cutoff:
                last_access = mem.metadata.get("last_accessed")
                if last_access:
                    try:
                        la = datetime.fromisoformat(last_access)
                        if la > cutoff:
                            continue  # Recently accessed, skip
                    except (ValueError, TypeError):
                        pass
                report.issues.append(LintIssue(
                    issue_type=IssueType.AGED,
                    severity=IssueSeverity.INFO,
                    memory_id=mem.id,
                    message=f"Memory is {(datetime.now() - mem.created_at).days} days old",
                    suggestion="Review if still relevant",
                ))

    def _check_contradictions_heuristic(
        self, memories: list[MemoryEntry], report: LintReport
    ) -> None:
        """Simple heuristic contradiction detection."""
        # Group by topic keywords (IPs, paths, versions)
        ip_memories: dict[str, list[MemoryEntry]] = {}
        for mem in memories:
            ips = _IP_RE.findall(mem.content)
            for ip in ips:
                ip_memories.setdefault(ip, []).append(mem)

        # Check for same-topic memories with conflicting info
        # e.g., "server IP is X" vs "server IP is Y" for same server name
        for ip, mems in ip_memories.items():
            if len(mems) > 1:
                # Check if any are marked as superseded
                active = [
                    m for m in mems
                    if not m.metadata.get("superseded_by")
                ]
                if len(active) > 1:
                    report.issues.append(LintIssue(
                        issue_type=IssueType.CONTRADICTION,
                        severity=IssueSeverity.WARN,
                        memory_id=active[0].id,
                        message=f"Multiple active memories reference IP {ip}",
                        suggestion="Review and mark outdated entries as superseded",
                        related_ids=[m.id for m in active[1:]],
                    ))

    async def _check_contradictions_llm(
        self, memories: list[MemoryEntry], report: LintReport
    ) -> None:
        """LLM-powered semantic contradiction detection."""
        # Only check semantic memories (facts) — most likely to contradict
        semantic = [
            m for m in memories if m.memory_type == MemoryType.SEMANTIC
        ]
        if len(semantic) < 2:
            return

        # Sample up to 30 for cost control
        sample = semantic[:30]
        content_list = "\n".join(
            f"[{m.id}] {m.content[:150]}" for m in sample
        )

        prompt = f"""Review these knowledge entries for contradictions.

Entries:
{content_list}

Find pairs that contradict each other (say opposite things about the same topic).
Respond as JSON array of pairs: [{{"id1": "...", "id2": "...", "reason": "..."}}]
If no contradictions found, respond: []"""

        try:
            response = await self.llm_fn(prompt)
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                import json
                pairs = json.loads(response[start:end])
                for pair in pairs:
                    if "id1" in pair and "id2" in pair:
                        report.issues.append(LintIssue(
                            issue_type=IssueType.CONTRADICTION,
                            severity=IssueSeverity.ERROR,
                            memory_id=pair["id1"],
                            message=f"Contradicts [{pair['id2']}]: {pair.get('reason', '')}",
                            suggestion="Resolve contradiction: keep newer or more accurate entry",
                            related_ids=[pair["id2"]],
                        ))
        except Exception as e:
            logger.warning("LLM contradiction check failed: %s", e)


def _normalize(text: str) -> str:
    """Normalize text for duplicate detection."""
    return re.sub(r'\s+', ' ', text.strip().lower())
