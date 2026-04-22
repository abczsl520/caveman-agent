"""OpenClaw Session Importer — extract knowledge from session conversations.

This is a separate importer from OpenClawImporter because:
1. Session extraction is opt-in (user decides per session)
2. It can use LLM for higher quality extraction (costs tokens)
3. It operates on a fundamentally different data format (JSONL conversations vs markdown files)

Usage:
    caveman import --from openclaw-sessions [--dry-run] [--use-llm] [--min-score 2.0]
    caveman import --from openclaw-sessions --session <session-id>
    caveman import --from openclaw-sessions --since 2026-04-01
    caveman import --from openclaw-sessions --interactive
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter,
    ImportItem,
    ImportManifest,
    ImportResult,
    write_import_log,
)
from .session_parser import ParsedSession, parse_session, scan_sessions
from .session_extractor import (
    ExtractionResult,
)
from .session_scoring import score_turn, cluster_turns

logger = logging.getLogger(__name__)

# Default paths
_OPENCLAW_SESSIONS = Path("~/.openclaw/agents/main/sessions").expanduser()
_OPENCLAW_SUB_AGENTS = [
    Path("~/.openclaw/agents/claude/sessions").expanduser(),
    Path("~/.openclaw/agents/claude-code/sessions").expanduser(),
]


class OpenClawSessionImporter(BaseImporter):
    """Import knowledge extracted from OpenClaw session conversations.

    Unlike the file-based OpenClawImporter, this one:
    - Parses JSONL session files
    - Scores conversation turns for knowledge density
    - Clusters related turns
    - Extracts knowledge (rule-based or LLM-enhanced)
    - Stores as memory entries with full provenance
    """

    def __init__(
        self,
        caveman_home: Path,
        dry_run: bool = True,
        include_secrets: bool = False,
        use_llm: bool = False,
        llm_complete: Any | None = None,
        min_score: float = 2.0,
        since: str | None = None,
        session_filter: str | None = None,
        include_sub_agents: bool = False,
        max_entries_per_session: int = 20,
    ) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.use_llm = use_llm
        self.llm_complete = llm_complete
        self.min_score = min_score
        self.since = since  # YYYY-MM-DD filter
        self.session_filter = session_filter  # specific session ID
        self.include_sub_agents = include_sub_agents
        self.max_entries_per_session = max_entries_per_session

        # Track extraction state
        self._parsed_sessions: list[ParsedSession] = []
        self._extraction_results: list[ExtractionResult] = []

    @property
    def source_name(self) -> str:
        return "OpenClaw Sessions"

    def detect(self) -> bool:
        return _OPENCLAW_SESSIONS.is_dir()

    def scan(self) -> ImportManifest:
        """Scan sessions and produce a manifest of extractable knowledge.

        This does the full parse + score + cluster pipeline but does NOT
        call LLM yet (that happens in execute).
        """
        manifest = ImportManifest(source="openclaw-sessions")
        if not self.detect():
            return manifest

        # Find session files
        session_paths = scan_sessions(_OPENCLAW_SESSIONS)
        if self.include_sub_agents:
            for sub_dir in _OPENCLAW_SUB_AGENTS:
                session_paths.extend(scan_sessions(sub_dir))

        # Apply filters
        session_paths = self._apply_filters(session_paths)

        logger.info("Scanning %d session files...", len(session_paths))

        for path in session_paths:
            session = parse_session(path)
            if session is None or not session.turns:
                continue

            # Score and cluster (no LLM yet)
            scored = [score_turn(t) for t in session.turns]
            extractable = [s for s in scored if s.score >= self.min_score and s.category]
            clusters = cluster_turns(scored, session)

            if not clusters:
                continue

            self._parsed_sessions.append(session)

            # Create manifest items for each cluster
            for cluster in clusters:
                # Use rule-based extraction for preview
                from .session_extractor import extract_knowledge_rule_based
                preview_entry = extract_knowledge_rule_based(cluster)
                if preview_entry is None:
                    continue

                item = ImportItem(
                    source_path=path,
                    target_type="memory",
                    memory_type=preview_entry.memory_type,
                    content=preview_entry.content,
                )
                # Store cluster metadata for execute phase
                item._cluster = cluster  # type: ignore[attr-defined]
                item._session = session  # type: ignore[attr-defined]
                manifest.items.append(item)

        return manifest

    def _apply_filters(self, paths: list[Path]) -> list[Path]:
        """Apply date and session ID filters."""
        filtered = paths

        # Filter by specific session ID
        if self.session_filter:
            sid = self.session_filter
            filtered = [
                p for p in filtered
                if sid in p.stem
            ]

        # Filter by date (from filename or session header)
        if self.since:
            since_date = self.since
            result = []
            for p in filtered:
                # Try to extract date from filename
                fname = p.name
                # Format: 2026-04-11T07-49-30-043Z_uuid.jsonl
                if fname[:4].isdigit() and fname[4] == "-":
                    file_date = fname[:10]
                    if file_date >= since_date:
                        result.append(p)
                        continue
                # For UUID-named files, we need to peek at the header
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                    obj = json.loads(first_line)
                    ts = obj.get("timestamp", "")
                    if ts[:10] >= since_date:
                        result.append(p)
                except Exception:
                    # If we can't determine date, include it
                    result.append(p)
            filtered = result

        return filtered

    async def execute(
        self,
        manifest: ImportManifest,
        memory_manager: Any,
    ) -> ImportResult:
        """Execute knowledge extraction and storage.

        If use_llm=True, this will call the LLM for each cluster.
        Otherwise, uses rule-based extraction (already computed in scan).
        """
        from .dedup import ImportDedup

        result = ImportResult()
        dedup = ImportDedup(memory_manager)

        for item in manifest.items:
            if item.skip_reason:
                result.skipped += 1
                continue

            cluster = getattr(item, "_cluster", None)

            try:
                # Determine extraction method
                if self.use_llm and self.llm_complete and cluster:
                    from .session_extractor import extract_knowledge_with_llm
                    entry = await extract_knowledge_with_llm(
                        cluster, self.llm_complete
                    )
                    if entry is None:
                        result.skipped += 1
                        result.details.append(
                            f"LLM skipped: {item.source_path.name}"
                        )
                        continue
                    content = entry.content
                    memory_type = entry.memory_type
                    metadata = entry.metadata
                else:
                    # Use the rule-based content from scan
                    content = item.content
                    memory_type = item.memory_type or MemoryType.SEMANTIC
                    metadata = {
                        "source": "import:openclaw-session",
                        "source_file": str(item.source_path),
                        "imported_at": datetime.now().isoformat(),
                    }

                # Dedup check
                if dedup.is_duplicate(content):
                    result.duplicates += 1
                    continue

                # Store
                if not self.dry_run:
                    await memory_manager.store(
                        content,
                        memory_type,
                        metadata=metadata,
                        trusted=self.include_secrets,
                    )
                result.imported += 1

            except Exception as e:
                result.failed += 1
                result.details.append(
                    f"Failed: {item.source_path.name}: {e}"
                )
                logger.warning(
                    "Import failed for %s: %s", item.source_path, e
                )

            result.files_processed += 1

        # Write import log
        if not self.dry_run and result.imported > 0:
            write_import_log(self.caveman_home, {
                "source": "openclaw-sessions",
                "imported": result.imported,
                "duplicates": result.duplicates,
                "skipped": result.skipped,
                "use_llm": self.use_llm,
                "min_score": self.min_score,
            })

        return result

    def get_session_preview(self, path: Path) -> dict[str, Any] | None:
        """Get a preview of what would be extracted from a single session.

        Useful for interactive mode where user picks sessions.
        """
        session = parse_session(path)
        if session is None:
            return None

        scored = [score_turn(t) for t in session.turns]
        extractable = [s for s in scored if s.score >= self.min_score and s.category]
        clusters = cluster_turns(scored, session)

        return {
            "session_id": session.metadata.session_id,
            "date": session.metadata.date,
            "topic": session.topic_hint,
            "total_turns": len(session.turns),
            "extractable_turns": len(extractable),
            "clusters": len(clusters),
            "stats": session.summary_stats,
            "top_scores": sorted(
                [(s.score, s.category.value if s.category else "none", s.turn.turn_index)
                 for s in extractable],
                reverse=True,
            )[:5],
        }
