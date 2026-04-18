"""Import report generator — pretty terminal output."""
from __future__ import annotations

from .base import ImportManifest, ImportResult


def format_manifest_report(manifest: ImportManifest, dry_run: bool = True) -> str:
    """Format a scan manifest as a readable report."""
    lines: list[str] = []
    mode = "dry-run" if dry_run else "execute"
    lines.append(f"\n🦴 Caveman Import Report")
    lines.append("=" * 40)
    lines.append(f"Source: {manifest.source}")
    lines.append(f"Mode: {mode}")
    lines.append("")

    # Group by target_type
    by_type: dict[str, list] = {}
    for item in manifest.items:
        by_type.setdefault(item.target_type, []).append(item)

    type_icons = {
        "workspace": "📁", "memory": "🧠", "skill": "🔧",
        "config": "⚙️", "cron": "⏰",
    }

    for ttype, items in by_type.items():
        icon = type_icons.get(ttype, "📄")
        actionable = [i for i in items if not i.skip_reason]
        skipped = [i for i in items if i.skip_reason]
        lines.append(f"{icon} {ttype.title()} ({len(actionable)} items)")

        # Show memory type breakdown for memory items
        if ttype == "memory":
            type_counts: dict[str, int] = {}
            for item in actionable:
                mt = item.memory_type.value if item.memory_type else "unknown"
                type_counts[mt] = type_counts.get(mt, 0) + 1
            for mt, count in sorted(type_counts.items()):
                lines.append(f"  {mt}: {count}")

        if skipped:
            lines.append(f"  ⏭️ Skipped: {len(skipped)}")
            for item in skipped[:3]:
                lines.append(f"    {item.source_path.name}: {item.skip_reason}")

        lines.append("")

    total_kb = manifest.total_size / 1024
    lines.append(f"📊 Total: {len(manifest.actionable)} items, {total_kb:.1f} KB")
    if manifest.skipped:
        lines.append(f"   Skipped: {len(manifest.skipped)}")

    if dry_run:
        lines.append("\nRun without --dry-run to execute.")

    return "\n".join(lines)


def format_result_report(result: ImportResult) -> str:
    """Format execution result as a readable report."""
    lines: list[str] = []
    lines.append(f"\n🦴 Import Complete")
    lines.append("=" * 40)
    lines.append(f"  Imported:   {result.imported}")
    lines.append(f"  Duplicates: {result.duplicates}")
    lines.append(f"  Skipped:    {result.skipped}")
    lines.append(f"  Failed:     {result.failed}")
    lines.append(f"  Files:      {result.files_processed}")

    if result.warnings:
        lines.append("\n⚠️ Warnings:")
        for w in result.warnings[:10]:
            lines.append(f"  {w}")

    if result.details:
        lines.append("\nDetails:")
        for d in result.details[:20]:
            lines.append(f"  {d}")

    return "\n".join(lines)


def format_detect_report(detected: dict[str, bool]) -> str:
    """Format detection results."""
    lines = ["\n🔍 Detected Sources:"]
    for source, found in detected.items():
        icon = "✅" if found else "❌"
        lines.append(f"  {icon} {source}")
    return "\n".join(lines)
