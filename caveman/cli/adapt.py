"""CLI: adapt-workspace command."""
from __future__ import annotations



def adapt_workspace(dry_run: bool = True) -> tuple[int, list[str]]:
    """Re-adapt workspace files. Returns (changed_count, warnings)."""
    from caveman.import_.workspace_adapter import adapt_workspace_content, validate_adapted_content
    from caveman.paths import CAVEMAN_HOME

    ws = CAVEMAN_HOME / "workspace"
    if not ws.is_dir():
        return 0, ["No workspace directory found"]

    changed = 0
    messages: list[str] = []

    for md in sorted(ws.glob("*.md")):
        content = md.read_text(encoding="utf-8")
        adapted = adapt_workspace_content(md.name, content)
        warnings = validate_adapted_content(md.name, adapted)
        if content != adapted:
            changed += 1
            messages.append(f"✏️  {md.name} — needs adaptation")
            if not dry_run:
                md.write_text(adapted, encoding="utf-8")
                messages.append(f"    → written")
        for w in warnings:
            messages.append(f"⚠️  {md.name}: {w}")

    return changed, messages
