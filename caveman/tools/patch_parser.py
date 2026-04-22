"""V4A Patch Format Parser.

Parses the V4A patch format used by Codex, Cline, and other coding agents.

V4A Format:
    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional context hint @@
     context line (space prefix)
    -removed line (minus prefix)
    +added line (plus prefix)
    *** Add File: path/to/new.py
    +new file content
    *** Delete File: path/to/old.py
    *** Move File: old/path.py -> new/path.py
    *** End Patch
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "PatchOp",
    "PatchOperation",
    "parse_v4a_patch",
    "apply_v4a_operations",
]


logger = logging.getLogger(__name__)


class PatchOp(str, Enum):
    """Patch operation type (insert, replace, delete)."""
    UPDATE = "update"
    ADD = "add"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class PatchOperation:
    """A single parsed patch operation with target location and content."""
    op: PatchOp
    path: str
    new_path: str = ""  # For move operations
    hunks: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""  # For add operations


def parse_v4a_patch(patch_text: str) -> tuple[list[PatchOperation], str | None]:
    """Parse a V4A patch into a list of operations.

    Returns (operations, error_message).
    """
    lines = patch_text.splitlines()
    operations: list[PatchOperation] = []
    current_op: PatchOperation | None = None
    current_hunk_removes: list[str] = []
    current_hunk_adds: list[str] = []
    current_hunk_context: list[str] = []
    in_patch = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "*** Begin Patch":
            in_patch = True
            continue

        if stripped == "*** End Patch":
            if current_op:
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                operations.append(current_op)
            in_patch = False
            break

        if not in_patch:
            continue

        # File operation headers
        if stripped.startswith("*** Update File:"):
            if current_op:
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                operations.append(current_op)
            path = stripped[len("*** Update File:"):].strip()
            current_op = PatchOperation(op=PatchOp.UPDATE, path=path)
            current_hunk_removes, current_hunk_adds, current_hunk_context = [], [], []
            continue

        if stripped.startswith("*** Add File:"):
            if current_op:
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                operations.append(current_op)
            path = stripped[len("*** Add File:"):].strip()
            current_op = PatchOperation(op=PatchOp.ADD, path=path)
            current_hunk_removes, current_hunk_adds, current_hunk_context = [], [], []
            continue

        if stripped.startswith("*** Delete File:"):
            if current_op:
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                operations.append(current_op)
            path = stripped[len("*** Delete File:"):].strip()
            operations.append(PatchOperation(op=PatchOp.DELETE, path=path))
            current_op = None
            continue

        if stripped.startswith("*** Move File:"):
            if current_op:
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                operations.append(current_op)
            rest = stripped[len("*** Move File:"):].strip()
            if " -> " in rest:
                old, new = rest.split(" -> ", 1)
                operations.append(PatchOperation(op=PatchOp.MOVE, path=old.strip(), new_path=new.strip()))
            current_op = None
            continue

        # Context hint line
        if stripped.startswith("@@") and stripped.endswith("@@"):
            if current_op and (current_hunk_removes or current_hunk_adds):
                _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                current_hunk_removes, current_hunk_adds, current_hunk_context = [], [], []
            continue

        # Diff lines
        if current_op:
            if line.startswith("-"):
                current_hunk_removes.append(line[1:])
            elif line.startswith("+"):
                if current_op.op == PatchOp.ADD:
                    current_op.content += line[1:] + "\n"
                else:
                    current_hunk_adds.append(line[1:])
            elif line.startswith(" "):
                if current_hunk_removes or current_hunk_adds:
                    _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
                    current_hunk_removes, current_hunk_adds = [], []
                current_hunk_context.append(line[1:])

    if current_op and in_patch:
        _flush_hunk(current_op, current_hunk_context, current_hunk_removes, current_hunk_adds)
        operations.append(current_op)

    if not operations:
        return [], "No valid patch operations found"

    return operations, None


def _flush_hunk(
    op: PatchOperation,
    context: list[str],
    removes: list[str],
    adds: list[str],
) -> None:
    """Flush accumulated hunk data into the operation."""
    if removes or adds:
        op.hunks.append({
            "context": list(context),
            "removes": list(removes),
            "adds": list(adds),
        })
    context.clear()
    removes.clear()
    adds.clear()


def apply_v4a_operations(
    operations: list[PatchOperation],
    base_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply parsed V4A operations to files.

    Returns dict with: applied, errors, skipped.
    """
    result: dict[str, list[str]] = {"applied": [], "errors": [], "skipped": []}

    for op in operations:
        try:
            target = base_dir / op.path

            if op.op == PatchOp.DELETE:
                if target.exists():
                    if not dry_run:
                        target.unlink()
                    result["applied"].append(f"delete: {op.path}")
                else:
                    result["skipped"].append(f"delete (not found): {op.path}")

            elif op.op == PatchOp.ADD:
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(op.content, encoding="utf-8")
                result["applied"].append(f"add: {op.path}")

            elif op.op == PatchOp.MOVE:
                new_target = base_dir / op.new_path
                if target.exists():
                    if not dry_run:
                        new_target.parent.mkdir(parents=True, exist_ok=True)
                        target.rename(new_target)
                    result["applied"].append(f"move: {op.path} -> {op.new_path}")
                else:
                    result["errors"].append(f"move (source not found): {op.path}")

            elif op.op == PatchOp.UPDATE:
                if not target.exists():
                    result["errors"].append(f"update (not found): {op.path}")
                    continue

                content = target.read_text(encoding="utf-8")
                for hunk in op.hunks:
                    old_text = "\n".join(hunk["removes"])
                    new_text = "\n".join(hunk["adds"])
                    if old_text in content:
                        content = content.replace(old_text, new_text, 1)
                    else:
                        result["errors"].append(f"update (hunk not found): {op.path}")
                        continue

                if not dry_run:
                    target.write_text(content, encoding="utf-8")
                result["applied"].append(f"update: {op.path}")

        except Exception as e:
            result["errors"].append(f"{op.op.value} {op.path}: {e}")

    return result
