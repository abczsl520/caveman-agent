"""Session lifecycle hooks — on_session_end knowledge extraction.

Inspired by Hermes MemoryProvider on_session_end hook.
Extracts final knowledge from a completed session before cleanup.
Integrates with Wiki Compiler for knowledge compilation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from caveman.engines.shield import CompactionShield, SessionEssence
from caveman.memory.nudge import MemoryNudge
from caveman.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)


async def on_session_end(
    shield: CompactionShield,
    nudge: Optional[MemoryNudge],
    trajectory: TrajectoryRecorder,
    task: str = "",
    wiki_compiler: Any = None,
) -> dict[str, Any]:
    """End-of-session hook: final Shield save + knowledge extraction + wiki ingest.

    Returns dict with extraction results.
    """
    result: dict[str, Any] = {
        "shield_saved": False,
        "memories_created": 0,
        "essence_turns": 0,
        "wiki_ingested": False,
    }

    # 1. Final Shield save
    try:
        await shield.save()
        result["shield_saved"] = True
        result["essence_turns"] = shield.essence.turn_count
        logger.info(
            "Session %s ended: %d turns, %d decisions, %d todos",
            shield.essence.session_id,
            shield.essence.turn_count,
            len(shield.essence.decisions),
            len(shield.essence.open_todos),
        )
    except Exception as e:
        logger.warning("Final Shield save failed: %s", e)

    # 2. Final knowledge extraction via Nudge
    if nudge:
        try:
            turns = trajectory.to_sharegpt()[-30:]
            created = await nudge.run(turns, task=task)
            result["memories_created"] = len(created) if created else 0
        except Exception as e:
            logger.warning("End-of-session nudge failed: %s", e)

    # 3. Wiki Compiler: ingest session summary
    if wiki_compiler and result["shield_saved"]:
        try:
            essence = shield.essence
            wiki_compiler.ingest_session(
                session_id=essence.session_id,
                task=essence.task or task,
                decisions=essence.decisions,
                progress=essence.progress,
                todos=essence.open_todos,
            )
            # Run compilation to promote/expire
            wiki_compiler.compile()
            result["wiki_ingested"] = True
            logger.info("Wiki: session ingested + compiled")
        except Exception as e:
            logger.warning("Wiki ingest failed: %s", e)

    return result


async def on_delegation_complete(
    parent_nudge: Optional[MemoryNudge],
    agent_name: str,
    agent_output: str,
    task: str = "",
) -> int:
    """Hook for when a sub-agent completes a delegated task.

    Extracts knowledge from the sub-agent's output into parent memory.
    Returns number of memories created.
    """
    if not parent_nudge or not agent_output:
        return 0

    try:
        # Format as a synthetic conversation for Nudge
        turns = [
            {"role": "user", "content": f"[Delegated to {agent_name}]: {task}"},
            {"role": "assistant", "content": agent_output[:2000]},
        ]
        created = await parent_nudge.run(turns, task=task)
        count = len(created) if created else 0
        if count:
            logger.info(
                "Extracted %d memories from %s delegation", count, agent_name,
            )
        return count
    except Exception as e:
        logger.warning("Delegation knowledge extraction failed: %s", e)
        return 0
