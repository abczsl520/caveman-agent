"""Clarify tool — ask the user a clarifying question before proceeding.

When the agent is unsure about the user's intent, it can use this tool
to ask a specific question and wait for the answer.
"""
from __future__ import annotations

import logging
from caveman.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="clarify",
    description=(
        "Ask the user a clarifying question when you're unsure about their intent. "
        "The question will be sent to the user and their response returned. "
        "Use this instead of guessing when the task is ambiguous."
    ),
    params={
        "question": {
            "type": "string",
            "description": "The clarifying question to ask the user",
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of choices for the user",
        },
    },
    required=["question"],
)
async def clarify(question: str, options: list[str] | None = None, source: dict | None = None) -> dict:
    """Send a clarifying question to the user via the gateway."""
    src = source or {}
    gw_name = src.get("gateway", "discord")
    channel_id = src.get("channel_id")

    if not channel_id:
        return {"ok": False, "error": "No channel context — can't ask user"}

    # Format the question
    msg = f"❓ {question}"
    if options:
        msg += "\n" + "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(options))

    # Send via gateway router
    try:
        router = src.get("gateway_router")
        if router:
            await router.send(gw_name, str(channel_id), msg)
        else:
            return {"ok": True, "question_sent": True, "message": msg,
                    "note": "Question formatted but no router available — include in your response instead"}
    except Exception as e:
        logger.warning("Failed to send clarification: %s", e)
        return {"ok": True, "question_sent": False, "message": msg,
                "note": "Include this question in your response to the user"}

    return {
        "ok": True,
        "question_sent": True,
        "message": "Question sent. The user's next message will contain their answer.",
    }
