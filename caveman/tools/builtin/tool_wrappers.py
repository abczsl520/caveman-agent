"""Tool wrappers — @tool-decorated bridge functions for auto-discovery.

Each wrapper provides:
- Input validation (type checks, required fields)
- Error handling with structured error responses
- Proper JSON serialization of results
- Lazy imports to avoid circular dependencies
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from caveman.tools.registry import tool
from caveman.aio import aio_exists, aio_is_dir

__all__ = ["file_replace", "file_create", "file_patch", "analyze_image_tool", "analyze_document_tool", "moa_tool", "transcribe_audio_tool", "generate_image_tool", "checkpoint_create_tool", "checkpoint_restore_tool", "security_audit_tool", "skills_sync_status_tool"]


logger = logging.getLogger(__name__)


def _error(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _success(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


# ── File Tools ─────────────────────────────────────────────────────────────

# file_read removed — canonical version in file_ops.py


# file_search removed — canonical version in file_ops.py


@tool(
    name="file_replace",
    description="Find and replace text in a file. Checks sensitive paths, detects staleness, requires unique match unless replace_all=true.",
    params={
        "path": {'type': 'string'},
        "old_string": {'type': 'string', 'description': 'Text to find (must be unique unless replace_all)'},
        "new_string": {'type': 'string', 'description': 'Replacement text'},
        "replace_all": {'type': 'boolean', 'default': False},
    },
    required=['path', 'old_string', 'new_string'],
)
async def file_replace(path: str, old_string: str, new_string: str,
                       replace_all: bool = False, **kwargs) -> str:
    """Replace text with staleness detection and sensitive path check."""
    if not path:
        return _error("path is required")
    if old_string is None:
        return _error("old_string is required")
    from caveman.tools.builtin.file_tools import replace_in_file
    result = replace_in_file(
        path, old_string, new_string, replace_all=replace_all,
        task_id=kwargs.get("task_id", "default"),
    )
    return json.dumps(result, ensure_ascii=False)


@tool(
    name="file_create",
    description="Create a new file. Refuses if file already exists. Checks sensitive paths.",
    params={
        "path": {'type': 'string'},
        "content": {'type': 'string', 'default': ''},
    },
    required=['path'],
)
async def file_create(path: str, content: str = "", **kwargs) -> str:
    """Create a new file with sensitive path check."""
    if not path:
        return _error("path is required")
    from caveman.tools.builtin.file_tools import create_file
    result = create_file(path, content)
    return json.dumps(result, ensure_ascii=False)


@tool(
    name="file_patch",
    description="Apply multiple find-and-replace patches to a file atomically. Checks staleness.",
    params={
        "path": {'type': 'string'},
        "patches": {'type': 'array', 'items': {'type': 'object', 'properties': {'old': {'type': 'string'}, 'new': {'type': 'string'}}, 'required': ['old', 'new']}},
    },
    required=['path', 'patches'],
)
async def file_patch(path: str, patches: list = None, **kwargs) -> str:
    """Apply multiple patches with staleness detection."""
    if not path:
        return _error("path is required")
    if not patches or not isinstance(patches, list):
        return _error("patches must be a non-empty list of {old, new} objects")
    from caveman.tools.builtin.file_tools import patch_file
    result = patch_file(path, patches, task_id=kwargs.get("task_id", "default"))
    return json.dumps(result, ensure_ascii=False)


# ── Media Tools ────────────────────────────────────────────────────────────

@tool(
    name="analyze_image",
    description="Analyze an image using vision model. Supports local paths and URLs.",
    params={
        "path": {'type': 'string', 'description': 'Image path or URL'},
        "prompt": {'type': 'string', 'description': 'What to analyze', 'default': 'Describe this image in detail.'},
    },
    required=['path'],
)
async def analyze_image_tool(path: str, prompt: str = "Describe this image in detail.", **kwargs) -> str:
    """Analyze image with vision model."""
    if not path:
        return _error("path is required")
    try:
        from caveman.tools.builtin.media_understanding import analyze_image
        result = analyze_image(path, question=prompt)
        if result.error:
            return _error(result.error)
        return _success({"description": result.description or result.text_content})
    except Exception as e:
        logger.error("analyze_image failed: %s", e)
        return _error(f"Image analysis failed: {e}")


@tool(
    name="analyze_document",
    description="Analyze a document (PDF, DOCX, etc.) using vision or text extraction.",
    params={
        "path": {'type': 'string', 'description': 'Document path'},
        "prompt": {'type': 'string', 'default': 'Summarize this document.'},
    },
    required=['path'],
)
async def analyze_document_tool(path: str, prompt: str = "Summarize this document.", **kwargs) -> str:
    """Analyze document with extraction + LLM."""
    if not path:
        return _error("path is required")
    try:
        from caveman.tools.builtin.media_understanding import analyze_document
        result = await analyze_document(path, prompt=prompt)
        return _success({"summary": result})
    except Exception as e:
        logger.error("analyze_document failed: %s", e)
        return _error(f"Document analysis failed: {e}")


# ── MoA Tool ───────────────────────────────────────────────────────────────

@tool(
    name="mixture_of_agents",
    description="Process complex queries using Mixture-of-Agents: multiple frontier models generate diverse responses, then an aggregator synthesizes the best answer. Best for: complex reasoning, math, algorithm design, multi-domain problems.",
    params={
        "prompt": {'type': 'string', 'description': 'The complex query to solve'},
        "reference_models": {'type': 'array', 'items': {'type': 'string'}, 'description': 'Custom reference models (optional)'},
        "aggregator_model": {'type': 'string', 'description': 'Custom aggregator model (optional)'},
    },
    required=['prompt'],
)
async def moa_tool(prompt: str, reference_models: list = None, aggregator_model: str = None, **kwargs) -> str:
    """Run Mixture-of-Agents with full error handling."""
    if not prompt or not isinstance(prompt, str):
        return _error("prompt is required and must be a non-empty string")
    from caveman.tools.builtin.mixture_of_agents import mixture_of_agents, check_moa_requirements
    if not check_moa_requirements():
        return _error("OPENROUTER_API_KEY not set. MoA requires OpenRouter API access.")
    try:
        result = await mixture_of_agents(
            prompt,
            reference_models=reference_models,
            aggregator_model=aggregator_model,
        )
        return json.dumps({
            "success": result.success,
            "response": result.response,
            "models_used": {
                "reference": [r.model for r in result.reference_responses],
                "aggregator": result.aggregator_model,
            },
            "processing_time": round(result.processing_time_seconds, 2),
            "error": result.error,
        }, ensure_ascii=False)
    except Exception as e:
        logger.error("MoA failed: %s", e, exc_info=True)
        return _error(f"Mixture-of-Agents failed: {e}")


# ── Transcription ──────────────────────────────────────────────────────────

@tool(
    name="transcribe_audio",
    description="Transcribe audio file to text using Whisper API.",
    params={
        "path": {'type': 'string', 'description': 'Path to audio file'},
        "language": {'type': 'string', 'description': 'Language hint (ISO 639-1)', 'default': ''},
    },
    required=['path'],
)
async def transcribe_audio_tool(path: str, language: str = "", **kwargs) -> str:
    """Transcribe audio with validation."""
    if not path:
        return _error("path is required")
    from pathlib import Path as _Path
    if not await aio_exists(_Path(path).expanduser()):
        return _error(f"Audio file not found: {path}")
    try:
        from caveman.tools.builtin.transcription import transcribe
        result = transcribe(path, language=language)
        return _success({"text": result.text, "language": result.language, "duration": result.duration_seconds})
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return _error(f"Transcription failed: {e}")


# ── Image Generation ───────────────────────────────────────────────────────

@tool(
    name="generate_image",
    description="Generate an image from a text prompt using DALL-E or similar.",
    params={
        "prompt": {'type': 'string', 'description': 'Image description'},
        "size": {'type': 'string', 'enum': ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792'], 'default': '1024x1024'},
        "style": {'type': 'string', 'enum': ['natural', 'vivid'], 'default': 'vivid'},
    },
    required=['prompt'],
)
async def generate_image_tool(prompt: str, size: str = "1024x1024", style: str = "vivid", **kwargs) -> str:
    """Generate image with validation."""
    if not prompt:
        return _error("prompt is required")
    if len(prompt) > 4000:
        return _error("prompt too long (max 4000 chars)")
    try:
        from caveman.tools.builtin.image_generation import generate_image
        result = await generate_image(prompt, size=size, style=style)
        return _success({"url": result.url, "path": result.local_path, "revised_prompt": result.revised_prompt})
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return _error(f"Image generation failed: {e}")


# ── Checkpoint ─────────────────────────────────────────────────────────────

@tool(
    name="checkpoint_create",
    description="Create a checkpoint (snapshot) of the current session state for later restoration.",
    params={
        "label": {'type': 'string', 'description': 'Human-readable label for this checkpoint'},
        "session_id": {'type': 'string', 'description': 'Session to checkpoint (default: current)'},
    },
    required=['label'],
)
async def checkpoint_create_tool(label: str, session_id: str = "", **kwargs) -> str:
    """Create checkpoint with validation."""
    if not label:
        return _error("label is required")
    try:
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        cp = mgr.create(label=label, session_id=session_id or "default")
        return _success({"checkpoint_id": cp.id, "label": cp.label, "created_at": cp.created_at})
    except Exception as e:
        logger.error("Checkpoint create failed: %s", e)
        return _error(f"Checkpoint creation failed: {e}")


@tool(
    name="checkpoint_restore",
    description="Restore session state from a previously created checkpoint.",
    params={
        "checkpoint_id": {'type': 'string', 'description': 'ID of checkpoint to restore'},
    },
    required=['checkpoint_id'],
)
async def checkpoint_restore_tool(checkpoint_id: str, **kwargs) -> str:
    """Restore checkpoint with validation."""
    if not checkpoint_id:
        return _error("checkpoint_id is required")
    try:
        from caveman.tools.builtin.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager()
        success = mgr.restore(checkpoint_id)
        if success:
            return _success({"restored": True, "checkpoint_id": checkpoint_id})
        return _error(f"Checkpoint not found: {checkpoint_id}")
    except Exception as e:
        logger.error("Checkpoint restore failed: %s", e)
        return _error(f"Checkpoint restore failed: {e}")


# ── Security Audit ─────────────────────────────────────────────────────────

@tool(
    name="security_audit",
    description="Run a security audit on a directory. Checks for hardcoded secrets, unsafe patterns, dependency vulnerabilities, and permission issues.",
    params={
        "path": {'type': 'string', 'description': 'Directory to audit', 'default': '.'},
        "checks": {'type': 'array', 'items': {'type': 'string', 'enum': ['secrets', 'permissions', 'dependencies', 'patterns']}, 'description': 'Which checks to run (default: all)'},
    },
)
async def security_audit_tool(path: str = ".", checks: list = None, **kwargs) -> str:
    """Run security audit with structured output."""
    from pathlib import Path as _Path
    audit_path = _Path(path).expanduser().resolve()
    if not await aio_exists(audit_path):
        return _error(f"Path not found: {path}")
    if not await aio_is_dir(audit_path):
        return _error(f"Path is not a directory: {path}")
    try:
        from caveman.gateway.security_audit import run_audit
        result = run_audit(str(audit_path), checks=checks or ["secrets", "permissions", "dependencies", "patterns"])
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("Security audit failed: %s", e)
        return _error(f"Security audit failed: {e}")


# ── Skills Sync ────────────────────────────────────────────────────────────

@tool(
    name="skills_sync_status",
    description="Check skill sync status — local count, bundled count, last sync time, and any errors.",
    params={},
)
async def skills_sync_status_tool(**kwargs) -> str:
    """Get comprehensive sync status."""
    try:
        from caveman.tools.builtin.skills_sync import get_sync_status
        return json.dumps(get_sync_status(), ensure_ascii=False)
    except Exception as e:
        logger.error("Skills sync status failed: %s", e)
        return _error(f"Skills sync status failed: {e}")
