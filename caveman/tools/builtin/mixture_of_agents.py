"""Mixture-of-Agents (MoA) — multi-model collaboration for complex reasoning.

Full port from Hermes: parallel reference models → aggregator synthesis,
retry + backoff, debug session tracking, reasoning extraction.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["REFERENCE_MODELS", "AGGREGATOR_MODEL", "REFERENCE_TEMPERATURE", "AGGREGATOR_TEMPERATURE", "MIN_SUCCESSFUL_REFERENCES", "MAX_RETRIES", "AGGREGATOR_SYSTEM_PROMPT", "MoAResponse", "MoAResult", "mixture_of_agents", "check_moa_requirements", "get_debug_session_info", "get_available_models", "get_moa_configuration"]

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

REFERENCE_MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "google/gemini-2.5-pro-preview-05-06",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
]
AGGREGATOR_MODEL = "anthropic/claude-opus-4-20250514"
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency
MIN_SUCCESSFUL_REFERENCES = 2
MAX_RETRIES = 6

AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various open-source models "
    "to the latest user query. Your task is to synthesize these responses into a single, "
    "high-quality response. It is crucial to critically evaluate the information provided "
    "in these responses, recognizing that some of it may be biased or incorrect. Your "
    "response should not simply replicate the given answers but should offer a refined, "
    "accurate, and comprehensive reply to the instruction. Ensure your response is "
    "well-structured, coherent, and adheres to the highest standards of accuracy and "
    "reliability.\n\nResponses from models:"
)

# ── Debug Session ──────────────────────────────────────────────────────────

class _DebugSession:
    """Tracks MoA debug information across calls."""

    def __init__(self, name: str, env_var: str = "MOA_TOOLS_DEBUG"):
        self.name = name
        self.active = os.getenv(env_var, "").lower() in ("true", "1", "yes")
        self.session_id = uuid.uuid4().hex if self.active else ""
        self._calls: List[Dict[str, Any]] = []
        self._log_dir = Path.home() / ".caveman" / "logs"

    def log_call(self, tool_name: str, data: Dict[str, Any]) -> None:
        """Log a tool call with its parameters and results."""
        if not self.active:
            return
        entry = {
            "tool": tool_name,
            "timestamp": time.time(),
            "session_id": self.session_id,
            **data,
        }
        self._calls.append(entry)
        logger.debug("MoA debug: %s — %s", tool_name, json.dumps(data, default=str)[:200])

    def save(self) -> None:
        """Persist debug log to disk."""
        if not self.active or not self._calls:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self._log_dir / f"moa_debug_{self.session_id}.json"
            log_file.write_text(
                json.dumps(self._calls, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug("Failed to save MoA debug log: %s", e)

    def get_session_info(self) -> Dict[str, Any]:
        """Get debug session metadata."""
        return {
            "active": self.active,
            "session_id": self.session_id,
            "call_count": len(self._calls),
            "log_dir": str(self._log_dir),
        }

_debug = _DebugSession("moa_tools")

# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class MoAResponse:
    """A single model's response."""
    model: str
    content: str
    success: bool
    latency_seconds: float = 0.0
    attempts: int = 1
    error: Optional[str] = None

@dataclass
class MoAResult:
    """Complete MoA execution result."""
    success: bool
    response: str
    reference_responses: List[MoAResponse] = field(default_factory=list)
    aggregator_model: str = ""
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

# ── LLM Client ─────────────────────────────────────────────────────────────

_client_cache: Dict[str, Any] = {}

def _get_openrouter_client():
    """Get or create OpenRouter async client (cached)."""
    if "client" not in _client_cache:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package required for MoA. Install: pip install openai")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _client_cache["client"] = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client_cache["client"]

def _extract_content_or_reasoning(response: Any) -> str:
    """Extract text content from API response, including reasoning-only responses.

    Handles multiple response formats:
    - Standard content field
    - Reasoning content (thinking models)
    - Reasoning details array
    """
    if not response or not response.choices:
        return ""
    msg = response.choices[0].message

    # Try standard content first
    content = getattr(msg, "content", "") or ""
    if content.strip():
        return content.strip()

    # Try reasoning_content (Claude thinking, DeepSeek reasoning)
    reasoning = getattr(msg, "reasoning_content", "") or ""
    if reasoning.strip():
        return reasoning.strip()

    # Try reasoning field (some providers)
    reasoning2 = getattr(msg, "reasoning", "") or ""
    if reasoning2.strip():
        return reasoning2.strip()

    # Try reasoning_details array (OpenAI o1-style)
    details = getattr(msg, "reasoning_details", None)
    if details and isinstance(details, list):
        parts = []
        for d in details:
            if isinstance(d, dict) and d.get("content"):
                parts.append(str(d["content"]))
            elif isinstance(d, str):
                parts.append(d)
        if parts:
            return "\n".join(parts).strip()

    return ""

# ── Reference model execution ──────────────────────────────────────────────

async def _run_reference_model(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> MoAResponse:
    """Run a single reference model with retry logic and graceful failure."""
    start = time.time()
    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %d/%d)", model, attempt + 1, max_retries)

            api_params: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            # GPT models don't support custom temperature in some configs
            if not model.lower().startswith("gpt-"):
                api_params["temperature"] = temperature

            # Enable reasoning for models that support it
            if any(x in model.lower() for x in ("claude", "deepseek", "gemini")):
                api_params["extra_body"] = {"reasoning": {"enabled": True, "effort": "xhigh"}}

            client = _get_openrouter_client()
            response = await client.chat.completions.create(**api_params)
            content = _extract_content_or_reasoning(response)

            if not content:
                logger.warning("%s returned empty content (attempt %d/%d)", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue

            logger.info("%s responded (%d chars)", model, len(content))
            return MoAResponse(
                model=model, content=content, success=True,
                latency_seconds=time.time() - start, attempts=attempt + 1,
            )

        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limited (attempt %d): %s", model, attempt + 1, error_str[:100])
            elif "invalid" in error_str.lower():
                logger.warning("%s invalid request (attempt %d): %s", model, attempt + 1, error_str[:100])
            else:
                logger.warning("%s error (attempt %d): %s", model, attempt + 1, error_str[:100])

            if attempt < max_retries - 1:
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying %s in %ds...", model, sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                logger.error("%s failed after %d attempts", model, max_retries, exc_info=True)
                return MoAResponse(
                    model=model, content="", success=False,
                    latency_seconds=time.time() - start, attempts=max_retries,
                    error=error_str[:200],
                )

    return MoAResponse(model=model, content="", success=False, latency_seconds=time.time() - start, attempts=max_retries)

# ── Aggregator ─────────────────────────────────────────────────────────────

def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    """Construct the aggregator system prompt with all reference responses."""
    response_text = "\n\n".join(
        f"--- Response {i + 1} ---\n{response}" for i, response in enumerate(responses)
    )
    return f"{system_prompt}\n\n{response_text}"

async def _run_aggregator(
    system_prompt: str,
    user_prompt: str,
    model: str = AGGREGATOR_MODEL,
    temperature: float = AGGREGATOR_TEMPERATURE,
) -> str:
    """Run the aggregator model to synthesize the final response."""
    logger.info("Running aggregator model: %s", model)

    api_params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not model.lower().startswith("gpt-"):
        api_params["temperature"] = temperature

    # Enable reasoning for aggregator
    if any(x in model.lower() for x in ("claude", "deepseek", "gemini")):
        api_params["extra_body"] = {"reasoning": {"enabled": True, "effort": "xhigh"}}

    client = _get_openrouter_client()
    response = await client.chat.completions.create(**api_params)
    content = _extract_content_or_reasoning(response)

    # Retry once on empty content (reasoning-only response with no output)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await client.chat.completions.create(**api_params)
        content = _extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%d chars)", len(content))
    return content

# ── Main entry point ───────────────────────────────────────────────────────

async def mixture_of_agents(
    user_prompt: str,
    *,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None,
    reference_temperature: float = REFERENCE_TEMPERATURE,
    aggregator_temperature: float = AGGREGATOR_TEMPERATURE,
    min_successful: int = MIN_SUCCESSFUL_REFERENCES,
) -> MoAResult:
    """Process a complex query using Mixture-of-Agents methodology.

    2-layer architecture:
    1. Layer 1: Multiple reference models generate diverse responses in parallel
    2. Layer 2: Aggregator model synthesizes the best elements into final response

    Best for: complex reasoning, math proofs, algorithm design, multi-domain problems.
    """
    start_time = time.time()
    ref_models = reference_models or REFERENCE_MODELS
    agg_model = aggregator_model or AGGREGATOR_MODEL

    # Debug tracking
    debug_data: Dict[str, Any] = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": ref_models,
            "aggregator_model": agg_model,
            "reference_temperature": reference_temperature,
            "aggregator_temperature": aggregator_temperature,
            "min_successful": min_successful,
        },
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "error": None,
    }

    logger.info("Starting MoA with %d reference models", len(ref_models))

    # Layer 1: Generate diverse responses in parallel
    responses = await asyncio.gather(*[
        _run_reference_model(model, user_prompt, reference_temperature)
        for model in ref_models
    ])

    successful = [r for r in responses if r.success and r.content]
    failed = [r for r in responses if not r.success]

    debug_data["reference_responses_count"] = len(successful)
    debug_data["failed_models_count"] = len(failed)
    debug_data["failed_models"] = [r.model for r in failed]

    logger.info("Reference results: %d successful, %d failed", len(successful), len(failed))
    if failed:
        logger.warning("Failed models: %s", ", ".join(r.model for r in failed))

    if len(successful) < min_successful:
        error_msg = (
            f"Insufficient successful reference models ({len(successful)}/{len(ref_models)}). "
            f"Need at least {min_successful}."
        )
        debug_data["error"] = error_msg
        debug_data["processing_time_seconds"] = time.time() - start_time
        _debug.log_call("mixture_of_agents", debug_data)
        _debug.save()
        return MoAResult(
            success=False, response=error_msg,
            reference_responses=list(responses),
            aggregator_model=agg_model,
            processing_time_seconds=time.time() - start_time,
            error="insufficient_references",
        )

    # Layer 2: Aggregate
    try:
        aggregator_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, [r.content for r in successful]
        )
        final_response = await _run_aggregator(
            aggregator_prompt, user_prompt, agg_model, aggregator_temperature,
        )
    except Exception as e:
        error_msg = f"Aggregation failed: {e}"
        debug_data["error"] = error_msg
        debug_data["processing_time_seconds"] = time.time() - start_time
        _debug.log_call("mixture_of_agents", debug_data)
        _debug.save()
        return MoAResult(
            success=False, response=error_msg,
            reference_responses=list(responses),
            aggregator_model=agg_model,
            processing_time_seconds=time.time() - start_time,
            error=str(e),
        )

    processing_time = time.time() - start_time
    logger.info("MoA completed in %.2fs", processing_time)

    debug_data["success"] = True
    debug_data["final_response_length"] = len(final_response)
    debug_data["processing_time_seconds"] = processing_time
    _debug.log_call("mixture_of_agents", debug_data)
    _debug.save()

    return MoAResult(
        success=True, response=final_response,
        reference_responses=list(responses),
        aggregator_model=agg_model,
        processing_time_seconds=processing_time,
    )

# ── Utility functions ──────────────────────────────────────────────────────

def check_moa_requirements() -> bool:
    """Check if all requirements for MoA are met."""
    return bool(os.getenv("OPENROUTER_API_KEY"))

def get_debug_session_info() -> Dict[str, Any]:
    """Get information about the current debug session."""
    return _debug.get_session_info()

def get_available_models() -> Dict[str, List[str]]:
    """Get available models for MoA processing."""
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_models": [AGGREGATOR_MODEL],
        "all": REFERENCE_MODELS + [AGGREGATOR_MODEL],
    }

def get_moa_configuration() -> Dict[str, Any]:
    """Get current MoA configuration."""
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "max_retries": MAX_RETRIES,
        "total_reference_models": len(REFERENCE_MODELS),
        "failure_tolerance": f"{len(REFERENCE_MODELS) - MIN_SUCCESSFUL_REFERENCES}/{len(REFERENCE_MODELS)} models can fail",
        "api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "debug_active": _debug.active,
    }
