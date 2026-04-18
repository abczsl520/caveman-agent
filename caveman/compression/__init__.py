"""Context compression — 3-layer pipeline with smart LLM summarization."""
from caveman.compression.context_engine import ContextEngine
from caveman.compression.pipeline import CompressionPipeline, CompressionStats
from caveman.compression.smart import SmartCompressor, SUMMARY_PREFIX
from caveman.compression.utils import (
    sanitize_tool_pairs,
    estimate_tokens,
    IDENTIFIER_PRESERVATION,
    align_forward,
    align_backward,
    build_template,
    serialize_turns,
)

__all__ = [
    "CompressionPipeline",
    "CompressionStats",
    "SmartCompressor",
    "ContextEngine",
    "sanitize_tool_pairs",
    "estimate_tokens",
    "IDENTIFIER_PRESERVATION",
    "SUMMARY_PREFIX",
]
