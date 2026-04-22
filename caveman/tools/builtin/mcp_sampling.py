"""MCP Sampling Handler — handles sampling/createMessage requests.

Extracted from Hermes mcp_tool.py SamplingHandler (350 lines).
Provides LLM-backed sampling for MCP servers that request it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "SamplingConfig",
    "SamplingMetrics",
    "STOP_REASON_MAP",
    "SamplingHandler",
]


logger = logging.getLogger("caveman.tools.mcp_sampling")


@dataclass
class SamplingConfig:
    """Configuration for MCP sampling handler."""
    max_rpm: int = 10
    timeout: float = 30
    max_tokens_cap: int = 4096
    max_tool_rounds: int = 5
    model_override: str = ""
    allowed_models: List[str] = field(default_factory=list)
    log_level: str = "info"


@dataclass
class SamplingMetrics:
    """Metrics for sampling handler."""
    requests: int = 0
    errors: int = 0
    tokens_used: int = 0
    tool_use_count: int = 0
    rate_limited: int = 0


STOP_REASON_MAP = {"stop": "endTurn", "length": "maxTokens", "tool_calls": "toolUse"}


class SamplingHandler:
    """Handles MCP sampling requests with rate limiting and tool loop governance."""

    def __init__(self, server_name: str, config: Optional[SamplingConfig] = None,
                 llm_fn: Optional[Callable] = None):
        self.server_name = server_name
        self._config = config or SamplingConfig()
        self._llm_fn = llm_fn
        self._rate_timestamps: List[float] = []
        self._tool_loop_count = 0
        self.metrics = SamplingMetrics()

    def check_rate_limit(self) -> bool:
        """Sliding-window rate limiter."""
        now = time.time()
        window = now - 60
        self._rate_timestamps[:] = [t for t in self._rate_timestamps if t > window]
        if len(self._rate_timestamps) >= self._config.max_rpm:
            self.metrics.rate_limited += 1
            return False
        self._rate_timestamps.append(now)
        return True

    def resolve_model(self, preferences: Any = None) -> str:
        """Resolve model: config override > server hint > empty."""
        if self._config.model_override:
            return self._config.model_override
        if preferences and hasattr(preferences, "hints"):
            for hint in getattr(preferences, "hints", []):
                name = getattr(hint, "name", "")
                if name:
                    if self._config.allowed_models and name not in self._config.allowed_models:
                        continue
                    return name
        return ""

    def convert_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP sampling messages to OpenAI format."""
        result = []
        for msg in messages:
            content = getattr(msg, "content", "")
            role = getattr(msg, "role", "user")

            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            blocks = content if isinstance(content, list) else [content]
            text_parts = []
            image_parts = []
            tool_results = []
            tool_uses = []

            for block in blocks:
                if hasattr(block, "toolUseId"):
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.toolUseId,
                        "content": self._extract_tool_text(block),
                    })
                elif hasattr(block, "name") and hasattr(block, "input"):
                    tool_uses.append({
                        "id": getattr(block, "id", f"call_{len(tool_uses)}"),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                        },
                    })
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "data") and hasattr(block, "mimeType"):
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                    })

            # Emit tool results
            result.extend(tool_results)

            # Emit tool uses
            if tool_uses:
                msg_dict: Dict[str, Any] = {"role": role, "tool_calls": tool_uses}
                if text_parts:
                    msg_dict["content"] = "\n".join(text_parts)
                result.append(msg_dict)
            elif text_parts and not image_parts:
                result.append({"role": role, "content": "\n".join(text_parts)})
            elif text_parts or image_parts:
                parts: List[Dict] = [{"type": "text", "text": t} for t in text_parts]
                parts.extend(image_parts)
                result.append({"role": role, "content": parts})

        return result

    @staticmethod
    def _extract_tool_text(block: Any) -> str:
        content = getattr(block, "content", None)
        if content is None:
            return ""
        items = content if isinstance(content, list) else [content]
        return "\n".join(getattr(item, "text", "") for item in items)

    def check_tool_loop(self) -> Optional[str]:
        """Check tool loop governance. Returns error message if exceeded."""
        if self._config.max_tool_rounds == 0:
            self._tool_loop_count = 0
            return f"Tool loops disabled for '{self.server_name}'"

        self._tool_loop_count += 1
        if self._tool_loop_count > self._config.max_tool_rounds:
            self._tool_loop_count = 0
            return f"Tool loop limit ({self._config.max_tool_rounds}) exceeded for '{self.server_name}'"
        return None

    def reset_tool_loop(self) -> None:
        """Reset tool loop counter (call on text response)."""
        self._tool_loop_count = 0

    async def handle_sampling(self, params: Any) -> Dict[str, Any]:
        """Handle a sampling request."""
        self.metrics.requests += 1

        if not self.check_rate_limit():
            self.metrics.errors += 1
            return {"error": f"Rate limit exceeded ({self._config.max_rpm}/min)"}

        if not self._llm_fn:
            self.metrics.errors += 1
            return {"error": "No LLM function configured"}

        model = self.resolve_model(getattr(params, "modelPreferences", None))
        messages = self.convert_messages(getattr(params, "messages", []))
        max_tokens = min(
            getattr(params, "maxTokens", self._config.max_tokens_cap),
            self._config.max_tokens_cap,
        )

        try:
            result = self._llm_fn(messages=messages, model=model, max_tokens=max_tokens)
            if hasattr(result, "__await__"):
                result = await result

            if isinstance(result, dict):
                tokens = result.get("usage", {}).get("total_tokens", 0)
                self.metrics.tokens_used += tokens
                text = result.get("text", result.get("content", ""))
                tool_calls = result.get("tool_calls", [])

                if tool_calls:
                    self.metrics.tool_use_count += 1
                    loop_err = self.check_tool_loop()
                    if loop_err:
                        return {"error": loop_err}
                    return {"role": "assistant", "tool_calls": tool_calls, "model": model}
                else:
                    self.reset_tool_loop()
                    return {"role": "assistant", "content": text, "model": model}

            return {"role": "assistant", "content": str(result), "model": model}

        except Exception as e:
            self.metrics.errors += 1
            return {"error": str(e)}
