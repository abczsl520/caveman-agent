"""Tool registry v2 — declarative self-registering tools.

Instead of 54 lines of manual registration, tools declare their own schema
via the @tool decorator. The registry auto-discovers them.

Usage:
    from caveman.tools.registry import ToolRegistry, tool

    @tool(
        name="bash",
        description="Execute a bash command",
        params={
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
        },
        required=["command"],
    )
    async def bash_exec(command: str, timeout: int = 30) -> dict:
        ...

    registry = ToolRegistry()
    registry.auto_discover()  # Finds all @tool-decorated functions
"""
from __future__ import annotations
import inspect
import logging
import asyncio
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Module-level registry for @tool decorated functions
_TOOL_REGISTRY: list[dict] = []


def tool(
    name: str,
    description: str,
    params: dict[str, dict],
    required: list[str] | None = None,
) -> Any:
    """Decorator: declare a function as a tool with its schema.

    The schema is attached to the function; ToolRegistry picks it up automatically.
    """
    schema = {
        "type": "object",
        "properties": params,
    }
    if required:
        schema["required"] = required

    def decorator(fn: Callable) -> Any:
        fn._tool_meta = {
            "name": name,
            "description": description,
            "schema": schema,
        }
        _TOOL_REGISTRY.append(fn)
        return fn
    return decorator


class ToolRegistry:
    """Central registry for all agent tools.

    Supports both:
    - Declarative: @tool decorator (recommended for built-in tools)
    - Imperative: registry.register(name, fn, description, schema)
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}
        self._context: dict = {}

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value that gets injected into tools accepting _context."""
        self._context[key] = value

    def register(self, name: str, fn: Callable, description: str, schema: dict) -> None:
        """Manually register a tool (for plugins, dynamic tools)."""
        # Pre-validate: cache signature info at registration time
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        needs_context = has_var_keyword or "_context" in params
        param_names = set(params.keys())
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "schema": schema,
            "is_async": inspect.iscoroutinefunction(fn),
            "needs_context": needs_context,
            "has_var_keyword": has_var_keyword,
            "param_names": param_names,
        }
        logger.debug("Tool registered: %s", name)

    def register_decorated(self, fn: Callable) -> None:
        """Register a @tool-decorated function."""
        meta = getattr(fn, "_tool_meta", None)
        if not meta:
            raise ValueError(f"{fn.__name__} is not decorated with @tool")
        self.register(meta["name"], fn, meta["description"], meta["schema"])

    def auto_discover(self) -> int:
        """Auto-register all @tool-decorated functions that were imported.

        Returns number of tools registered.
        """
        count = 0
        for fn in _TOOL_REGISTRY:
            meta = fn._tool_meta
            if meta["name"] not in self._tools:
                self.register(meta["name"], fn, meta["description"], meta["schema"])
                count += 1
        return count

    def visible_tool_names(self) -> list[str]:
        """Return tool names intended for model/user-facing listings."""
        return [name for name, info in self._tools.items() if not info.get("hidden_from_schema")]

    def get_schemas(self) -> list[dict]:
        """Return tool schemas for LLM API calls."""
        return [{"name": k, "description": v["description"], "input_schema": v["schema"]}
                for k, v in self._tools.items() if not v.get("hidden_from_schema")]

    def register_alias(self, alias: str, target: str) -> None:
        """Register a dispatch-only alias that is hidden from LLM tool schemas."""
        if target not in self._tools:
            raise ValueError(f"Cannot alias unknown tool: {target}")
        self._tools[alias] = {**self._tools[target], "hidden_from_schema": True}

    async def dispatch(self, name: str, args: dict[str, Any], timeout: float = 120.0) -> Any:
        from caveman.errors import ToolNotFoundError, ToolTimeoutError
        if name not in self._tools:
            # Try to repair the tool name before failing
            repaired = self._repair_tool_name(name)
            if repaired:
                logger.warning("Repaired tool name: '%s' → '%s'", name, repaired)
                name = repaired
            else:
                raise ToolNotFoundError(f"Unknown tool: {name}", context={"tool": name})
        tool_info = self._tools[name]
        fn = tool_info["fn"]

        # Use pre-cached signature info (no inspect at dispatch time)
        if tool_info["needs_context"]:
            args = {**args, "_context": self._context}
        if not tool_info["has_var_keyword"]:
            args = {k: v for k, v in args.items() if k in tool_info["param_names"]}

        start = time.monotonic()
        try:
            if tool_info["is_async"]:
                result = await asyncio.wait_for(fn(**args), timeout=timeout)
            else:
                result = fn(**args)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            raise ToolTimeoutError(
                f"Tool '{name}' timed out after {elapsed:.1f}s",
                context={"tool": name, "timeout": timeout},
            )
        elapsed = time.monotonic() - start

        if elapsed > 1.0:
            logger.warning("Slow tool dispatch: %s took %.2fs", name, elapsed)

        return result

    def _repair_tool_name(self, name: str) -> str | None:
        """Attempt to repair a mismatched tool name. Ported from Hermes."""
        from difflib import get_close_matches
        valid = set(self._tools.keys())
        # 1. Lowercase
        if name.lower() in valid:
            return name.lower()
        # 2. Normalize (hyphens/spaces → underscores)
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if normalized in valid:
            return normalized
        # 3. Fuzzy match (cutoff=0.7)
        matches = get_close_matches(name.lower(), valid, n=1, cutoff=0.7)
        return matches[0] if matches else None

    def _register_builtins(self) -> None:
        """Auto-register built-in tools via @tool decorator."""
        # Import modules to trigger @tool decorations
        import caveman.tools.builtin.bash  # noqa: F401
        import caveman.tools.builtin.file_ops  # noqa: F401
        import caveman.tools.builtin.web_search  # noqa: F401
        import caveman.tools.builtin.browser  # noqa: F401
        import caveman.tools.builtin.coding_agent  # noqa: F401
        import caveman.tools.builtin.memory_tool  # noqa: F401
        import caveman.tools.builtin.process_tool  # noqa: F401
        import caveman.tools.builtin.delegate_tool  # noqa: F401
        import caveman.tools.builtin.todo_tool  # noqa: F401
        import caveman.tools.builtin.skill_manager_tool  # noqa: F401
        import caveman.tools.builtin.vision_tool  # noqa: F401
        import caveman.tools.builtin.mcp_tool  # noqa: F401
        import caveman.tools.builtin.gateway_tool  # noqa: F401
        import caveman.tools.builtin.checkpoint_tool  # noqa: F401
        import caveman.tools.builtin.sandbox_tool  # noqa: F401
        import caveman.tools.builtin.transcribe_tool  # noqa: F401
        import caveman.tools.builtin.image_gen_tool  # noqa: F401
        import caveman.tools.builtin.url_safety_tool  # noqa: F401
        import caveman.tools.builtin.acp_tool  # noqa: F401
        import caveman.tools.builtin.flywheel_tool  # noqa: F401
        import caveman.tools.builtin.progress_tool  # noqa: F401
        import caveman.tools.builtin.metrics_tool  # noqa: F401
        import caveman.tools.builtin.session_search_tool  # noqa: F401
        import caveman.tools.builtin.branch_tool  # noqa: F401
        import caveman.tools.builtin.cron_tool  # noqa: F401
        import caveman.tools.builtin.clarify_tool  # noqa: F401
        import caveman.tools.builtin.moa_tool  # noqa: F401
        import caveman.tools.builtin.tts_tool  # noqa: F401
        import caveman.tools.builtin.homeassistant_tool  # noqa: F401
        import caveman.tools.builtin.terminal_tool  # noqa: F401
        import caveman.tools.builtin.tool_wrappers  # noqa: F401

        # Wire remaining orphan tool modules (non-fatal)
        _orphan_tools = [
            "caveman.tools.builtin.skills_hub",
            "caveman.tools.builtin.mcp_client",
            "caveman.tools.builtin.mcp_lifecycle",
            "caveman.tools.builtin.mcp_sampling",
            "caveman.tools.builtin.code_execution",
            "caveman.tools.builtin.voice_mode",
            "caveman.tools.builtin.llm_router",
            "caveman.tools.builtin.process_registry",
            "caveman.tools.builtin.approval",
            "caveman.tools.builtin.skills_guard",
            "caveman.tools.builtin.web_research",
            "caveman.tools.builtin.browser_providers",
            "caveman.tools.builtin.cronjob",
            "caveman.tools.builtin.website_policy",
            "caveman.tools.builtin.vision_tools",
            "caveman.tools.builtin.send_message_tool",
            "caveman.tools.builtin.browser_v2",
            "caveman.tools.builtin.web_fetch_v2",
            "caveman.tools.builtin.terminal_v2",
            "caveman.tools.builtin.file_ops_v2",
            "caveman.tools.credential_files",
            "caveman.tools.env_passthrough",
            "caveman.tools.debug_helpers",
            "caveman.tools.budget_config",
            "caveman.tools.interrupt",
            "caveman.tools.patch_parser",
        ]
        for mod_name in _orphan_tools:
            try:
                __import__(mod_name)
            except Exception as exc:
                logger.debug("unknown: suppressed %s", exc)

        count = self.auto_discover()
        if "todo_finish" in self._tools and "todo_done" not in self._tools:
            self.register_alias("todo_done", "todo_finish")
        logger.debug("Auto-discovered %d built-in tools", count)

    @property
    def tool_count(self) -> int:
        return len(self.visible_tool_names())

    def list_tools(self, include_hidden: bool = False) -> list[str]:
        if include_hidden:
            return list(self._tools.keys())
        return self.visible_tool_names()
