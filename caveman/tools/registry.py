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
):
    """Decorator: declare a function as a tool with its schema.

    The schema is attached to the function; ToolRegistry picks it up automatically.
    """
    schema = {
        "type": "object",
        "properties": params,
    }
    if required:
        schema["required"] = required

    def decorator(fn: Callable):
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

    def get_schemas(self) -> list[dict]:
        """Return tool schemas for LLM API calls."""
        return [{"name": k, "description": v["description"], "input_schema": v["schema"]}
                for k, v in self._tools.items()]

    def get_tool_by_name(self, name: str) -> dict | None:
        return self._tools.get(name)

    async def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        from caveman.errors import ToolNotFoundError
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name}", context={"tool": name})
        tool_info = self._tools[name]
        fn = tool_info["fn"]

        # Use pre-cached signature info (no inspect at dispatch time)
        if tool_info["needs_context"]:
            args = {**args, "_context": self._context}
        if not tool_info["has_var_keyword"]:
            args = {k: v for k, v in args.items() if k in tool_info["param_names"]}

        start = time.monotonic()
        if tool_info["is_async"]:
            result = await fn(**args)
        else:
            result = fn(**args)
        elapsed = time.monotonic() - start

        if elapsed > 1.0:
            logger.warning("Slow tool dispatch: %s took %.2fs", name, elapsed)

        return result

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

        count = self.auto_discover()
        logger.debug("Auto-discovered %d built-in tools", count)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
