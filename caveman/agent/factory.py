"""Factory — create AgentLoop from config file."""
from __future__ import annotations
import logging

from caveman.config.loader import load_config
from caveman.agent.loop import AgentLoop
from caveman.engines.flags import EngineFlags
from caveman.providers.registry import resolve_provider
from caveman.memory.manager import MemoryManager
from caveman.skills.manager import SkillManager
from caveman.trajectory.recorder import TrajectoryRecorder

logger = logging.getLogger(__name__)


def _make_llm_fn(provider):
    """Create a simple prompt->response async callable from a provider.

    Returns ``async (prompt: str) -> str`` that wraps the provider's
    streaming interface into a single-shot call.
    """
    async def llm_fn(prompt: str) -> str:
        result = []
        async for event in provider.safe_complete(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            if event.get("type") == "delta":
                result.append(event.get("text", ""))
        return "".join(result)
    return llm_fn


def create_loop(
    model: str | None = None,
    config_path: str | None = None,
    max_iterations: int | None = None,
    surface: str = "cli",
) -> AgentLoop:
    """Create an AgentLoop from config. Resolves provider, memory, skills from config."""
    config = load_config(config_path)
    agent_cfg = config.get("agent", {})
    providers_cfg = config.get("providers", {})

    from caveman.paths import (
        DEFAULT_MODEL, DEFAULT_MAX_ITERATIONS,
        DEFAULT_MAX_TOKENS_ANTHROPIC, DEFAULT_MAX_TOKENS_OPENAI,
    )

    # Resolve model
    final_model = model or agent_cfg.get("default_model", DEFAULT_MODEL)
    final_max = max_iterations or agent_cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # Resolve provider via registry (replaces 80-line if-elif chain)
    provider = resolve_provider(
        model=final_model,
        providers_cfg=providers_cfg,
        default_max_tokens={
            "anthropic": DEFAULT_MAX_TOKENS_ANTHROPIC,
            "openai": DEFAULT_MAX_TOKENS_OPENAI,
        },
    )

    # Resolve memory/skills dirs from config
    mem_cfg = config.get("memory", {})
    mem_dir = mem_cfg.get("local_dir")  # None = use paths.py default

    # Try to get embedding function for vector memory
    embedding_fn = None
    try:
        from caveman.memory.embedding import get_embedding_fn
        emb_backend = mem_cfg.get("embedding_backend", "auto")
        embedding_fn = get_embedding_fn(emb_backend)
    except Exception as e:
        logger.debug("Embedding function unavailable: %s", e)

    skills_cfg = config.get("skills", {})
    skills_dir = skills_cfg.get("local_dir")  # None = use paths.py default

    # Optionally connect OpenClaw bridge
    bridges_cfg = config.get("bridges", {})
    openclaw_cfg = bridges_cfg.get("openclaw", {})
    openclaw_bridge = None
    if openclaw_cfg.get("enabled"):
        from caveman.bridge.openclaw_bridge import OpenClawBridge
        from caveman.paths import OPENCLAW_GATEWAY_PORT
        port = int(openclaw_cfg.get("port", OPENCLAW_GATEWAY_PORT))
        token = openclaw_cfg.get("token", "")
        openclaw_bridge = OpenClawBridge(gateway_port=port, token=token)

    # Engine flags
    engine_flags = EngineFlags(config)

    # LLM function for engines (nudge, shield)
    llm_fn = _make_llm_fn(provider)

    # Create memory manager (SQLite + FTS5 by default)
    scorer_config = mem_cfg.get("scorer", {})  # e.g. {"trust_weight": 0.3}
    memory_manager = MemoryManager.with_sqlite(
        base_dir=mem_dir, embedding_fn=embedding_fn,
        scorer_config=scorer_config,
    )

    skill_manager = SkillManager(skills_dir=skills_dir)

    # Create all engines via EngineManager (unified lifecycle)
    from caveman.engines.manager import EngineManager
    engine_mgr = EngineManager(
        flags=engine_flags,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        llm_fn=llm_fn,
    )
    engines = engine_mgr.create_all()

    loop = AgentLoop(
        model=final_model,
        max_iterations=final_max,
        provider=provider,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        trajectory_recorder=TrajectoryRecorder(),
        engine_flags=engine_flags,
        llm_fn=llm_fn,
        lint_engine=engines.lint,
        shield=engines.shield,
        recall_engine=engines.recall,
        nudge_engine=engines.nudge,
        reflect_engine=engines.reflect,
        surface=surface,
    )

    # Wire engines that need loop-level access
    if engines.ripple:
        loop.set_ripple(engines.ripple)
    if engines.lint:
        loop.set_lint(engines.lint)

    # Store bridge reference for later use
    loop._openclaw_bridge = openclaw_bridge
    return loop
