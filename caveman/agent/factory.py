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
    # Build credential pool from config (multi-key rotation)
    from caveman.providers.credential_pool import CredentialPool
    credential_pool = CredentialPool.from_config(config)
    pool_summary = credential_pool.status_summary()
    if pool_summary:
        for prov, counts in pool_summary.items():
            if counts["total"] > 1:
                logger.info("Credential pool: %s has %d keys", prov, counts["total"])

    provider = resolve_provider(
        model=final_model,
        providers_cfg=providers_cfg,
        default_max_tokens={
            "anthropic": DEFAULT_MAX_TOKENS_ANTHROPIC,
            "openai": DEFAULT_MAX_TOKENS_OPENAI,
        },
        credential_pool=credential_pool,
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
    quality_cfg = mem_cfg.get("quality_gate", {}) or {}
    q_mode = str(quality_cfg.get("mode", "heuristic")).lower()
    use_llm_quality_gate = bool(quality_cfg.get("use_llm", False)) or q_mode == "llm"
    # "off" disables only the optional LLM judge; security and hard heuristics
    # still run in SQLiteMemoryStore to protect the flywheel from garbage/secrets.
    if q_mode == "off":
        use_llm_quality_gate = False

    # RetrievalLog — records every memory search for embedding training (PRD §5.2 Ring 6)
    from caveman.training.retrieval_log import RetrievalLog
    retrieval_log = RetrievalLog()  # default path: ~/.caveman/training/retrieval_log.jsonl

    memory_manager = MemoryManager.with_sqlite(
        base_dir=mem_dir, embedding_fn=embedding_fn,
        scorer_config=scorer_config,
        retrieval_log=retrieval_log,
        quality_llm_fn=llm_fn if use_llm_quality_gate else None,
        use_llm_quality_gate=use_llm_quality_gate,
    )

    # WorkspaceMemorySync: keep MEMORY.md etc. synced to vector DB (PRD §8.8.1)
    # Runs on every session start, fast no-op if nothing changed.
    _workspace_sync = None
    try:
        from caveman.agent.workspace_memory_sync import WorkspaceMemorySync
        from caveman.paths import CAVEMAN_HOME
        _workspace_sync = WorkspaceMemorySync(CAVEMAN_HOME, memory_manager)
        import asyncio
        try:
            loop_obj = asyncio.get_running_loop()
            # Already in async context — schedule as task
            loop_obj.create_task(_workspace_sync.sync())
        except RuntimeError:
            # No running loop — run synchronously
            asyncio.run(_workspace_sync.sync())
        logger.debug("WorkspaceMemorySync completed")
    except Exception as e:
        logger.debug("WorkspaceMemorySync unavailable: %s", e)

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

    # --- Wire orphan modules into live system ---

    # Prompt builder: structured system prompt assembly
    try:
        from caveman.agent.prompt_builder import build_system_prompt, PromptConfig
        loop._build_system_prompt = lambda: build_system_prompt(PromptConfig(
            surface=surface, model=final_model, skills_dir=skills_dir,
        ))
        logger.debug("PromptBuilder wired")
    except Exception as e:
        logger.debug("PromptBuilder unavailable: %s", e)

    # Context engine: manages what the agent remembers
    try:
        from caveman.agent.context_engine import DefaultContextEngine
        loop._context_engine = DefaultContextEngine()
        logger.debug("ContextEngine wired")
    except Exception as e:
        logger.debug("ContextEngine unavailable: %s", e)

    # Smart model routing: cheap model for simple queries
    routing_cfg = agent_cfg.get("smart_routing", {})
    if routing_cfg.get("enabled", False):
        try:
            from caveman.agent.smart_model_routing import classify_message_complexity, RoutingConfig
            loop._classify_complexity = lambda text: classify_message_complexity(
                text, RoutingConfig(**routing_cfg))
            logger.info("SmartModelRouter enabled")
        except Exception as e:
            logger.debug("SmartModelRouter unavailable: %s", e)

    # Prompt caching: reduce API costs
    try:
        from caveman.agent.prompt_caching import PromptCache
        loop._prompt_cache = PromptCache()
    except Exception as e:
        logger.debug("PromptCache unavailable: %s", e)

    # Title generator: auto-generate session titles
    try:
        from caveman.agent.title_generator import _heuristic_title
        loop._generate_title = _heuristic_title
    except Exception as e:
        logger.debug("TitleGenerator unavailable: %s", e)

    # Redaction: strip secrets from outbound messages
    try:
        from caveman.gateway.redaction import redact_all
        loop._redact_output = redact_all
    except Exception as e:
        logger.debug("Redaction unavailable: %s", e)

    # Secrets manager: secure credential storage
    secrets_cfg = config.get("secrets", {})
    if secrets_cfg:
        try:
            from caveman.gateway.secrets import SecretsManager
            loop._secret_manager = SecretsManager(secrets_cfg)
        except Exception as e:
            logger.debug("SecretsManager unavailable: %s", e)

    # Wire fallback chain from config
    fallback_cfg = agent_cfg.get("fallback_chain", [])
    if fallback_cfg:
        from caveman.providers.fallback_chain import FallbackChain
        loop._fallback_chain = FallbackChain(fallback_cfg)
        logger.info("Fallback chain configured: %d entries", len(fallback_cfg))

    # Store bridge reference for later use
    loop._openclaw_bridge = openclaw_bridge

    # Store workspace sync reference for manual resync
    if _workspace_sync is not None:
        loop._workspace_sync = _workspace_sync

    # Compression feasibility check (Hermes pattern)
    try:
        ctx_len = getattr(provider, 'context_length', 200_000)
        threshold = int(ctx_len * 0.75)
        if threshold > 100_000:
            logger.debug("Compression threshold: %d tokens (context: %d)", threshold, ctx_len)
    except Exception as exc:
        logger.debug("unknown: suppressed %s", exc)

    return loop
