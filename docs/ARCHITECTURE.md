# Caveman — Architecture Design Document

> An agent that learns, executes, and evolves.

## 1. Design Philosophy

```
┌─────────────────────────────────────────────────────────┐
│                    CAVEMAN FLYWHEEL                      │
│                                                          │
│   Use → Execute tasks → Produce trajectories → Learn     │
│    ↑                                              ↓      │
│    └──────── More tasks ← Stronger execution ←───┘      │
│                                                          │
│   Long-term: trajectories → compress → fine-tune model   │
└─────────────────────────────────────────────────────────┘
```

Three pillars:
1. **Learning Flywheel** (from Hermes) — Skill auto-creation, self-improvement, memory nudge
2. **Execution Depth** (from OpenClaw) — ACP orchestration, device control, browser automation
3. **Engineering Discipline** (from Claude Code) — Coordinator orchestration, verification, anti-rationalization

## 2. System Overview

```
caveman/                    # 20 subsystems, 159 Python files
├── agent/                  # Core agent loop + execution
├── bridge/                 # External agent bridges (Hermes, OpenClaw, CLI agents)
├── cli/                    # CLI entry points + TUI
├── compression/            # 3-layer context compression pipeline
├── config/                 # YAML config loader + validation
├── coordinator/            # Multi-agent orchestration + verification
├── engines/                # 6 cognitive engines (the flywheel)
├── gateway/                # Messaging gateways (Telegram, Discord)
├── hub/                    # Hub client for agent registry
├── mcp/                    # MCP server + client
├── memory/                 # SQLite+FTS5 memory with hybrid retrieval
├── plugins/                # Plugin system
├── providers/              # LLM providers (Anthropic, OpenAI, Ollama)
├── security/               # Sandbox, encryption, PII redaction, scanning
├── skills/                 # Skill management, execution, RL routing
├── tools/                  # 30 built-in tools
│   └── builtin/            # Tool implementations
├── training/               # Embedding fine-tuning, SFT/RL data export
├── trajectory/             # Trajectory recording, scoring, compression
└── wiki/                   # Wiki Compiler (Karpathy pattern)
```

## 3. Agent Loop

The core execution engine lives in `caveman/agent/`.

```
User Task
    │
    ▼
┌──────────────────────────────────────────────┐
│  AgentLoop (loop.py)                         │
│                                              │
│  1. Build system prompt (prompt.py)          │
│     ├── Base identity                        │
│     ├── Wiki context (compiled knowledge)    │
│     ├── Recall context (restored memories)   │
│     └── Tool schemas (30 tools)              │
│                                              │
│  2. LLM call (streaming)                     │
│     └── Provider abstraction (providers/)    │
│                                              │
│  3. Parse response                           │
│     ├── Text → display to user               │
│     └── Tool calls → dispatch (tools_exec.py)│
│                                              │
│  4. Tool execution                           │
│     └── ToolRegistry.dispatch()              │
│         └── Context injection (_context)     │
│                                              │
│  5. Feed result back → goto 2               │
│                                              │
│  6. Post-task hooks (session_hooks.py)       │
│     ├── Shield: save session essence         │
│     ├── Reflect: extract skills              │
│     ├── Nudge: extract knowledge             │
│     └── Trajectory: record for training      │
└──────────────────────────────────────────────┘
```

Key files:
| File | Purpose |
|------|---------|
| `loop.py` | Main agent loop, iteration control, tool dispatch |
| `factory.py` | Creates configured AgentLoop instances |
| `prompt.py` | System prompt builder with layered token budgets |
| `tools_exec.py` | Tool call parsing and execution |
| `phases.py` | Multi-phase task execution |
| `phased_coordinator.py` | Phase orchestration |
| `context.py` | Context management |
| `context_refs.py` | Context reference tracking |
| `checkpoint.py` | Conversation state save/restore |
| `session_store.py` | Session persistence |
| `session_hooks.py` | Post-task engine triggers |
| `workspace.py` | Workspace file loading |
| `auxiliary.py` | Auxiliary LLM client for background tasks |
| `title_generator.py` | Auto-generate session titles |
| `display.py` | Output formatting |

## 4. Cognitive Engines

Six engines form the learning flywheel in `caveman/engines/`:

```
Shield → Nudge → Reflect → Ripple → Lint → Recall
  ↑                                           │
  └───────────── continuous loop ─────────────┘
```

### 4.1 Shield (`shield.py`)
Preserves conversation essence before context compression. When the context window fills up and compression is needed, Shield extracts the critical information that must survive.

- Runs pre-compression
- Extracts key facts, decisions, and context
- Stores as structured essence for later restoration
- Prevents the "amnesia" problem of naive truncation

### 4.2 Nudge (`nudge.py`, in `memory/`)
Background knowledge extraction from every conversation turn.

- Monitors conversation for new facts, preferences, corrections
- Classifies into memory types (semantic, episodic, procedural)
- Stores via MemoryManager with confidence scores
- User preferences → SEMANTIC (permanent, not ephemeral)

### 4.3 Reflect (`reflect.py`)
Post-task skill extraction and evolution.

- Analyzes completed tasks for patterns and anti-patterns
- Creates new skills or evolves existing ones (version bumps)
- Extracts tool usage patterns as procedural memories
- Skills stored as JSON in `~/.caveman/wiki/skills/`

### 4.4 Ripple (`ripple.py`)
Knowledge propagation across related memories.

- When a memory is updated, finds related memories
- Propagates updates, detects contradictions
- Uses LLM to verify conflicts before persisting changes
- Defers persistence until after verification (no premature writes)

### 4.5 Lint (`lint.py`)
Memory quality and consistency scanning.

- Detects stale or outdated memories
- Identifies contradictions between memories
- Weakens confidence of problematic entries
- Distinguishes contradiction vs. supersession

### 4.6 Recall (`recall.py`)
Context restoration at session start.

- Retrieves relevant memories for the current task
- Generates narrative summaries for LLM-optimized context
- Handles project identity matching
- Sorts by recency (mtime) for freshness

### Engine Scheduler (`scheduler.py`)
Orchestrates engine execution based on priority and available resources.

### Engine Flags (`flags.py`)
Feature flags to enable/disable individual engines via config.

### Project Identity (`project_identity.py`)
Caveman's answer to context loss — maintains project-level identity across sessions.

## 5. Memory System

`caveman/memory/` provides persistent, searchable memory with hybrid retrieval.

```
┌─────────────────────────────────────────┐
│           MemoryManager                 │
│                                         │
│  store() ──▶ SecurityScan ──▶ SQLite    │
│  recall() ──▶ HybridRetrieval ──▶ rank  │
│  recent() ──▶ FTS5 query ──▶ results    │
└─────────────────────────────────────────┘
```

| File | Purpose |
|------|---------|
| `manager.py` | Central memory API (store, recall, recent, search) |
| `sqlite_store.py` | SQLite+FTS5 backend with vector search |
| `retrieval.py` | Hybrid retrieval (keyword + vector + trust scoring) |
| `embedding.py` | Embedding generation (fastembed, CJK/jieba support) |
| `confidence.py` | Memory confidence scoring and decay |
| `drift.py` | Drift detection (stale/contradictory memories) |
| `nudge.py` | Background knowledge extraction |
| `provider.py` | MemoryProvider ABC for pluggable backends |
| `refiner.py` | Memory refinement and deduplication |
| `security.py` | Memory-level security scanning |
| `obsidian.py` | Obsidian-compatible markdown export |
| `types.py` | Memory type definitions (working/episodic/semantic/procedural) |

### Memory Types (4-tier pyramid)

```
Procedural  ← workflows and patterns (months, never expires)
Semantic    ← cross-session facts (weeks)
Episodic    ← session summaries (days)
Working     ← recent observations (hours)
```

## 6. Compression Pipeline

`caveman/compression/` handles context window management.

```
Context too large?
    │
    ▼
┌──────────────────────────────────────┐
│  Compression Pipeline                │
│                                      │
│  1. Safeguard check                  │
│     └── Preserve identifiers,        │
│         critical context (OpenClaw)  │
│                                      │
│  2. Shield pre-save                  │
│     └── Extract session essence      │
│                                      │
│  3. Compression strategy             │
│     ├── Micro: trim old messages     │
│     ├── Normal: summarize chunks     │
│     └── Smart: LLM-guided (14K)     │
│                                      │
│  4. Context engine                   │
│     └── Manage token budgets         │
└──────────────────────────────────────┘
```

## 7. Tools System

`caveman/tools/` provides 30 built-in tools with a registry pattern.

```
ToolRegistry
    │
    ├── register(@tool decorator)
    ├── dispatch(name, args)
    │   └── inject _context (memory_manager, trajectory_recorder)
    └── get_schemas() → JSON schemas for LLM
```

Tools are auto-registered via `_register_builtins()` which imports all modules in `builtin/`.

Context injection: tools that need access to internal systems (e.g., memory_tool needs MemoryManager) receive it via `_context` dict, set by AgentLoop at init time.

## 8. LLM Providers

`caveman/providers/` abstracts LLM access with production-grade reliability.

| File | Purpose |
|------|---------|
| `anthropic_provider.py` | Anthropic Claude (primary) |
| `anthropic_adapter.py` | Streaming adapter with retry |
| `openai_provider.py` | OpenAI GPT models |
| `ollama_provider.py` | Local Ollama models (training loop closure) |
| `llm.py` | Provider ABC |
| `model_router.py` | Smart model selection |
| `model_metadata.py` | Model capabilities and pricing |
| `credential_pool.py` | Multi-key rotation |
| `error_classifier.py` | Error classification (retryable vs fatal) |
| `retry.py` | Exponential backoff with jitter |
| `rate_limit.py` | Rate limit tracking |
| `prompt_cache.py` | Prompt caching for repeated prefixes |
| `usage_pricing.py` | Token usage and cost tracking |
| `insights.py` | Provider performance analytics |

## 9. MCP (Model Context Protocol)

`caveman/mcp/` implements both server and client.

**Server** (`server.py`): Exposes Caveman's memory, skills, wiki, and engines as MCP tools. Any MCP-compatible agent (Claude Code, Codex, Gemini CLI) can connect.

**Client** (`client.py` + `manager.py`): Connects to external MCP servers, making their tools available to Caveman. Managed via `mcp_connect`/`mcp_call`/`mcp_disconnect` tools.

## 10. Gateway

`caveman/gateway/` provides messaging platform integration.

- `telegram_gw.py` — Telegram bot gateway
- `discord_gw.py` — Discord bot gateway
- `router.py` — Message routing between gateways
- `runner.py` — Gateway lifecycle management
- `base.py` — Gateway ABC

Session isolation and error info leak prevention are enforced.

## 11. Bridge

`caveman/bridge/` connects Caveman to external agent frameworks.

| File | Purpose |
|------|---------|
| `hermes_bridge.py` | Bridge to Hermes agent |
| `openclaw_bridge.py` | Bridge to OpenClaw (CLI subprocess transport) |
| `cli_agents.py` | CLI agent runner (5 agents) |
| `cli_transport.py` | CLI subprocess transport |
| `acp.py` | Agent Communication Protocol |
| `uds_transport.py` | Unix Domain Socket transport |
| `ws_transport.py` | WebSocket transport |

## 12. Coordinator

`caveman/coordinator/` handles multi-agent orchestration.

- `orchestrator.py` — Task decomposition and agent assignment
- `engine.py` — Task execution engine with dependency validation
- `verification.py` — Verification agent (anti-rationalization)

## 13. Security

`caveman/security/` provides defense-in-depth.

- `sandbox.py` — Command execution sandboxing (fail-closed)
- `encryption.py` — E2E encryption for sensitive memories
- `redact.py` — PII redaction
- `scanner.py` — Security scanning for stored content
- `permissions.py` — Permission management

## 14. Training Pipeline

`caveman/training/` enables the long-term vision: fine-tuning a personal model.

```
Trajectories → Compress → Export (SFT/RL format) → Fine-tune
                                                        │
Embedding fine-tune ← Retrieval logs ← Memory queries ◀┘
```

- `sft.py` — Supervised fine-tuning data export
- `rl.py` — Reinforcement learning data export
- `embedding.py` — Embedding model fine-tuning
- `eval_embedding.py` — Embedding quality evaluation
- `retrieval_log.py` — Retrieval quality logging
- `stats.py` — Training statistics
- `cli_handler.py` — CLI interface for training commands

## 15. Skills System

`caveman/skills/` manages learned capabilities.

- `manager.py` — Skill CRUD, versioning, evolution
- `executor.py` — Skill execution (wired into AgentLoop)
- `harness.py` — Skill testing harness
- `rl_router.py` — RL-based skill selection routing
- `types.py` — Skill type definitions
- `utils.py` — Skill utilities

Skills are JSON files stored in `~/.caveman/wiki/skills/`, created by the Reflect engine and evolved over time.

## 16. Self-Improvement Flywheel

`caveman/cli/flywheel.py` — Caveman's ability to audit and fix its own code.

```
Discover subsystems → Audit (P0/P1/P2) → Fix → Test → Commit → Stats → Next round
```

Features:
- Auto-discovery of all subsystems
- Parallel audit mode (asyncio.gather)
- Stats tracking (JSON history)
- Configurable rounds and targets

## 17. Data Flow

### Session Lifecycle

```
1. Session Start
   └── Recall engine restores context
       └── Wiki context injected into prompt

2. Task Execution
   └── Agent loop: LLM ↔ Tools
       └── Nudge extracts knowledge in background
       └── Trajectory recorded

3. Context Compression (when needed)
   └── Shield saves essence
       └── Compression pipeline runs
           └── Safeguards preserve critical info

4. Session End
   └── Reflect extracts skills
       └── Ripple propagates updates
           └── Lint checks consistency
```

### Knowledge Flow

```
Conversation → Nudge → Working Memory
                           ↓ (hours)
                      Episodic Memory
                           ↓ (days)
                      Semantic Memory
                           ↓ (weeks)
                      Procedural Memory
                           ↓
                      Wiki Compilation
```

## 18. Configuration

`caveman/config/` manages YAML-based configuration.

```yaml
# ~/.caveman/config.yaml
model: claude-opus-4-6
max_iterations: 50
engines:
  shield:
    enabled: true
  recall:
    enabled: true
  reflect:
    enabled: true
  ripple:
    enabled: true
  lint:
    enabled: true
memory:
  embedding_backend: fastembed
```

Config validation ensures type safety and prevents YAML injection.

## 19. Dependencies

Core (minimal):
- `typer` — CLI framework
- `rich` — Terminal formatting
- `httpx` — HTTP client
- `pyyaml` — Config parsing
- `pydantic` — Data validation
- `aiofiles` — Async file I/O

Optional:
- `anthropic` — Anthropic provider
- `openai` — OpenAI provider
- `mcp` — MCP protocol
- `discord.py` — Discord gateway
- `python-telegram-bot` — Telegram gateway
- `fastembed` — Local embeddings
- `jieba` — CJK tokenization

## 20. Testing

1,253 tests across 71 test files:
- Unit tests with mocks for isolated component testing
- Integration tests verifying real interactions between subsystems
- E2E tests for full lifecycle verification
- NFR compliance tests for non-functional requirements

Run: `pytest tests/ -x -q`
