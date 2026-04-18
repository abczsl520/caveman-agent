# 🦴 Caveman — The Self-Evolving AI Agent Framework

> An AI agent that learns, remembers, and improves itself.

Caveman is an AI agent operating system built around a 6-engine memory flywheel. Unlike agents that forget between sessions, Caveman's knowledge gets richer, more confident, and more useful over time. It audits its own code, learns skills from experience, and compiles knowledge into a structured wiki — all automatically.

## What Makes Caveman Different

Most agent frameworks are static tools: you build them, they run, they forget. Caveman is a living system:

- **Self-Evolving Skills** — The Reflect engine learns patterns from every completed task, creates skills, and evolves them over time
- **3-Layer Knowledge Pyramid** — A Wiki Compiler (inspired by [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)) distills conversations into structured, tiered knowledge
- **Knowledge Drift Detection** — Detects when memories become stale or contradictory, weakens outdated knowledge automatically
- **Self-Auditing Flywheel** — Audits and fixes its own code: find bugs → fix → test → commit → learn
- **30 Built-in Tools** — From memory search to MCP client to process management to browser automation
- **MCP Ecosystem** — Both server and client, connecting to thousands of external tools

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      CAVEMAN AGENT OS                        │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Agent   │   │ Engines  │   │  Memory  │   │  Tools   │  │
│  │  Loop    │──▶│ (6-core) │──▶│ (SQLite  │◀──│ (30      │  │
│  │         │   │          │   │  +FTS5)  │   │ built-in)│  │
│  └────┬────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │                                                      │
│  ┌────▼────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │Compress-│   │  Wiki    │   │ Training │   │ Gateway  │  │
│  │  ion    │   │ Compiler │   │ Pipeline │   │ (TG/DC)  │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │   MCP   │   │ Security │   │  Bridge  │   │Coordinat-│  │
│  │ Server  │   │ (sandbox │   │ (Hermes/ │   │  or      │  │
│  │+Client  │   │  +crypto)│   │ OpenClaw)│   │          │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install caveman-agent
caveman run "What files are in this directory?"
```

Or interactive mode:

```bash
caveman run -i
```

Other commands:

```bash
caveman status          # Dashboard: engines, memory, skills
caveman skills          # List learned skills
caveman flywheel        # Run self-improvement loop
caveman wiki status     # Knowledge stats per tier
caveman wiki compile    # Compile knowledge (promote + expire)
caveman audit           # Static code quality checks
caveman bench           # Memory performance benchmarks
caveman self-test       # Full lifecycle verification
```

## Cognitive Engines

The 6 engines form a continuous learning flywheel:

```
Shield → Nudge → Reflect → Ripple → Lint → Recall
  ↑                                           │
  └───────────── continuous loop ─────────────┘
```

| Engine | Purpose |
|--------|----------------------------------------------|
| Shield | Preserves conversation essence across context compressions |
| Nudge | Extracts new knowledge from every interaction |
| Reflect | Learns skills from completed tasks, evolves them over time |
| Ripple | Propagates knowledge updates across related memories |
| Lint | Detects stale or contradicted knowledge, weakens confidence |
| Recall | Restores relevant context at session start |

Plus a scheduler that orchestrates engine execution based on priority and resource availability.

## Wiki Compiler

Inspired by [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), the Wiki Compiler organizes all knowledge into a 4-tier pyramid:

```
Procedural  ← workflows and patterns (months, never expires)
Semantic    ← cross-session facts (weeks)
Episodic    ← session summaries (days)
Working     ← recent observations (hours)
```

Knowledge automatically promotes upward as it proves useful and expires downward when stale.

## Tools (30)

| Category | Tools |
|----------|-------|
| Shell | `bash` |
| Files | `file_read`, `file_write`, `file_edit`, `file_search`, `file_list` |
| Web | `web_search`, `browser` |
| Memory | `memory_search`, `memory_store`, `memory_recent` |
| Process | `process_start`, `process_list`, `process_output`, `process_kill` |
| Agent | `delegate`, `coding_agent` |
| Todo | `todo_add`, `todo_list`, `todo_done`, `todo_remove` |
| Skills | `skill_list`, `skill_show`, `skill_delete` |
| Vision | `vision_describe` |
| MCP | `mcp_connect`, `mcp_list_tools`, `mcp_call`, `mcp_disconnect` |
| Gateway | `gateway_send`, `gateway_list` |
| Checkpoint | `checkpoint_save`, `checkpoint_restore`, `checkpoint_list` |

## MCP Server

Expose Caveman's memory to any MCP-compatible agent (Claude Code, Codex, Gemini CLI):

```json
{
  "caveman": {
    "command": "caveman",
    "args": ["mcp", "serve"]
  }
}
```

Tools exposed: `memory_store`, `memory_search`, `memory_recall`, `shield_save`, `shield_load`, `reflect`, `skill_list`, `skill_get`, `wiki_search`, `wiki_context`.

## Self-Improvement Flywheel

Caveman can audit and fix its own code:

```
┌─────────────────────────────────────────┐
│           FLYWHEEL LOOP                 │
│                                         │
│  Discover subsystems                    │
│       ↓                                 │
│  Audit each (P0/P1/P2 findings)         │
│       ↓                                 │
│  Fix P0 + P1 issues                     │
│       ↓                                 │
│  Run tests (must pass)                  │
│       ↓                                 │
│  Commit                                 │
│       ↓                                 │
│  Record stats → learn from patterns     │
│       ↓                                 │
│  Next round                             │
└─────────────────────────────────────────┘
```

Run it: `caveman flywheel --rounds 3 --target memory`

Parallel mode: `caveman flywheel --parallel --targets memory,engines,tools`

## Standing on Giants

Caveman incorporates battle-tested patterns from open-source projects:

- [Hermes](https://github.com/NousResearch/hermes) (MIT) — Compression, retrieval, error classification, credential management
- [OpenClaw](https://github.com/openclaw/openclaw) (MIT) — Compaction safeguards, identifier preservation
- [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — Wiki compilation pattern
- [Memento-Skills](https://arxiv.org/abs/2603.18743) — Reflect-Write skill evolution

## Stats

- 159 Python files (core)
- 24,600+ lines of code
- 1,253 tests (unit + integration)
- 20 subsystems
- 30 tools
- 6 cognitive engines
- 170 commits
- Self-audited through 94 rounds

## License

MIT
