# Changelog


## Ungrouped
- `5c471b1` fix: flywheel env — venv python path + bash tool PATH injection
- `22dfd91` fix(tools): file_edit accepts old_string/new_string aliases
- `449dfd2` fix: complete audit of remaining 5 subsystems (Rounds 82-86)
- `75a3bb5` feat(discord): trigger modes (all/prefix/thread) + Intents.all() for threads
- `d65dc07` feat: flywheel tool + gateway router injection
- `58751d6` feat(memory): pluggable backend + SQLite default — P0 fix
- `0972cd4` feat: progress tool + SOUL.md reporting rules
- `99d6a3e` feat: persistent sessions + multi-turn context in gateway
- `37ef586` feat: disk-backed session persistence (PRD §4.5/§5.3)
- `ca71ada` fix: migrate 3666 JSON memories to SQLite + migration script
- `f272077` perf: async post-task engines + 15min timeout
- `2f119d4` fix: enforce progress reporting in system prompt
- `eaa1a40` feat: streaming gateway — auto-push LLM output to Discord
- `699c4ab` fix: remove tool_result noise from Discord streaming
- `c582976` feat: SmartBuffer + typing indicator — complete Discord UX overhaul

## Round 65
- `4ea50a0` fix(training): P1 embedding robustness — global limit + I/O error handling + dead imports , self-fix)
- `a763e94` fix(training): P1 remaining — rl/sft/stats IO guards + JSONL integrity b, self-fix)

## Round 66
- `7f349e1` fix(security): P0 decrypt_file data destruction + P1 hardening , self-fix)

## Round 67
- `861d4c3` fix(memory): P0 SQL injection in sqlite_store._vector_search — parameterized query , audit)

## Round 68
- `3d93a1c` fix(agent): P0 path traversal + title_generator API mismatches , self-audit)

## Round 69
- `8ca5115` fix(engines): P0 recall.py created_at AttributeError + file_edit file_path alias

## Round 70
- `9176311` fix(tools): file_read/file_list parameter aliases + tools audit

## Round 71
- `6a33543` fix(tools): P0 dispatch arg filtering + fsync for file_write/file_edit , self-fix)

## Round 72
- `5b1697a` fix(providers): P0 context window mismatch for Claude 4.6 models (200K→1M) in AnthropicProvider , self-fix)

## Round 73
- `b4230d7` fix(cli): P0 YAML injection in setup command , self-audit)

## Round 74
- `03c08b3` fix(bridge): P0 ws_transport pending futures hang on disconnect + P1/P2 cleanup

## Round 75
- `c21721b` fix(config): P0 strict validation crash on startup + P1 sentinel ordering

## Round 77
- `568fdcf` fix(mcp): P0 datetime serialization + P1 reflect persistence + min_score filter

## Round 78
- `3085f7d` fix(wiki): P0 immortal corrupt timestamps + P1 auto_resolve count + P1 cross-tier add

## Round 79
- `2b67126` fix(utils): P1 retry_async raise None + split_message infinite loop

## Round 80
- `3596a96` fix(gateway): P0 shutdown never fires + P1 swallowed exceptions

## Round 81
- `ea4f082` fix(compression): P0 compression crash kills agent loop

## Round 87
- `48327ff` feat(tools): memory_tool + process_tool + delegate_tool + context injection
- `9b88168` fix(tools): audit and harden -90 tools

## Round 88
- `061dc25` feat(tools): todo_tool + skill_manager_tool + vision_tool

## Round 89
- `62a5f7e` feat(mcp): MCP client + manager + mcp_tool — connect to external tool servers

## Round 90
- `cc867c2` feat: checkpoint system + gateway abstraction layer

## Round 92
- `8fb2c2f` feat(flywheel): parallel mode + auto-discovery + stats tracker + CLI

## Round 93
- `87b7e57` fix: parallel audit — 9 P0 fixes across 5 subsystems

## Round 94
- `7000b97` test: integration tests for memory/agent/checkpoint/flywheel/tools

## Round 95
- `1c08672` docs: README + ARCHITECTURE + CHANGELOG
- `d088425` feat: meta-flywheel -101 + progress dedup fix

## Round 96
- `862e07b` feat(providers): GeminiProvider + OllamaProvider inherits LLMProvider

## Round 97
- `e88255d` feat(tools): sandbox + transcribe + image_gen + url_safety

## Round 98
- `8a53469` feat(acp): ACP server + client + tool — agent interoperability

## Round 99
- `f720ed0` perf: connection pooling + memory cache + batch ops + metrics

## Round 100
- `572ca33` docs: self-assessment milestone report

## Round 101
- `1f67ab9` test: coverage boost for training/transcribe/web_search/image_gen/browser

## Round 102
- `b4502ed` feat(providers): Groq + DeepSeek + Mistral + Together

## Round 103
- `81730cd` feat(streaming): StreamEvent + run_stream + SSE + gateway streaming
All notable changes to Caveman, organized by flywheel round.

## Round 94 — Integration Tests
- `test`: Integration tests for memory/agent/checkpoint/flywheel/tools pipelines
- 25 new integration tests, 1253 total

## Round 93 — Parallel Audit
- `fix`: 9 P0 fixes across 5 subsystems (parallel audit)

## Round 92 — Flywheel Upgrade
- `feat(flywheel)`: Parallel mode + auto-discovery + stats tracker + CLI

## Round 91 — Tool Hardening
- `fix(tools)`: Audit and harden Round 87-90 tools (22 findings, 15 P0 + 7 P1)
- Path traversal fixes in checkpoint, skill_manager
- Unhandled exception guards in gateway, mcp tools

## Round 90 — Checkpoint + Gateway
- `feat`: Checkpoint system (save/restore conversation state)
- `feat`: Gateway abstraction layer (gateway_send, gateway_list tools)

## Round 89 — MCP Client
- `feat(mcp)`: MCP client + manager + mcp_tool
- Connect to external MCP tool servers (mcp_connect, mcp_call, mcp_disconnect)

## Round 88 — Todo + Skills + Vision Tools
- `feat(tools)`: todo_tool (add/list/done/remove with JSON persistence)
- `feat(tools)`: skill_manager_tool (list/show/delete skills)
- `feat(tools)`: vision_tool (image base64 payload prep)

## Round 87 — Memory + Process + Delegate Tools
- `feat(tools)`: memory_tool (search/store/recent via context injection)
- `feat(tools)`: process_tool (start/list/output/kill background processes)
- `feat(tools)`: delegate_tool (sub-agent spawning)
- `feat`: Context injection in ToolRegistry (_context dict + set_context)

## Rounds 82–86 — Complete Audit
- `fix`: Complete audit of remaining 5 subsystems

## Round 81 — Compression Safety
- `fix(compression)`: P0 compression crash kills agent loop

## Round 80 — Gateway Fixes
- `fix(gateway)`: P0 shutdown never fires + P1 swallowed exceptions

## Round 79 — Utils Fixes
- `fix(utils)`: P1 retry_async raise None + split_message infinite loop

## Round 78 — Wiki Fixes
- `fix(wiki)`: P0 immortal corrupt timestamps + P1 auto_resolve count + P1 cross-tier add

## Round 77 — MCP Fixes
- `fix(mcp)`: P0 datetime serialization + P1 reflect persistence + min_score filter

## Round 75 — Config Fixes
- `fix(config)`: P0 strict validation crash on startup + P1 sentinel ordering

## Round 74 — Bridge Fixes
- `fix(bridge)`: P0 ws_transport pending futures hang on disconnect + P1/P2 cleanup

## Round 73 — CLI Security
- `fix(cli)`: P0 YAML injection in setup command

## Round 72 — Provider Fixes
- `fix(providers)`: P0 context window mismatch for Claude 4.6 models (200K→1M)

## Round 71 — Tools Dispatch
- `fix(tools)`: P0 dispatch arg filtering + fsync for file_write/file_edit

## Round 70 — Tools Audit
- `fix(tools)`: file_read/file_list parameter aliases

## Round 69 — Engine Fixes
- `fix(engines)`: P0 recall.py created_at AttributeError + file_edit file_path alias

## Round 68 — Agent Fixes
- `fix(agent)`: P0 path traversal + title_generator API mismatches

## Round 67 — SQL Injection Fix
- `fix(memory)`: P0 SQL injection in sqlite_store._vector_search — parameterized query

## Round 66 — Security Fixes
- `fix(security)`: P0 decrypt_file data destruction + P1 hardening

## Round 65 — Training Robustness
- `fix(training)`: P1 embedding robustness — global limit + I/O error handling
- `fix(training)`: P1 rl/sft/stats IO guards + JSONL integrity

## Round 63 — Status Dashboard
- `feat(cli)`: Project stats in status command

## Round 62 — Encryption Fix
- `fix(security)`: Validate EncryptedBlob.from_bytes input

## Round 61 — Memory Security
- `fix(memory)`: Security scan in MemoryManager.store + dead code cleanup

## Round 60 — Changelog CLI
- `feat(cli)`: `caveman changelog` — auto-generate from git log

## Round 59 — Self-Test
- `feat(cli)`: `caveman self-test` — full lifecycle verification

## Round 57–58 — Bench + Version
- `feat(cli)`: `caveman bench` — memory performance benchmarks
- `chore`: Add `__version__`, update README stats

## Round 55–56 — Audit CLI
- `feat(cli)`: `caveman audit` — static code quality checks
- `test(audit)`: CI-integrated static quality gates

## Round 53 — Trajectory Fix
- `fix(trajectory)`: Session ID UUID truncation [:8] → [:12]

## Round 52 — SQLite Fixes
- `fix(memory)`: sqlite_store logging + UUID collision + restore commit/return

## Round 50–51 — Encoding Fixes
- `fix`: Add `encoding='utf-8'` to all file opens across codebase

## Round 49 — Coordinator Fix
- `fix(coordinator)`: Dependency validation — missing deps block execution

## Round 48 — Flywheel Params
- `chore`: Flywheel max_iterations param + mark dead trajectory code deprecated

## Round 46 — Ripple Fix
- `fix(ripple)`: Defer persistence until after LLM conflict verification

## Round 45 — Memory Import Fix
- `fix(cli)`: Load existing memories before import to prevent data loss

## Round 44 — Bridge Fix
- `fix(bridge)`: Add gateway_port/token params to OpenClawBridge

## Round 43 — Plugin Fix
- `fix(plugins)`: sys.modules cleanup on load failure + spec null guard

## Round 42 — Flywheel CLI
- `feat(cli)`: `caveman flywheel` command — self-improvement loop

## Round 41 — MCP stdio
- `feat(mcp)`: stdio entry point + Claude Code config + docs

## Round 40 — MCP Wiring
- `fix(mcp)`: Wire memory_store/search correctly + deduplicate imports

## Round 39 — Trajectory Cleanup
- `fix(trajectory+training)`: Empty save returns None, log corrupt files, remove dead code

## Round 38 — Trajectory Audit
- `audit(trajectory)`: 3/4 files dead code, role schema mismatch — 1×P0, 4×P1, 3×P2

## Round 37 — Wiki Atomic Writes
- `fix(wiki)`: Atomic writes for WikiStore + ProvenanceTracker

## Round 36 — Wiki + Recall Wiring
- `fix(phases)`: Wire wiki context + recall_context into system prompt

## Round 35 — Wiki Audit
- `audit(wiki)`: Subsystem mostly dead/manual — 5×P1, 4×P2

## Round 34 — ToolResult Tests
- `test(toolresult)`: Integration tests for ToolResult through file_ops

## Round 33 — ToolResult Adoption
- `feat(tools)`: Adopt ToolResult in file_ops + tools_exec handler

## Round 31 — Config Fix
- `fix(config)`: Add fastembed to embedding_backend choices

## Round 30 — Gateway Security
- `fix(gateway)`: Session isolation + error info leak prevention

## Round 28 — CJK Support
- `feat(memory)`: CJK tokenization (jieba) + multilingual embedding (fastembed)

## Round 26 — Compression Fallback
- `feat(compression)`: Heuristic summary fallback when no LLM

## Round 25 — Status Command
- `feat(cli)`: Add `caveman status` command

## Round 24 — Reflect Patterns
- `feat(reflect)`: Extract tool usage patterns as procedural memories

## Round 22 — Dead Code Audit
- `audit(coordinator)`: Dead code analysis — TaskEngine/Verification not wired

## Round 21 — Status + Config
- `feat(cli)`: Status dashboard + enable Ripple/Lint in config
- `fix(factory)`: Correct Ripple/Lint init param names
- `fix(loop)`: Add set_ripple() + _ripple init for factory wiring

## Round 20 — Production Polish
- `feat`: Model Metadata + Usage Pricing (from Hermes)
- `feat`: Context Engine ABC + Auxiliary Client
- `feat`: Session Persistence (from OpenClaw)
- `feat`: ProjectIdentity (Caveman's answer to context loss)
- `fix`: Critical tool loop bug — tool results lost by vacuous truth
- `feat`: Title Generator
- `feat`: Wire all engines to real LLM + REPL polish
- `feat`: Skill utils + gateway/rate-limit tests
- `feat`: Rate limit wired into provider + REPL /ratelimit
- `fix(memory+shield)`: P0 fixes + LLM extraction robustness
- `fix(memory)`: Superseded metadata persistence + recall() race condition
- `fix(drift)`: Reduce false positives, distinguish contradiction vs supersession
- `fix(nudge)`: User preferences → SEMANTIC (permanent, not ephemeral)
- `fix(agent)`: P0 tool error detection + bash guardrails + provider error handling
- `fix(recall)`: P0 task='' bypass + mtime sorting + memory truncation + project matching
- `feat(recall)`: Narrative summary for LLM-optimized context restoration
- `feat`: Wire Ripple + Lint engines into runtime
- `feat(bridge)`: Rewrite OpenClaw bridge — CLI subprocess transport
- `fix(providers)`: P0 stream retry corruption + P1 timeout
- `feat(providers)`: Wire CredentialPool into Anthropic provider
- `fix(compression)`: P0 token counting + Shield ordering + type safety
- `fix(security)`: Sandbox hardening + permissions fail-closed + secret leak reduction
- `fix(tools)`: Atomic file writes + web_search sanitization
- `fix(config)`: API key validation + strict config + Ripple/Lint wiring
- `fix(shield)`: Message→dict conversion in pre-compression path
- `feat(skills)`: Wire SkillExecutor into AgentLoop runtime

## Round 19 — E2E Verification
- `feat`: Real-world E2E flywheel verification
- `feat`: LLM Judge (multi-dimensional quality assessment)
- `feat`: CI/CD + Docker
- `feat`: Ollama Provider (training loop closure)

## Round 18 — Wiki + MCP + Package
- `feat`: Wiki Compiler Engine (Karpathy LLM Wiki pattern)
- `feat`: CLI commands (wiki + mcp)
- `feat`: README + LICENSE + pyproject v0.3.0
- `feat`: Wiki flywheel integration + MCP wiki tools

## Round 17 — Production Providers + Tools + MCP
- `feat`: Production LLM providers (Anthropic adapter + smart retry)
- `feat`: Production tools (bash + file_ops upgrade)
- `feat`: MCP Server (highest-leverage distribution)

## Round 16 — Reflect Engine
- `feat`: Reflect Engine (6th engine) + Memory Confidence

## Round 15 — Flywheel First Turn
- `feat`: Flywheel first turn (E2E verification + session hooks)

## Round 14 — Compaction + Prompt Cache
- `feat`: Compaction Safeguard + Prompt Cache + Recall upgrade

## Round 13 — Reliability
- `feat`: Reliability chassis (error classifier + retry + credential pool)
- `feat`: PII redaction + context references
- `feat`: PromptBuilder upgrade with layered token budgets

## Round 12 — REPL + Bridge + Security
- `feat`: REPL + Bridge 5-agent + Audit Log + Sandbox + E2E Encryption

## Round 11 — Loop Refactor
- `feat`: loop.py refactor + E2E integration tests + self-bootstrap

## Round 10 — Ripple + RL + Obsidian
- `feat`: Ripple Engine + RL Router + Obsidian Export + `__all__` declarations

## Round 9 — Scheduler + Lint + Verification
- `feat`: LLM Scheduler + Lint Engine + Nudge Phase 2 + Verification + Harness
- `feat`: Doctor v2 + Trajectory Scorer + Config Compat + Memory Import + Skill Auto-Create

## Round 8 — Feature Flags + Shield + Recall
- `feat`: Feature Flags + EventBus Persistence + Workspace Loader + CLI Agent Runner
- `feat`: Compaction Shield + Nudge LLM Integration
- `feat`: Recall Engine (context restoration)

## Rounds 5–7 — Core Refactoring
- `refactor`: Root-cause treatment — 6 systemic fixes
- `refactor`: Structural tech debt — 7 systemic treatments
- `refactor`: Complete constant centralization
- `feat`: Event-driven architecture + AgentLoop decomposition
- `feat`: Declarative tool system + config validation + provider unification
- `feat`: Result types + error hierarchy + lifecycle manager
- `feat`: Browser @tool migration, provider streaming fix

## Phases 1–10 — Initial Build
- Phase 1+2: AgentLoop, providers, tools, memory, skills, compression, gateway, bridge, security
- Phase 3: CLI fully functional + builtin auto-register + integration tests
- Phase 4: Config loader + factory + setup wizard
- Phase 5: OpenClaw MCP bridge + Memory v2 (hybrid vector/keyword)
- Phase 6: Skill framework v2 + Hermes bridge + system prompt builder
- Phase 7: 3-layer compression pipeline + real API verified
- Phase 8-9: Trajectory v2 + Gateway (Discord/Telegram) + CLI serve/export
- Phase 10: README + pyproject polish + pip installable package
- Phase 2 (v2): Memory Nudge + Drift Detection + Doctor
- Phase 2 (v2): SQLite+FTS5 memory + TUI interactive REPL
- Phase 3 (v2): UDS transport + ACP + Browser + Coordinator
- Phase 4+5 (v2): SFT/RL training pipeline + Hub + Plugin system
