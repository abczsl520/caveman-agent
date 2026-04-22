# Caveman Architecture Refactor Plan

> 10 findings, 6 phases, zero regressions.
> **Status: ✅ COMPLETED** — Branch `arch/refactor-v1`, 2864 passed, 0 regressions.
> Commits: 45b08f7 → a39e181 → ff951a8 → 0974b75 → f110bf1

## Baseline
- Branch: `audit/main-fixes`
- Tests: 2788 passed, 3 pre-existing failures
- runner.py: exactly 450 lines (hard limit)

## Phase 1: Dead Code & Duplication Cleanup (Low Risk)

### 1A: Memory Dead Code Chain
- `caveman/agent/memory_manager.py` (309L) — imported by wiring.py (lazy load), tui.py, info.py
  - BUT: tui.py and info.py access `agent.memory_manager` ATTRIBUTE (from memory.manager.MemoryManager)
  - wiring.py just does `__import__` for side effects
  - The CLASS `agent.memory_manager.MemoryManager` is NEVER instantiated
- `caveman/agent/memory_provider.py` (237L) — only imported by agent/memory_manager.py
- `caveman/memory/provider.py` (131L) — only imported by wiring.py (lazy load)
- Action: Mark all 3 as deprecated, remove from wiring.py imports

### 1B: Dead Token Estimator
- `caveman/gateway/agent_memory.py:estimate_prompt_tokens` — never called externally
- Action: Remove

### 1C: split_message Consolidation
- Keep: `caveman/utils.py:split_message` (canonical, 66L, code-fence aware)
- Keep: `caveman/gateway/message_splitting.py:split_message` (platform-aware facade)
- Redirect: `caveman/gateway/display_config.py:split_message` → import from message_splitting
- Redirect: `caveman/gateway/outbound.py:chunk_message` → import from utils or message_splitting

## Phase 2: Token Estimation Unification (Medium Risk)

### 2A: Establish Single API
- `caveman/utils.py:estimate_tokens(text)` — string-level, CJK-aware (KEEP as-is)
- `caveman/compression/utils.py:estimate_tokens(messages)` — message-list level (KEEP, already delegates)
- `caveman/compression/utils.py:estimate_msg_tokens(msg)` — single message (KEEP, already delegates)

### 2B: Redirect Variants
- `caveman/agent/context_compressor.py:estimate_tokens_rough` → use compression/utils.estimate_tokens
- `caveman/gateway/agent_memory_depth.py:estimate_tokens_for_model` → enhance utils.estimate_tokens with optional model param
- `caveman/gateway/agent_memory_depth.py:estimate_transcript_tokens` → use compression/utils.estimate_tokens

## Phase 3: ContextEngine Rename (Low Risk)
- Rename `caveman/compression/context_engine.py:ContextEngine` → `CompressionEngine`
- Update `caveman/compression/__init__.py` import
- No other files import it

## Phase 4: Dependency Inversion Fixes (Medium Risk)

### 4A: Move shared utilities out of gateway/cli
- `caveman/gateway/redaction.py:redact_all` → `caveman/security/redaction.py`
- `caveman/gateway/secrets.py:SecretsManager` → `caveman/security/secrets.py`
- `caveman/gateway/security_audit.py:run_audit` → `caveman/security/audit.py`
- `caveman/cli/flywheel.py:FlywheelStats` → `caveman/metrics/flywheel.py`
- `caveman/gateway/router.py:GatewayRouter` — keep, but agent/loop.py should use Protocol

### 4B: Update all importers

## Phase 5: Exception Handling Triage (High Volume, Low Risk per change)
- gateway/: 45 bare except pass → logger.debug
- tools/: 25 bare except pass → logger.debug
- Prioritize: only the ones that swallow actionable errors

## Phase 6: Documentation
- Update any affected __init__.py exports
- Deprecation notices on dead code files
