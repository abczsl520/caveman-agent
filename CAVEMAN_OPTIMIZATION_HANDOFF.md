# Caveman 优化 HANDOFF

更新时间: 2026-04-30 15:15 CST

## 当前最终状态
- Round 8 已完成、提交并推送到 `main`。
- 最新 commit: `9624ce069d74efa063e2c8c2aa4fef0feef80604` (`Normalize imported memory source metadata`)。
- `origin/main` 已同步到最新 SHA。
- GitHub Actions 对最新 SHA 全绿：`docs` success、`test (3.12)` success、install smoke macOS/Ubuntu/Windows success。
- Round 7 handoff commit: `be73746`；Round 7 code commits: `92d6819` (`[verified] fix quarantine lifecycle CI gates`)、`b2c0169` (`[verified] add reversible memory quarantine lifecycle`)。
- Round 6 code commit: `5e98357f06f34a4e6b28716ab5414c436378dea4` (`[verified] surface source governance in flywheel dashboard`)；Round 6 handoff commit: `1ceef53`。
- Round 5 commit: `7de8ff65a156c924fc9af2d74452bd24e8e39f77` (`[verified] add source-aware memory quarantine policy`)。
- Round 4 code commit: `90ee17f75cbe5ca78a8c7335c6051ccf19962829`；Round 4 handoff commit: `a70d3f5552d6aed51e3f07549915283f001b727a`。
- Gateway 最后已知未运行：`curl http://localhost:4201/health` 失败；日志显示 09:37 用户停止并移除 PID。这不是代码失败，但下轮若需要交互验证要按 gateway SOP 安全启动。

## 下次启动时做
1. Round 9/50：helpfulness/retrieval 反馈质量与 decay scheduling/observability 闭环：确认 quarantine 后 recall 候选减少和 helpful memory 不被误伤。优先从真实 memory DB/dashboard 指标出发，闭环“检索→helpful feedback→decay/quarantine protection→dashboard 可观测”。
2. Round 10：quarantine restore 的 operator guardrails：dry-run bulk restore、source/reason scoped restore preview、恢复后 dashboard impact report。
3. Round 11：import/source taxonomy 更严格治理：把 normalized source 枚举、导入入口、dashboard/decay allowlist 统一成单一来源，避免未来拼写漂移。
4. Rounds 12-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

## Round 8 做了什么
- 聚焦 import metadata normalization/backfill，修复 dashboard 中大量 `<missing>` source 导致治理策略、source breakdown 和 quarantine policy 无法稳定聚合的问题。
- 将 memory schema 升到 `SCHEMA_VERSION = 3`，新增事务性 migration `v3: normalize import memory source metadata`。
- 新增 `normalize_import_metadata()`：
  - 对缺失/空字符串/`<missing>`/`unknown` source 的 imported memories 回填规范 source；
  - 保留原始 `source_file` 等 provenance；
  - 追加 `source_normalized_at`、`source_normalization_reason`、`source_normalization_previous`，确保可审计。
- v3 migration heuristics：
  - `source_file` 路径含 `openclaw` → `import:openclaw`；
  - 含 `hermes` → `import:hermes`；
  - legacy task-result 内容形态 `Task: ... Result:` → `legacy:task-result`；
  - 非 import/非 legacy task-result 的 organic memory 不强行写 source，避免制造假 provenance。
- 对 malformed `metadata_json` 保持 legacy tolerance：不崩溃、不重写坏 JSON，只推进 schema version。
- 更新 `docs/API_REFERENCE.md`，使 docs CI gate 与 schema v3/新增函数一致。

## Round 8 验证结果
- TDD/focused migration tests：`tests/test_memory_migrations.py` 共 `10 passed`。
- Focused regression subset：`tests/test_memory_migrations.py tests/test_memory.py tests/test_import_system.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_metadata_quality_wiring.py` 共 `94 passed`。
- Full test suite：`3308 passed, 8 skipped`。
- Coverage gate：`3308 passed, 8 skipped`；observed coverage `69.23%` > baseline `68.25%`，80% 长期债务继续可见。
- Ruff CI parity：`ruff check --select E9,F63,F7,F82 caveman tests` pass。
- Ruff changed files：`caveman/memory/store_helpers.py tests/test_memory_migrations.py` pass。
- Mypy baseline-aware gate：full-project historical baseline 仍可见；changed Python file `caveman/memory/store_helpers.py` 无 mypy errors。
- Docs generation：`scripts/generate_api_reference.py` 更新并提交 `docs/API_REFERENCE.md`；remote `docs` job success。
- Live DB copy smoke：对 `/Users/yeren64g/.caveman/memory/caveman.db` copy 执行 v3，`<missing>` source 从 624 降到 547，v3 changed 77，未直接修改生产 DB。
- Security scan：changed files pattern scan clean；push hook safety checks passed。
- Independent review：第一次建议避免 organic `Task:` 文本误判；已增加负例并把 legacy task-result 判定收紧到显式 `\nResult:` 或 ` Result:`。第二次 re-review passed，无 blocker。
- Remote CI：最新 SHA `9624ce069d74efa063e2c8c2aa4fef0feef80604` GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

## Round 7 做了什么
- 补齐 reversible quarantine lifecycle operator path，避免 Round 4-6 只会自动隔离、缺少安全查看/恢复/审计路径。
- 新增 `caveman.memory.quarantine`，把 quarantine list/restore lifecycle 从 `sqlite_store.py` 拆出，保持 no-god-file 上限；`sqlite_store.py` 仍为 450 行。
- `SQLiteMemoryStore` 现在挂载 `list_quarantined()` / `restore_quarantined()`：
  - list 支持按 `source`、`reason`、`limit` 查询 quarantined memory；
  - source 过滤使用 `CASE WHEN json_valid(metadata_json) THEN json_extract(...) ELSE 0 END`，避免 malformed metadata crash；
  - restore 保留并追加 `quarantine_audit`，写入 `restored_at`、`restored_by`、`restore_reason`、`previous_governance_state`、`previous_quarantine_reason`。
- 新增 `caveman memory-quarantine list/restore` CLI，支持 `--db-path`、`--source`、`--reason`、`--limit`、`--restored-by`、`--restore-reason`，给 operator 可观测/可审计的恢复入口。
- CLI 主入口接入 `memory-quarantine` 子命令，并更新 API reference。
- 新增 TDD 回归覆盖 store lifecycle 和 CLI lifecycle：list、source/reason filter、restore 后不再 active quarantine、audit metadata 保留。

## Round 7 验证结果
- Focused lifecycle tests：`2 passed in 0.10s`。
- Regression subset：`tests/test_memory.py tests/test_memory_decay.py tests/test_flywheel_dashboard_boundaries.py` 共 `37 passed in 0.46s`。
- Coverage gate：`scripts/ci_coverage_gate.py --maxfail=1 --tb=short -q` 通过（此前 ruff+coverage gate exit 0）。
- Ruff focused：`caveman/cli/main.py caveman/cli/memory_quarantine.py caveman/memory/quarantine.py caveman/memory/sqlite_store.py caveman/memory/store_helpers.py tests/test_memory.py` pass。
- CI lint parity：`ruff check --select E9,F63,F7,F82 caveman tests` pass。
- Docs generation：`scripts/generate_api_reference.py` 更新并提交 `docs/API_REFERENCE.md`，CI docs gate pass。
- Mypy baseline-aware gate：full-project historical baseline 仍可见；Round 7 touched Python files 无 blocking mypy errors。发现 `caveman/cli/main.py` 既有 `yaml` import-untyped 会因本轮触碰而阻塞，已加 `# type: ignore[import-untyped]`，没有绕过新增错误。
- Security scan：changed/untracked files pattern scan clean；push hook safety checks passed。
- Independent review：发现 source filter 直接 `json_extract` 会对 malformed `metadata_json` 崩溃；已改为 `CASE WHEN json_valid(...)` 并重新验证。
- Remote CI：最新 SHA `92d68192e404060a6b1261f43aee27e93074780b` GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

## Round 6 做了什么
- 把 flywheel dashboard 从 source/type skew 展示升级为 source-governance 行动面板。
- `source_breakdown` 现在展示每个 source 的：`active`、`quarantined`、`eligible_for_source_policy`，并在报告里显示 `noise` 与 `recall-reduction`。
- 新增 `source_governance` actions：按所有 sources 聚合，不受 top-12 source_breakdown 截断影响；优先展示 eligible/quarantined 的高噪声来源。
- Dashboard eligibility 与 Round 5 `MemoryDecay` 策略对齐：复用 decay 常量，检查 30-89 天 age window，并按 decay 后 `new_trust` 估算是否会进入 source policy。
- 保持 legacy schema 容错：只有存在 `metadata_json`、`created_at`、`trust_score`、`retrieval_count`、`helpful_count` 时才启用 source diagnostics；partial legacy schema 不再因缺 `created_at` 报错。
- 新增回归：malformed metadata、长 source label、已 quarantined/eligible source、new/old age boundary、top-12 截断外 actionable source、partial legacy source schema。

## Round 6 验证结果
- Dashboard boundaries：`9 passed in 0.05s`。
- Focused suite：`43 passed in 0.47s`（dashboard boundaries + memory decay + memory + event_chain）。
- Coverage gate：`3300 passed, 8 skipped`；observed coverage `69.10%` > baseline `68.25%`，80% 长期债务继续可见。
- Docs generation idempotent：`scripts/generate_api_reference.py` 后 docs 无 diff。
- Mypy baseline-aware gate：full-project baseline debt 仍可见；changed Python files `caveman/training/_flywheel_memory_diagnostics.py`、`caveman/training/flywheel_dashboard.py` 无 mypy errors。
- Ruff：changed files pass。
- Security scan：added-line pattern scan clean；push hook safety checks passed。
- Independent review：第一轮发现 top-12 truncation 与 age policy mismatch；第二轮发现 partial legacy schema 与 post-decay trust mismatch；均已修复。最终 re-review passed，无 blocker。
- Remote CI：GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

## Round 5 做了什么
- 基于 dashboard 证据，把 `MemoryDecay` 从只按通用 age/trust 规则治理，升级为 source-aware lifecycle。
- 新增高噪声来源 allowlist：`import:openclaw`、`import:openclaw-session`、`import:hermes`、`import:hermes-skill-ref`。
- 对这些来源新增 early quarantine 策略：age >= 30 天且 < 90 天、new_trust <= 0.08、retrieval_count=0、helpful_count=0、尚未 quarantined 时，标记 `governance_state=quarantined`。
- 90 天以上仍走 Round 4 的既有 stale import quarantine 路径，保持原因 `stale_low_signal_import`，避免破坏历史语义。
- 新增可解释 metadata：`quarantine_reason=source_policy_low_signal_import`、`quarantine_policy={source,min_age_days,trust_threshold,requires_retrieval_count,requires_helpful_count}`、`previous_trust_score`、`quarantined_at`。
- `DecayResult` 新增 dry-run/impact counters：`eligible_by_source`、`quarantined_by_source`，用 `field(default_factory=dict)` 避免 mutable default。
- 新增 TDD 回归：高噪声来源 45 天可提前 quarantine；generic import 不被误伤；dry-run 只报告不 mutate；retrieved/helpful imports 受保护。

## Round 5 验证结果
- TDD RED：新增 3 个 source-aware tests 初始 `2 failed, 1 passed`，确认旧实现不会提前 quarantine 或报告 source impact。
- GREEN：`tests/test_memory_decay.py` 全部 `16 passed`。
- Focused suite：`41 passed in 0.45s`（memory decay + memory/event/dashboard boundaries）。
- Docs generation idempotent：`scripts/generate_api_reference.py` 后 docs 无 diff。
- Mypy baseline-aware gate：full-project baseline debt 仍可见；changed Python file `caveman/memory/decay.py` 无 mypy errors。
- Coverage gate：`3300 passed, 8 skipped`；observed coverage `69.13%` > baseline `68.25%`，80% 长期债务继续可见。
- Ruff：changed files pass。
- No-god-file gates：pass；`decay.py` 273 行，`test_memory_decay.py` 344 行。
- Security scan：added-line pattern scan clean；push hook safety checks passed。
- Independent pre-commit review：passed，无 blocker。建议后续可补 exactly 30/90 day boundary 与 already-quarantined idempotence 测试。
- Remote CI：GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

## Round 4 做了什么
- Round 4/50 聚焦 dashboard 证据中最高噪声源：imported memories（`import:openclaw` n=950 never=94% helpful=0%，`import:openclaw-session` n=145 never=95% helpful=1%，`import:hermes*` never=100%）。
- 将 `governance_state=quarantined` 从“metadata 标记”接入真实 SQLite active recall/exposure 路径：FTS、LIKE、vector candidate query、fallback recall、`search_sync()`、`search_by_entity()`、`recent()`、`all_entries()` 均应用 active-memory SQL predicate，并保留 Python `is_quarantined()` 防线。
- active-memory SQL predicate 已集中到 `store_helpers.active_memory_sql()`，使用 `CASE WHEN json_valid(...) THEN json_extract(...) ELSE 1 END`，避免 SQLite `OR` 不短路导致坏 JSON 仍触发 `malformed JSON`。
- 修复 recall 更新 `last_accessed` 时对坏 `metadata_json` 的 JSONDecodeError 容错，保持 row_to_entry 既有 legacy tolerance。
- 将 decay 单次扫描上限从 500 提升到 2000，避免 bulk import 噪声只能 500-row trickle 治理；新增 bulk import quarantine 测试。
- LOOP_END decay integration 日志增加 `memories_quarantined`，否则 import governance 发生时 observability 仍显示“0 decayed/0 pruned”而沉默。
- 把 delete-memory cross-ref cleanup 抽到 `store_helpers.cleanup_related_refs()`，使 `sqlite_store.py` 降到 no-god-file gate 上限内（450 行）。
- 补齐 TDD/回归测试：quarantined recall candidate 排除、fallback leak、sync search leak、recent/all_entries leak、FTS LIMIT crowding、malformed metadata tolerance、decay bulk scan、decay quarantine logging。

## Round 4 验证结果
- Docs generation idempotent：`scripts/generate_api_reference.py` 后 docs diff hash 不变。
- Mypy baseline-aware gate：full-project baseline debt 仍可见；changed Python files 无 mypy errors。
- 聚焦测试：`38 passed in 0.43s`（`tests/test_memory.py tests/test_memory_decay.py tests/test_event_chain.py tests/test_flywheel_dashboard_boundaries.py`）。
- Coverage gate：`3297 passed, 8 skipped`；observed coverage `69.16%` > baseline `68.25%`，80% 长期债务继续可见。
- Ruff：changed files pass。
- No-god-file gates：pass；`sqlite_store.py` 降到 450 行。
- Security scan：added-line pattern scan clean；push hook safety checks passed。
- Independent pre-commit review：先发现 SQL `OR` 短路假设问题；已改为 `CASE WHEN json_valid(...)` 并用 sqlite malformed predicate proof 验证。
- Remote CI：GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

## 已知坑
- 不要用裸 `python`，它可能指向 Hermes venv；Caveman 验证一律用项目 `.venv/bin/python`。
- 不要用 `nohup caveman serve &` 从 Hermes terminal 启动 gateway；历史上会触发 exit-130 loop。需要启动 gateway 时用 `subprocess.Popen(..., start_new_session=True)` 或现有 gateway SOP。
- `scripts/ci_mypy_gate.py | tail` 普通管道会隐藏前段 exit status；需要用脚本自身 exit code 或 `set -o pipefail`。
- `json_extract(metadata_json, ...)` 不能依赖 `OR NOT json_valid(metadata_json)` 短路；SQLite 可先求值 `json_extract`。必须使用 `CASE WHEN json_valid(...) THEN json_extract(...) ELSE ... END`。
- 对 quarantine 这类治理状态，不能只在 Python 层 “LIMIT 后过滤”；必须尽量下推 SQL predicate，否则 bulk import 噪声会挤占 top-k candidate 页。
- Round 5 source-policy 边界：30-89 天走 `source_policy_low_signal_import`；>=90 天保留旧的 `stale_low_signal_import` 语义。后续补 boundary/idempotence 测试时不要误改原因语义。
- Round 6 dashboard source-governance 不能基于 top-N displayed breakdown 生成；必须扫描所有 sources，否则小但 actionable 的 source 会被隐藏。
- Dashboard source policy 口径要和 `MemoryDecay` 对齐：使用 decay 后 `new_trust`、30-89 天 age window、retrieval/helpful 保护，而不是只看当前 trust。
- Round 7 CLI touched `caveman/cli/main.py`，触发 mypy baseline-aware gate 对该文件的既有 `yaml` import-untyped 债务；已加 targeted ignore。后续触碰 baseline-heavy 文件时要先跑 `scripts/ci_mypy_gate.py`，不要只跑局部 mypy。
- Quarantine list/restore 的任何 metadata JSON 查询都必须使用 malformed-safe `CASE WHEN json_valid(...)`；review 已抓到一次 source filter 直接 `json_extract` 的 regression。
- API reference 是 CI docs gate 的 committed artifact。新增/导出函数或改模块 docstring 后要运行 `scripts/generate_api_reference.py` 并提交 `docs/API_REFERENCE.md`，不要把 docs diff 当成失败回滚。
- Legacy task-result source normalization 不能只看 `content.startswith("Task:")`；必须要求显式 `\nResult:` 或 ` Result:`，否则会误伤 organic “Task:” 笔记。
