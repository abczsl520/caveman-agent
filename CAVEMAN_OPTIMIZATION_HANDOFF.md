# Caveman 优化 HANDOFF

更新时间: 2026-04-30 23:08 CST

## 当前最终状态
- Round 13 已完成、提交并推送到 `main`。
- 最新 code commit: `edae36bf747baa7cb55e5addce47a9ba1044ba7e` (`[verified] report restorable quarantine impact`)。
- 最新 handoff/docs commit: `60161563a3a5e242749604a0a7214bc535c6b2f5` (`docs: update api reference for quarantine preview`)。
- `origin/main` 已同步到最新 SHA。
- GitHub Actions 对 Round 13 docs/handoff 最新 SHA 全绿：run `25173026185`，`https://github.com/abczsl520/caveman-agent/actions/runs/25173026185`。
- 自动续跑已配置：cron job `36500447cc33` (`Caveman 50轮自动续跑`)，每 30 分钟触发，最多 48 次，目标回发当前 Discord thread；preflight 脚本 `/Users/yeren64g/.hermes/scripts/caveman_50round_preflight.py`，互斥锁 `/tmp/caveman-50round.lock`。
- Round 9 code commit: `23df73debb9c113251eb0390515c47dbca9d5aa5` (`Protect decay with canonical access timestamps`)；Round 9 handoff commit: `ba76a2d93f5db3f08d60e6e34d17798555ea619d`。
- Round 8 handoff commit: `f4a07bcde9f67e3d283dc43fda4dbe559a174a36`；Round 8 code commit: `9624ce069d74efa063e2c8c2aa4fef0feef80604` (`Normalize imported memory source metadata`)。
- Round 7 handoff commit: `be73746`；Round 7 code commits: `92d6819` (`[verified] fix quarantine lifecycle CI gates`)、`b2c0169` (`[verified] add reversible memory quarantine lifecycle`)。
- Round 6 code commit: `5e98357f06f34a4e6b28716ab5414c436378dea4` (`[verified] surface source governance in flywheel dashboard`)；Round 6 handoff commit: `1ceef53`。
- Round 5 commit: `7de8ff65a156c924fc9af2d74452bd24e8e39f77` (`[verified] add source-aware memory quarantine policy`)。
- Gateway 最后已知未运行：需要交互验证时按 gateway SOP 安全启动，避免 `nohup caveman serve &` 触发 Hermes terminal exit-130 loop。

## 下次启动时做
1. Round 14/50：优先考虑 reviewer 留下的 operator semantics 问题：restorable quarantine preview 是否应脱离 `MemoryDecay` dry-run 成功路径独立展示；或继续补 source impact trend / quarantine candidate drift。先用 dashboard 数据/测试证明缺口，再 TDD 小步修。
2. Round 14 也可处理 source taxonomy 后续大小写策略/unknown source report；Round 11 reviewer 仅建议文档化 unknown casing，不是 blocker。
3. Rounds 15-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

## Round 13 做了什么
- 聚焦 quarantine restore observability：Round 10 已有 dry-run preview API，但 dashboard 只展示 decay 会新增 quarantine 的影响，缺少“当前已 quarantined 且可恢复”的 operator 视角。
- 新增 `caveman.training._flywheel_quarantine_preview.collect_restorable_quarantine_preview()`，从 memory DB 的 quarantined metadata 中统计 restorable candidates 的 `source` 与 `quarantine_reason`。
- `FlywheelDashboard.collect_memory_stats()` 在 decay dry-run 成功时追加 `restorable_quarantine_by_source` / `restorable_quarantine_by_reason`，report 输出对应 source/reason impact 行。
- 为避免 `flywheel_dashboard.py` 继续膨胀，统计逻辑抽到独立 helper module；malformed/non-dict metadata best-effort 跳过，固定 SQL 无用户输入。
- CI 首轮在 docs job 失败，根因是新增 API module 后 `docs/API_REFERENCE.md` 未更新；已运行 `scripts/generate_api_reference.py` 并提交 docs fix。

## Round 13 验证结果
- RED：新增 dashboard boundary test 初始证明旧 dashboard 缺少 restorable quarantine source/reason impact；后续 CI docs job 对 code commit `edae36bf747baa7cb55e5addce47a9ba1044ba7e` 失败，证明 API reference stale。
- GREEN focused tests：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` 共 `40 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3298 passed, 8 skipped`。
- Py compile：`caveman/training/flywheel_dashboard.py caveman/training/_flywheel_quarantine_preview.py tests/test_flywheel_dashboard_boundaries.py` pass。
- Ruff changed files：pass。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：code review passed，无 blocker；docs fix review passed，确认唯一 diff 为 generated API reference 更新。
- Remote CI：code commit `edae36bf747baa7cb55e5addce47a9ba1044ba7e` 的 run `25172545287` docs failure（API reference 未提交），已修复；docs commit `60161563a3a5e242749604a0a7214bc535c6b2f5` 的 run `25173026185` completed success。

## Round 12 做了什么
- 聚焦 decay scheduling/observability 的 operator report，避免 memory governance 只在后台静默执行。
- `FlywheelDashboard.collect_memory_stats()` 现在复用 `MemoryDecay(db_path).run(dry_run=True)` 生成只读 decay preview，输出 `scanned`、`would_decay`、`would_prune`、`would_quarantine`、`trust_total_reduced`、`would_quarantine_by_source`、`eligible_by_source`。
- dashboard report 新增 `Decay dry-run: scan=..., would_decay=..., would_prune=..., would_quarantine=...` 行，便于 operator 在运行实际 decay 前看到影响面。
- 新增 `already_quarantined` 直接全表统计，避免只看 top source breakdown 导致 omitted source 下 quarantined 数量被低估。
- dry-run preview 对 malformed metadata / sqlite lock / IO error 失败保持 best-effort：跳过 preview，不破坏基础 dashboard memory stats。

## Round 12 验证结果
- RED：新增 decay dry-run dashboard test 初始失败，错误为 `KeyError: 'decay_dry_run'`，证明旧 dashboard 没有 operator preview。
- GREEN focused tests：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` 共 `39 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3297 passed, 8 skipped`。
- Py compile：`caveman/training/flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py` pass。
- Ruff changed files：pass。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：第一次指出 sqlite lock preview 会破坏 dashboard，已 catch `sqlite3.Error`/`OSError` 并加 regression；第二次指出 `already_quarantined` top-N undercount 与 fixed-date test，已改为全表 query 和相对日期；最终 review passed，无 blocker。
- Remote CI：Round 12 commit `19fc4348b8dbaae6cba5119c34bf04f5358d04aa` GitHub Actions run `25170776976` completed success。

## Round 11 做了什么
- 聚焦 import/source taxonomy 漂移：把 decay allowlist、dashboard source-governance、migration normalization 的 source 字符串收敛到单一模块 `caveman.memory.sources`。
- 新增 `canonicalize_memory_source()`、`SOURCE_ALIASES`、`SOURCE_POLICY_LOW_SIGNAL_IMPORTS`、`IMPORT_SOURCE_PREFIX`，覆盖 `import:openclaw_sessions` / `import:openclaw-sessions` / `openclaw_sessions` 等 legacy 拼写。
- `MemoryDecay.run()` 现在对 metadata source 先 canonicalize，再做 source policy / prune 判断；非 dry-run 时会把 normalization audit metadata 持久化，即使该 row 只是 decayed、没有被 quarantined。
- `store_helpers.normalize_import_metadata()` 显式 source 回填路径复用同一 canonicalizer，避免 migration 与 decay/dashboard 各自维护字符串口径。
- `_flywheel_memory_diagnostics` dashboard 复用同一 taxonomy，并修复 review 抓到的 display label 与 policy identity 混用问题：展示 label 可截断，但 eligibility 使用 canonical identity。
- `docs/API_REFERENCE.md` 已由 `scripts/generate_api_reference.py` 更新，包含新模块 API。

## Round 11 验证结果
- RED：新增 canonical source tests 后初始失败：`import:openclaw_sessions` 未被 quarantine/source grouping 识别；review 补充的两个 regression 初始失败，分别证明非 quarantined normalization 未持久化、dashboard policy 被 display truncation 影响。
- GREEN focused tests：`tests/test_memory_decay.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_migrations.py` 共 `40 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3295 passed, 8 skipped`。
- Py compile：`caveman/memory/sources.py caveman/memory/decay.py caveman/memory/store_helpers.py caveman/training/_flywheel_memory_diagnostics.py` pass。
- Ruff changed files：pass。
- Mypy touched source：pass。
- Docs/API：`scripts/generate_api_reference.py` 已运行并提交 API reference diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：第一次 failed，指出 non-quarantined normalization 没持久化、dashboard display label 与 policy identity 耦合；已按 review 修复并新增 regression。第二次 review passed，无 blocker；仅建议后续文档化 unknown source casing policy。
- Remote CI：Round 11 commit `fe013ad84e58cb5c4ebbeef97d34c9465268a8ff` GitHub Actions run `25169126875` completed success。

## Round 10 做了什么
- 聚焦 quarantine restore 的 operator guardrails，避免“能恢复单条”演变成未来误批量恢复事故。
- 新增 `QuarantineRestorePreview` dry-run impact report，包含匹配 entries、`total_matches`、`by_source`、`by_reason`。
- 新增 `preview_restore_quarantined(store, source=None, reason=None, limit=500)`：
  - 只执行 `SELECT`，不写 DB；
  - 支持 source/reason 双重精确 scope；
  - 复用 malformed-safe `CASE WHEN json_valid(metadata_json) THEN json_extract(...) ELSE 0 END` predicate；
  - 使用 SQLite 参数绑定，避免 SQL injection。
- `list_quarantined()` 改为复用 `_quarantine_where()` / `_row_to_memory_entry()`，保持既有 source list 行为，同时减少后续 predicate 漂移。
- 新增 CLI：`caveman memory-quarantine preview-restore --source ... --reason ... --limit ...`，输出 `would_restore=N`、sources/reasons impact、候选 memory 列表；这是批量恢复前的只读预检入口。
- 为“为什么老停下来”做系统排查并加自动续跑机制：
  - 原因不是代码阻塞，而是 Hermes 单次对话/上下文/任务收口后不会天然无限自驱；需要外部 scheduler 重新唤醒。
  - 已创建 cron job `36500447cc33`，每 30 分钟自动续跑，最多 48 次；prompt 明确禁止递归 schedule，要求每次按 SOP/TDD/review/CI/handoff 推进下一轮。
  - 已写 preflight 脚本 `/Users/yeren64g/.hermes/scripts/caveman_50round_preflight.py`，注入 git/gateway/lock 状态；使用 `/tmp/caveman-50round.lock` 防并发。

## Round 10 验证结果
- RED：新增 preview tests 后初始失败，错误为 `ImportError: cannot import name 'preview_restore_quarantined'`，确认旧实现缺少 dry-run preview API。
- GREEN focused tests：新增 direct preview + CLI preview 测试通过：`2 passed`。
- Focused quarantine/memory gate：`6 passed`。
- Expanded regression：`tests/test_memory.py tests/test_memory_decay.py tests/test_flywheel_dashboard_boundaries.py` 共 `41 passed`。
- Ruff changed files：`caveman/memory/quarantine.py caveman/cli/memory_quarantine.py tests/test_memory.py` pass。
- Mypy baseline-aware：changed source 无新增 mypy 错误；full invocation 仍暴露既有 baseline debt（`caveman/providers/error_classifier.py`、`caveman/utils.py`），非本轮新增。
- API reference：首次 code CI docs job 失败，根因是新增导出函数后 `docs/API_REFERENCE.md` 未提交；已运行 `scripts/generate_api_reference.py` 并提交 docs commit `07c576fc20c6ec083eeb0b1db8a2d41a013eee6d`。
- Security scan：changed files/docs added-line scan clean；push hook safety checks passed。
- Independent review：passed，无 blocker/important；确认 dry-run 非 mutation、SQL 参数化、source/reason scope 正确、CLI 只读安全。
- Remote CI：
  - code commit `3ba164e651b39f04e49ef4d69437d807edb63c2f` 的 run `25167169827` docs job failure（API reference artifact 未提交），已修复。
  - docs/API commit `07c576fc20c6ec083eeb0b1db8a2d41a013eee6d` 的 run `25167433914` completed success。

## Round 9 做了什么
- 聚焦 helpfulness/retrieval feedback 对 decay protection 的真实闭环：此前 decay 只读取 `metadata_json.last_accessed`，但实际 SQLite schema 有 canonical `memories.last_accessed` 列。若 recall/update 只写 canonical column、metadata 没同步，最近访问的 helpful/retrieved memory 可能被 decay 误伤。
- `MemoryDecay.run()` 现在会优先读取 `memories.last_accessed` canonical column，并保留 legacy `metadata_json.last_accessed` fallback。
- 为旧 DB/schema copy 增加容错：运行前通过 `PRAGMA table_info(memories)` 判断是否存在 `last_accessed` 列；不存在时用 `NULL AS last_accessed`，避免 `no such column: last_accessed` 崩溃，同时保持旧 metadata fallback/age-based decay 行为。
- 新增 TDD 覆盖：
  - `test_last_accessed_column_without_metadata_is_immune`：canonical column 有近期 access、metadata 为空时不 decay。
  - `test_decay_tolerates_legacy_schema_without_last_accessed_column`：legacy schema 没有 `last_accessed` 列时 decay 不崩溃且仍治理旧未访问 memory。
- 测试 helper `_create_test_db()` / `_insert_memory()` 补齐 `last_accessed` 列，后续 decay tests 更贴近生产 schema。

## Round 9 验证结果
- RED：新增 legacy schema test 初始失败，错误为 `sqlite3.OperationalError: no such column: last_accessed`，确认不是无效测试。
- Focused regression：`tests/test_memory_decay.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory.py` 共 `39 passed`。
- Expanded focused gate：`tests/test_memory_decay.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory.py tests/test_memory_metadata_quality_wiring.py` 共 `43 passed`。
- Ruff changed files：`caveman/memory/decay.py tests/test_memory_decay.py` pass。
- Ruff CI parity：`ruff check --select E9,F63,F7,F82 caveman tests` pass。
- Mypy changed-file sanity：`caveman/memory/decay.py tests/test_memory_decay.py` pass。
- Coverage gate：`3310 passed, 8 skipped`；observed coverage `69.23%` > baseline `68.25%`，80% 长期债务继续可见。
- API reference generation：`scripts/generate_api_reference.py` 后 `docs/API_REFERENCE.md` 无 diff。
- Live DB dry-run smoke：对 `/Users/yeren64g/.caveman/memory/caveman.db` dry-run，`Decay: scanned=2000, decayed=0, pruned=0, quarantined=0, trust_reduced=0.000`，无 mutation。
- Security scan：changed files added-line pattern scan clean；push hook safety checks passed。
- Independent review：passed，无 blocker/important；确认 f-string SQL 只在固定 literal `last_accessed` / `NULL AS last_accessed` 中选择，低风险。
- Remote CI：Round 9 code commit `23df73debb9c113251eb0390515c47dbca9d5aa5` GitHub Actions run `25154718044` completed success。

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
- Remote CI：code SHA `9624ce069d74efa063e2c8c2aa4fef0feef80604` GitHub Actions 全绿（docs、test 3.12、install smoke macOS/Ubuntu/Windows）。

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
- Round 7 CLI touched `caveman/cli/main.py`，触发 mypy baseline-aware gate 对该文件的既有 `yaml` import-untyped` 债务；已加 targeted ignore。后续触碰 baseline-heavy 文件时要先跑 `scripts/ci_mypy_gate.py`，不要只跑局部 mypy。
- Quarantine list/restore 的任何 metadata JSON 查询都必须使用 malformed-safe `CASE WHEN json_valid(...)`；review 已抓到一次 source filter 直接 `json_extract` 的 regression。
- API reference 是 CI docs gate 的 committed artifact。新增/导出函数或改模块 docstring 后要运行 `scripts/generate_api_reference.py` 并提交 `docs/API_REFERENCE.md`，不要把 docs diff 当成失败回滚。
- Legacy task-result source normalization 不能只看 `content.startswith("Task:")`；必须要求显式 `\nResult:` 或 ` Result:`，否则会误伤 organic “Task:” 笔记。
- Round 9 decay 不能只读 `metadata_json.last_accessed`；生产 schema 的 canonical `memories.last_accessed` 才是 recall/access 更新主路径。查询时也要兼容没有该列的旧 DB copy。
- Quarantine restore 未来若做批量 mutation，不要绕过 Round 10 `preview_restore_quarantined()`；必须先 dry-run impact report，再用明确 source/reason scope，并保留 audit metadata。
- Hermes 不会在单次 Discord turn 结束后天然无限继续；长期 50 轮需要 scheduler 外部唤醒。当前自动续跑 job 为 `36500447cc33`，preflight 脚本负责状态注入，cron run 禁止递归创建 cron。
- Round 11 source taxonomy：policy 判断必须使用 canonical identity，展示层 label 可截断但不能反向参与治理判断；normalization audit metadata 对非 quarantined rows 也要持久化，否则每轮 decay 会重复“发现”同一 alias。

