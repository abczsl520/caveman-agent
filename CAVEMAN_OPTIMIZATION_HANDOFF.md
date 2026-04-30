# Caveman 优化 HANDOFF

更新时间: 2026-04-30 10:53 CST

## 当前最终状态
- Round 5 已完成、提交并推送到 `main`。
- 最新 commit: `7de8ff65a156c924fc9af2d74452bd24e8e39f77` (`[verified] add source-aware memory quarantine policy`)。
- `origin/main` 已同步到该 SHA。
- GitHub Actions 对该 SHA 全绿：`docs` success、`test (3.12)` success、install smoke macOS/Ubuntu/Windows success。
- Round 4 代码 commit: `90ee17f75cbe5ca78a8c7335c6051ccf19962829`；Round 4 handoff commit: `a70d3f5552d6aed51e3f07549915283f001b727a`。
- Gateway 当前未运行：`curl http://localhost:4201/health` 失败；日志显示 09:37 用户停止并移除 PID。这不是代码失败，但下轮若需要交互验证要按 gateway SOP 安全启动。

## 下次启动时做
1. Round 6/50：把 dashboard 从“展示 source/type skew”升级为“行动面板”：显示 active/quarantined/eligible_by_source、source-policy dry-run impact、quarantine 后 recall 候选减少量，并优先展示 `import:openclaw` / `import:openclaw-session` / `import:hermes*`。
2. Round 7：补齐 quarantine lifecycle 的可逆操作面板/CLI：按 source/reason 查看、恢复单条或批量恢复、输出审计日志。
3. Round 8-10：继续沿 memory flywheel 数据流做闭环质量：import metadata 规范化、missing source 回填、helpfulness/retrieval 反馈质量、decay scheduling 与 observability。
4. Rounds 11-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

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
