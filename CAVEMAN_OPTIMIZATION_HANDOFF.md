# Caveman 优化 HANDOFF

更新时间: 2026-04-30 09:30 CST

## 下次启动时做
1. Round 5/50：在 `MemoryDecay` 上加 source-aware lifecycle 策略，把 dashboard 的高噪声来源（`import:openclaw`、`import:openclaw-session`、`import:hermes`、`import:hermes-skill-ref`）做成可解释的分级治理：年龄/recall/helpful/trust/source 权重、dry-run 统计、可逆 quarantine、审计日志。
2. Round 6：把 dashboard 从“展示 source/type skew”升级为“行动面板”：显示 active/quarantined/eligible_by_source、预计治理影响、quarantine 后 recall 候选减少量。
3. Round 7-10：继续沿 memory flywheel 数据流做闭环质量：import metadata 规范化、missing source 回填、helpfulness/retrieval 反馈质量、decay scheduling 与 observability。
4. Rounds 11-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

## 本轮做了什么
- Round 4/50 聚焦 dashboard 证据中最高噪声源：imported memories（`import:openclaw` n=950 never=94% helpful=0%，`import:openclaw-session` n=145 never=95% helpful=1%，`import:hermes*` never=100%）。
- 将 `governance_state=quarantined` 从“metadata 标记”接入真实 SQLite active recall/exposure 路径：FTS、LIKE、vector candidate query、fallback recall、`search_sync()`、`search_by_entity()`、`recent()`、`all_entries()` 均应用 active-memory SQL predicate，并保留 Python `is_quarantined()` 防线。
- active-memory SQL predicate 使用 `json_valid(metadata_json)` 防护；legacy/corrupt metadata 不会因 `json_extract()` 抛 `malformed JSON` 而击穿 recall/recent/all_entries。
- 修复 recall 更新 `last_accessed` 时对坏 `metadata_json` 的 JSONDecodeError 容错，保持 row_to_entry 既有 legacy tolerance。
- 将 decay 单次扫描上限从 500 提升到 2000，避免 bulk import 噪声只能 500-row trickle 治理；新增 bulk import quarantine 测试。
- LOOP_END decay integration 日志增加 `memories_quarantined`，否则 import governance 发生时 observability 仍显示“0 decayed/0 pruned”而沉默。
- 补齐 TDD/回归测试：quarantined recall candidate 排除、fallback leak、sync search leak、recent/all_entries leak、FTS LIMIT crowding、malformed metadata tolerance、decay bulk scan、decay quarantine logging。

## 验证结果
- 聚焦测试：`38 passed in 0.43s`（`tests/test_memory_decay.py tests/test_memory.py tests/test_event_chain.py tests/test_flywheel_dashboard_boundaries.py`）。
- Selected memory regression：`126 passed in 11.33s`（training pivot/fixes、memory self-audit/provider/bridge/metadata/migrations/decay/memory/event_chain）。
- Ruff：changed files pass。
- Security scan：added-line grep for common secrets/shell/eval/pickle/SQL-format patterns clean；internal `SECRET_PATTERNS` scan `pattern_matches []`。
- Dashboard smoke：runs successfully; current data still shows import-source skew because dry-run does not mutate production memory DB.
- Decay dry-run：`Decay: scanned=2000, decayed=0, pruned=0, quarantined=0, trust_reduced=0.000`（current import memories only 6-12 days old; existing age threshold prevents immediate quarantine）。
- Gateway health：running PID 33057, Discord connected ✅, slash commands synced ✅, no gateway log alerts。

## 独立 review 结论
- Review 1 blocked ship：recall fallback and `search_sync()` could still return quarantined memories. Fixed with regression tests.
- Review 2 blocked ship：`recent()` / `all_entries()` still exposed quarantined entries, and filtering after SQL `LIMIT` could let quarantined rows crowd out active matches. Fixed by pushing active-memory predicate into SQL and adding LIMIT-crowding regression.
- Review 3 blocked ship：raw `json_extract()` on malformed legacy metadata could raise `OperationalError: malformed JSON`; recall `last_accessed` update also assumed valid JSON. Fixed with `json_valid()` SQL guard and JSONDecodeError fallback.
- Review 4 passed: no real security concerns or logic errors; only maintainability suggestion to centralize SQL predicate construction later.

## 已知坑
- 不要用裸 `python`，它可能指向 Hermes venv；Caveman 验证一律用项目 `.venv/bin/python`。
- 不要用 `nohup caveman serve &` 从 Hermes terminal 启动 gateway；历史上会触发 exit-130 loop。需要启动 gateway 时用 `subprocess.Popen(..., start_new_session=True)` 或现有 gateway SOP。
- `scripts/ci_mypy_gate.py | tail` 普通管道会隐藏前段 exit status；需要用脚本自身 exit code 或 `set -o pipefail`。
- `json_extract(metadata_json, ...)` 必须先 guard `json_valid(metadata_json)`；历史/损坏 memory metadata 不能让 recall/search/recent 崩溃。
- 对 quarantine 这类治理状态，不能只在 Python 层 “LIMIT 后过滤”；必须尽量下推 SQL predicate，否则 bulk import 噪声会挤占 top-k candidate 页。
