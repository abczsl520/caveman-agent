# Caveman 优化 HANDOFF

更新时间: 2026-05-01 03:54 CST

## 当前最终状态
- Round 26 已完成、提交并推送到 `main`；GitHub Actions run `25185877026`，`https://github.com/abczsl520/caveman-agent/actions/runs/25185877026`，completed success。
- Round 25 已完成、提交并推送到 `main`；GitHub Actions run 待补查（当前仅补查了 Round 26）。
- Round 24 已完成、提交并推送到 `main`；GitHub Actions run 待补查。
- Round 23 已完成、提交并推送到 `main`；GitHub Actions run 待补查。
- Round 22 已完成、提交并推送到 `main`。
- Round 21 已完成、提交并推送到 `main`。
- Round 20 已完成、提交并推送到 `main`。
- Round 19 已完成、提交并推送到 `main`。
- Round 18 已完成、提交并推送到 `main`。
- Round 17 已完成、提交并推送到 `main`。
- Round 16 已完成、提交并推送到 `main`。
- 最新 code commit: `2ee79fa` (`[verified] share operator literal formatter`)。
- Round 26 code commit: `2ee79fa9c0675cf97cb6e662eb5feee5d8708e48` (`[verified] share operator literal formatter`)；GitHub Actions run `25185877026` completed success。
- Round 25 code commit: `3d393ae` (`[verified] escape source drift operator literals`)；GitHub Actions run 待补查（本机无 GitHub token/gh auth）。
- Round 24 code commit: `4c8ea7cbfa8e08309a3a6da3955533cbd6e6c341` (`[verified] document source governance literal safety`)；GitHub Actions run 待补查（GitHub API rate limit）。
- Round 23 code commit: `2578fbabe2f1a99b1f3030667f8761610cc2ba04` (`[verified] centralize source governance literals`)；GitHub Actions run 待补查（GitHub API rate limit）。
- Round 22 code commit: `163417f8e70fa7b72153bc0dc636d5fc38bb0b93` (`[verified] harden source governance preview output`)；GitHub Actions run `25182925025`，`https://github.com/abczsl520/caveman-agent/actions/runs/25182925025`，completed success。
- Round 21 code commit: `4cfa33540927fc9a7cbd97bbaed1f16a928c6444` (`[verified] add source drift review checklist`)；GitHub Actions run `25182107564` completed success。
- Round 20 code commit: `fd234095a4f0d6ab20ca9bf33a0c099418b31382` (`[verified] preserve source drift rerun scope`)；GitHub Actions run `25180742729` completed success（Round 22 已补查）。
- Round 19 code commit: `7062e3414fc52d4e5e11773570fa0e3020aab4de` (`[verified] add source drift policy workflow`)；GitHub Actions run `25179439795` completed success。
- Round 14 handoff/docs content commit: `da1059ba3a61aba476a576547c2a4898627b082a` (`docs: update caveman handoff after round 14`)；本文件可能随后有 metadata-only 校正提交。
- `origin/main` 已同步到最新 SHA（除非下一轮发现 CI/handoff commit 待补）。
- 自动续跑已配置：cron job `36500447cc33` (`Caveman 50轮自动续跑`)，每 5 分钟触发，最多 240 次，目标回发当前 Discord thread；preflight 脚本 `/Users/yeren64g/.hermes/scripts/caveman_50round_preflight.py`，互斥锁 `/tmp/caveman-50round.lock`。preflight 已具备 stale lock 自愈：lock pid 不存在或 lock 超过 90 分钟会自动清理，避免死锁后永久跳过。
- Gateway 最后已知未运行：需要交互验证时按 gateway SOP 安全启动，避免 shell background 启动触发 Hermes terminal exit-130 loop。

## 下次启动时做
1. 先补查 Round 25 code SHA `3d393ae`、Round 24 code SHA `4c8ea7cbfa8e08309a3a6da3955533cbd6e6c341` 与 Round 23 code SHA `2578fbabe2f1a99b1f3030667f8761610cc2ba04` 的 GitHub Actions run id/结论；Round 26 CI 已确认 success。
2. 继续 Round 27/50：扫描其它 operator-facing DB-derived plaintext output（memory quarantine list/preview、diagnostics/dashboard sections 等）是否仍有 raw control-character/newline spoofing 风险；优先复用 `caveman.operator_output.operator_literal()`，保持 CLI/dashboard 只读，不自动 mutate allowlist/quarantine。
3. Dashboard 主文件已在 450 行 hard limit；继续 dashboard 方向必须优先抽 helper，不要在 `flywheel_dashboard.py` 主文件堆逻辑。
4. Rounds 27-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

## Round 26 做了什么
- 先按 handoff 检查真实状态：main 已到 Round 25，工作区干净；`gh` 不可用，但 unauthenticated GitHub Actions API 可查，Round 26 后续 run 可监控。
- 聚焦 Round 25 下一步：source-governance CLI 与 dashboard 已分别有 `_operator_literal()`，但安全边界仍分散在两个模块，后续新增 operator-facing DB-derived output 容易再次复制/漂移。
- RED 新增 regression：要求 CLI 和 dashboard literal formatter 都依赖共享 `caveman.operator_output.operator_literal()`；旧代码初始失败 `ModuleNotFoundError: No module named 'caveman.operator_output'`。
- 实现：新增 `caveman/operator_output.py`，提供共享 `operator_literal(value, max_length=None)`；source-governance CLI 与 flywheel dashboard formatter 都委托该 helper；同步更新 API reference。

## Round 26 验证结果
- Baseline focused before change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `68 passed`。
- RED：`tests/test_memory.py::test_operator_literal_helper_is_shared_across_cli_and_dashboard` 初始失败，旧代码没有共享模块。
- GREEN focused：shared helper + docstring tests → `2 passed`；expanded focused suite → `69 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3313 passed, 8 skipped`。
- Py compile：`caveman/operator_output.py caveman/cli/source_governance.py caveman/training/_flywheel_dashboard_formatters.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` 生成 `caveman.operator_output` API 条目并已纳入本轮提交。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；非阻塞建议：若 public helper 后续被更多调用，可考虑显式校验/文档说明 `max_length` 必须为正。
- Remote CI：code commit `2ee79fa9c0675cf97cb6e662eb5feee5d8708e48` GitHub Actions run `25185877026` completed success。

## Round 25 做了什么
- 聚焦 Round 24 后续 operator-facing literal 安全：dashboard `Source policy drift` report 仍把 DB-derived source label / candidate 直接拼进 plaintext report，遇到 ANSI ESC/control characters 会造成 terminal spoofing 风险。
- RED 新增 regression：构造 `import:evil\x1b[31mspoof` drift source，要求 dashboard report 中 `candidate=` 与 `label=` 使用 repr-style escaped literal；旧代码初始失败，显示 raw ESC 进入 report。
- 实现：在 `_flywheel_dashboard_formatters.py` 增加 `_operator_literal()`，对 source policy drift label/candidate 使用 repr-style literal；同步更新既有 drift report test 的期望格式。

## Round 25 验证结果
- RED：`tests/test_flywheel_dashboard_boundaries.py::test_source_policy_drift_escapes_control_characters_in_operator_report` 初始失败，旧 report 缺少 escaped `candidate='...\\x1b...'`。
- GREEN focused：新增 regression 通过；dashboard focused suite `tests/test_flywheel_dashboard_boundaries.py tests/test_flywheel_dashboard.py` → `24 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3312 passed, 8 skipped`。
- Py compile：`caveman/training/_flywheel_dashboard_formatters.py` pass。
- Ruff changed files：pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker。
- Remote CI：code commit `3d393ae` 已 push；本机无 GitHub token/gh auth，无法查询 Actions run id/结论，待下一轮补查。

## Round 24 做了什么
- 先补查 Round 23 CI，但 `gh` 未认证且 unauthenticated GitHub API rate limit 仍用尽，无法读取 run id/结论。
- 聚焦 Round 23 independent review 的非阻塞建议：`_operator_literal()` 已集中 operator-facing literal formatting，但 helper 本身没有说明它是安全边界，后续维护者可能把它当普通 repr wrapper，削弱 control-character/terminal spoofing 防护语义。
- RED 新增 regression：要求 `_operator_literal.__doc__` 明确包含 control / repr / operator 语义；旧代码无 docstring，测试先失败。
- 实现：给 `_operator_literal()` 增加 purpose docstring：`Return repr() for operator output so control characters stay escaped.`；不改 runtime 行为、不写 memory rows、不修改 allowlist、不触碰 quarantine state。

## Round 24 验证结果
- Baseline focused before change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `66 passed`。
- RED：`tests/test_memory.py::test_operator_literal_has_security_purpose_docstring` 初始失败，旧 helper docstring 为空。
- GREEN focused：docstring regression + shared literal formatter test → `2 passed`；expanded focused suite → `67 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3311 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` 生成无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker。
- Remote CI：code commit `4c8ea7cbfa8e08309a3a6da3955533cbd6e6c341` 已 push；GitHub API rate limit exceeded，Actions run id/结论待下一轮补查。

## Round 23 做了什么
- 聚焦 Round 22 reviewer 建议：source-governance preview 已用 `!r` 防控制字符 spoofing，但 candidate detail、copy/paste workflow、review checklist 的 Python literal 输出分散在多个 inline `!r`，后续新增 operator-facing DB-derived output 容易漏掉统一安全语义。
- RED 新增 regression：monkeypatch 期望存在并调用 shared `_operator_literal()`，旧代码无该 helper，测试先失败 `AttributeError`。
- 实现：新增 `_operator_literal(value: object) -> str`，当前语义保持 `repr(value)`；把 source/reason/candidate_policy_entry/workflow/checklist 的 literal 输出全部改为调用该 helper，保持 CLI 只读，不写 memory rows、不修改 allowlist、不触碰 quarantine state。

## Round 23 验证结果
- Baseline focused before change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `65 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_uses_shared_literal_formatter` 初始失败，旧模块没有 `_operator_literal`，证明 literal formatting 仍是散落 inline。
- GREEN focused：shared literal + existing escaping/checklist/preview rows tests → `5 passed`；expanded focused suite → `66 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3310 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` 生成无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；建议给 `_operator_literal` 补 purpose docstring，后续可在 Round 24 做。
- Remote CI：code commit `2578fbabe2f1a99b1f3030667f8761610cc2ba04` 已 push；GitHub API rate limit exceeded，Actions run id/结论待下一轮补查。

## Round 22 做了什么
- 先补查历史 CI：Round 20 run `25180742729` 与 Round 21 run `25182107564` 都已 completed success。
- 聚焦 Round 21 reviewer 非阻塞建议：`source-governance preview-drift` checklist 已安全输出，但顶部 candidate detail rows 仍直接插入 DB-derived `source` / `reason` / `candidate_policy_entry`，遇到 newline/ANSI control chars 会造成 terminal/output spoofing。
- RED 新增 regression：构造带 `\n` 与 ANSI escape 的 source，并 monkeypatch drift reason 为带 control chars 的值；初始失败显示旧 preview rows 会拆出伪造行 `spoof`。
- 实现：candidate detail rows 的 source 改为 canonical `candidate_policy_entry!r`，reason 与 candidate_policy_entry 也用 `!r` 输出；保持 CLI 只读，不写 memory rows、不修改 source allowlist、不触碰 quarantine state。

## Round 22 验证结果
- Baseline focused before change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `64 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_preview_rows_escape_control_characters` 初始失败，旧 output 出现 data-controlled spoof line。
- GREEN focused：preview rows / candidate count / workflow / checklist / control-char tests → `5 passed`；expanded focused suite → `65 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3308 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；建议后续抽 dedicated escaping/sanitization helper，避免更多 inline `!r` 分散。
- Remote CI：Round 22 code commit `163417f8e70fa7b72153bc0dc636d5fc38bb0b93` GitHub Actions run `25182925025` completed success。

## Round 21 做了什么
- 聚焦 Round 20 handoff 指定的 `source-governance preview-drift` operator workflow：旧 output 有 copy/paste allowlist 与 re-run command，但没有每个 candidate 的 reviewed checklist，operator 多候选审查时容易漏项或无法标记审核状态。
- RED 新增 CLI regression：两个 drift candidates、`--limit 2` 时必须输出 `Review checklist:`，并为每个展示 candidate 生成 `[ ] <source> — reason=... total=...` checklist；初始失败证明旧 CLI 没有 checklist。
- 实现：preview workflow 在 re-run command 后输出 per-candidate checklist，保持只读，不写 memory rows、不修改 source allowlist、不触碰 quarantine state。
- Independent review 首轮/二轮指出 checklist 里的 DB-derived source/reason 可能造成 terminal/output spoofing；按 review 加了 control-character regression，并把 checklist source 与 reason 改为 `!r` 安全 Python literal 输出。

## Round 21 验证结果
- 先查 Round 20 CI：本环境 `gh` 未认证且 unauthenticated GitHub API rate limit 仍用尽，无法读取 run `25180742729` 结论。
- Baseline focused before change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `62 passed`；`tests/test_memory.py` → `19 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_prints_review_checklist_for_each_candidate` 初始失败，旧输出缺少 `Review checklist:`。
- Review-driven RED：`test_source_governance_cli_checklist_escapes_control_characters` 初始失败，旧 checklist raw source 允许 newline/ANSI escape 进入 output；修复后通过。
- Focused after change：source governance checklist tests `2 passed`；expanded focused suite → `64 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3308 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：最终 passed，无 security/logic blocker；非阻塞建议是后续把 pre-existing preview rows 也统一 safe display。
- Remote CI：code commit `4cfa33540927fc9a7cbd97bbaed1f16a928c6444` 的 GitHub Actions run `25182107564` 已完成 success。

## Round 20 做了什么
- 聚焦 Round 19 handoff 指定的 `source-governance preview-drift` operator workflow：旧 copy/paste re-run command 没保留显式 `--db`，也没保留 `--limit`，operator 审查临时/非默认 DB 时复制命令会悄悄切回默认 memory DB。
- RED 新增 CLI regression：传入 custom `--db`、`--min-rows 3`、`--limit 1` 时，preview output 必须打印可复制的 re-run command，保留同一个 DB scope 与 limit。初始失败显示旧输出只有 `--min-rows 3`。
- 实现：`preview_drift(ctx, ...)` 使用 Click `ParameterSource` 只在 operator 显式提供 `--db` 时把 DB path 写入 re-run command；同时保留 `--min-rows` 与 `--limit`。
- Independent review 首轮建议 re-run command 用 shell quoting 而不是 Python `repr`；按建议改为 `shlex.quote(str(db))`，并把 regression 加强为包含单引号路径的 shell-quote case。
- 保持 CLI 只读：不写 memory rows、不修改 source allowlist、不触碰 quarantine state。

## Round 20 验证结果
- Baseline focused before change：`tests/test_memory.py` → `18 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_rerun_command_preserves_custom_db_path` 初始失败，旧输出缺少 `--db` / `--limit`。
- Review-driven RED：改为 `test_source_governance_cli_rerun_command_shell_quotes_custom_db_path`，包含单引号路径；先确认 repr 方案不满足 shell-quote 期望，再改 `shlex.quote`。
- GREEN source-governance focused tests：5 passed。
- Expanded focused suite：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `62 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3306 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：首轮 passed 但建议 shell quoting；已按建议修正并二轮 review passed，无 security/logic blocker。
- Remote CI：code commit `fd234095a4f0d6ab20ca9bf33a0c099418b31382` 已 push；run `25180742729` 监控到 in_progress，随后 GitHub unauthenticated API rate limit exceeded，最终结论待下一轮先查。

## Round 19 做了什么
- 聚焦 Round 18 handoff 指定的 `source-governance preview-drift` operator workflow：旧 CLI 已有 candidate identity，但 operator 仍需要自己判断如何安全复制到 `SOURCE_POLICY_LOW_SIGNAL_IMPORTS`，容易误贴或遗漏只读边界。
- RED 新增 CLI regression：要求 preview 输出 `Policy workflow (copy/paste)`、明确 read-only review 步骤、给出可直接复制的 allowlist entry、re-run 命令与 `auto_mutation=disabled`。初始失败显示旧输出只有 candidate 行，没有 workflow block。
- Independent review 首轮指出 copy/paste entry 直接插入双引号字符串存在 malformed source label / quote / backslash 破坏 Python literal 的风险。
- 按 review 再 TDD 新增 unsafe source regression，先看失败，再改为 `{candidate_policy_entry!r}` 输出安全 Python literal；保持 CLI 只读，不写 memory rows、不修改 source allowlist、不触碰 quarantine state。

## Round 19 验证结果
- Baseline focused before change：`tests/test_memory.py` → `16 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_prints_copy_paste_policy_workflow` 初始失败，旧输出缺少 `Policy workflow (copy/paste)`。
- Review-driven RED：`tests/test_memory.py::test_source_governance_cli_escapes_copy_paste_policy_entries` 初始失败，旧 output 对 quote/backslash source 不是安全 Python literal。
- GREEN focused：新增 workflow / escaping / limit tests → `3 passed`。
- Focused suite after change：`tests/test_memory.py tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` → `61 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3305 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：首轮 failed（unsafe literal），已修复并加 regression；二轮 passed，无 security/logic blocker，仅建议 future 可以把 diagnostic `candidate_policy_entry=` 也 quote 以减少 operator confusion。
- Remote CI：Round 19 code commit `7062e3414fc52d4e5e11773570fa0e3020aab4de` GitHub Actions run `25179439795` completed success。

## Round 18 做了什么
- 聚焦 Round 17 handoff 指定的 `source-governance preview-drift` limit/排序 regression：旧 CLI 在 `--limit N` 后输出 `candidate_count=len(shown)`，operator 会误以为未展示的 drift candidates 不存在。
- RED 新增 CLI regression：构造 3 个 unmanaged low-signal import sources，使用 `--limit 1` 时要求输出 `candidate_count=3` 与 `showing_count=1`，并只展示 total 最大的 candidate。初始失败显示旧输出为 `candidate_count=1`。
- 实现：`_collect_memory_source_policy_drift(limit=None)` 支持返回完整 ordered candidate list；CLI 先取 all candidates 计数，再按 `--limit` 切片展示。
- 保持治理模型只读：不写 memory rows、不修改 source allowlist、不触碰 quarantine state。

## Round 18 验证结果
- Baseline focused before change：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py tests/test_memory.py` → `58 passed`。
- RED：`tests/test_memory.py::test_source_governance_cli_reports_total_candidates_separately_from_limit` 初始失败，旧输出 `candidate_count=1`。
- GREEN focused：新增 test + 既有 CLI/source drift tests → `4 passed`。
- Focused suite after change：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py tests/test_memory.py` → `59 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3303 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py caveman/training/_flywheel_memory_diagnostics.py tests/test_memory.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；确认 helper default limit 行为保留。
- Remote CI：code commit `c3196e429e70cba5d737db25970418b9b2b5a517` 已 push；本环境 `gh` 未认证且 unauthenticated GitHub API rate limit 已用尽（HTTP 403），无法读取 run id/结论，下一轮先检查该 SHA 的 Actions 状态。

## Round 17 做了什么
- 聚焦 Round 16 的下一步：`source_policy_drift` candidate 已在 dashboard/report 中出现，但 operator 仍缺少无需打开 Python/DB 的只读 CLI 入口来复制 candidate allowlist entry 与 impact。
- 新增 `caveman.cli.source_governance` Typer sub-app，注册为 `caveman source-governance preview-drift`。
- CLI 复用 `_collect_memory_source_policy_drift()`，支持 `--db`、`--min-rows`、`--limit`，输出 `candidate_count`、source、total/active、avg_trust、never/helpful pct、reason、recommended_action、`candidate_policy_entry`。
- 保持治理模型为 operator-assisted：preview 只读，不写 memory rows、不修改 source policy、不触碰 quarantine state。
- 更新 `docs/API_REFERENCE.md` 以满足 docs/API gate。

## Round 17 验证结果
- RED：新增 CLI regression 初始失败：测试不存在/命令模块未注册，随后实现后通过，证明旧系统缺少 operator CLI preview 入口。
- Focused baseline before change：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py tests/test_memory.py` → `57 passed`。
- Focused tests after change：同一集合 → `58 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3302 passed, 8 skipped`。
- Py compile：`caveman/cli/source_governance.py caveman/cli/main.py` pass。
- Ruff changed files：`caveman/cli/source_governance.py tests/test_memory.py` pass；`caveman/cli/main.py` 仍有既有 E402/F841/F541 baseline，未作为本轮新增 blocker。
- Docs/API：`scripts/generate_api_reference.py` 更新 `docs/API_REFERENCE.md`，已提交 docs commit。
- Security scan：changed added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；确认 CLI 只读、Typer option validation、SQLiteStore close、固定 SQL diagnostic helper。
- Remote CI：本环境 `gh` 未认证且 unauthenticated GitHub API rate limit 已用尽（HTTP 403），无法读取 run id；code/docs commits 已成功 push，下一轮先检查 `d5f854f` 的 Actions 状态。

## Round 16 做了什么
- 聚焦 Round 15 的下一步：`source_policy_drift` 已能发现 unmanaged low-signal import source，但 operator report 仍只显示“有漂移”，缺少可复制/审批的 policy candidate identity，尤其长 source label 被截断时无法安全加入 allowlist。
- TDD 新增期望：drift rows 必须携带 `recommended_action=review_for_low_signal_allowlist` 与 canonical `candidate_policy_entry`，report 必须展示 candidate identity。
- 实现：`_collect_memory_source_policy_drift()` 输出 canonical full identity 作为 `candidate_policy_entry`，并保留 display `label` 只用于人类可读展示；`_format_source_policy_drift()` 展示 candidate，同时用 `.get()` 做 defensive formatting。
- 保持 dashboard 主文件不增长：只改 helper modules，`flywheel_dashboard.py` 仍为 450 行。

## Round 16 验证结果
- RED：新增 source drift candidate tests 初始失败，旧 rows 缺少 `recommended_action` / `candidate_policy_entry`，证明缺口存在。
- Focused tests：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py tests/test_round11.py::TestLoopRefactor::test_no_file_over_400_lines` → `44 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3301 passed, 8 skipped`。
- Py compile：`_flywheel_memory_diagnostics.py`、`_flywheel_dashboard_formatters.py`、`tests/test_flywheel_dashboard_boundaries.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；建议 HTML/terminal escaping 与 malformed percentage 防御，当前 output 为 plaintext dashboard 且 formatter 已补 `.get()`，非阻塞。
- Remote CI：Round 16 code commit `f51d66939b96ea5a2f8504f76b599c72d7070a4b` GitHub Actions run `25177036454` completed success。

## Round 15 做了什么
- 聚焦 Round 11 reviewer 留下的 source taxonomy/unknown policy 明文化后续：dashboard 能治理已知 allowlist source，但无法提示“看起来低信号、但还没进入 allowlist 的 import source”，operator 不知道 allowlist 是否漂移。
- 新增 `_collect_memory_source_policy_drift()`：对 canonical source identity 聚合，识别 `import:` 前缀、未在 `SOURCE_POLICY_LOW_SIGNAL_IMPORTS`、active rows>=3、never_recalled>=90%、helpful=0、avg_trust<=0.1 的 bulk import source。
- 保持 display label 与 policy identity 解耦：展示可截断，policy/drift 判断使用 canonical full identity，避免长 source 被 truncation 误判。
- `FlywheelDashboard.collect_memory_stats()` 输出 `source_policy_drift`，report 展示 `Source policy drift` 行。
- 为避免 `flywheel_dashboard.py` 超过 NFR-502 450 行限制，新格式化逻辑抽到 `_flywheel_dashboard_formatters.py`，主文件保持 450 行。

## Round 15 验证结果
- RED：新增 unmanaged low-signal import drift test 初始失败，证明旧 dashboard 没有 `source_policy_drift`。
- 修复过程发现 NFR-502 regression：`flywheel_dashboard.py` 曾到 452/461 行；根因是 dashboard 主文件继续堆格式化逻辑，已抽 helper 并把主文件压回 450 行。
- Focused tests：`tests/test_round11.py::TestLoopRefactor::test_no_file_over_400_lines` 与 source drift regression `2 passed`；dashboard related suite `43 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3301 passed, 8 skipped`。
- Py compile：`flywheel_dashboard.py`、`_flywheel_memory_diagnostics.py`、`_flywheel_dashboard_formatters.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py --check` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches。
- Independent review：passed，无 blocker；建议 `_format_source_policy_drift()` 可更 defensive 使用 `.get()`，当前 rows 由 collector 生成，非阻塞。

## Round 14 做了什么
- 聚焦 Round 13 reviewer 留下的 operator semantics 问题：restorable quarantine preview 不应依赖 `MemoryDecay` dry-run 成功才展示。
- 根因：`collect_restorable_quarantine_preview(cur)` 原本只在 `decay_preview is not None` 分支里执行；如果 decay dry-run 因 sqlite lock/schema/IO 错误被 best-effort 跳过，dashboard 会同时丢失“当前已 quarantined 且可恢复”的 operator 视角。
- 新增 regression：模拟 `MemoryDecay.run(dry_run=True)` 抛 `sqlite3.OperationalError("database is locked")`，要求 dashboard 仍输出 top-level `restorable_quarantine_by_source/reason` 并在 report 展示 `Restorable quarantine: ...`。
- 实现：把 restorable quarantine collection 移到 decay preview 成功路径之外；成功时仍回填到 `decay_dry_run` 兼容旧消费者，同时 top-level stats 也提供独立字段。
- 为满足 NFR-502，压缩 `flywheel_dashboard.py` 到 450 行以内；没有新增 API docs 变更。

## Round 14 验证结果
- RED：新增 `test_restorable_quarantine_report_survives_decay_preview_failure` 初始失败，错误为 `KeyError: 'restorable_quarantine_by_source'`，证明旧实现把 restore observability 绑死在 decay dry-run 成功路径上。
- GREEN focused tests：`tests/test_flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py tests/test_memory_decay.py` + god-file gates 共 `43 passed`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3299 passed, 8 skipped`。
- Py compile：`caveman/training/flywheel_dashboard.py tests/test_flywheel_dashboard_boundaries.py` pass。
- Ruff changed files：pass。
- Docs/API：`scripts/generate_api_reference.py` 后 `docs/API_REFERENCE.md` 无 diff。
- Security scan：added-line hardcoded secret/shell/eval/pickle/SQL-string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 security/logic blocker；仅建议可读性/ordering 文档化，非阻塞。
- Remote CI：Round 14 code commit `8575c9e191ba3a4c392d9ac319c4ef1a9f571dbe` GitHub Actions run `25174683621` completed success。

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


