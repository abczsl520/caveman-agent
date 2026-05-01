# Caveman 优化 HANDOFF

更新时间: 2026-05-01 15:10 CST

## 当前最终状态

- Round 51 已完成、提交并推送到 `main`；code commit `5d68fac` (`[verified] escape embedding auto-select output`)。
- 本轮起始确认 local/main HEAD 为 `dde7806`，Round 50 follow-up CI 已知 green；pre-run gateway health 不可达，本轮未重启 gateway。
- 本轮 push 到 `origin/main` 成功；push hook public-repo safety checks passed。`origin/main` 已包含 `5d68fac`。
- 当前 cron 环境没有 `gh`，也没有 GitHub API token；未能查询 GitHub Actions run id/结论。credential 状态只记录为 `[REDACTED]`，未泄露 token。
- 自动续跑 cron job 保持不变；本 cron run 未创建/修改/删除 cron jobs。

## 下次启动时做
1. 先确认 Round 51 code commit `5d68fac` 与本 handoff docs commit 的 GitHub Actions 状态；如果 credential 恢复，补查 run id/结论。
2. 继续下一轮：扫描 training CLI operator-facing 输出，优先 `_run_sft` / `_run_rl` 非 dry-run train result 输出是否也需要 `operator_literal()`；每次只做一个 TDD 小切片。
3. 继续按“证据→TDD RED→实现 GREEN→focused/full suite→ruff/security→independent review→commit/push→CI→handoff”推进。
4. Gateway health 当前不可达但本自动续跑不依赖 gateway；除非任务明确需要，不要重启 gateway。

## Round 51 做了什么
- 确认真实状态：`main` 起始为 `dde7806`，`CAVEMAN_OPTIMIZATION_HANDOFF.md` 已显示 Round 50 follow-up CI 修复完成；pre-run gateway health 不可达，本轮未重启 gateway。
- 继续 operator-facing output 安全边界扫描，聚焦 `caveman.training.cli_handler._run_embedding()` 的 auto-select 分支。
- 发现 root cause：auto-select 分支中 `evaluator.compare(...)` 产出的 `report`、`evaluator.improvement_decision(...)` 产出的 `reason`、以及 selected path 都直接拼入最终 CLI/operator message。若 evaluator/report/path 含换行或 ANSI/control chars，可伪造后续行。
- RED：新增 `test_embedding_auto_select_escapes_report_and_reason`，fake evaluator 返回含 `\nP0:` 和 `\x1b[31m` 的 report/reason，旧实现 raw 输出并失败。
- GREEN：在 auto-select selected/not-changed 两条返回路径上复用共享 `operator_literal()`，只在最终 message 边界 escape `report`、`reason`、`selected_path`。
- 按 reviewer 建议补充 success branch regression：`test_embedding_auto_select_escapes_report_and_selected_path` 覆盖 selected=true 时 report 与 selected path escaping。
- 该切片不改 pair extraction、dataset build、embedding train/eval/selection 决策逻辑，不改 raw evaluator 数据层，只锁住 CLI auto-select status 的 operator boundary。

## Round 51 验证结果
- Baseline：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3348 passed, 8 skipped in 117.53s`。
- RED：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py::test_embedding_auto_select_escapes_report_and_reason -q` 旧实现失败；失败原因为 message raw 包含 `## Report\nP0: forged\x1b[31m` 与 `quality\nP0: reason\x1b[31m`。
- Focused GREEN：`test_embedding_auto_select_escapes_report_and_reason` → `1 passed`。
- Reviewer suggestion follow-up RED/GREEN：新增 selected branch test，先因未覆盖/未安全处理 selected path 失败，修正测试 fixture 后通过。
- Focused gates：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py tests/test_flywheel_dashboard_operator_output.py tests/test_training_stats_operator_output.py -q` → `15 passed in 0.09s`。
- Py compile：`.venv/bin/python -m py_compile caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` pass。
- Ruff：`.venv/bin/python -m ruff check caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` → pass。
- Full suite：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3350 passed, 8 skipped in 113.94s`。
- Security scan：staged diff hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook public safety checks passed。
- Independent review：passed；无 blocking `security_concerns`，无 `logic_errors`。
- Remote CI：push 到 `origin/main` 成功；CI 查询因 cron 环境缺少 `gh` / token，未能获取 run id/结论。

## Round 51 什么 work 了
- 小切片 TDD 有效：report/reason 先失败再修复；selected path 根据 independent review 建议追加成功分支覆盖。
- 继续复用单一 `operator_literal()` 边界，保持“最终 formatter/message 边界 escape，数据层保留 raw 值”的原则。
- Full suite baseline 3348 passed / 8 skipped，完成后 3350 passed / 8 skipped（排除 NFR），新增 2 个 operator-output regression tests。

## Round 51 什么没做/没work
- 未能确认 GitHub Actions run id/结论：当前 cron 环境没有 `gh` / GitHub API token；输出 credential 状态只写 `[REDACTED]`。
- 本轮只处理 embedding auto-select report/reason/selected_path；未继续处理 `_run_sft` / `_run_rl` 非 dry-run train result 输出。
- 未重启 gateway（当前自动续跑不依赖 gateway）。

## Round 51 已知坑
- Training CLI 的 dry-run/status/eval/auto-select 字符串都是 operator-facing 输出；来自 CLI 参数、文件 path、模型名、dataset path、训练结果 dict、eval report/reason/selection path 的动态值都应在最终输出边界 escape。
- Python f-string 对非-dict object 会调用 `__str__`；dict 子类也可覆盖 `__str__`，不能假设 result/report/reason 是安全可打印对象。
- Repo-wide secret regex 会命中既有 redaction/scanner 测试 canary 与示例 pattern；本轮未新增这些内容。push 前应至少做 staged/changed-file sensitive scan，并保留 push hook 的 public-repo safety check。
- GitHub Actions API credential 在 cron 环境可能缺失；push 可成功不代表 API polling 可用。不要打印 token 值。
