# Caveman 优化 HANDOFF

更新时间: 2026-05-01 13:09 CST

## 当前最终状态

- Round 45 已完成、提交并推送到 `main`；code commit `6c800d8` (`[verified] escape train target output`)。
- 本轮 GitHub push 成功，`origin/main` 已到 `6c800d8`；当前 cron 环境没有可用 GitHub token（`GITHUB_TOKEN`/`~/.hermes/.env` 均未提供），因此未能查询 GitHub Actions run id/结论。未泄露 credential，token 状态只记录为 `[REDACTED]`。
- Round 44 code commit `7006fd7` 与 docs commit `c2263ab` 已在本轮起始状态确认位于 `main`。
- 自动续跑 cron job 保持不变；本 cron run 未创建/修改/删除 cron jobs。
- Gateway health 起始不可达；本轮未重启 gateway，优化任务不依赖 gateway。

## 下次启动时做
1. 先确认 Round 45 code commit `6c800d8` 与本 handoff docs commit 的 GitHub Actions 状态；当前 cron 环境缺少 token，如果下轮 token 恢复，补查 run id/结论。
2. 继续 Round 46：扫描剩余 training CLI / dashboard / report 的 operator-facing 输出，优先 `_run_embedding` / `_run_sft` / `_run_rl` dry-run 或 result message 中由 path、model、dataset、stats 派生的动态输出；每次只做一个 TDD 小切片。
3. 继续按“证据→TDD RED→实现 GREEN→focused/full suite→ruff/security→independent review→commit/push→CI→handoff”推进。
4. Gateway health 当前不可达但本自动续跑不依赖 gateway；除非任务明确需要，不要重启 gateway。

## Round 45 做了什么
- 先确认真实状态：`main` 起始为 `c2263ab`，handoff 显示 Round 44 已完成；pre-run 中 gateway health 不可达，本轮未重启 gateway。
- 继续 operator-facing output 安全边界扫描，聚焦 `caveman.training.cli_handler.run_train()` 的 unknown target 输出。该路径直接把 untrusted `target` 拼入错误消息，可能让换行、ANSI escape、BEL 等控制字符伪造后续 operator-facing 行。
- RED：新增 `tests/test_training_cli_handler_operator_output.py::test_unknown_train_target_escapes_target_name`，构造 `embedding\nP0: forged\x1b[31m`，确认旧实现会 raw 输出并失败。
- GREEN：在 `caveman/training/cli_handler.py` 引入共享 `operator_literal()`，仅对 unknown-target message 的 `target` 字段做最终输出边界 escaping。
- 该切片不改训练逻辑、不改目标分派语义，只锁住 CLI 错误输出的 operator boundary。

## Round 45 验证结果
- RED：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py::test_unknown_train_target_escapes_target_name -q` 先失败，失败原因为旧 message raw 包含 `embedding\nP0: forged\x1b[31m`。
- Focused GREEN：同一测试通过。
- Focused gates：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py tests/test_flywheel_dashboard_operator_output.py tests/test_training_stats_operator_output.py -q` → `8 passed in 0.06s`。
- Py compile：`.venv/bin/python -m py_compile caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` pass。
- Ruff：`.venv/bin/python -m ruff check caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` → pass。
- Full suite：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3343 passed, 8 skipped in 114.21s`。
- Security scan：staged diff hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook public-repo safety checks passed；未泄露 credential。
- Independent review：passed；无 blocking `security_concerns`，无 `logic_errors`。
- Remote CI：push 到 `origin/main` 成功；CI 查询因 cron 环境缺少 GitHub token 未能获取 run id/结论。

## Round 45 什么 work 了
- TDD 小切片有效：unknown target 输出先失败再修复，避免把当前 operator escaping 波次扩大到训练实现细节。
- 继续复用单一 `operator_literal()` 边界，行为与 dashboard/stats/source governance 既有安全边界一致。
- Full suite baseline 增至 3343 passed / 8 skipped（排除 NFR）。

## Round 45 什么没做/没work
- 未能确认 GitHub Actions run id/结论：当前 cron 环境没有可用 GitHub token；输出 credential 状态只写 `[REDACTED]`。
- 本轮只处理 unknown target；未继续处理 `_run_embedding` / `_run_sft` / `_run_rl` 中 path/model/result/stat 派生的动态输出。
- 未重启 gateway（当前自动续跑不依赖 gateway）。

## Round 45 已知坑
- Training CLI 的错误/status 字符串也是 operator-facing 输出；任何来自 CLI 参数、文件 path、模型名、dataset path、训练结果 dict 的动态值都应在最终输出边界 escape，而不是假设“只给开发者看”。
- GitHub Actions API credential 在 cron 环境可能缺失；push 可成功不代表 API polling 可用。不要打印 token 值。
- 继续保持“最终 formatter/message 边界 escape，数据层保留 raw 值”的原则，避免 pre-escaped 字符串在后续处理里造成二次 escaping 或绕过。
