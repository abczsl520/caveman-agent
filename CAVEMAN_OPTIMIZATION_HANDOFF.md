# Caveman 优化 HANDOFF

更新时间: 2026-05-01 13:38 CST

## 当前最终状态

- Round 48 已完成、提交并推送到 `main`；code commit `b9f1145` (`[verified] escape embedding dry-run output`)。
- 本轮 GitHub push 成功，`origin/main` 已到 `b9f1145`；当前 cron 环境没有可用 GitHub token，且未安装/未认证 `gh` CLI，因此未能查询 GitHub Actions run id/结论。未泄露 credential，token 状态只记录为 `[REDACTED]`。
- Round 47 code commit `775dc2e` 与 docs commit `45d76af` 已在本轮起始状态确认位于 `main`。
- 自动续跑 cron job 保持不变；本 cron run 未创建/修改/删除 cron jobs。
- Gateway health 起始不可达；本轮未重启 gateway，优化任务不依赖 gateway。

## 下次启动时做
1. 先确认 Round 48 code commit `b9f1145` 与本 handoff docs commit 的 GitHub Actions 状态；当前 cron 环境缺少 token/gh，如果下轮 credential 恢复，补查 run id/结论。
2. 继续 Round 49：扫描剩余 `_run_embedding` operator-facing 输出，优先 eval-only `baseline_eval`、train result `result`、auto-select `report` / `selected_path` / `reason` 派生的动态输出；每次只做一个 TDD 小切片。
3. 继续按“证据→TDD RED→实现 GREEN→focused/full suite→ruff/security→independent review→commit/push→CI→handoff”推进。
4. Gateway health 当前不可达但本自动续跑不依赖 gateway；除非任务明确需要，不要重启 gateway。

## Round 48 做了什么
- 先确认真实状态：`main` 起始为 `45d76af`，handoff 显示 Round 47 已完成；pre-run 中 gateway health 不可达，本轮未重启 gateway。
- 继续 operator-facing output 安全边界扫描，聚焦 `caveman.training.cli_handler._run_embedding()` 的 dry-run 输出。该路径直接把 `dataset_path` 拼入 operator-facing status message，若 `output_dir` 含换行、ANSI escape、BEL 等控制字符，可伪造后续行。
- RED：新增 `tests/test_training_cli_handler_operator_output.py::test_embedding_dry_run_escapes_dataset_path`，用 monkeypatch 注入含 `\nP0:` 与 `\x1b[31m` 的 embedding output dir，确认旧实现 raw 输出并失败。
- GREEN：复用共享 `operator_literal()`，仅在 embedding dry-run 的最终输出边界 escape `dataset_path`。
- 该切片不改 pair extraction、dataset build、embedding train/eval/auto-select 逻辑，不改 raw 数据层，只锁住 CLI dry-run status 的 operator boundary。

## Round 48 验证结果
- Baseline：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3345 passed, 8 skipped in 116.89s`。
- RED：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py::test_embedding_dry_run_escapes_dataset_path -q` 先失败，失败原因为旧 message raw 包含 `embedding\nP0: forged\x1b[31m/embedding_pairs.jsonl`。
- Focused GREEN：`tests/test_training_cli_handler_operator_output.py::test_embedding_dry_run_escapes_dataset_path` → `1 passed in 0.04s`。
- Focused file：`tests/test_training_cli_handler_operator_output.py` → `4 passed in 0.04s`。
- Focused gates：`.venv/bin/python -m pytest tests/test_training_cli_handler_operator_output.py tests/test_flywheel_dashboard_operator_output.py tests/test_training_stats_operator_output.py -q` → `11 passed in 0.08s`。
- Py compile：`.venv/bin/python -m py_compile caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` pass。
- Ruff：`.venv/bin/python -m ruff check caveman/training/cli_handler.py tests/test_training_cli_handler_operator_output.py` → pass。
- Full suite：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py` → `3346 passed, 8 skipped in 116.28s`。
- Security scan：staged diff hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；changed-file high-confidence sensitive scan passed；repository push hook public safety checks passed。
- Independent review：passed；无 blocking `security_concerns`，无 `logic_errors`。
- Remote CI：push 到 `origin/main` 成功；CI 查询因 cron 环境缺少 GitHub token 且无 authenticated `gh`，未能获取 run id/结论。

## Round 48 什么 work 了
- TDD 小切片有效：embedding dry-run 输出先失败再修复，范围小、证据明确。
- 继续复用单一 `operator_literal()` 边界，保持“最终 formatter/message 边界 escape，数据层保留 raw 值”的原则。
- Full suite baseline 增至 3346 passed / 8 skipped（排除 NFR），新增 1 个 operator-output regression test。

## Round 48 什么没做/没work
- 未能确认 GitHub Actions run id/结论：当前 cron 环境没有可用 GitHub token，且无 authenticated `gh`；输出 credential 状态只写 `[REDACTED]`。
- 本轮只处理 embedding dry-run dataset path；未继续处理 `_run_embedding` eval-only、train result、auto-select compare/report/selection/reason 中的动态输出。
- 未重启 gateway（当前自动续跑不依赖 gateway）。

## Round 48 已知坑
- Training CLI 的 dry-run/status/eval/auto-select 字符串都是 operator-facing 输出；来自 CLI 参数、文件 path、模型名、dataset path、训练结果 dict、eval report/reason 的动态值都应在最终输出边界 escape。
- Repo-wide secret regex 会命中既有 redaction/scanner 测试 canary 与示例 pattern；本轮未新增这些内容。push 前应至少做 staged/changed-file sensitive scan，并保留 push hook 的 public-repo safety check。
- GitHub Actions API credential 在 cron 环境可能缺失；push 可成功不代表 API polling 可用。不要打印 token 值。
