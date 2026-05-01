## 🚀 启动指令
Phase: 日常优化 | Gate: 自动续跑 | 更新: 2026-05-01
类型: 🅳️内部 | 复杂度: L
你的第一个动作：读取 HANDOFF，按 Caveman 50轮连续优化 SOP 继续下一轮。

### ⏳ 待元宝确认
| 决策 | AI选择 | 理由 | 状态 |

## HANDOFF
### 下次启动时做
继续 Round 38/50：先确认 Round 37 handoff docs commit 的 CI；若 code commit 60a456c 的 run 25197774691 已 success 不要重复长等，然后继续扫描 CLI/dashboard/report config/file/DB-derived operator-facing 输出，优先 `format_*report`、`typer.echo(format_...)`、Rich/TUI table rows 中未使用 `operator_literal` 的字符串 label。
### 上次做了什么
Round 37 完成：为 `caveman/import_/report.py` 的 import detect/manifest/result report 增加 `operator_literal()` escaping，覆盖 source label、source_path.name、skip_reason、warnings、details；新增 `tests/test_import_report_operator_output.py` 3 个 regression；提交 code commit 60a456c 并 push；更新 CAVEMAN_OPTIMIZATION_HANDOFF.md。
### 什么work了
RED regression 3 failed 按预期失败；focused 3 passed；full suite `3329 passed, 8 skipped`；ruff/py_compile/API docs/security scan/independent review 全通过；code CI run 25197774691 success。
### 什么没做/没work
未重启 gateway（本轮不依赖）；Reviewer 建议的 `target_type` label literal 化未做，风险低，可作为下一轮小切片；本次 handoff docs commit 需要 push 后监控 CI。
### 已知坑
`git add -N` 后 diff 才包含 untracked 测试文件；长 CI polling 放进 execute_code 可能被 300s wrapper timeout 截断，截断后用短 GitHub API 查询；import report 的 source/path/warnings/details 属于 external/file/scan-derived operator output。
