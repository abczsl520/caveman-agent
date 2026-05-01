## 🚀 启动指令
Phase: 日常优化 | Gate: 自动续跑 | 更新: 2026-05-01
类型: 🅳️内部 | 复杂度: L
你的第一个动作：读取 HANDOFF，按 Caveman 50轮连续优化 SOP 继续下一轮。

### ⏳ 待元宝确认
| 决策 | AI选择 | 理由 | 状态 |

## HANDOFF
### 下次启动时做
继续 Round 36/50：先确认 Round 35 handoff docs commit 的 CI；若 code commit 46301e9 的 run 25196099823 已 success 不要重复长等，然后继续扫描其他 CLI/dashboard config/file/DB-derived operator-facing 输出，优先寻找未使用 `operator_literal` 的字符串 label，按 TDD 小切片推进。
### 上次做了什么
Round 35 完成：为 `caveman.cli.status._format_gateway_status()` 的 gateway log diagnostic `boundary` 与 alert pattern labels 增加 `operator_literal()` escaping，防止 log-derived 换行或 ANSI 控制字节伪造 status 输出；提交 code commit 46301e9 并 push；更新 CAVEMAN_OPTIMIZATION_HANDOFF.md。
### 什么work了
Baseline `3323 passed, 8 skipped`；RED regression 按预期失败；focused status/gateway diagnostics 18 passed；full suite `3324 passed, 8 skipped`；ruff/py_compile/API docs/security scan/independent review 全通过；code CI run 25196099823 success。
### 什么没做/没work
未重启 gateway（本轮不依赖）；尚未系统扫描其他 CLI/dashboard config/file/DB-derived 输出；本次 handoff docs commit 需要 push 后监控 CI。
### 已知坑
`operator_literal` 会给普通 label 加引号，tests 需同步期待 literal 边界；长 CI polling 放在 execute_code 可能被 300s wrapper timeout 截断，截断后用短查询确认状态；本环境无 `gh` CLI，CI 监控用 GitHub public REST API。
