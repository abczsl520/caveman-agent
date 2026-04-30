## 🚀 启动指令
Phase: 日常优化 | Gate: 自动续跑 | 更新: 2026-05-01
类型: 🅳️内部 | 复杂度: L
你的第一个动作：按 Caveman 50轮连续优化 SOP 执行下一轮：开工确认→baseline/证据→TDD→实现→验证→review→commit/push/CI→更新 HANDOFF。

### ⏳ 待元宝确认
| 决策 | AI选择 | 理由 | 状态 |
| --- | --- | --- | --- |
| 自动续跑方向 | operator-facing output 安全边界小步优化 | 延续 CAVEMAN_OPTIMIZATION_HANDOFF Round 30 下一步 | 自动执行中 |

## HANDOFF
### 下次启动时做
先确认 Round 32 handoff/docs commit 的 CI；继续 Round 33/50：扫描 status/utility/changelog/audit/migrate/self-test 等 operator-facing config/file/DB-derived 输出，优先挑一个未走 operator_literal 的可控字符串做 TDD 小切片。
### 上次做了什么
Round 32 完成：status dashboard 的配置派生 model 名称改为 operator_literal(model)，防止换行/ANSI spoof；code commit 8c09262 已 push，GitHub Actions run 25193311855 success；本次更新 CAVEMAN_OPTIMIZATION_HANDOFF.md 和项目记忆，待 docs commit/push/CI。
### 什么work了
RED regression 先证明原 status_text 会输出真实换行和 raw ANSI；修复后 tests/test_cli_status.py 9 passed；full suite 3321 passed, 8 skipped；py_compile/ruff/security/API/independent review 均通过；GitHub Actions success。
### 什么没做/没work
尚未系统扫描 status_text 的 mem_detail/Home 以及 utility commands 其他输出；Round 33 继续。未重启 gateway（当前自动续跑不依赖 gateway）。
### 已知坑
必须使用 /Users/yeren64g/projects/caveman/.venv/bin/python；cron run 禁止递归创建/修改 cron jobs；结束前释放 /tmp/caveman-50round.lock；GitHub Actions run 可能 push 后数分钟才出现在 head_sha 查询；status_text 的 model/provider 来自配置，模型名等 operator-facing config 值应走 operator_literal。
