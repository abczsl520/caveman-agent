# Caveman 优化 HANDOFF

更新时间: 2026-05-01 06:31 CST

## 当前最终状态
- Round 31 已完成、提交并推送到 `main`；code commit `60a42f2e92bb00989c46a6bf3b4ea658bc421fbb` (`[verified] document ansi operator literals`)；GitHub Actions run `25192393917`，`https://github.com/abczsl520/caveman-agent/actions/runs/25192393917`，completed success。
- Round 30 已完成、提交并推送到 `main`；code commit `596c7d8ff2a3ac01ce583a0c161d14ab7ae4b95d` (`[verified] escape wiki search operator output`)；GitHub Actions run `25191456061`，`https://github.com/abczsl520/caveman-agent/actions/runs/25191456061`，completed success。
- Round 29 handoff/docs commit `8490c84` (`docs: update caveman handoff after round 29`) 已在当前 git log 中可见；Round 29 后续 fix commit `e3d5114` (`fix: satisfy flywheel typing gate`) 也已在 main。
- Round 29 code commit `82dccd1a0cbea776ac5750c71031bcf4d14574e9` (`[verified] enforce operator literal bound types`)；GitHub Actions run `25189817175` completed success。
- Round 28 handoff/docs commit `6a3ea3037ada3ec0ba2f2e1c6b823996c7f86e02` (`docs: update caveman handoff after round 28`) 已推送；GitHub Actions run `25189463356` completed success。
- Round 28 code commit `bc8f32f59831853bbdf10fa309e5cd76f72580cc` (`[verified] validate operator literal truncation bounds`)；GitHub Actions run `25189144126` completed success。
- Round 27 code commit `c2aa9379646f6d6b1917215ff7a49d5277ead967` (`[verified] escape quarantine operator output`)；GitHub Actions run `25187256050` completed success。
- Round 26 code commit `2ee79fa9c0675cf97cb6e662eb5feee5d8708e48` (`[verified] share operator literal formatter`)；GitHub Actions run `25185877026` completed success。
- Round 26 handoff/docs commit `30303b1013d53428de88810eafba492c89bf88bb` (`docs: update caveman handoff after round 26`)；GitHub Actions run `25186258099` completed success。
- Round 25 code commit: `3d393ae` (`[verified] escape source drift operator literals`)；GitHub Actions run 待补查。
- Round 24 code commit: `4c8ea7cbfa8e08309a3a6da3955533cbd6e6c341` (`[verified] document source governance literal safety`)；GitHub Actions run 待补查。
- Round 23 code commit: `2578fbabe2f1a99b1f3030667f8761610cc2ba04` (`[verified] centralize source governance literals`)；GitHub Actions run 待补查。
- Round 22 code commit: `163417f8e70fa7b72153bc0dc636d5fc38bb0b93` (`[verified] harden source governance preview output`)；GitHub Actions run `25182925025` completed success。
- `origin/main` 已同步到最新 SHA（除非下一轮发现 CI/handoff commit 待补）。
- 自动续跑已配置：cron job `36500447cc33` (`Caveman 50轮自动续跑`)，每 5 分钟触发，最多 240 次，目标回发当前 Discord thread；preflight 脚本 `/Users/yeren64g/.hermes/scripts/caveman_50round_preflight.py`，互斥锁 `/tmp/caveman-50round.lock`。cron run 禁止递归创建/修改/删除 cron jobs。
- Gateway 最后已知未运行；本轮未重启 gateway，优化任务不依赖 gateway。

## 下次启动时做
1. 先确认本 handoff docs commit 的 GitHub Actions 结论；如果 code commit `60a42f2` 的 run `25192393917` 已记录 success，不要重复等待。
2. 继续 Round 32/50：沿 operator-facing DB/file-derived output 安全边界深挖，扫描 wiki/source-governance/flywheel dashboard 之外 plaintext diagnostics 是否还有 raw DB/file-derived fields 未统一委托 `operator_literal`；优先做能 TDD 小切片验证的输出边界。
3. Dashboard 主文件仍有 450 行 hard limit；继续 dashboard 方向必须优先抽 helper，不要在 `flywheel_dashboard.py` 主文件堆逻辑。
4. Rounds 32-50：按“证据→TDD→实现→门禁→review→commit/push→监控”小步推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。

## Round 30 做了什么
- 先确认 Round 30 前真实状态：`main` 最新包含 `e3d5114`、`1ee23a5`、Round 29 handoff/docs commit `8490c84`，工作树起始干净；gateway health 不可达但本轮不依赖 gateway。
- 扫描 operator-facing 输出边界后，选中 `caveman/cli/wiki_mcp.py` 的 `wiki search`：它直接输出 `WikiEntry.title` 和 `entry.content[:120]`，这些字段来自 wiki DB/编译内容，可能包含换行、控制字符或 ANSI escape，存在终端/日志欺骗风险。
- RED 新增 regression：`test_wiki_search_cli_escapes_entry_title_and_preview` 使用 fake `WikiStore`/`WikiCompiler` 返回带换行的 `WikiEntry(title="Safe title\nSPOOF_TITLE", content="Safe content\nSPOOF_CONTENT")`，要求 CLI 输出 literal escaped `\\n`，且不产生真实下一行 spoof。
- 实现：`wiki search` 复用共享 `caveman.operator_output.operator_literal`；title 走 `operator_literal(entry.title)`，preview 走 `operator_literal(entry.content, max_length=120)`，避免局部 replace 和截断后残留控制字符。

## Round 30 验证结果
- RED/GREEN focused：`tests/test_memory.py::test_wiki_search_cli_escapes_entry_title_and_preview` → `1 passed`。
- Focused operator-output suite：wiki search regression + `operator_literal` shared/non-positive/non-integer tests → passed。
- Py compile：`caveman/cli/wiki_mcp.py tests/test_memory.py` pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3319 passed, 8 skipped in 116.12s`。
- Ruff changed files：`ruff check caveman/cli/wiki_mcp.py tests/test_memory.py` → pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` 生成检查通过；`docs/API_REFERENCE.md` 无需提交 diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是后续可补 ANSI escape sequence 覆盖。
- Remote CI：code commit `596c7d8ff2a3ac01ce583a0c161d14ab7ae4b95d` GitHub Actions run `25191456061` completed success。

## Round 30 什么 work 了
- 沿共享 `operator_literal` 安全边界继续推进，小切片只改 wiki search 的 operator-facing 输出，复用现有 helper，避免新增第二套 escaping 语义。
- Fake store/compiler + Typer `CliRunner` 可稳定覆盖 wiki CLI，不依赖真实 wiki DB 或文件系统状态。
- 本地 full suite、ruff、security scan、independent review、push 和 GitHub Actions 均通过；CI polling 修复了 heredoc/stdin 与 `set -e` 提前退出问题。

## Round 30 什么没做/没work
- 本 handoff 更新已准备单独 docs commit；提交后需 push 并监控 CI。
- 尚未补 independent reviewer 建议的 ANSI escape sequence 专项测试；可作为 Round 31 的低风险 TDD 入口。
- 尚未补查 Round 25/24/23 的历史 Actions run id/结论；优先级低于当前轮推进，可在后续有 API quota 时补。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 30 已知坑
- `curl | python - <<'PY'` 会让 heredoc 占用 Python stdin，导致 JSON 没传给 Python；CI polling 需要把 API 响应写 temp file 或用 `python -c`。
- `set -e` 下 Python 用 exit 2 表示“CI 仍在运行”会提前中断 shell loop；轮询脚本必须显式捕获 rc 后再决定 sleep/retry。
- `WikiEntry` title/content 属于 DB/file-derived operator output；任何 CLI/dashboard 输出这些字段时都必须使用共享 `operator_literal` 或等效统一 formatter。
- 必须使用 `/Users/yeren64g/projects/caveman/.venv/bin/python`；cron run 禁止递归创建/修改 cron jobs；结束前释放 `/tmp/caveman-50round.lock`。

## Round 31 做了什么
- 先确认真实状态：`main` 最新为 Round 30 handoff/docs commit `003d4d0`，工作树起始干净；gateway health 不可达但本轮不依赖 gateway；项目文件 `memory/projects/caveman.md` 已存在并读取。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3319 passed, 8 skipped`。
- 延续 Round 30 independent reviewer 建议，给共享 `operator_literal` 补 ANSI/C1 escape 专项 regression：输入包含 ESC CSI (`\x1b[31m`, `\x1b[0m`) 与 C1 CSI (`\x9b31m`)，要求 formatter 输出中没有 raw control bytes，只保留可见的 escaped literal。
- RED：新增测试先要求 `operator_literal.__doc__` 明确包含 `ANSI`，当前实现 docstring 只写 control characters，测试按预期失败。
- GREEN：仅更新 `caveman/operator_output.py` docstring 为“control and ANSI escape bytes stay visible”，实际 escaping 仍复用 Python `repr()` 的既有安全语义，避免新建第二套 ANSI parser。

## Round 31 验证结果
- Baseline full suite（before change）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3319 passed, 8 skipped in 111.02s`。
- RED：`tests/test_memory.py::test_operator_literal_documents_and_escapes_ansi_sequences` → expected failure on missing `ANSI` docstring assertion。
- GREEN focused：ANSI regression + operator_literal bound/shared/wiki tests → `5 passed`。
- Py compile：`caveman/operator_output.py tests/test_memory.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/operator_output.py tests/test_memory.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3320 passed, 8 skipped in 116.55s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`、无 blocking suggestions。
- Remote CI：code commit `60a42f2e92bb00989c46a6bf3b4ea658bc421fbb` GitHub Actions run `25192393917` completed success。

## Round 31 什么 work 了
- 选择“文档化安全边界 + regression”是低风险高复利：所有 CLI/dashboard 调用共享 `operator_literal`，测试明确覆盖 ANSI/C1 终端控制字节不会作为真实 terminal instruction 输出。
- TDD RED 不是伪红：同一行为的 escaping 已存在，但安全边界文档缺失；测试先失败于 docstring contract，修复后 focused/full/CI 均通过。
- 不引入自定义 ANSI stripping/parser，避免破坏既有 repr-style literal 语义。

## Round 31 什么没做/没work
- 本 handoff 更新已准备单独 docs commit；提交后需 push 并监控 CI。
- 尚未扫描所有剩余 plaintext diagnostics；Round 32 继续。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 31 已知坑
- `ruff` 不在 PATH；需用 `.venv/bin/ruff` 或 `.venv/bin/python -m ruff`。
- Bash 输出中包含 auth mode label 时可能被安全过滤显示为星号，不要误判为真实 token 泄露。
- `set -e` + CI polling 的 Python exit 2 会提前退出；轮询脚本必须关闭 `set -e` 或显式捕获 rc。

## 历史摘要
- Round 31：补 `operator_literal` ANSI/C1 escape regression 并明确 docstring 安全边界，commit `60a42f2`，CI success。
- Round 29：`operator_literal` 拒绝 non-integer `max_length`（含 bool/float/str），commit `82dccd1`，CI success。
- Round 28：`operator_literal` 拒绝 non-positive `max_length`，commit `bc8f32f`，CI success。
- Round 27：复用 `operator_literal` escape memory-quarantine list/preview 的 source/reason/content，commit `c2aa937`，CI success。
- Round 26：新增共享 `caveman.operator_output.operator_literal()` 并让 source-governance CLI 与 dashboard formatter 委托，commit `2ee79fa`，CI success。
- Round 25：dashboard source policy drift report 对 source label/candidate 做 repr-style escaped literal，commit `3d393ae`。
- Round 24：给 `_operator_literal()` 补安全目的 docstring，commit `4c8ea7c`。
- Round 23：source-governance CLI 抽 shared `_operator_literal()`，commit `2578fba`。
- Round 22：source-governance preview rows escape control chars，commit `163417f`，CI success。
- Round 21：source-governance preview checklist + escaping，commit `4cfa335`，CI success。
- Round 20：preview-drift re-run command 保留 custom `--db`/`--limit` 并 shell quote，commit `fd23409`，CI success。
- Round 19：preview-drift copy/paste workflow + safe Python literal allowlist entries，commit `7062e34`，CI success。
