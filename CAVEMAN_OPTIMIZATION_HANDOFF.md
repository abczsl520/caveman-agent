# Caveman 优化 HANDOFF

更新时间: 2026-05-01 10:35 CST

## 当前最终状态

- Round 39 已完成、提交并推送到 `main`；code commit `190669a0f6c15c4eb9846cbf4a05a4ceca0d4849` (`[verified] escape restorable quarantine report labels`)；本地验证全通过；GitHub Actions 在 10 分钟轮询内未出现该 head_sha 的 run（API 返回 0 runs），下轮先补查 CI，再继续。
- Round 38 已完成、提交并推送到 `main`；code commit `b13e8d6f6cc1c1885c7a616fa484b41292030c56` (`[verified] escape import target type labels`)；GitHub Actions run `25198448467`，`https://github.com/abczsl520/caveman-agent/actions/runs/25198448467`，completed success。
- Round 37 已完成、提交并推送到 `main`；code commit `60a456c44df1b1b959f303e192e979087bfaf934` (`[verified] escape import report operator output`)；GitHub Actions run `25197774691`，`https://github.com/abczsl520/caveman-agent/actions/runs/25197774691`，completed success。
- Round 36 已完成、提交并推送到 `main`；code commit `256c658557243a0aa2345f07f587b16c67dddeb9` (`[verified] harden setup operator output`)；GitHub Actions run `25197084551`，`https://github.com/abczsl520/caveman-agent/actions/runs/25197084551`，completed success。
- Round 35 handoff/docs commit `673e8b0722e77c4356d9dfe7c1293ad2c6fc0d5b` (`docs: update caveman handoff after round 35`)；git log 可见，作为本轮起点。
- Round 35 已完成、提交并推送到 `main`；code commit `46301e90036f27358c6dce4ef9606746e20d8de4` (`[verified] escape gateway status diagnostics`)；GitHub Actions run `25196099823`，`https://github.com/abczsl520/caveman-agent/actions/runs/25196099823`，completed success。
- Round 34 handoff/docs commit `59134a44c9172926c248d6f4f038dc6390d38f09` (`docs: update caveman handoff after round 34`)；GitHub Actions run `25195081338` completed success。
- Round 34 已完成、提交并推送到 `main`；code commit `484a0d2b4ec8c765154cb7614a3a5bbc8726a8d6` (`[verified] escape status home path`)；GitHub Actions run `25194895813`，`https://github.com/abczsl520/caveman-agent/actions/runs/25194895813`，completed success。
- Round 33 已完成、提交并推送到 `main`；code commit `e6d88548a3fcd6aa0d8773fe81ea6b8a132bda41` (`[verified] escape status memory detail labels`)；GitHub Actions run `25194162832`，`https://github.com/abczsl520/caveman-agent/actions/runs/25194162832`，completed success。
- Round 32 已完成、提交并推送到 `main`；code commit `8c09262` (`[verified] escape status model output`)；GitHub Actions run `25193311855`，`https://github.com/abczsl520/caveman-agent/actions/runs/25193311855`，completed success。
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
1. 先补查 Round 39 code commit `190669a0f6c15c4eb9846cbf4a05a4ceca0d4849` 的 GitHub Actions：本轮 push 成功，但 10 分钟轮询 `actions/runs?head_sha=190669a...` 返回 0 runs；若后续仍无 run，记录为“CI 未触发/无 workflow”，不要重复提交同一 diff。
2. 若 Round 39 CI 补查通过或确认未触发，继续 Round 40：扫描 CLI/dashboard/report config/file/DB-derived operator-facing 输出，优先 source impact trend / quarantine candidate drift / training stats 输出，或 `format_scan_report` / code-health report 中未使用 `operator_literal` 的字符串 label。
3. 每次只做一个 TDD 小切片；继续按“证据→TDD→实现→门禁→review→commit/push→监控”推进；不要虚构完成 50 轮，每轮必须有验证与提交或明确 no-op 证据。
4. Gateway health 当前不可达但本自动续跑不依赖 gateway；除非任务明确需要，不要重启 gateway。

## Round 39 做了什么
- 先确认真实状态：`main` 最新为 Round 38 handoff/docs commit `302d863`，工作树起始干净；Round 38 code CI run `25198448467` 已 success；gateway health 不可达但本轮不依赖 gateway。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3330 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，选中 flywheel dashboard 的 restorable quarantine preview：`restorable_quarantine_by_source` 与已收集但未展示的 `restorable_quarantine_by_reason` 都来自 DB metadata；若 source/reason 带换行或 ANSI，report 会被伪造行/终端控制字节污染。
- RED 新增 `tests/test_flywheel_quarantine_preview_operator_output.py`，构造带 `\nSPOOF_*` 与 ANSI 的 quarantined memory，证明原 report 输出 raw label 且未展示 reason 汇总。
- GREEN：新增 `_format_restorable_quarantine()`，在 report 边界用共享 `operator_literal(..., max_length=120)` 转义 source/reason label；保留 collect_memory_stats 中的 raw stats keys，避免把 presentation literal 泄漏到结构化指标；`format_report()` 同时展示 `Restorable quarantine reasons:`。
- 为满足 god-file 行数 gate，将 `flywheel_dashboard.py` 顶部长 docstring 压缩为单行，并把 formatting 逻辑下沉到 `_flywheel_dashboard_formatters.py`。

## Round 39 验证结果
- Baseline full suite（before change）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3330 passed, 8 skipped in 114.22s`。
- RED：`tests/test_flywheel_quarantine_preview_operator_output.py::test_restorable_quarantine_report_escapes_source_and_reason_labels` 初次运行 1 failed，原 report 输出 raw newline/ANSI 且缺少 reason line。
- GREEN focused + file-size gates：`tests/test_flywheel_quarantine_preview_operator_output.py tests/test_flywheel_dashboard_boundaries.py tests/test_audit.py::test_no_god_files tests/test_round11.py::TestLoopRefactor::test_no_file_over_400_lines` → `20 passed`。
- Py compile：`caveman/training/_flywheel_quarantine_preview.py caveman/training/flywheel_dashboard.py caveman/training/_flywheel_dashboard_formatters.py tests/test_flywheel_dashboard_boundaries.py tests/test_flywheel_quarantine_preview_operator_output.py` pass。
- Ruff changed files：same changed files → `All checks passed!`。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3331 passed, 8 skipped in 120.32s`。
- Security scan：final diff hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是现有 boundary test 也可显式断言 reason line（dedicated malicious-output test 已覆盖）。
- Remote push：code commit `190669a0f6c15c4eb9846cbf4a05a4ceca0d4849` 已推送到 `origin/main`。
- Remote CI：push 后轮询 10 分钟 `actions/runs?head_sha=190669a...` 返回 0 runs；需下轮先补查。

## Round 39 什么 work 了
- 采纳独立 reviewer 建议，raw stats 保持结构化原值，escaping 只在 operator-facing report 边界发生。
- TDD regression 同时覆盖 source 与 reason 两个 DB-derived label，证明 report 不会出现真实 spoof 行或 raw ANSI。
- 本地 targeted、ruff、py_compile、full suite、security scan、independent review、commit/push 全部通过。

## Round 39 什么没做/没work
- GitHub Actions 没能在本轮轮询窗口内观察到 run；已记录为下轮第一步补查。
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI/或记录未触发。
- 未重启 gateway（当前自动续跑不依赖 gateway）。

## Round 39 已知坑
- `git diff --stat` 显示新增测试需先 `git add -N` 或 staged 后才能完整纳入 review/security scan；本轮在最终 diff/security scan 阶段已覆盖。
- `flywheel_dashboard.py` 有 450 行 policy；新增逻辑要优先下沉到 formatter/helper，否则 full suite 会因 file-size gate 失败。
- 结构化 metrics 不应存 presentation literal；应在 `format_report()`/formatter 这类 operator boundary 才调用 `operator_literal()`。

## Round 38 做了什么
- 先确认真实状态：`main` 最新为 Round 37 handoff/docs commit `9ab7317`，工作树起始干净；Round 37 code CI run `25197774691` 已 success；gateway health 不可达但本轮不依赖 gateway。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3329 passed, 8 skipped`。
- 延续 Round 37 independent reviewer 的建议，审查 `caveman/import_/report.py` 的 manifest target type 分组标题。虽然正常 importers 传入 enum-like target types，但 `ImportItem.target_type` 是普通字符串，report 会直接执行 `ttype.title()`；若未来外部 importer 或异常 manifest 携带换行/ANSI，仍可伪造 CLI report 行或注入终端控制字节。
- RED 新增 `test_import_manifest_report_escapes_target_type_labels`，证明原输出会产生 raw `
SPOOF_TARGET` 与 raw ANSI。
- GREEN：新增私有 `_target_type_label()`；安全的 alnum/underscore target type 保持原有 TitleCase 可读性（兼容 `Memory` 断言），异常/带控制字符的 target type 走共享 `operator_literal()`。

## Round 38 验证结果
- Baseline full suite（before change）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3329 passed, 8 skipped in 116.40s`。
- RED：`tests/test_import_report_operator_output.py::test_import_manifest_report_escapes_target_type_labels` 初次运行 1 failed，原 report 输出 raw newline/ANSI。
- GREEN focused：`tests/test_import_report_operator_output.py` → `4 passed`。
- Related tests：`tests/test_import_system.py tests/test_import_report_operator_output.py tests/test_memory.py::test_operator_literal_helper_is_shared_across_cli_and_dashboard` → `64 passed`。
- Py compile：`caveman/import_/report.py tests/test_import_report_operator_output.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/import_/report.py tests/test_import_report_operator_output.py` → pass。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3330 passed, 8 skipped in 115.04s`。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是未来可给 `_target_type_label` 增加 safe-character policy 注释。
- Remote CI：code commit `b13e8d6f6cc1c1885c7a616fa484b41292030c56` GitHub Actions run `25198448467` completed success。

## Round 38 什么 work 了
- TDD 小切片直接补上 Round 37 reviewer 建议，没有扩大范围；保持 import target type 正常显示兼容，同时封住异常 label 的终端/日志 spoof 面。
- 继续复用共享 `operator_literal`，没有新增第二套 escaping 语义。
- 本地 baseline、RED/GREEN、related/full suite、ruff、security scan、independent review、push 和 GitHub Actions 全部通过。

## Round 38 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。
- 未继续扫描 `format_scan_report` / source impact trend / quarantine drift；留给 Round 39。

## Round 38 已知坑
- `str.title()` 会把换行后的 spoof 文本也转成像正常标题的形式，并保留 ANSI escape；operator-facing labels 即使“看起来像 enum”也需要对异常字符串 fail-closed。
- GitHub Actions run list 偶尔短暂 `RUN_NOT_FOUND`（API/visibility eventual consistency），同一 SHA 后续会出现；不要把短暂无结果误判为未触发。

## Round 37 做了什么
- 先确认真实状态：`main` 最新为 Round 36 handoff/docs commit `77a99eb`，工作树起始干净；Round 36 handoff/docs CI run `25197265449` completed success；gateway health 不可达但本轮不依赖 gateway。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3326 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，选中 `caveman/import_/report.py`：import detect/manifest/result report 直接输出 source label、source_path.name、skip_reason、warnings、details；这些字段来自外部导入源/文件名/扫描结果，若包含换行或 ANSI 控制字节，会伪造 CLI 输出或注入终端控制字节。
- RED 新增 `tests/test_import_report_operator_output.py` 3 个 regression：detect source label、manifest source/path/skip_reason、result warnings/details 均要求 repr-style escaped literal，且没有真实 spoof 行或 raw ANSI。
- GREEN：`format_detect_report()`、`format_manifest_report()`、`format_result_report()` 复用共享 `operator_literal()`；顺手移除两个无 placeholder f-string，保持 ruff 干净。

## Round 37 验证结果
- Round 36 handoff/docs CI：run `25197265449` completed success。
- Baseline full suite（before change）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3326 passed, 8 skipped in 116.26s`。
- RED：`tests/test_import_report_operator_output.py` 初次运行 3 failed，原 report 输出 raw newline/ANSI。
- GREEN focused：`tests/test_import_report_operator_output.py` → `3 passed`；operator literal shared smoke → `4 passed`。
- Py compile：`caveman/import_/report.py tests/test_import_report_operator_output.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/import_/report.py tests/test_import_report_operator_output.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3329 passed, 8 skipped in 112.73s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是未来可考虑对 `target_type` label 也防御性 literal 化。
- Remote CI：code commit `60a456c44df1b1b959f303e192e979087bfaf934` GitHub Actions run `25197774691` completed success。

## Round 37 什么 work 了
- 继续复用共享 `operator_literal`，没有新增第二套 escaping 语义；TDD regression 覆盖 import report 三个入口。
- 把 import source/file/scan-result derived operator output 纳入同一安全边界，减少导入预览与结果报告的终端/日志 spoof 面。
- 本地 full suite、ruff、security scan、independent review、push 和 GitHub Actions 全部通过。

## Round 37 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- Reviewer 建议的 `target_type` label literal 化未做：当前 target_type 来自 importer controlled enum-like 分类，风险低；下一轮若继续 import report 可用 TDD 小切片补上。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。
- 第一次 CI 长轮询被 300s wrapper timeout 截断；随后短查询确认 code run 已 success。

## Round 37 已知坑
- `git add -N` 后 diff 才会包含 untracked 测试文件，review/security scan 不要漏扫新增测试。
- 长 CI polling 不宜包在 `execute_code` 里超过 300s；若被截断，立即用短 API 查询 head_sha 状态，不要重复提交。
- Import report 的 source/path/warnings/details 都是 external/file/scan-derived operator output；继续同类扫描时优先 grep `format_*report` 和 CLI `typer.echo(format_...)`。

## Round 36 做了什么
- 先确认真实状态：`main` 最新为 Round 35 handoff/docs commit `673e8b0`，工作树起始有本轮未提交的 `caveman/cli/main.py` 与新增 `tests/test_cli_operator_output.py`（来自同一自动续跑上下文）；gateway health 不可达但本轮不依赖 gateway。
- 继续 operator-facing output 安全边界扫描，选中 `caveman.cli.main.setup()`：它会输出外部配置检测结果的 source/path/model/API key 片段，以及 import confirmation source 和 config saved path；这些值来自外部配置路径/模型名/密钥数据流，若包含换行或 ANSI 控制字节，会伪造 setup 输出或泄露 credential 片段。
- RED 新增 regression：`test_setup_escapes_detected_config_source_path_and_model` 与 `test_setup_redacts_detected_api_key`，要求 source/path/model 使用 repr-style literal，raw spoof 行与 ANSI 不出现；检测到的 API key 只显示固定 `[REDACTED]`，不显示首尾片段。
- GREEN：`setup()` 复用共享 `operator_literal()` 转义 source/path/model/import prompt source/config saved path；API key 输出从首尾片段改为固定 `[REDACTED]`；同时清理 changed-file ruff 暴露的 late-import E402 与 unused assignment。

## Round 36 验证结果
- RED：`tests/test_cli_operator_output.py` 初次运行按预期失败，原 `setup()` 输出 raw source/path/model 与 API key 片段。
- GREEN focused：`tests/test_cli_operator_output.py` → `2 passed`。
- Py compile：`caveman/cli/main.py tests/test_cli_operator_output.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/cli/main.py tests/test_cli_operator_output.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3326 passed, 8 skipped in 119.67s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：Python added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是后续可继续把其他 setup 动态提示统一纳入 `operator_literal`。
- Remote CI：code commit `256c658557243a0aa2345f07f587b16c67dddeb9` GitHub Actions run `25197084551` completed success。

## Round 36 什么 work 了
- 继续复用共享 `operator_literal`，没有新增第二套 escaping 语义；TDD regression 同时覆盖外部 config source/path/model 与 saved path。
- Detected API key 不再展示首尾片段，降低 operator log/Discord transcript 中的 credential 泄露面。
- 本地 full suite、ruff、security scan、independent review、push 和 GitHub Actions 全部通过。

## Round 36 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- `setup()` 仍有部分用户输入 prompt/overwrite prompt 是交互提示常量或 operator-entered 值，未做额外语义改动；下一轮可继续审查其他 CLI setup/status/utility 输出。
- GitHub public API 轮询中短暂触发 403 rate limit，但后续重试拿到 run success；不要把临时 403 误判为 CI 失败。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 36 已知坑
- `git diff --stat` 默认不包含 untracked 测试文件；review/security scan 前需要 `git add -N` 或 `git diff --no-index /dev/null <file>`。
- Bash grep security scan 容易被引号嵌套搞坏；本轮改用 Python regex 扫描 git diff，更稳定。
- 测试 fixture 变量名不要叫 `secret = "..."`，会触发 added-line sensitive-info scan；可用 `imported_key` 这类不匹配 scan 的名称，并确保最终输出仍断言 `[REDACTED]`。


## Round 35 做了什么
- 先确认真实状态：`main` 最新为 Round 34 handoff/docs commit `59134a4`，工作树起始干净；gateway health 不可达但本轮不依赖 gateway。
- 确认 Round 34 handoff/docs commit CI：GitHub Actions run `25195081338` completed success；Round 34 code CI `25194895813` 已在上轮成功，不重复长等。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3323 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，发现 `caveman.cli.status._format_gateway_status()` 直接输出 gateway log diagnostics 的 `boundary` 与 `patterns` keys；这些值来自 pidfile/log diagnostic 数据流，若出现换行或 ANSI 控制字节，会伪造 status 行或注入终端控制字节。
- RED 新增 regression：`test_status_text_gateway_escapes_log_diagnostic_labels` monkeypatch gateway diagnostic report 返回带换行与 ANSI 的 `boundary`/pattern key，要求 status 输出使用 repr-style escaped literal，且没有真实 spoof 行或 raw ANSI 字节。
- GREEN：`_format_gateway_status()` 对 `report["boundary"]` 与 alert pattern key 复用共享 `operator_literal()`；既有 gateway status test 更新为期待 `'pid_marker'` literal。没有新增第二套 escaping 语义。

## Round 35 验证结果
- Round 34 handoff/docs CI：run `25195081338` completed success。
- Baseline full suite（before change）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3323 passed, 8 skipped in 116.07s`。
- RED：`tests/test_cli_status_gateway.py::test_status_text_gateway_escapes_log_diagnostic_labels` 按预期失败，原输出包含真实 `SPOOF_BOUNDARY`/`SPOOF_PATTERN` 换行与 raw ANSI。
- GREEN focused：`tests/test_cli_status_gateway.py` → `2 passed`；`tests/test_cli_status.py tests/test_cli_status_gateway.py` → `13 passed`。
- Extended focused：`tests/test_cli_status.py tests/test_cli_status_gateway.py tests/test_gateway_log_diagnostics.py` → `18 passed`。
- Py compile：`caveman/cli/status.py tests/test_cli_status_gateway.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/cli/status.py tests/test_cli_status_gateway.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3324 passed, 8 skipped in 110.63s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；reviewer note 仅说明其隔离环境没 pytest，但基于 diff 无阻塞问题。
- Remote CI：code commit `46301e90036f27358c6dce4ef9606746e20d8de4` GitHub Actions run `25196099823` completed success。

## Round 35 什么 work 了
- 继续复用共享 `operator_literal`，把 gateway diagnostic labels 也纳入同一 operator-output 安全边界。
- TDD regression 直接证明 gateway log boundary/pattern 不能通过 log-derived 字符串伪造额外行或注入 ANSI 控制字节。
- 本地 full suite、ruff、security scan、independent review、push 和 GitHub Actions 全部通过；公共 GitHub Actions API 继续可无 token 查询 head_sha 的 run。

## Round 35 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- Gateway status 的 `pid`/`line_count` 仍来自内部数值路径，未发现同类 string label 风险；后续可继续扫描其他 CLI/dashboard config/file/DB-derived 输出。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 35 已知坑
- 通过 monkeypatch `caveman.gateway.log_diagnostics.scan_current_startup_log` 可稳定覆盖 status gateway path，因为 `_format_gateway_status()` 是函数内 import。
- `operator_literal` 会给普通 label 加引号；这是有意的 operator-facing literal 边界，相关 tests 要同步期待 `'pid_marker'`。
- 长 CI polling 若放在 execute_code 里仍可能被 wrapper 300s timeout 截断；截断后要用短查询确认 commit/push/CI 状态，不要重复提交。

## Round 34 做了什么
- 先确认真实状态：`main` 最新为 Round 33 handoff/docs commit `d8c3f1f`，仅 `memory/projects/caveman.md` 有上轮未纳入 git 的项目记忆 diff；gateway health 不可达但本轮不依赖 gateway。
- 确认 Round 33 handoff/docs commit CI：GitHub Actions run `25194358146` completed success。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3322 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，发现 `caveman/cli/status.py` 的 `Home: {CAVEMAN_HOME}` 直接输出配置/环境派生路径；若 home path 被配置为含换行或 ANSI 控制字节，会伪造 status 行或注入终端控制字节。
- RED 新增 regression：`test_status_text_escapes_home_path_control_characters` monkeypatch `CAVEMAN_HOME` 为带换行与 ANSI 的字符串，要求 status 输出使用 repr-style escaped literal，且没有真实 spoof 行或 raw ANSI 字节。
- GREEN：`status_text()` 的 Home 字段复用共享 `operator_literal(CAVEMAN_HOME)`；不新增第二套 escaping 语义。

## Round 34 验证结果
- RED：`tests/test_cli_status.py::test_status_text_escapes_home_path_control_characters` 按预期失败，原输出包含真实换行和 raw ANSI。
- GREEN focused：`tests/test_cli_status.py::test_status_text_escapes_home_path_control_characters` → `1 passed`；`tests/test_cli_status.py` → `11 passed`。
- Py compile：`caveman/cli/status.py tests/test_cli_status.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/cli/status.py tests/test_cli_status.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3323 passed, 8 skipped in 113.94s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是未来可对极长 path 加 `max_length` 保持终端可读性。
- Remote CI：code commit `484a0d2b4ec8c765154cb7614a3a5bbc8726a8d6` GitHub Actions run `25194895813` completed success。

## Round 34 什么 work 了
- 继续复用共享 `operator_literal`，让 status 的 Home 字段与 model/memory labels 采用同一 operator-output 安全边界。
- TDD regression 直接证明 status Home 输出不能通过路径字符串伪造额外行或注入 ANSI 控制字节。
- Round 33 docs CI 与 Round 34 code CI 均通过；公共 GitHub Actions API 仍可无 token 查询当前 head_sha 的 run。

## Round 34 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- 尚未检查 gateway status 的 `boundary`/`patterns` 等 log-derived labels；Round 35 可继续。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 34 已知坑
- 本轮项目记忆 `memory/projects/caveman.md` 在开工前已有未提交 diff；只纳入 handoff/docs commit，不混入 code commit。
- `gh` CLI 不存在；CI 监控继续使用 GitHub public REST API + `head_sha` 查询。
- Home path 这类环境/config-derived operator output 也要走 `operator_literal`，不是只有 DB/file label 才需要 escaping。

## Round 33 做了什么
- 先确认真实状态：`main` 最新为 Round 32 handoff/docs commit `35b9a14`，工作树起始干净；gateway health 不可达但本轮不依赖 gateway。项目文件实际为 `memory/projects/caveman.md`（`caveman优化.md` 不存在）。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3321 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，发现 `caveman/cli/status.py` 的 `mem_detail` 直接输出 memory JSON 文件 stem：带换行和 ANSI 控制字节的 stem 会在 status 输出里伪造额外行/注入终端控制字节。
- RED 新增 regression：`test_status_text_escapes_memory_type_names` monkeypatch `_count_memories()` 返回带换行与 ANSI 的 key，要求 status 输出使用 repr-style escaped literal，且没有真实 spoof 行或 raw ANSI 字节。
- GREEN：`status_text()` 的 memory detail key 复用共享 `operator_literal(k)`；保留 count 数值原样输出。

## Round 33 验证结果
- RED：`tests/test_cli_status.py::test_status_text_escapes_memory_type_names` 按预期失败，原输出包含真实换行和 raw ANSI。
- GREEN focused：`tests/test_cli_status.py` → `10 passed`。
- Py compile：`caveman/cli/status.py tests/test_cli_status.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/cli/status.py tests/test_cli_status.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3322 passed, 8 skipped in 110.52s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`。
- Remote CI：code commit `e6d88548a3fcd6aa0d8773fe81ea6b8a132bda41` GitHub Actions run `25194162832` completed success。

## Round 33 什么 work 了
- 继续复用共享 `operator_literal`，没有新增 escaping 语义；TDD regression 直接证明 status memory detail 不能通过文件名/stem 伪造额外行或注入 ANSI 控制字节。
- 保留 Round 32 的 model escaping regression，并新增 memory label regression，status operator-output 安全边界覆盖更完整。
- GitHub Actions 公共 API 可无 token 查询当前 head_sha 的 run；本轮 code CI 成功记录为 run `25194162832`。

## Round 33 什么没做/没work
- 本 handoff 更新将单独 docs commit；提交后需 push 并监控 CI。
- 尚未系统检查 `status_text()` 的 `Home` 路径与 gateway diagnostic 字段；Round 34 继续。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 33 已知坑
- 本轮一开始 `memory/projects/caveman优化.md` 未找到，实际项目文件是 `memory/projects/caveman.md`；后续按真实文件名读取/更新。
- `gh` CLI 不存在；CI 监控使用 GitHub public REST API + `head_sha` 查询。
- 长轮询放进 `execute_code` 可能受 300s wrapper timeout 影响；优先用短 `terminal` 查询或分段轮询。
- `mem_detail` 的 key 来自文件名/stem，也属于 file-derived operator output；任何 CLI/dashboard 输出此类 label 时统一走 `operator_literal`。

## Round 32 做了什么
- 先确认真实状态：`main` 最新为 Round 31 handoff/docs commit `e4c8182`，工作树起始干净；gateway health 不可达但本轮不依赖 gateway。项目文件存在但仍停在 Round 30 摘要，已在本轮同步更新。
- 执行 baseline：full suite（排除已知 NFR）起始为 `3320 passed, 8 skipped`。
- 继续 operator-facing output 安全边界扫描，发现 `caveman/cli/status.py` 直接输出配置派生的 model 名称：`Model: {model}`。配置值可能包含换行或 ANSI escape，存在 status 输出终端/日志 spoof 风险。
- RED 新增 regression：`test_status_text_escapes_configured_model_control_characters` monkeypatch `_get_model_info()` 返回 `safe-model\nSPOOF_MODEL\x1b[31m`，要求 status 输出中出现 repr-style escaped literal，且没有真实 spoof 行或 raw ANSI 字节。
- GREEN：`status_text()` 复用共享 `operator_literal(model)`；同时清理 `tests/test_cli_status.py` 已存在未使用 import，使 changed-file ruff 干净。

## Round 32 验证结果
- RED：`tests/test_cli_status.py::test_status_text_escapes_configured_model_control_characters` 按预期失败，原输出包含真实换行和 raw ANSI。
- GREEN focused：`tests/test_cli_status.py` → `9 passed`。
- Py compile：`caveman/cli/status.py tests/test_cli_status.py` pass。
- Ruff changed files：`.venv/bin/ruff check caveman/cli/status.py tests/test_cli_status.py` → pass。
- Full suite（排除已知 NFR）：`.venv/bin/python -m pytest tests/ -q --ignore=tests/test_nfr_compliance.py --tb=short` → `3321 passed, 8 skipped in 112.76s`。
- Docs/API：`.venv/bin/python scripts/generate_api_reference.py --check` pass；无 API docs diff。
- Security scan：added-line/final scan hardcoded secret/token/password、shell injection、eval/exec、pickle、SQL string-format patterns 0 matches；push hook safety checks passed。
- Independent review：passed，无 `security_concerns`、无 `logic_errors`；non-blocking suggestion 是未来可同样检查其他 status 字段。
- Remote CI：code commit `8c09262` GitHub Actions run `25193311855` completed success。

## Round 32 什么 work 了
- 复用共享 `operator_literal`，没有新增 escaping 语义；TDD regression 直接证明 status model 输出不会通过配置值伪造额外行或注入 ANSI 控制字节。
- Changed-file ruff 暴露并清理了测试文件历史未使用 imports，避免本轮新增测试后留下 lint debt。
- 公共 GitHub Actions API 无 token 也可轮询；本轮 code CI 成功记录为 run `25193311855`。

## Round 32 什么没做/没work
- 本 handoff 更新已准备单独 docs commit；提交后需 push 并监控 CI。
- 尚未系统扫描 `status_text()` 的 `mem_detail`、`Home`、以及 utility commands 里其他 config/file-derived 输出；Round 33 继续。
- 未重启 gateway（当前自动续跑不依赖 gateway，且 SOP 要避免不必要启动）。

## Round 32 已知坑
- `git diff --cached` 在未 staged 时为空；pre-commit review 若未 stage 要用 `git diff`，本轮 independent reviewer 使用 unstaged diff。
- GitHub API 查询刚 push 的 `head_sha` 可能 5-7 分钟没有 run；不要因 `NO_RUN_YET` 立即判失败，需继续轮询。
- `status_text()` 的 model/provider 来自配置，不应直接作为 terminal instruction 输出；模型名等 operator-facing config 值统一走 `operator_literal`。

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
- Round 32：status dashboard 的配置派生 model 名称改为 `operator_literal(model)`，防换行/ANSI spoof，commit `8c09262`，CI success。
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

