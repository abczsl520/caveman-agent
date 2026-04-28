# Caveman 优化 HANDOFF

更新时间: 2026-04-28 22:30 CST

## 下次启动时做
1. 先确认 `origin/main` CI 是否已经 green：优先查最新 commit 的 GitHub Actions/check-runs；如果 API/gh 不可用，再用 `git status --short --branch` + `git log -1 --oneline` 恢复本地状态。
2. 如果本轮提交尚未完成/CI 未过，继续从 retrieval telemetry flywheel 变更收尾：security scan → diff review → commit/push → monitor CI。
3. 后续高复利方向：继续审计 memory/retrieval/training 数据流，重点看 adoption/positive-negative pair 生成是否能从 retrieval log 闭环到训练样本，而不是做 cosmetic cleanup。

## 本轮做了什么
- 针对 SQLite retrieval log / memory flywheel 做数据流审计，发现 `MemoryManager.recall_scored()` 和 MCP `memory_search()` 走 scored retrieval 路径时没有写 retrieval log，外部 agent 检索会绕过 training flywheel。
- 修复 `MemoryManager.with_sqlite()`：默认挂载 `RetrievalLog()`，让生产 SQLite memory manager 自动进入 retrieval telemetry；同时对显式 `base_dir`/`db_path` 的隔离 store，把默认 retrieval log 放到对应 store 旁边的 `training/retrieval_log.sqlite`，避免测试/doctor/import/sandbox 污染全局 `~/.caveman/training/retrieval_log.sqlite`。
- 修复 `MemoryManager.recall_scored()`：backend/SQLite 路径现在会记录非空 scored recall，source=`memory_search_scored`，并捕获 logging 异常避免影响检索主流程。
- MCP 侧保持走 `MemoryManager.with_sqlite()`，新增测试确认 MCP memory tools 实际调用该入口；不再在 MCP 层重复传 retrieval log，减少 wiring 分叉。
- 顺手修复 `caveman/mcp/server.py` 的 ruff E741（`l` → `lesson`），并给 FastMCP runtime 支持但 typing 不完整的 `mcp.run(... host=..., port=...)` 加 `# type: ignore[call-arg]`。

## 验证结果
- 聚焦测试：`4 passed in 0.31s`（default retrieval log wiring、custom db isolation、scored recall logging、MCP memory_search）。
- 全量测试：`3209 passed, 8 skipped, 7 xfailed in 103.23s`。
- Coverage gate：observed `68.64%` >= baseline `68.25%`；长期 target `80%` 仍保持可见债务。
- Ruff：changed files pass；CI subset `E9,F63,F7,F82` pass；API reference regenerated and unchanged。
- Mypy gate：full-project 仍有历史 baseline `416 errors in 154 files`，baseline-aware gate exit 0，changed Python files `caveman/mcp/server.py` / `caveman/memory/manager.py` 无 mypy errors。
- CLI sanity：`.venv/bin/python -m caveman version` → `Caveman v0.3.0`，`--help` 可运行。
- Security scan：changed files non-entropy secret findings `0`；diff secret scan findings `0`；dangerous pattern grep 只有 `tokenize` 的 false positive。

## 独立 review 结论
- 初始 review 发现 medium 风险：`with_sqlite(db_path=...)` 默认 `RetrievalLog()` 会把 sandbox/custom DB 检索写入全局训练库。
- 已修复为：无显式路径的生产 manager 使用全局默认；显式 `base_dir`/`db_path` 的 manager 使用相邻隔离 retrieval log，并新增测试锁住。
- 当前可 ship；剩余低风险说明：`recall_scored()` 延续既有行为，不走 recall cache，重复搜索会重复记录 telemetry；这是 MCP 可观测性/训练数据完整性的可接受取舍。

## 已知坑
- 不要用裸 `python`，它可能指向 Hermes venv；Caveman 验证一律用项目 `.venv/bin/python`。
- 不要用 `nohup caveman serve &` 从 Hermes terminal 启动 gateway；历史上会触发 exit-130 loop。需要启动 gateway 时用 `subprocess.Popen(..., start_new_session=True)` 或现有 gateway SOP。
- `scripts/ci_mypy_gate.py | tail` 普通管道会隐藏前段 exit status；需要用脚本自身 exit code 或 `set -o pipefail`。
- coverage gate 会生成 `coverage.json`，提交前删除。
