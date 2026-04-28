# Caveman 优化 HANDOFF

更新时间: 2026-04-28 23:18 CST

## 下次启动时做
1. 先确认 `origin/main` 最新 commit 的 GitHub Actions/check-runs 是否 green；如果 API/gh 不可用，再用 `git status --short --branch` + `git log -1 --oneline` 恢复本地状态。
2. 如果本轮 hard-negative 提交尚未完成/CI 未过，继续从 security scan → diff review → commit/push → monitor CI 收尾。
3. 后续高复利方向：继续审计 memory/retrieval/training 数据流，优先看 feedback 事件是否能稳定、可解释地进入训练样本与评估，而不是做 cosmetic cleanup。

## 本轮做了什么
- 延续 retrieval telemetry flywheel，审计 `retrieval_log -> embedding pair export -> embedding trainer` 数据流，发现 adoption 反馈只把 adopted memory 当 positive，检索返回的未采用相邻结果没有作为 hard negative 进入训练，导致元飞轮缺少“选它而不是它”的对比信号。
- 修复 `RetrievalLog.generate_training_pairs()`：有 adoption 事件时，为每个 adopted positive 配对每个有效的非 adopted retrieval result 作为显式 `negative`；多 adoption 时排除所有 adopted ids，避免一个正确 adopted answer 被训练成另一个正确 answer 的负样本。
- 修复 `PairExtractor.extract_from_retrieval_log()` / dataset 导出链路：保留 retrieval log 生成的 optional `negative` 字段到 `QueryMemoryPair`，`build_dataset()` 输出 JSONL 时写入 `negative`，保持旧 `query`/`positive` schema 向后兼容。
- 修复 `EmbeddingTrainer.train()`：读取 JSONL 时拆出 pair examples 与 triplet examples；若存在显式 hard negatives，则只把 3-text triplets 喂给 `TripletLoss`，避免 sentence-transformers 在混合 2/3 text `InputExample` dataloader 中丢列或运行时报错；无 triplet 时保持原 `MultipleNegativesRankingLoss` 行为。
- `embedding-train` 与 `all` optional extras 增加 `datasets>=2.19`，覆盖 sentence-transformers 3+/5.x 实际训练路径依赖。
- 为 base model 加载失败返回 `skip`，避免 optional embedding training 在离线/模型不可达环境下把验证变成网络脆弱失败；CLI 仍按 non-success 处理。

## 验证结果
- 聚焦测试：`47 passed, 1 xfailed`（training pivot + training pivot fixes；覆盖 hard-negative 生成、多 adoption 排除、PairExtractor negative 保留、TripletLoss 选择、混合 pair/triplet 过滤）。
- 全量测试：`3214 passed, 8 skipped, 7 xfailed, 2 warnings in 233.30s`。
- Coverage gate：observed `68.68%` >= baseline `68.25%`；长期 target `80%` 仍保持可见债务。
- Ruff：changed files pass。
- Mypy gate：full-project 仍有历史 baseline `415 errors in 153 files`，baseline-aware gate exit 0，changed Python files `caveman/training/embedding.py` / `caveman/training/retrieval_log.py` 无 mypy errors。
- Security scan：changed files 仅 entropy false positives；diff grep common secret patterns `0`。

## 独立 review 结论
- 第一轮独立 review 判定 no-ship：仅在 JSONL/schema 层保留 negative 还不够，trainer 如果把混合 2-text/3-text examples 一起交给 `TripletLoss`，真实 sentence-transformers 训练可能丢 negative 列或报错；还指出 multi-adoption 会互相作为 negative、`datasets` packaging 缺口。
- 已修复上述 blockers：TripletLoss 只训练 triplet examples；multi-adoption 排除所有 adopted ids；`embedding-train` 与 `all` 均加入 `datasets>=2.19`；清理 unreachable adoption branch。
- 最终独立 review：逻辑可 ship；仅剩低风险说明：base model load 的 broad exception 会把本地模型配置问题归类为 `skip`，这是 optional/离线训练场景的可接受取舍，后续可改成更细的异常与日志。

## 已知坑
- 不要用裸 `python`，它可能指向 Hermes venv；Caveman 验证一律用项目 `.venv/bin/python`。
- 不要用 `nohup caveman serve &` 从 Hermes terminal 启动 gateway；历史上会触发 exit-130 loop。需要启动 gateway 时用 `subprocess.Popen(..., start_new_session=True)` 或现有 gateway SOP。
- `scripts/ci_mypy_gate.py | tail` 普通管道会隐藏前段 exit status；需要用脚本自身 exit code 或 `set -o pipefail`。
- coverage gate 会生成 `coverage.json`，提交前删除。
