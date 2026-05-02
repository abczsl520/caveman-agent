# caveman优化2 项目记忆

更新时间: 2026-05-02 17:56 CST

## 当前状态
- Round 54 code 已完成并推送到 `main`。
- Latest HEAD/origin-main: `2dfb93d` `[verified] escape training stats warning output`。
- CI: GitHub Actions run `25249169155` success。
- Repo: `/Users/yeren64g/projects/caveman` clean on main after code commit before docs update。
- 自动续跑 cron job `36500447cc33` 仍 paused/disabled，除非元宝明确要求不要恢复。

## 最近完成
- Round 53: `caveman train` entrypoint Target banner 使用 `operator_literal(effective_target)`，防止 `--target/--method` newline/ANSI 伪造 CLI 输出。
- Round 54: `caveman train --stats` unreadable trajectory warning 对 path 与 exception 使用 `operator_literal()`，防止恶意文件名 newline/ANSI 伪造 operator log。

## 续跑指令
继续 Caveman optimization：每轮找一个真实 operator-facing 输出边界，TDD 小切片修复，跑 focused/full tests、ruff、mypy touched paths、security scan、独立 review、commit/push、监控 CI、更新 handoff。

## 下一轮建议
继续 training 子系统 remaining outputs：
- `embedding.py` logger warning/info：unreadable/malformed files、dataset_path、output_path、stats。
- `sft.py` / `rl.py` logger warning/info：trajectory file path、dataset output path、model/method metadata。
- `eval_embedding.py` report/selection metadata：model_path/reason/report 是否可能 raw 输出到 operator/docs。
- `flywheel_dashboard.py` report lines中 task/source/tier/name/path 类字段是否统一 literal 化。

## 固定注意事项
- 不要把 API key/token/password/secret/credential/connection string 写入总结或提交。
- push 前至少扫描 changed added lines 的 secret/shell/eval/pickle/SQL 注入模式；push hook 是最后防线。
- 使用 `.venv/bin/python` 执行 pytest/ruff/mypy。
- Gateway health 不是本轮依赖；不要无故重启 gateway。
