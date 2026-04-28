# Caveman 优化 HANDOFF

更新时间: 2026-04-28 10:03 CST

## 下次启动时做
1. 不要再从历史全日志判断 gateway；先跑 `caveman status --gateway` 看 current-startup bounded window。
2. 当前 gateway 已恢复在线，优先继续代码收敛：检查 `tests/conftest.py` 删除 xfail 的变更是否要提交；若要 push，先跑 security scan。
3. 若继续优化，下一步建议是做 full/near-full pytest 分层验证，而不是再局部修 gateway 热重载（当前已是 full restart）。

## 上次做了什么
- 读取历史会话，确认上次未完成项是 Caveman/wildman gateway runtime verification + handoff。
- 运行 current runtime 诊断：当时无 `~/.caveman/gateway.pid`，health `localhost:4201` 连接失败，`caveman status --gateway` 显示 Gateway stopped。
- 确认当前工作区只有 1 个变更：`tests/conftest.py` 删除 `test_core_files_under_400_lines` 的 known-failure/xfail 条目。
- 验证核心回归：
  - `tests/test_logging_config_idempotent.py`
  - `tests/test_permission_hot_reload.py`
  - `tests/test_hot_reload_safety.py`
  - `tests/test_gateway_lifecycle.py`
  - 结果：66 passed, 1 skipped
- 验证 file-limit 相关测试：4 passed。
- 扩大验证：`tests/test_prd_audit.py tests/test_round108_dedup.py tests/test_round11.py tests/test_logging_config_idempotent.py tests/test_permission_hot_reload.py tests/test_hot_reload_safety.py tests/test_gateway_lifecycle.py tests/test_gateway_false_done.py tests/test_gateway_auto_continue_final_suppression.py`
  - 结果：134 passed, 2 skipped
- 使用 `subprocess.Popen` 安全启动 gateway（不是 nohup）：PID 95818。
- 验证 health：`{"status":"ok","uptime_s":9}`。
- 验证日志：Discord connected `wildman#1416`，synced 59 slash commands。
- 验证 status：Gateway running PID 95818；bounded log window 17 lines；Gateway log alerts none；Discord connected ✅；Slash commands synced ✅。

## 什么 work 了
- `caveman status --gateway` 的 bounded pid_marker 窗口能避免被历史 ERROR/Traceback/Permission/no such column 噪音误导。
- SIGUSR2 当前启动日志显示 `SIGUSR2 handler installed for full restart`，不是旧的 broad hot-reload。
- gateway 启动方式：`.venv/bin/python -c subprocess.Popen([... '-m', 'caveman', 'serve'], start_new_session=True)` 正常。
- 日志没有当前启动期重复 handler；当前启动期 alerts none。

## 什么没做/没 work
- 未提交 `tests/conftest.py` 的变更；当前仍是 dirty working tree。
- 已跑完整 pytest：3199 passed, 8 skipped, 7 xfailed in 111.22s。
- 注意：完整 pytest 期间 gateway 在 2026-04-28 11:15:49 收到 shutdown 并停止；已用 subprocess.Popen 重启，当前 PID 5119，health ok，status alerts none。
- 未 push；如需 push，按用户偏好必须先做敏感信息/PII/security scan。

## 已知坑
- 不要用 `nohup caveman serve &` 从 Hermes terminal 启动，历史上会触发 exit-130 loop；继续用 subprocess.Popen。
- `ps aux | grep '[c]aveman.*serve'` 可能匹配当前诊断命令本身；以 pidfile/health/status 为准。
- 历史日志有大量旧 `hot-reload`、Permission DENIED、`no such column`、重复日志；不要用 raw tail 全历史做当前状态判断。
