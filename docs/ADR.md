# Architecture Decision Records (ADR)

> 记录 Caveman 框架7轮深度优化中的关键架构决策。每条决策包含：为什么做、对比了什么方案、最终选择了什么。

---

## ADR-001: 事件驱动解耦 AgentLoop（Round 4）

**背景：** AgentLoop.run() 是一个 117 行、圈复杂度 26 的单片方法。日志、指标、审计、显示全部硬编码在里面。加新功能（比如 metrics 采集）需要改 loop 本体。

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 保持现状 + 更多 if/else | 简单 | 复杂度持续膨胀 |
| B. Middleware 链（Express风格） | 灵活 | 过度工程，Python async 中间件链不自然 |
| **C. EventBus + 订阅者** | **解耦彻底，新功能=订阅事件** | 轻微的间接性 |
| D. Hooks 系统（Claude Code 风格） | 用户可扩展 | 实现更复杂 |

**决策：** 方案 C — EventBus。定义 22 种事件（EventType enum），AgentLoop 只负责 emit，观察者（日志/指标/审计/显示）各自订阅。

**结果：** 
- AgentLoop.run() 复杂度从 26 降到 7
- MetricsCollector 作为普通订阅者实现，零耦合
- 新增功能（比如 audit log）只需 `bus.on(EventType.TOOL_CALL, audit_handler)`

---

## ADR-002: @tool 声明式装饰器替代手动注册（Round 5）

**背景：** 添加新工具需要在 3 个地方改代码：① 写工具函数 ② 写 schema ③ 在 registry 手动注册。容易遗漏。

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 手动注册（现状） | 明确 | 三处修改，容易遗漏 |
| **B. @tool 装饰器 + auto_discover** | **一处定义，自动注册** | 全局状态（_TOOL_REGISTRY list） |
| C. 基于文件名约定自动扫描 | 零配置 | 隐式，不够明确 |

**决策：** 方案 B — 装饰器。`@tool(name, description, params)` 附着在函数上，`ToolRegistry.auto_discover()` 自动收集所有被装饰的函数。

**结果：**
- 新增工具只需 1 处改动（写函数 + @tool 装饰）
- 7 个内置工具全部声明式：bash, file_read, file_write, file_edit, file_list, web_search, browser
- `_register_builtins()` 只需 import 模块触发装饰器

---

## ADR-003: Provider 真流式 — 去掉 buffer-retry 反模式（Round 7）

**背景：** 两个 Provider 的 `complete()` 方法用 `retry_async` 包裹了整个流消费过程：先把所有事件收集到 list，retry 成功后再 re-yield。这完全破坏了流式体验——用户要等整个响应完成才能看到第一个字。

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 保持 buffer-retry（现状） | 简单重试 | **不是真流式，内存膨胀** |
| **B. 流式直��� yield + 仅 non-stream 重试** | **真流式，低延迟** | 流式中途出错需上层处理 |
| C. 流式 checkpoint 重试 | 最完美 | 实现极复杂，需跟踪已yield事件 |

**决策：** 方案 B — stream 模式直接 yield 事件，不包裹 retry。non-stream 模式用 `retry_async` 包裹 API 调用（不是事件收��）。

**结果：**
- 用户看到第一个 token 的延迟从"等全部完成"变为"API首token到达即显示"
- 代码量减少（去掉了两处 `results = []` + `results.append` + re-yield 模式）
- 非流式调用仍然有3次重试保护

---

## ADR-004: Browser 从 class 迁移到 @tool 函数（Round 7）

**背景：** BrowserTool 是一个独立的 class，有自己的 `dispatch()` + `schema()` 方法，没有接入 @tool 系统。添加浏览器功能需要手动实例化 class 并注册。

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 保持 BrowserTool class | OOP 风格一致 | 与其��6个工具风格不同，不被 auto_discover 发现 |
| **B. 模块级 @tool 函数 + set_bridge()** | **统一风格，自动发现** | 模块级状态（_bridge, _playwright_ctx） |
| C. class 内部用 @tool | 两全其美 | 装饰器 + class 方法的交互复杂 |

**决策：** 方案 B — 模块级函数。`browser_dispatch()` 用 @tool 装饰，`set_bridge()` / `close_browser()` 管理状态。

**结果：**
- 7 个工具风格完全统一，全部走 auto_discover
- `close_browser()` 可注册到 Lifecycle.shutdown，实现生命周期管理
- dispatch table 模式（dict → lambda）替代 if-elif 链

---

## ADR-005: 统一结果类型 ToolResult/Ok/Err（Round 6）

**背景：** 21+ 处返回 ad-hoc `{"error": ...}` dict，有的用 `"error"` key，有的用 `"ok": False`，有的用 `"success": False`。调用者不知道该检查哪个 key。

**方案对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. 统一 dict 约定（文档规范） | 改动最小 | 无法强制执行 |
| **B. frozen dataclass ToolResult + Ok/Err 工厂** | **类型安全，不可变** | 需要全面替换 |
| C. Exception 统一 | Pythonic | 正常返回和错误返回混用异常不直观 |

**决策：** 方案 B — `ToolResult(ok, data, error, metadata)` frozen dataclass + `Ok(data)` / `Err(msg)` 工厂函数。

**结果：**
- 调用者统一用 `r.ok` 判断，`r.data` 取数据，`r.error` 取错误
- `to_dict()` 序列化为一致的 JSON shape（给 LLM tool_result 用）
- frozen=True 防止意外修改

---

## ADR-006: 结构化异常层次替代裸 Exception（Round 6）

**背景：** 全框架使用 `raise Exception("...")` 或 `raise RuntimeError("...")`，调用者无法区分"API限流"和"工具不存在"和"配置错误"。

**决策：** 建立 CavemanError 基类 + 11 个子类的层次结构，每个异常携带 `context` dict 和 `to_dict()` 方法。

```
CavemanError
├── ConfigError
├── ProviderError → RateLimitError, AuthError
├── ToolError → ToolNotFoundError, ToolPermissionError, ToolTimeoutError
├── MemoryError
├── BridgeError
└── SecurityError
```

**结果：** 调用者可以 `except RateLimitError as e: sleep(e.retry_after)` 精确处理。

---

## ADR-007: Lifecycle Manager 统一资源生命周期（Round 6）

**背景：** SQLite、Browser、UDS Server、Gateway 等资源散落在各模块，没有统一的启动/关闭顺序。Ctrl+C 杀进程时资源不会清理。

**决策：** 中央 Lifecycle Manager：
- `register(name, startup, shutdown)` 注册资源
- `start_all()` 按注册顺序启动
- `shutdown_all()` 逆序（LIFO）清理
- 支持 `async with lifecycle:` context manager
- 自动安装 SIGINT/SIGTERM 处理器

**结果：** 资源管理从"祈祷没泄漏"变为"框架保证清理"。

---

*最后更新: 2026-04-14 | 7 条 ADR*
