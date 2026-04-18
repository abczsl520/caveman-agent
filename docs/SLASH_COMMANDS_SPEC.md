## Caveman Slash Command System — 产品级 Spec

### 设计哲学
不是"够用就行"，是做一个比 OpenClaw 和 Hermes 都更好的命令系统。
- 注册式架构（像 Hermes 的 CommandDef，但更强）
- 支持 CLI + Gateway + Discord/Telegram 全平台
- 自动补全 + 子命令 + 参数提示
- 插件可扩展

### 竞品分析

#### OpenClaw (18 commands, switch-case 硬编码)
/help, /gateway-status, /agent, /agents, /session, /sessions, /model, /models,
/think, /verbose, /fast, /reasoning, /usage, /elevated, /activation, /new, /reset,
/abort, /settings, /exit, /quit

优点：model selector, thinking levels, fast mode
缺点：硬编码 switch-case，不可扩展，无自动补全

#### Hermes (40+ commands, 注册式架构)
Session: /new, /clear, /history, /save, /retry, /undo, /title, /branch, /compress,
         /rollback, /stop, /approve, /deny, /background, /btw, /queue, /status,
         /resume, /sethome
Config:  /model, /provider, /personality, /statusbar, /verbose, /yolo, /reasoning,
         /fast, /skin, /voice
Tools:   /tools, /toolsets, /skills, /cron, /reload-mcp, /browser, /plugins
Info:    /commands, /help, /restart, /usage, /insights, /platforms, /paste, /image, /update
Exit:    /quit

优点：CommandDef 注册式、分类、别名、子命令、自动补全、多平台适配
缺点：没有 fuzzy match，没有命令历史统计，没有快捷键绑定

### Caveman 命令清单（超集 + 独有）

#### Session (15)
| 命令 | 别名 | 参数 | 说明 | 来源 |
|------|------|------|------|------|
| /new | /reset | | 新建会话 | OC+H |
| /clear | | | 清屏+新会话 | H |
| /history | /h | [n] | 显示对话历史 | H |
| /retry | | | 重试上一条 | H |
| /undo | | | 撤销上一轮对话 | H |
| /title | | [name] | 设置会话标题 | H |
| /branch | /fork | [name] | 分支会话 | H |
| /compress | | [topic] | 手动压缩上下文 | H |
| /rollback | | [n] | 文件系统检查点回滚 | H |
| /stop | /abort | | 停止所有后台进程 | OC+H |
| /approve | | [session\|always] | 批准危险命令 | H |
| /deny | | | 拒绝危险命令 | H |
| /background | /bg | <prompt> | 后台运行 | H |
| /btw | | <question> | 旁路提问（不存历史） | H |
| /queue | /q | <prompt> | 排队下一轮 | H |

#### Configuration (12)
| 命令 | 别名 | 参数 | 说明 | 来源 |
|------|------|------|------|------|
| /model | | [name] [--global] | 切换模型 | OC+H |
| /provider | | | 显示可用 provider | H |
| /reasoning | /think | [level\|show\|hide] | 推理等级 | OC+H |
| /fast | | [on\|off\|status] | 快速模式 | OC+H |
| /verbose | | [on\|off] | 详细输出 | OC+H |
| /personality | | [name] | 预设人格 | H |
| /yolo | | | 跳过所有审批 | H |
| /voice | | [on\|off\|tts\|status] | 语音模式 | H |
| /skin | /theme | [name] | 主题切换 | H |
| /config | | [key] [value] | 查看/修改配置 | H |
| /elevated | | [on\|off\|ask] | 提权模式 | OC |
| /activation | | [mention\|always] | 群聊激活模式 | OC |

#### Tools & Skills (8)
| 命令 | 别名 | 参数 | 说明 | 来源 |
|------|------|------|------|------|
| /tools | | [list\|disable\|enable] [name] | 管理工具 | H |
| /skills | | [list\|search\|install\|info] | 管理技能 | OC+H |
| /engines | | [list\|enable\|disable] [name] | 管理引擎 | Caveman独有 |
| /cron | | [list\|add\|edit\|pause\|resume\|run\|remove] | 定时任务 | H |
| /reload-mcp | | | 重载 MCP 服务器 | H |
| /browser | | [connect\|disconnect\|status] | 浏览器工具 | H |
| /plugins | | | 插件列表 | H |
| /import | | [--from\|--detect\|--all] | 数据导入 | Caveman独有 |

#### Info & Diagnostics (12)
| 命令 | 别名 | 参数 | 说明 | 来源 |
|------|------|------|------|------|
| /status | | | 会话状态 | OC+H |
| /help | /? | [command] | 帮助（支持单命令详情） | OC+H |
| /commands | | [page] | 分页浏览所有命令 | H |
| /usage | | [off\|tokens\|full] | Token 用量 | OC+H |
| /insights | | [days] | 使用分析 | H |
| /doctor | | | 健康检查 | Caveman独有 |
| /memory | | [stats\|search\|forget] | 记忆管理 | Caveman独有 |
| /recall | | [topic] | 恢复上下文 | Caveman独有 |
| /shield | | | Shield 状态 | Caveman独有 |
| /reflect | | | 反思记录 | Caveman独有 |
| /audit | | [path] | 导出审计日志 | Caveman独有 |
| /ratelimit | | | 速率限制状态 | Caveman独有 |

#### System (5)
| 命令 | 别名 | 参数 | 说明 | 来源 |
|------|------|------|------|------|
| /restart | | | 重启网关 | H |
| /update | | | 更新 Caveman | H |
| /profile | | | 当前 profile | H |
| /sethome | /set-home | | 设为主频道 | H |
| /quit | /exit, /q | | 退出 | OC+H |

#### Caveman 独有增强
| 命令 | 参数 | 说明 |
|------|------|------|
| /flywheel | [status\|trigger\|history] | 飞轮状态 |
| /trajectory | [list\|replay\|score] | 轨迹管理 |
| /bench | [run\|results] | 性能基准测试 |
| /selftest | | 自测 |
| /wiki | [search\|compile] | 知识 Wiki |

### 架构

```
caveman/commands/
├── __init__.py          — COMMAND_REGISTRY + resolve_command()
├── types.py             — CommandDef dataclass
├── registry.py          — 注册、查找、分类、别名解析
├── handlers/
│   ├── __init__.py
│   ├── session.py       — /new, /clear, /history, /retry, /undo, /branch...
│   ├── config.py        — /model, /reasoning, /fast, /verbose, /personality...
│   ├── tools.py         — /tools, /skills, /engines, /cron, /import...
│   ├── info.py          — /status, /help, /usage, /doctor, /memory...
│   └── system.py        — /restart, /update, /quit...
├── completer.py         — 自动补全（命令+子命令+参数+路径+@引用）
├── dispatcher.py        — 统一分发：CLI/Gateway/Discord/Telegram
├── formatter.py         — 多平台输出格式化（Rich/Markdown/Plain）
└── gateway_adapter.py   — Discord/Telegram 命令注册适配
```

### CommandDef 增强（比 Hermes 更强）

```python
@dataclass(frozen=True)
class CommandDef:
    name: str                          # 命令名（不含 /）
    description: str                   # 描述
    category: str                      # 分类
    handler: str                       # handler 函数路径 "session.handle_new"
    aliases: tuple[str, ...] = ()      # 别名
    args_hint: str = ""                # 参数提示
    subcommands: tuple[str, ...] = ()  # 子命令
    cli_only: bool = False             # 仅 CLI
    gateway_only: bool = False         # 仅 Gateway
    hidden: bool = False               # 隐藏（不在 /help 显示）
    dangerous: bool = False            # 危险操作（需确认）
    cooldown_seconds: int = 0          # 冷却时间（防刷）
    examples: tuple[str, ...] = ()     # 使用示例
    since: str = "0.3.0"              # 引入版本
```

### Handler 接口

```python
@dataclass
class CommandContext:
    """Passed to every command handler."""
    command: str           # 原始命令名
    args: str              # 参数字符串
    agent: Any             # CavemanAgent 实例
    surface: str           # "cli" | "discord" | "telegram" | "gateway"
    session_key: str       # 当前会话 key
    respond: Callable      # 回复函数（自动适配平台格式）

async def handle_xxx(ctx: CommandContext) -> None:
    """每个 handler 的签名。"""
    ctx.respond("result text")
```

### 自动补全增强（比 Hermes 更强）

1. 命令名 fuzzy match（/mod → /model）
2. 子命令补全（/model <tab> → 显示可用模型）
3. 路径补全（/image <tab> → 文件浏览）
4. @引用补全（@file:, @folder:, @diff, @staged）
5. 模型名补全（从 provider registry 动态获取）
6. 记忆搜索补全（/recall <tab> → 最近话题）

### 多平台适配

- CLI: Rich Panel + Table 格式
- Discord: Markdown 格式，注册 Discord slash commands
- Telegram: BotCommands 菜单注册
- Gateway: 纯文本格式

### 测试要求

1. 每个 handler 的单元测试
2. CommandDef 注册完整性测试（无重复名/别名）
3. 别名解析测试
4. 子命令补全测试
5. 多平台格式化测试
6. 危险命令确认测试
7. 冷却时间测试
8. /help [command] 单命令详情测试
