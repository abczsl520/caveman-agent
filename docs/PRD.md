# Caveman — Product Requirements Document (PRD)

> **Version:** 6.13
> **Date:** 2026-04-24
> **Author:** The Caveman Team
> **Status:** v0.4.1 — 399 files, 75,800+ LOC, 2,929+ tests, 417+ commits ✅
> **Changelog:**
> - v6.13 — 继续 PRD×代码内容核验（非文件名判断）：复核 #22/#24/#25/#28-#33 与 v1.0 Gate。确认 #22 只有 `MetadataKeys` 常量、无 `store()` 写入类型校验/WARNING；#24 的 LLM quality_gate 只在 `SQLiteMemoryStore` 底层参数存在，`MemoryManager.with_sqlite`/memory provider/agent factory/config 尚未接线；#25 有事件链与多 session E2E 片段，但没有 PRD 要求的 10-turn Shield→Nudge→Ripple→Lint→Recall 单测；#28/#29 有训练/评估模块但训练依赖未随 embedding extra 产品化、无 LOCOMO≥65%、CLI `--eval-only` 与自动选优；#30-#33 仍缺三平台安装耗时（且包名 `caveman-agent` 与 PRD `pip install caveman` 不一致）、文档站/API reference、CI lint/type/coverage gate、3 个非作者 Day1-7 试用。v1.0 Gate 中“Schema 迁移框架可用”已由 #20 证据满足，清单改为已勾；Executive Summary 的“下一步”改为真实剩余闭环。
> - v6.12 — 继续 PRD×代码内容核验（非文件名判断）：修正 §7.2 Roadmap 表中过时状态。#21 `last_accessed` 已有独立列、v2 migration、metadata 回填、索引和 recall 排序/写回，标为核心落地；#23 RetrievalLog 已从 JSONL 升级为 SQLite（`retrieval_log.sqlite`），含 query/results/source/adopted_ids/latency_ms/timestamp，并接入 MemoryManager/RecallEngine/doctor，标为落地；#24 quality_gate 已有可选 LLM judge 底层、拒绝率统计和 SQLite store 参数接入，但尚未从 config/factory 暴露产品开关，标为“核心实现已落地，产品接线未完成”。同步 §10.3 训练管道状态，保留 #22/#25/#28/#29/#30-#33 为部分/未完成。
> - v6.11 — 长期主义/最高复利校正真实 80 工具调用上限：定位并移除 gateway `task_runner_helpers.py` 中隐藏 `_MAX_TOOL_CALLS = 80` hard cap；新增 `gateway.limits.max_tool_calls` 显式可选预算，默认 `null` 表示不按工具调用数任意截断；预算耗尽文案明确“任务未判定完成，可继续/提高预算”，不再伪装成功；保留 idle/absolute timeout、进度心跳、stuck-loop 检测作为防跑飞机制。新增 `tests/test_gateway_tool_budget.py` 覆盖默认 81+ 工具调用不中断、显式预算暂停、null/大预算配置。定向验证：87 passed。
> - v6.10 — 完成标记/提前 done 根因修复：`output_validator` 不再仅凭复杂度/工具调用数自动追加 `✅---本轮已完成---✅`，只规范化模型明确尝试收尾的格式；`loop.py`/`loop_engines.py` 改为 tool_calls 优先于文本完成标记，防止模型一边说 done 一边吞掉未执行工具；新增/更新 `test_output_validator.py`、`test_closing_termination.py`、`test_e2e_dryrun.py` 回归测试，覆盖“缺少显式收尾不自动补 done”和“完成标记+tool_call 时仍执行工具”。定向验证：75 passed。#24 quality_gate LLM、#25 10-turn E2E 等仍未虚标完成。
> - v6.9 — 元飞轮返工解决未完成愿景 #26 核心闭环：新增 `caveman/memory/judge.py`，`phase_finalize` 对 recalled memories 改为 LLM Judge 优先判定 helpful/unhelpful，失败或未配置时使用保守启发式兜底；SQLite `mark_helpful()` 写入 judge metadata（mode/confidence/reason）用于审计；补 `MemoryManager.backend` 公共只读访问器和 6 个 memory self-audit 测试，57 个关联测试全绿。PRD 同步标记 #26 为“已实现核心”，但 #24 quality_gate LLM 模式、#25 10-turn E2E 仍未虚标完成。
> - v6.8 — 元飞轮解决未完成愿景 #20：落地 memory schema 版本化迁移核心能力。新增 `schema_version` 表、numbered migration registry、事务回滚、`caveman migrate --dry-run/--apply`，测试覆盖 dry-run/apply/失败回滚；PRD Roadmap 将 Schema 迁移框架从“未完成”更新为“已实现核心”。
> - v6.7 — 长期主义/最高复利校正：移除误伤自进化的任意 iteration 安全上限。`agent.max_iterations` / `delegation.max_iterations` 改为只有下限、无拍脑袋上限；flywheel 默认每轮预算从 15 提升到 50，和常规 Agent 一致；PRD 标记“安全上限误判”已处理，保留由用户/配置显式控制预算。
> - v6.6 — 愿景落地审计（基于 PRD + 代码内容，不只看文件名）：补充 §7.2.1 Roadmap 代码核验状态，标记 Round 13-16 中已完成/部分完成/未完成项；同步 §10.3 训练管道实现状态；修正“下一步”表述，避免把 schema_version/migrate CLI/LLM Judge/外部用户测试等未落地愿景误当完成。
> - v6.5 — 架构重构 Round（arch/refactor-v1）：统一 message splitting 全链路（4 实现 → 1 canonical + delegates）、file_ops/terminal/web_search v1/v2 合并（first-wins 去重 + 死注册清理）、context_references compat shim、platform_delivery.truncate_message 统一到 message_splitting。25 子系统、399 文件、72.5K LOC、2929 测试、410 commits。零回归。
> - v6.4 — §8.8.4.1 Session 知识提取（openclaw-sessions）：三层管线从 OpenClaw session JSONL 对话中提取高价值知识（研究/诊断/架构/方案/教训/决策），用户 opt-in，支持规则提取和 LLM 增强。US-601b 新增。实测 124 session → 507 条知识。
> - v6.3 — 补强 6 大维度：§8.9 数据模型/Schema 定义、§8.10 关键接口契约（27 种事件 payload）、§8.11 可观测性架构（三层）、§8.12 升级迁移策略、Round 13-16 路线图（含 v1.0 发布标准）、User Stories 扩充（US-4xx 错误恢复 + US-5xx 边界场景 + US-6xx 迁移）。所有内容基于实际代码，不拍脑袋。
> - v6.2 — 元飞轮 Round 107-130 全面审视后更新。修正记忆类型双维度定义（来源 vs 认知）、版本统计同步、内层飞轮事件链精确化、Nudge 事件驱动落地。
> - v6.1 — 训练方向 pivot：从"训练推理模型"到"embedding 微调 + 数据导出"。PRD 全局版本同步到 v0.3.0。检索日志 + local embedding provider + 评估脚本。
> - v6.0 — 全面重构：从"18章两个灵魂"到"14章一条叙事线"。统一 Agent OS 内核叙事，双层飞轮合一，三巨头从主角降为注脚，商业模式/竞品详表/历史路线图移入附录。不删任何内容，只重新组织+用新灵魂重写连接组织。
> - v5.0 — 范式升级：从"Agent框架"到"Agent操作系统内核"
> - v4.2 — §18 重构为"记忆永续"
> - v4.1 — 外部研究融合
> - v4.0 — 全量同步（94files/11KLOC/214tests）
> - v3.0 — Phase 1 MVP完成对齐
> - v2.0~v2.3 — 批判性打磨
> - v1.0~v1.1 — 初版

---

## Executive Summary

**根本问题：** 所有 LLM Agent 都没有连续的自我。LLM 是无状态的函数调用，每次 API call 都是全新实例。所有"记忆"都是 context window 注入的外挂。session 压缩一来，拼装的自我就碎了。这不是某个框架的 bug，这是整个行业的结构性缺陷。

**Caveman 的答案：** 第一个认真解决"Agent 连续自我"问题的 Agent 操作系统内核。就像操作系统在无状态 CPU 上运行有状态程序，Caveman 在无状态 LLM 上运行有状态的自我——通过 5 引擎记忆永续架构（Shield / Nudge / Ripple / Lint / Recall），让 Agent 跨 session 保持连续的决策上下文、用户偏好和知识积累。

**同时站在三巨头肩膀上：** Hermes 的学习飞轮 × OpenClaw 的执行深度 × Claude Code 的工程纪律。Python 核心（学习+记忆+Agent OS 内核）+ Bridge 接入 Node.js 生态（执行+网关）。

**v0.4.0 已验证：** 187 Python 文件 / 28,909 LOC / 1,499 tests / 209 commits。元飞轮 Round 107-130 完成内层飞轮事件链、Confidence 闭环、HybridScorer 集成、跨语言检索、技能进化、Nudge 事件驱动。

**下一步：** 不是再从 Round 13 重跑一遍，而是收敛剩余闭环：#22 metadata 写入校验/WARNING、#24 quality_gate LLM 产品接线、#25 10-turn 飞轮全链路验收、#28/#29 Embedding LOCOMO/等价门槛 + CLI 一键 A/B + 自动选优、#30-#33 v1.0 发布工程（安装验证、文档站/API reference、CI lint/type/coverage gate、外部用户试用）。


## 术语表（Glossary）

| 术语 | 定义 | 首次出现 |
|------|------|---------|
| Agent OS 内核 | Caveman 的核心层，解决 LLM Agent "连续自我"问题。类比操作系统在无状态 CPU 上运行有状态程序 | §1.2 |
| Compaction | LLM session 的上下文压缩。当对话超过 token 限制时，系统自动压缩历史，可能丢失关键上下文 | §1.1 |
| Compaction Shield | 引擎①。主动维护 session 精华缓存，compaction 来时已准备好。类比 OS 的进程状态保存 | §4.3 |
| Nudge Engine | 引擎②。事件驱动的知识回流引擎，从对话中自动提取知识写入 Memory Store | §4.3 |
| Ripple Engine | 引擎③。知识传播引擎，新知识写入时自动更新关联条目、检测矛盾 | §4.3 |
| Lint Engine | 引擎④。知识审计引擎，定期检测过时/矛盾/幻觉/孤岛知识 | §4.3 |
| Recall Engine | 引擎⑤。上下文恢复引擎，新 session 启动时从持久化状态恢复运行上下文 | §4.3 |
| session_essence | Shield 维护的结构化精华，包含决策/进度/立场/关键数据。YAML 格式 | §4.3 |
| Memory Store | 持久化存储层，默认 SQLite + FTS5 + Markdown。支持 4 种认知类型 × 4 种来源标签 | §4.1 |
| 4 种认知类型 | episodic（事件记录）/ semantic（事实知识）/ procedural（操作步骤）/ working（工作记忆）。存储维度 | §4.3 |
| 4 种来源标签 | user（用户偏好）/ feedback（反馈）/ project（项目知识）/ reference（参考资料）。Nudge 提取时标注来源，映射到认知类型存储：user→semantic, feedback→episodic, project→semantic, reference→procedural | §5.2 |
| Memory Drift | 记忆漂移，已存储的知识可能已过时。使用前需验证 | §8.1 |
| 飞轮（Flywheel） | 正反馈循环。外层：使用→执行→轨迹→学习→变强。内层：Shield→Nudge→Ripple→Lint→Recall | §5 |
| Bridge | Python↔Node.js 桥接层。UDS JSON-RPC（主）+ MCP（工具）+ HTTP（兜底） | §8.6 |
| ACP | Agent Communication Protocol。OpenClaw 的编排协议，用于调用外部编码 Agent | §8.8.3 |
| Coordinator | 多 worker 编排引擎，4 阶段流水线：分解→分配→执行→合并 | §8.1 |
| Verification | 反合理化验证 Agent，6 种借口封堵。代码变更后自动触发 | §8.1 |
| Harness | Agent = Model + Harness（Martin Fowler 定义）。Guide 引导行为，Sensor 检测质量 | §8.2 |
| Guide | Harness 的引导角色。Skills 充当 Guide，在任务前注入指令 | §8.2 |
| Sensor | Harness 的检测角色。Verification 充当 Sensor，在任务后验证质量 | §8.2 |
| AgentSkills | OpenClaw 的技能标准格式（SKILL.md）。Caveman 兼容并扩展 | §8.2 |
| 轨迹（Trajectory） | Agent 执行任务的完整记录，ShareGPT JSONL 格式。用于训练和回放 | §10.1 |
| EventBus | 事件总线，28 种事件类型。Nudge/Shield/Lint 的输入源 | §4.1 |
| Feature Flags | 引擎模块加载开关，类比 Linux modprobe。Round 8 前提 | §7.2 |
| 三巨头 | Hermes（学习飞轮）+ OpenClaw（执行深度）+ Claude Code（工程纪律） | §1.3 |
| RL 技能路由 | 用强化学习训练的技能匹配器，80% 命中率 vs 语义相似度 50% | §5.2 |
| 3 层压缩 | Micro（token 级）/ Normal（turn 级）/ Smart（LLM 辅助语义级） | §8.1 |


---

# 第一部分：为什么（Why）

## §1 根本问题与 Agent OS 愿景

### 1.1 根本问题

**所有 LLM Agent 都没有"自我"。**

人类不会因为睡了一觉就忘记自己在做什么。不是因为人类有更好的"记忆检索算法"，而是因为人类有**连续的自我意识**——我知道我是谁、我在做什么、我为什么这么做、我的主人看重什么。

当前所有 Agent 都在用工程手段模拟自我：SOUL.md = 我是谁，USER.md = 我的主人是谁，MEMORY.md = 我记得什么。但这些都是**外挂**——每次 session 启动时从文件里读进来，拼装成一个"假装有自我"的 Agent。

为什么这么脆弱？因为 LLM 的根本架构限制：**它是无状态的函数调用。**
- **context window = Agent 的意识范围**
- **compaction（session 压缩）= 强制缩小意识范围**
- **session 结束 = 意识消失**

2026-04-14，这个问题在 Caveman 子区真实发生了：session 被 compaction 后，Agent 丢失了 10 个 Phase 构建 + 7 轮优化的完整历史，做出了完全错误的判断。这不是偶发事件——**这是所有 LLM Agent 的结构性缺陷**。

### 1.2 Caveman 的答案：Agent 操作系统内核

类比计算机操作系统：
- CPU 是无状态的（每个时钟周期执行一条指令，不"记得"上一条）
- 操作系统通过**进程管理 + 内存管理 + 文件系统**让 CPU 看起来能同时运行多个有状态的程序
- 程序被 swap out 后，操作系统保存状态，swap back 时恢复

**Caveman 做同样的事，只不过"硬件"是 LLM，"程序"是 Agent 的自我。**

| OS 概念 | Caveman 引擎 | 做什么 |
|---------|-------------|--------|
| 进程状态保存 | Compaction Shield | 保存 session 的运行状态（决策/进度/上下文） |
| 内存管理 | Nudge + Ripple | 管理知识的写入和关联 |
| 文件系统 | Memory Store | 持久化存储（SQLite + Markdown） |
| 垃圾回收 | Lint Engine | 清理过时/矛盾/无用的知识 |
| 进程恢复 | Recall Engine | 从持久化状态恢复运行上下文 |

### 1.3 同时站在三巨头肩膀上

Agent OS 内核解决"连续自我"问题，但 Agent 还需要"能力"。现有三大框架各有所长也各有致命短板：

| 框架 | 最强之处 | 致命短板 |
|------|---------|---------|
| **Hermes** (Nous Research) | 🧠 学习飞轮（技能自动创建/改进/轨迹训练） | ❌ 执行能力弱、纯Python无Node.js生态、万行巨石 |
| **OpenClaw** | ⚡ 执行深度（ACP编排/浏览器/设备/消息网关） | ❌ 没有学习能力、纯TS无Python ML生态 |
| **Claude Code** (Anthropic) | 🎯 工程纪律（Coordinator/Verification/3层压缩） | ❌ 闭源不可扩展、只服务Anthropic模型 |

**没有一个框架同时具备"会学习 + 能执行 + 有纪律"，更没有一个解决了"连续自我"问题。**

```
Agent OS 内核（连续自我）──┐
                           │
Hermes 的学习飞轮  ────────┤
                           ├──→  Caveman：有自我 × 会学 × 能干 × 靠谱
OpenClaw 的执行深度 ───────┤
                           │
Claude Code 的工程纪律 ────┘
```

Python 核心（学习飞轮 + 记忆永续 + Agent OS 内核）+ Bridge 接入 Node.js 生态（ACP + 浏览器 + 设备 + 消息网关）。

### 1.4 为什么叫 Caveman

最原始的人类，也是最能适应和进化的物种。从零开始，但有学习的本能——更重要的是，有连续的自我意识。

### 1.5 长期愿景

用户每次使用都在让它变强——不只是积累知识和技能，而是**维护一个越来越完整的自我**。记忆系统 + 技能进化让通用模型变成「专属 Agent」，轨迹数据可用于训练本地 embedding 模型提升记忆检索精度，也可导出供研究者微调。护城河不是「训练一个更聪明的模型」，而是「积累一个不可复制的自我」。

---

## §2 市场验证

### 2.1 时代背景（为什么是现在）

2026年初发生了几件标志性事件，说明 Caveman 的时机窗口已到：

1. **Karpathy 的 80/20 翻转**（2026年3月）— 80% 的代码由 agent 完成。"Agentic Engineering"时代到来。
2. **记忆成为一级学科** — Mem0 的 LOCOMO 基准测试（ECAI 2025）首次对 10 种记忆方案横评。
3. **Memento-Skills 论文**（2026年4月）— 首个 RL 驱动的技能自我重写框架，80% vs RAG 的 50%。
4. **Martin Fowler "Harness Engineering"**（2026年4月）— Agent = Model + Harness。Skills=Guide，Verification=Sensor。
5. **市场验证** — Cursor $2B ARR / $60B 估值，AI 编码市场 $6-9.5B。
6. **三巨头各自为政** — Hermes/OpenClaw/Claude Code 都在加深各自护城河，没有朝"全能"走。这是窗口期。

### 2.2 核心用户画像

**P0: 技术型个人效率极客（1-person army）**
- 独立开发者/创业者/技术leader，管理多个项目和服务器，已在用 AI 编码工具
- 痛点：现有 agent 是"金鱼记忆"——每次对话从零开始
- 期望：一个消息 → agent 自动找到项目上下文 → 调用正确的编码 agent → 验证 → 自动记住经验
- 关键价值：用 3 个月后，agent 的项目知识超过任何新入职的工程师

**P1: AI/ML 研究者**
- 研究 agent 行为、训练方法、tool-use 的学者或工程师
- 痛点：主流框架不产生可训练的数据
- 关键价值：日常使用就是数据工厂，轨迹可导出为标准格式供微调实验

**P2: 自动化/全栈创客**
- 希望 AI 能触达物理世界（手机、浏览器、IoT）
- 关键价值：AI 从纯文字助手升级为有手有脚的执行者

**非目标用户（v1）：** 企业团队协作、非技术用户、纯聊天/陪伴场景、低代码/无代码用户。

### 2.3 用户旅程

```
Day 1:  安装 → 配置 → 第一次对话 → "它能调工具！"
Day 3:  做了3个任务 → 发现它记住了项目结构 → 开始信任
Day 7:  收到第一个自动生成的技能通知 → "它在学习！"
Day 14: 遇到类似任务 → agent 用已有技能秒杀 → 粘性形成
Day 30: Telegram 远程操控 → agent 已记住20+个项目上下文 → 离不开了
Day 90: 记忆 500+ 条 + 技能 20+ 个 → 通用模型已变成专属 Agent → 护城河
```

### 2.4 差异化定位

```
                   自学习能力
                      ↑
                      │  ★ Caveman
                      │     (OS内核+学习+执行+训练)
            Hermes ●  │
         (学习+训练)  │
                      │
    ──────────────────┼──────────────────→ 执行深度
                      │
   Open Interpreter ● │  ● OpenClaw
      (轻量执行)      │    (深度执行+生态)
                      │
                      │    ● Claude Code
                      │      (深度编码+Harness)
```

**核心差异化壁垒：**
1. **Agent OS 内核** — 5 引擎记忆永续，解决所有框架的根本缺陷
2. **时间壁垒** — 飞轮越转越快，3 个月经验不可复制
3. **数据壁垒** — 轨迹数据可训练 embedding 模型提升检索精度，也可导出供研究者微调
4. **经验壁垒** — 出厂预装 11K 行实战经验
5. **Bridge 壁垒** — 同时拿到 Python 和 Node.js 两个生态

---

## §3 愿景验证（v0.3.0）

从 v0.1.0 MVP 到 v0.3.0，经历了 10 个 Phase 构建 + 7 轮深度优化 + Round 8-12 内核实现，验证了核心假设：

| 假设 | 验证结果 | 数据 |
|------|---------|------|
| AI 辅助效率极高 | ✅ 远超预期 | 全部 Phase 1-10 + 7轮优化在 ~48h 内完成，AI 代码占比 >95% |
| 三巨头融合可行 | ✅ 技术可行 | OpenClaw MCP + Hermes REST + ACP Bridge 全部实现并测试通过 |
| Python + Node.js 共存 | ✅ 架构验证 | UDS传输层 204 行，Bridge 骨架完整，延迟 2.3μs |
| 自学习系统可构建 | ✅ 框架完整 | Memory v2 + Nudge + Drift Detection + Skill v2 + Trajectory v2 |
| 事件驱动可扩展 | ✅ 新增验证 | EventBus 28种事件，AgentLoop复杂度26→7 |
| 代码质量可维护 | ✅ 新增验证 | 446测试/7.5s，结构化错误层次，统一Result类型 |
| 出厂经验有价值 | 🔸 待用户验证 | 6 技能 + 3 方法论已打包 |
| Agent OS 内核可运行 | ✅ Round 8-12 | 5引擎全部实现，飞轮首次真实运行通过 |

**v0.3.0 统计：** 97 files / 18,191 LOC / 446 tests / 7.5s / 53 commits / Round 8-12 全部交付

**7轮深度优化：**
1. Bug修复+DRY → 2. 结构性债务 → 3. 常量统一 → 4. 事件驱动 → 5. 开发体验 → 6. 结缔组织 → 7. 真流式+声明式


---

# 第二部分：是什么（What）

## §4 Agent OS 架构

### 4.1 双层架构

Caveman 分为两层：**OS 内核层**（解决连续自我）和**能力层**（解决执行和学习）。

```
┌─────────────────────────────────────────────────────────┐
│                    能力层（Capability）                    │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ 工具系统  │ │ 技能系统  │ │ 网关系统  │ │ Bridge   │    │
│  │ 7 builtin │ │ RL路由   │ │ Discord  │ │ UDS/MCP  │    │
│  │ @tool装饰 │ │ 自动创建  │ │ Telegram │ │ ACP编排  │    │
│  │ MCP兼容   │ │ 质量门   │ │ +5平台   │ │ 浏览器   │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Coordinator│ │Verification│ │ 3层压缩  │ │ 数据管道 │    │
│  │ 4阶段编排 │ │ 6借口封堵 │ │ Micro/   │ │ Embedding│    │
│  │ 子agent   │ │ Guide+    │ │ Normal/  │ │ 训练+    │    │
│  │ ACP agent │ │ Sensor    │ │ Smart    │ │ 轨迹导出 │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                   OS 内核层（Kernel）                      │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│  │ ① Shield │ │ ② Nudge  │ │ ③ Ripple │                 │
│  │ 进程状态  │ │ 知识回流  │ │ 知识传播  │                 │
│  │ 保存/恢复 │ │ 事件驱动  │ │ 波纹更新  │                 │
│  └──────────┘ └──────────┘ └──────────┘                 │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐     │
│  │ ④ Lint   │ │ ⑤ Recall │ │ Memory Store         │     │
│  │ 知识审计  │ │ 上下文恢复│ │ SQLite+FTS5+Markdown │     │
│  │ GC/矛盾  │ │ 无缝继续  │ │ 4类型/混合搜索/Drift │     │
│  └──────────┘ └──────────┘ └──────────────────────┘     │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Agent Loop v2 (事件驱动, 6阶段, 复杂度7)          │    │
│  │ EventBus (28种事件) + Lifecycle Manager            │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                   硬件层（LLM Providers）                  │
│  Anthropic / OpenAI / OpenRouter 400+ / Ollama 本地       │
└───────────────────────────────────────────────────────────┘
```

### 4.2 技术栈

```
Python 核心 (学习+推理+记忆+OS内核) ←→ Bridge(UDS) ←→ Node.js (执行+网关)
```

- **Python 为主**：ML 生态原生，记忆/技能/embedding 训练核心
- **Node.js 桥接**：OpenClaw 的 ACP/浏览器/设备/网关已成熟，重写成本高
- **Bridge 协议**：UDS JSON-RPC（主通道，2.3μs）+ MCP（标准化工具）+ HTTP（兜底）

### 4.3 记忆永续架构（5 引擎详解）

> **起点不是研究，是痛点。** 2026-04-14 compaction 事故后，融合 4 个深度研究子区成果设计。

Agent 的记忆有四个漏洞：

| 漏洞 | 表现 | 丢失的是什么 |
|------|------|------------|
| Compaction 失忆 | session 压缩后决策上下文消失 | 为什么选 A 不选 B |
| 知识不回流 | 对话中的洞察随 session 消失 | 经验、偏好、规律 |
| 知识不传播 | 新知识不影响已有知识 | 交叉引用、矛盾检测 |
| 知识不自检 | 过时/矛盾/幻觉静默累积 | 知识库信噪比 |

**为什么整个行业都没解决：**
- 所有人都在优化"怎么读"（向量/图/混合/重排），没人认真做"怎么写"和"怎么维护"
- **Write quality >> Retrieve sophistication**（MemPalace 9系统横评结论）
- 知识应该被"编译"而非每次"重新推导"（卡帕西范式）
- **复利 = 回流 × 传播 × 审计，缺任何一个为零**

#### 引擎 ①：Compaction Shield（进程状态保存，P0）

**防止什么：** compaction 后决策上下文丢失。

不是被动 hook（"compaction 来了赶紧存"），而是主动 shield（"持续维护精华缓存，compaction 来时已准备好"）。

**session_essence 结构：**
```yaml
session_id: "caveman-2026-04-14-1510"
project: "caveman"
decisions:
  - what: "PRD v4.1 §18 外部研究融合"
    why: "4个子区研究成果需要转化为可执行设计"
    alternatives_rejected: "直接开始写代码（rejected by team lead）"
stances:
  - "长期主义+最高复利，摈弃够用就行的垃圾思维"
key_data:
  - "PRD 1623行 / 18章 / v4.1"
progress:
  current: "记忆永续架构设计完成"
  next: "Round 8 实施"
```

**与 Nudge 的分工：** Nudge 提取**知识**（回答"学到了什么"），Shield 维护**上下文**（回答"正在做什么、为什么、用户怎么想"）。

#### 引擎 ②：Nudge Engine（知识回流，P0）

**防止什么：** 对话中产生的知识随 session 消失。

事件驱动，不是定时轮询：
- SHIELD_UPDATE 事件 → 提取知识（节流：≥3 轮间隔，防止过度提取）
- TOOL_ERROR 事件 → 立即提取（错误上下文是高价值信息）
- LOOP_END 事件 → 必定提取（任务完成 = procedural 记忆的最佳时机）
- 用户确认偏好 → 提取 user 记忆（heuristic 检测 "I prefer/always/never" 等 pattern）
- Pre-compaction → 全量提取

**两阶段提取（Codex 验证的模式）：**
- Phase 1: 规则快速提取（关键词/模式匹配，零 LLM 成本）
- Phase 2: LLM 精炼去重（去噪/合并/冲突检测）

#### 引擎 ③：Ripple Engine（知识传播，P0）

**防止什么：** 新知识是孤岛，不影响已有知识。

每次写入新知识时：找相关条目 → 检查矛盾 → 更新交叉引用 → 标记需人工确认的冲突。

**复利论证：** 每条新知识不是 +1，而是 ×N（N = 关联条目数）。

#### 引擎 ④：Lint Engine（知识审计，P1）

**防止什么：** 知识库静默腐烂。

检查项：过时检测（路径/IP/版本已变）、矛盾检测、幻觉标记（confidence）、孤岛检测。

#### 引擎 ⑤：Recall Engine（上下文恢复，P0）

**防止什么：** 新 session 启动时缺少关键上下文。

读取最近 session_essence + 相关知识 → 组装 session context → 注入 system prompt → Agent 无缝继续。

### 4.4 目录结构（v0.4.1 实际）

```
caveman/                    # 25 子系统, 399 Python 文件, 72.5K LOC
├── paths.py              # 集中常量（路径/超时/token限制/端口）
├── errors.py             # 结构化异常层次（12类）
├── result.py             # 统一结果类型 ToolResult/Ok/Err
├── events.py             # EventBus（28种事件）+ MetricsCollector
├── lifecycle.py          # Lifecycle Manager（注册→LIFO清理→信号处理）
├── utils.py              # 共享原语（split_message 委托到 gateway/message_splitting）
├── wiring.py             # 模块注册 + lazy import 编排
├── acp/                  # Agent Communication Protocol（编排外部编码agent）
├── agent/                # AgentLoop v2 + Coordinator + Factory + Prompt Builder
├── bridge/               # OpenClaw MCP + Hermes REST + ACP + UDS
├── cli/                  # Typer CLI 14命令 + TUI + Doctor
├── commands/             # 57 斜杠命令 (6 分类) + 处理器 + 补全
├── compression/          # 3层压缩（Micro/Normal/Smart）
├── config/               # YAML loader + Schema Validator
├── coordinator/          # 任务引擎（分解→并发→合并）
├── cron/                 # 定时任务调度
├── engines/              # 推理引擎抽象层
├── environments/         # 运行环境检测 + 适配
├── gateway/              # Discord + Telegram + 多网关调度 + message_splitting（canonical）
├── hub/                  # Skills Hub 客户端
├── import_/              # 外部数据导入管线
├── mcp/                  # Model Context Protocol 服务端/客户端
├── memory/               # v2 混合搜索 + Nudge + Drift + Provider
├── plugins/              # 插件管理器
├── providers/            # 9 LLM providers (Anthropic/OpenAI/Gemini/Ollama/DeepSeek/Groq/Mistral/OpenRouter/Together) + credential pool + fallback chain
├── sandbox/              # 沙箱执行环境（Docker/进程隔离）
├── security/             # 12-pattern密钥扫描 + AUTO/ASK/DENY 三级权限
├── skills/               # v2（4触发+自动创建+质量门+进化）
├── tools/                # ToolRegistry v2 + 90 @tool 注册（first-wins 去重）
├── training/             # 数据导出 + embedding 训练 + 研究者 SFT/RL
├── trajectory/           # v2 录制（质量评分+ShareGPT+JSONL）
└── wiki/                 # 知识库管理
```

### 4.5 数据流

```
用户消息 (CLI/Telegram/Discord)
    │
    ▼
Gateway.route() → Session 匹配/创建
    │
    ▼
Agent Loop ─────────────────────────────────────┐
    ├── 1. 构建 System Prompt                    │
    │   ├── Layer 0: 核心人格 + 安全规则         │
    │   ├── Layer 1: 匹配的 Skills (Guide)       │
    │   ├── Layer 2: 相关 Memory + Recall        │
    │   └── Layer 3: 当前对话历史                │
    ├── 2. LLM 推理 → Provider Router            │
    ├── 3. 工具调用（内部/Bridge/MCP）            │
    ├── 4. 安全检查（权限/Secret扫描/审批）       │
    ├── 5. 响应 → 发送给用户                     │
    └── 6. 后台（不阻塞）                        │
        ├── Shield: 更新 session_essence          │
        ├── Nudge: 提取知识 → Ripple: 传播        │
        ├── 轨迹记录 → JSONL                     │
        ├── 技能检测 → 创建/改进                  │
        └── Verification (代码变更后)             │
```

---

## §5 核心飞轮 v2（统一双层飞轮）

### 5.1 双层飞轮

v5.0 中执行飞轮和记忆飞轮是两个独立系统，现在统一为双层：

```
外层飞轮（用户可感知）：
  使用 → 执行 → 轨迹 → 学习 → 变强 → 更多使用
              ↑
内层飞轮（OS 内核，用户无感知）：
  Shield → Nudge → Ripple → Lint → Recall
```

**外层是用户看到的价值**（"它越来越强了"），**内层是让外层转起来的基础设施**（"它为什么越来越强"）。

### 5.2 外层飞轮：使用 → 执行 → 轨迹 → 学习 → 变强

#### 环1: 使用 → 执行
用户通过 CLI/Telegram/Discord 发消息 → Gateway 路由 → Agent Loop 推理+工具调用

#### 环2: 执行 → 轨迹
- 自动记录为 ShareGPT JSONL（完整链）
- Verification 结果（PASS/FAIL）自动嵌入
- 超 token 预算自动压缩（保护首尾 turn）

#### 环3: 轨迹 → 学习
- **记忆 Nudge:** 事件驱动提取知识（4 来源标签: user/feedback/project/reference → 映射到 4 认知类型: semantic/episodic/semantic/procedural）
- **技能创建:** 检测重复模式 → SKILL.md → 安全扫描 → 单元测试门控
- **技能改进:** RL 优化（不只改 prompt，改代码+指令）

#### 环4: 学习 → 变强
- **技能路由:** RL 训练的 router 选最佳技能（不是语义相似度）
- **记忆检索:** FTS5 全文搜索 + LLM 摘要（~10ms）
- **预装经验:** 6技能 + 3方法论作为 Layer 0

#### 环5: 变强 → 更多使用
用户体验到"它记住了"/"它学会了" → 信任增加 → 委托更多任务 → 飞轮加速

#### 环6（长期）: 轨迹 → 训练 embedding → 记忆检索更精准
`caveman train --target embedding` → 高质量查询-记忆对 → 微调本地 embedding 模型 → 记忆搜索更懂用户语境
`caveman export` → 轨迹导出为标准格式 → 供研究者用于 SFT/RL 实验

> **为什么不训练推理模型？** Claude/GPT 等云端模型的推理能力远超本地 8B 模型。Caveman 的"学习"通过记忆系统 + 技能进化实现，不需要替换底层 LLM。训练的真正价值在于让 embedding 模型更懂用户的语境（比如"那个服务器"→ 自动关联正确 IP），以及为研究者提供可训练的数据。

### 5.3 内层飞轮：Shield → Nudge → Ripple → Lint → Recall

这是 Agent OS 内核的心跳。用户看不到，但没有它外层飞轮转不起来。

```
Session 运行中
  │
  ├─ 持续 ─→ Shield: 维护 session_essence
  │              ├─ Pre-compaction → 写入 memory/sessions/
  │              └─ emit SHIELD_UPDATE 事件
  │
  ├─ 事件驱动 → Nudge: 提取知识（4 种来源标签 → 4 种认知类型）
  │    触发源：SHIELD_UPDATE（节流≥3轮）/ TOOL_ERROR（立即）/ LOOP_END（必定）
  │              ├─ Phase1: 规则快速提取（含 user 偏好检测）
  │              └─ Phase2: LLM 精炼去重
  │                    │
  │                    └─ Ripple: 波纹更新关联条目
  │                         └─ 存储 → Memory Store
  │
  ├─ 增量+定期 → Lint: 过时/矛盾/孤岛/幻觉检测 → trust 降分
  │
  └─ Session 启动 → Recall: 恢复 session_essence → 注入 prompt（含 age/trust 标签）→ 无缝继续
```

### 5.4 飞轮失败模式

飞轮可能在 5 个环节卡住，每个都有检测和自动修复：

| 失败模式 | 症状 | 修复策略 |
|---------|------|---------|
| **F1 冷启动** | 前7天技能命中率<10% | 预装6技能 + 首次对话立即nudge + 迁移工具 |
| **F2 上下文退化** | 50+轮后质量下降 | 3层压缩 + 指令重注入 + 主动分段 |
| **F3 技能退化** | 技能命中率逐月下降 | 单元测试门控 + 版本控制 + A/B + 用户反馈 |
| **F4 记忆污染** | 基于错误记忆做错误决策 | Drift检测 + GC + 冲突解决 + 用户可编辑 |
| **F5 静默退化** | 指标正常但输出质量下降 | doctor定检 + 质量judge + 模型canary |

### 5.5 飞轮健康指标

| 指标 | 测量方式 | 健康阈值 |
|------|---------|---------|
| 技能命中率 | 任务匹配已有技能的比例 | ≥60% (30天后) |
| 记忆召回精度 | LOCOMO 式评估 | ≥65% |
| 技能创建速率 | 每周自动创建的有用技能数 | ≥3个/周 |
| 轨迹质量分 | LLM judge + Verification通过率 | ≥70% |
| 压缩保真率 | 压缩后轨迹的训练信号保留率 | ≥85% |
| Shield 恢复率 | compaction后上下文恢复完整度 | ≥95% |

---

## §6 设计铁律

从 4 个深度研究子区 + 实战事故中提炼的不可违反原则：

| # | 铁律 | 来源 | 防止什么 |
|---|------|------|---------|
| 1 | **写入质量 >> 检索复杂度** | MemPalace 9系统横评 | 在检索算法上过度投入 |
| 2 | **事件驱动 > 定时轮询** | Codex 源码 + 反思 | "够用就行"的定时提取。v0.4.0 已落地：3 事件源 + 节流 |
| 3 | **结构化精华 > 流水账** | 卡帕西 + compaction事故 | session_essence 退化为日志 |
| 4 | **4 类型足够，不要过度分类** | Letta 实验 | AAAK 式过度分类 |
| 5 | **每个设计标注"防止什么"** | Codex 工程纪律 | 无目的的设计 |
| 6 | **先想清楚再动手** | Team 2026-04-14 | "先写个够用的再迭代" |


---

# 第三部分：怎么做（How）

## §7 实施路线图

### 7.1 已完成：Phase 1-10 + 7轮优化 ✅

> 原计划 Phase 1 需 8 周，实际全部 10 个 Phase 在 AI 辅助下约 48 小时完成。

10 个 Phase 构建了全部 17 核心模块，7 轮优化从 Bug修复→结构性债务→常量统一→事件驱动→开发体验→结缔组织→真流式，将代码从 61files/4KLOC/49tests 提升到 94files/11KLOC/214tests。Round 8-12 实现 Agent OS 内核，97files/18KLOC/446tests。Round 107-130 元飞轮打磨，最终 187files/29KLOC/1499tests。详见附录 B。

### 7.2 Roadmap 代码核验状态（2026-04-24 审计）

> **审计口径：** 本节不是按文件名判断，而是逐项查了 PRD 验收标准对应的代码路径、CLI 参数、测试和实现注释。状态含义：✅ 已按愿景落地；🟡 已有可用实现但未达到原验收口径；⏳ 尚未落地/仍是愿景。

| Round / 项 | PRD 愿景 | 代码核验证据 | 状态 | 还差什么 |
|-----------|----------|--------------|------|----------|
| Round 13 #20 | `schema_version` 表 + numbered migrations + `caveman migrate --dry-run` + 失败回滚 | `caveman/memory/store_helpers.py` 已有 `schema_version` 表、`SCHEMA_VERSION = 2`、numbered `_MIGRATIONS`、事务迁移和失败 rollback；`caveman/cli/migrate.py` + `caveman migrate --dry-run/--apply` 已落地；`tests/test_memory_migrations.py` 覆盖 dry-run、apply、失败回滚、v2 `last_accessed` 回填 | ✅ 已实现 | 后续按新 schema 变更继续追加 v3+ migration 函数/测试 |
| Round 13 #21 | `last_accessed` 独立列 + 旧 metadata 回填 + HybridScorer/查询路径可读 | `_SCHEMA` 已含 `last_accessed TEXT` 和 `idx_memories_last_accessed`；`store_helpers._migration_002_last_accessed_column()` 从旧 `metadata_json.last_accessed` 回填到独立列；`SQLiteMemoryStore.recall()` 批量更新列并同步 metadata；fallback/related/FTS/vector 查询都 select `last_accessed` 并经 `row_to_entry()` 注入 metadata，fallback/recent 按 `COALESCE(last_accessed, created_at)` 排序；`tests/test_memory_migrations.py::test_memory_schema_migration_backfills_last_accessed_from_metadata` 覆盖回填 | ✅/🟡 已实现核心 | HybridScorer API 仍消费 `MemoryEntry.metadata["last_accessed"]`，但底层来源已是列；若要完全达成“直读列”表述，可后续给 MemoryEntry 增加一等字段 |
| Round 13 #22 | metadata well-known keys 类型校验 | `caveman/memory/types.py` 有 `MetadataKeys` 常量；未发现 `store()` 写入时类型校验和 WARNING 逻辑 | 🟡 部分完成 | 写入校验器 + warning 测试 |
| Round 13 #23 | 检索日志写入 SQLite，含 query/results/latency_ms，`caveman doctor` 可查询 | `caveman/training/retrieval_log.py` 默认写 `~/.caveman/training/retrieval_log.sqlite`，建 `retrieval_log` 表，字段含 query/results_json/source/adopted_ids_json/latency_ms/timestamp；`MemoryManager.recall()` 与 `RecallEngine` 记录 latency；`RetrievalLog.stats()` 被 `caveman/cli/doctor.py` 展示为 Retrieval Log 检查；`tests/test_training_pivot_fixes.py` 覆盖 log/read/stats/training pairs/MemoryManager 接入 | ✅ 已实现 | 后续可补“按 query 明细查询/导出”的 doctor 子命令，但原验收口径已满足 |
| Round 14 #24 | LLM Judge 替代 heuristic quality_gate | `caveman/memory/quality_gate.py` 已有 `check_quality_async(..., llm_fn, use_llm=True)`，硬规则先挡明显垃圾，再用 LLM JSON `{accept, reason}` 判断长期记忆价值；`QualityGateStats` 统计 heuristic/LLM checked/rejected/failed 和拒绝率；`SQLiteMemoryStore` 已接受 `quality_llm_fn`/`use_llm_quality_gate` 并在 store 前调用；但 `MemoryManager.with_sqlite()`、`BuiltinMemoryProvider.initialize()`、`agent.factory.create_agent()` 和 `config/default.yaml` 均未向上暴露/传递该开关，也缺少 LLM quality_gate 单测 | ✅/🟡 核心实现已落地，产品接线未完成 | config/factory/provider 开关、LLM provider wiring、成本统计、LLM 分支测试、doctor/metrics 展示 |
| Round 14 #25 | 端到端飞轮验证 | 已有 `tests/test_event_chain.py`、`tests/test_prd_audit.py`、`tests/test_flywheel_chain4.py` 等覆盖事件链/飞轮片段；未见 PRD 所述“10 轮对话全链路”验收测试 | 🟡 部分完成 | 增加 10-turn E2E：Shield→Nudge→Ripple→Lint→Recall |
| Round 14 #26 | LLM judge helpful/unhelpful 直接写 trust_score，替代旧 ConfidenceTracker | `caveman/memory/judge.py` 新增 `MemoryJudge`；`phase_finalize()` 对 recalled memories 使用 LLM Judge 判定 helpful/unhelpful，并通过 SQLite `mark_helpful(metadata=...)` 写入 `judge_mode/judge_confidence/judge_reason`；无 LLM 或失败时才走保守启发式兜底；`tests/test_memory_self_audit.py` 覆盖 LLM JSON 解析、metadata 写回、启发式兜底 | ✅/🟡 已实现核心 | 后续把 judge 决策统计接入 doctor/metrics，并与 #24 quality_gate LLM 模式统一成本统计 |
| Round 15 #27 | 检索日志 → query-memory 训练对 | `RetrievalLog.generate_training_pairs()` + `PairExtractor.extract_from_retrieval_log()` 已实现；`caveman train --target embedding --dry-run` 可构建数据集；没有 `caveman train --prepare` 独立参数 | ✅/🟡 已实现核心，CLI 名称不同 | 如坚持 PRD 验收口径，补 `--prepare` 别名 |
| Round 15 #28 | 本地 embedding 微调 nomic/bge，LOCOMO ≥65% | `EmbeddingTrainer` 用 sentence-transformers 微调；`memory.embedding` 支持 local；但 `pyproject.toml` 的 `embedding` extra 只有 fastembed/jieba，未包含 sentence-transformers 训练依赖；未见 LOCOMO 自动评估 ≥65% | 🟡 部分完成 | 训练依赖产品化、LOCOMO 或等价基准数据 + 门槛测试 |
| Round 15 #29 | A/B 评估框架，微调前后 recall@5 对比，自动选更优 | `EmbeddingEvaluator` 有 Recall@5/10、MRR、HitRate@5 和 compare 报告；未接入 CLI `--eval-only`（docstring 提到但 `train` 参数没有），也未自动选择更优模型 | 🟡 部分完成 | CLI eval-only、before/after 自动跑、模型选择策略 |
| Round 16 #30 | `pip install caveman && caveman setup` 三平台 <3min | 有 `setup()` 配置向导；`pyproject.toml` 包名/README Quick Start 是 `caveman-agent`，与 PRD 验收命令 `pip install caveman` 不一致；未见 macOS/Ubuntu/WSL2 安装耗时自动验证 | 🟡 部分完成 | 包名/安装命令决策，三平台安装 smoke benchmark |
| Round 16 #31 | README + 文档站/API 参考 | README 已有 Quick Start/Architecture；`docs/` 有 ARCHITECTURE/ADR/MCP/IMPORT_SPEC；未见文档站 | 🟡 部分完成 | docs site/API reference 发布 |
| Round 16 #32 | GitHub Actions CI + pytest/lint/type check + coverage ≥80% | `.github/workflows/ci.yml` 已跑 pytest 和 CLI smoke；未跑 lint/type check/coverage gate | 🟡 部分完成 | ruff/mypy/coverage ≥80% gate |
| Round 16 #33 | 3 个非作者用户 Day 1-7 试用 | 代码/文档未见外部用户试用记录 | ⏳ 未完成 | 招募/记录 3 个用户 7 天反馈 |

**当前最大未完成愿景（按产品价值排序）：**
1. **LLM Judge quality_gate 产品接线（#24）** — 底层 `check_quality_async` + LLM JSON judge + 拒绝率统计 + SQLite store 参数已经落地，但还没从 `MemoryManager.with_sqlite`/provider/config/factory 暴露开关，也缺 LLM 分支测试和成本/doctor 展示；下一步是把“能用的内核能力”变成默认可配置产品能力。
2. **端到端飞轮验证（#25）** — 需要补 10-turn 自动化验收：Shield→Nudge→Ripple→Lint→Recall 全链路通过，证明飞轮不是片段可用。
3. **v1.0 Gate 的发布工程** — CI 质量门、安装验证、文档站、外部用户试用仍未完成。
4. **Embedding 评估闭环** — 训练/评估模块已可用，检索日志也已 SQLite 化；但训练依赖未随产品 extra 完整声明，还没形成 CLI 一键 A/B + 自动选优，也没有 LOCOMO/等价基准门槛。

**2026-04-24 复利校正：安全上限误判已处理 ✅**
- 问题：`agent.max_iterations` 被配置校验硬性限制在 1000，`delegation.max_iterations` 被限制在 200，flywheel CLI 默认每轮只有 15 次 LLM/tool 迭代；另有真实运行层面的 gateway 隐藏 `_MAX_TOOL_CALLS = 80`，会在单次任务执行到 80 个工具调用时直接暂停。对普通短任务看似“安全”，但对元飞轮、全局审计、跨模块修复这类长期复利任务，会把深度执行误判为风险并提前截断。
- 决策：不保留拍脑袋的任意上限。安全不靠隐藏 hard cap，而靠用户/配置显式预算、进度汇报、测试验证、权限边界、stuck-loop 检测、idle/absolute timeout 和可观测性。
- 代码状态：`caveman/config/validator.py` 已支持 `(min, None)` 范围；`agent.max_iterations` / `delegation.max_iterations` 只有下限无任意上限；`caveman/cli/flywheel.py` 与 `caveman/cli/utility_commands.py` 的 flywheel 默认预算已提升到 50；`caveman/gateway/task_runner_helpers.py` 已移除隐藏 80 工具调用 hard cap；`caveman/config/default.yaml` 新增 `gateway.limits.max_tool_calls: null`，仅当用户显式设置正整数预算时才按工具调用数暂停，且文案明确“任务未判定完成，可继续/提高预算”。测试覆盖“大预算不报警”、flywheel 默认预算、默认 81+ 工具调用不中断、显式工具预算暂停、null 表示无限制。
- 原则：长期主义 + 最高复利 > 过度保守的局部安全感。预算应可调、可审计、可扩展，而不是阻断自进化。

### 7.3 下一步：Agent OS 内核路线图

> **排序原则：** 内核先稳定，用户态再丰富。先让 kernel 能 boot，再让 shell 好用。

#### Round 8 — 内核启动 ✅

| # | 做什么 | OS 类比 | 状态 |
|---|--------|---------|------|
| 1 | Feature Flags（引擎开关） | modprobe | ✅ |
| 2 | EventBus 持久化 | journald | ✅ |
| 3 | Compaction Shield | 进程状态保存 | ✅ |
| 4 | Nudge Phase1（规则提取） | 内存写入 | ✅ |
| 5 | Recall Engine | 进程恢复 | ✅ |
| 6 | Workspace 文件兼容 | /etc 读取 | ✅ |
| 7 | CLI Agent 调用 | exec() | ✅ |

**验收：** 自举测试通过 — Caveman 跑完整任务 → compaction → Shield 恢复 → 飞轮首次真实运行。

#### Round 9 — 内核稳定 ✅

| # | 做什么 | OS 类比 | 状态 |
|---|--------|---------|------|
| 6 | LLM 调度器（优先级+配额） | I/O 调度器 | ✅ |
| 7 | Nudge Phase2（LLM 精炼） | 内存页面整理 | ✅ |
| 8 | 飞轮健康度指标 | 性能监控 | ✅ |
| 9 | Lint Engine | GC | ✅ |

#### Round 10 — 用户态 ✅

| # | 做什么 | OS 类比 | 状态 |
|---|--------|---------|------|
| 10 | Ripple Engine | 页表关联 | ✅ |
| 11 | `__all__` 声明 + 模块 docstring | API 文档 | ✅ |
| 12 | Obsidian 兼容输出 | 文件系统驱动 | ✅ |

#### Round 11 — 自举验证 ✅

| # | 做什么 | 意义 | 状态 |
|---|--------|------|------|
| 13 | 用 Caveman 管理 Caveman 开发知识 | 第一个用户 = 自己 | ✅ |
| 14 | 端到端集成测试 | 接口稳定后写才不白写 | ✅ |
| 15 | loop.py 重构 | 代码稳定后重构才有意义 | ✅ |

#### Round 12 — 安全与交互 ✅

| # | 做什么 | 意义 | 状态 |
|---|--------|------|------|
| 16 | REPL 交互增强 | 开发体验 | ✅ |
| 17 | Bridge 5-Agent + PTY | 执行生态 | ✅ |
| 18 | 审计日志 | 可观测性 | ✅ |
| 19 | 沙箱 + E2E 加密 | 安全基线 | ✅ |

#### Round 13 — 数据基础设施

> **排序理由：** Schema 版本化是"记忆永续"的工程保障。没有它，后续所有涉及 schema 变更的 Round 都是在赌博。PRD §8.12 已设计完整迁移框架。

| # | 做什么 | OS 类比 | 依赖 | 验收标准 |
|---|--------|---------|------|---------|
| 20 | Schema 版本化 + 迁移框架 | 文件系统版本 | Round 12 ✅ | `schema_version` 表存在；`caveman migrate --dry-run` 输出 pending 迁移；迁移失败自动回滚 |
| 21 | `last_accessed` 提升为独立列 | — | #20 | 列存在；旧 metadata 数据回填完成；HybridScorer 直接读列而非解析 JSON |
| 22 | metadata well-known keys 校验 | — | #20 | `store()` 写入时校验 well-known keys 类型；非法类型 → WARNING 日志 |
| 23 | 检索日志（retrieval_log） | 审计追踪 | Round 12 ✅ | 每次 recall 记录 query/results/latency_ms 到 SQLite；`caveman doctor` 可查询 |

#### Round 14 — LLM Judge + 飞轮闭环验证

> **排序理由：** Heuristic scorer 是飞轮的瓶颈——它不能判断"这条记忆对任务有没有帮助"。LLM judge 是从"够用"到"真正有复利"的关键跳跃。

| # | 做什么 | 意义 | 依赖 | 验收标准 |
|---|--------|------|------|---------|
| 24 | LLM Judge 替代 heuristic quality_gate | 写入质量飞跃 | Round 13 ✅ | quality_gate 支持 LLM 模式；LLM 拒绝率 vs heuristic 拒绝率有统计对比 |
| 25 | 端到端飞轮验证 | 证明飞轮在转 | #24 | 自动化测试：10 轮对话 → Shield 保存 → Nudge 提取 → Ripple 传播 → Lint 审计 → Recall 恢复，全链路通过 |
| 26 | Confidence 闭环升级 | trust 信号更准 | #24 | LLM judge 的 helpful/unhelpful 判定直接写入 trust_score；替代旧的 ConfidenceTracker |

#### Round 15 — Embedding 训练管道

> **排序理由：** 飞轮验证通过后，积累的检索日志就是训练数据。embedding 微调让记忆检索从"关键词匹配"升级到"语义理解"。

| # | 做什么 | 意义 | 依赖 | 验收标准 |
|---|--------|------|------|---------|
| 27 | 检索日志 → 训练对生成 | 数据管道 | Round 14 ✅ | `caveman train --prepare` 从检索日志提取 query-memory 正负样本对 |
| 28 | 本地 embedding 微调 | 检索精度提升 | #27 | `caveman train --target embedding` 微调 nomic/bge 模型；LOCOMO 评估 ≥ 65% |
| 29 | A/B 评估框架 | 量化提升 | #28 | 微调前后 recall@5 对比报告；自动选择更优模型 |

#### Round 16 — 发布准备（v1.0 Gate）

> **排序理由：** 前面 3 个 Round 让内核稳定、飞轮可验证、检索可训练。这个 Round 解决"敢不敢让外部用户用"的问题。

| # | 做什么 | 意义 | 依赖 | 验收标准 |
|---|--------|------|------|---------|
| 30 | 安装体验优化 | 第一印象 | Round 15 ✅ | `pip install caveman && caveman setup` 在 macOS/Ubuntu/WSL2 上 < 3 分钟完成 |
| 31 | 文档站 + README 重写 | 开源门面 | #30 | README 有 Quick Start / Architecture / Contributing；文档站有 API 参考 |
| 32 | GitHub Actions CI | 质量门 | #30 | PR 自动跑 pytest + lint + type check；覆盖率 ≥ 80% |
| 33 | 首批外部用户测试 | 真实验证 | #31, #32 | 3 个非作者用户完成 Day 1-7 旅程；收集反馈 |

**v1.0 发布标准（Round 16 验收后）：**
- [ ] 安装 < 3 分钟（3 个 OS）
- [ ] 飞轮端到端测试通过
- [x] Schema 迁移框架可用（#20 已落地：`schema_version`、numbered migrations、`caveman migrate --dry-run/--apply`、失败回滚测试）
- [ ] 3 个外部用户完成 7 天试用
- [ ] README + 文档站上线
- [ ] CI 绿灯 + 覆盖率 ≥ 80%

**Round 13-16 明确不做的事（1 人团队生死线）：**

| 不做 | 为什么 |
|------|--------|
| 多用户/团队协作 | v1 只服务个人用户，团队功能是 v2 |
| Web UI / Dashboard | CLI + Telegram/Discord 够用，UI 是资源黑洞 |
| 自建 Hub 平台 | 先用 GitHub Releases，Hub 是有社区后的事 |
| 付费功能 | 1000+ 活跃用户前不考虑商业化 |
| Windows 原生支持 | WSL2 兜底，原生 Windows 是 v2 |

### 7.3 里程碑依赖图

```
Phase 1 (✅)     Round 8 (✅)      Round 9 (✅)      Round 10 (✅)     Round 11 (✅)     Round 12 (✅)
┌──────────┐    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ AgentLoop│    │ Flags    │     │ Scheduler│     │ Ripple   │     │ 自举验证 │     │ REPL增强 │
│ Memory v2│───→│ EventBus │────→│ Nudge P2 │────→│ __all__  │────→│ E2E测试  │────→│ Bridge5  │
│ Skill v2 │    │ Shield   │     │ Doctor v2│     │ Obsidian │     │ loop重构 │     │ 审计日志 │
│ CLI 14cmd │    │ Nudge P1 │     │ Lint     │     │ RL Router│     │ NFR合规  │     │ 沙箱加密 │
│ 3层压缩  │    │ Recall   │     │ Scorer   │     └──────────┘     └──────────┘     └──────────┘
│ Bridge   │    │ Workspace│     │ Config   │
│ Trajectory│   │ CLI Agent│     │ Import   │
│ 7轮优化  │    └──────────┘     └──────────┘
└──────────┘
                                                                                     ↓
Round 13          Round 14           Round 15           Round 16 (v1.0 Gate)
┌──────────┐     ┌──────────┐      ┌──────────┐      ┌──────────────┐
│ Schema版本│     │ LLM Judge│      │ 训练对生成│      │ 安装体验优化 │
│ last_accs │────→│ E2E飞轮  │─────→│ Embedding│─────→│ 文档站+README│
│ meta校验  │     │ Confidence│     │ A/B评估  │      │ CI + 外部测试│
│ 检索日志  │     │ 闭环升级  │      └──────────┘      └──────────────┘
└──────────┘     └──────────┘
```

### 7.4 人力与资源

| 资源 | 假设 | 实测 |
|------|------|------|
| 核心开发者 | 1人 | 确认 |
| AI 辅助比例 | 60-80% | **实测 >90%** |
| LLM API 费用 | ~$100-300/月 | Phase 1: ~$3-5 |
| 硬件 | MacBook M3 Max + VPS | 确认 |

---

## §8 技术实现

### 8.1 核心模块清单

| 模块 | 来源 | 职责 | 关键决策 |
|------|------|------|---------|
| Agent Loop | Hermes | 消息→推理→工具→学习 | 事件驱动6阶段，复杂度7 |
| Coordinator | Claude Code | 多worker编排 | 4阶段流水线 |
| Verification | Claude Code | 反合理化验证 | 6借口封堵 + Guide+Sensor |
| Compression | Claude Code | 3层上下文压缩 | Micro/Normal/Smart |
| Memory | 三者融合 | 多provider + 4类型 + 漂移检测 | SQLite+FTS5 默认 |
| Training | Hermes + veRL | 轨迹导出 + embedding 训练 | 导出为主，embedding 微调为辅 |
| Bridge | 新创 | Python↔Node.js | UDS 2.3μs + MCP + HTTP |
| ACP | OpenClaw | 编排外部编码agent | Codex/Claude Code/Gemini |
| Browser | OpenClaw | Playwright 自动化 | snapshot→act |
| Gateway | OpenClaw | 消息路由 | 7+平台 |
| Skills | 三者融合 | AgentSkills + 自动创建 | RL路由 + 质量门 |
| Tools | 三者融合 | 90 内置工具（first-wins 去重） | @tool声明式注册 |

### 8.2 出厂预装经验

> 别人的 Caveman 从零开始学走路。我们的 Caveman 出厂就会跑。

**6 个内置技能（~1,700 行）：**

| Skill | 核心能力 | Harness 角色 |
|-------|---------|-------------|
| debug-methodology | 5步调试法 + PUA压力升级 + 根因铁律 | Sensor: 卡住时触发 |
| project-arch-core | 语言无关架构原则 + 文件分拆标准 | Guide: 写代码前 |
| codex-review | 三级代码质量防线 | Sensor: 代码完成后 |
| subagent-review | 4维并行审查 | Sensor: 子agent返回后 |
| bug-audit | 基于项目代码构建检查矩阵 | Sensor: 审计触发 |
| project-sop | 开工确认→干活→更新HANDOFF | Guide: 开发任务前 |

**3 套方法论（~940 行）：** critical-analysis / validated-approaches / multi-agent-quickstart

**3 套编排规则（~290 行）：** coding.md / spawn.md / safety.md

**SKILL.md 规范（AgentSkills 兼容 + Caveman 扩展）：**

```yaml
name: debug-methodology          # 唯一标识 (kebab-case)
version: 1.2.0                   # 语义化版本
description: |                   # 触发描述（匹配器使用）
  Systematic debugging with PUA-style pressure escalation.
triggers:                        # 触发条件（OR 逻辑）
  - pattern: "failed 2+ times"
  - sentiment: frustration
harness_role: sensor             # guide | sensor | hybrid
permissions: [file.read, file.write, terminal.exec]
tests:                           # 单元测试（门控）
  - input: "代码报错了，试了3次"
    expect_trigger: true
quality_score: 0.85              # 自动评分 (0-1)
created_by: bundled              # bundled | auto | user | hub
```

ClawHub / Hermes 技能可直接导入（AgentSkills 标准兼容）。Caveman 扩展字段可选。

### 8.3 CLI 入口

| 命令 | 描述 | 状态 |
|------|------|------|
| `caveman run "task"` | 单次任务执行 | ✅ |
| `caveman` | CLI 交互模式 | ✅ |
| `caveman serve` | 启动消息网关 | ✅ |
| `caveman setup` | 配置向导 | ✅ |
| `caveman tools` | 列出工具 | ✅ |
| `caveman skills` | 管理技能 | ✅ (list) |
| `caveman export` | 导出轨迹 | ✅ |
| `caveman train` | 数据导出 + embedding 训练 | ✅ |
| `caveman doctor` | 飞轮健康检查 | ✅ |
| `caveman version` | 版本信息 | ✅ |
| `caveman import` | 从 OpenClaw/Hermes/Claude Code 导入数据 | ✅ |
| `caveman adapt-workspace` | 适配工作区配置 | ✅ |
| `caveman hub` | Skills Hub 交互 | ✅ |
| `caveman plugins` | 插件管理 | ✅ |

### 8.4 异常处理

设计原则：永不静默失败 / 降级而非崩溃 / 自动恢复优先 / 用户始终掌控。

| 异常 | 用户看到什么 | 自动处理 |
|------|------------|---------|
| 网络断开 | ⚠️ 本地工具仍可用，联网功能暂停 | 联网请求排队等恢复 |
| LLM 超时 | ⏳ 重试中(1/3)... | 3次退避重试（5s→15s→30s） |
| API 余额不足 | 💰 切换免费模型 | Credential Pool 自动轮转备用 key |
| API 429 限流 | 🔄 等待 {retry_after}s | 自动等待 Retry-After；有 Pool 时轮转 |
| 模型拒绝（安全） | 🚫 可以换个说法或切换模型 | 记录到轨迹（训练信号）；不自动重试 |
| 工具执行失败 | ❌ [tool] 失败: exit code 1 | 显示错误输出；agent 自动分析并建议修复 |
| Bridge 断开 | 🔌 核心功能正常 | 自动重启，3次失败→降级纯Python |
| 磁盘空间不足 | 💾 轨迹记录暂停，运行 caveman gc | 暂停轨迹（核心不停）；提示清理 |
| 记忆数据库损坏 | 🔧 从备份恢复中... | 自动恢复；备份也坏→重建索引 |
| 权限被拒 | 🔒 [Y]允许 [n]拒绝 [a]始终允许 | 等待确认；超时30s→自动拒绝 |

### 8.5 技术约束

| 约束 | 决策 | 理由 |
|------|------|------|
| 主语言 | Python 3.12+ | 记忆/技能/embedding 生态原生 |
| 桥接语言 | Node.js 22+ | OpenClaw 执行层已成熟 |
| 最小硬件 | $5 VPS (1核/1GB) | 不排斥低端用户 |
| 模型无关 | OpenRouter 400+ | 不绑定任何 provider |
| 开源协议 | MIT | 最大生态兼容 |
| 数据库 | SQLite + FTS5 | 零依赖，单文件，10ms检索 |
| 技能标准 | AgentSkills | 兼容 ClawHub/Hermes 生态 |
| 轨迹格式 | ShareGPT JSONL | 研究者训练工具链通用格式 |

### 8.6 Bridge 层设计

Python↔Node.js 桥接是 Caveman 的独有创新：

| 协议 | 延迟 | 用途 |
|------|------|------|
| UDS JSON-RPC | 2.3μs p50 | 主通道（同机高频） |
| MCP (stdio) | ~1-5ms | 工具层（标准化） |
| HTTP REST | ~1-10ms | 兜底（跨机/降级） |
| WebSocket | <100ms | OpenClaw Gateway 直连（绕过 CLI 子进程 ~3s 开销） |

**降级策略：** Node.js未安装→纯Python / UDS不可用→WebSocket→HTTP / Bridge崩溃→自动重启3次→降级

### 8.7 实战发现的约束（2026-04-13）

| 约束 | 发现方式 | 应对 |
|------|---------|------|
| pyenv 不预装 | macOS 无 pyenv/brew，只有 Python 3.9.6 | 安装脚本检测 + `curl pyenv.run` 兜底 |
| `lzma` 模块缺失 | pyenv 编译时 `_lzma` 未编译 | 标记 non-critical warning |
| Anthropic 代理模型名 | 第三方代理要求精确 ID（`claude-opus-4-6`） | factory.py model alias 映射 |
| 子 agent 沙箱隔离 | `sessions_spawn` 默认沙箱，文件蒸发 | **必须 `sandbox: "inherit"`** |
| async generator 兼容 | Anthropic SDK streaming 非标准 async generator | Provider 层统一适配 |
| Tool schema 格式差异 | Anthropic `input_schema` vs OpenAI `parameters` | ToolRegistry 统一输出，provider 层转换 |


### 8.8 生态兼容层

> **设计原则：** Caveman 不是孤岛。用户可能已经在用 OpenClaw / Hermes / Claude Code，迁移成本必须趋近于零。

#### 8.8.1 Workspace 文件兼容（P0）

**防止什么：** 用户已有的 SOUL.md / USER.md / MEMORY.md / AGENTS.md 等 workspace 文件无法被 Caveman 读取，导致人格/偏好/记忆全部丢失。

| 文件 | OpenClaw | Hermes | Caveman 兼容策略 |
|------|----------|--------|------------------|
| SOUL.md | ✅ 人格定义 | ❌ | 读取并注入 Layer 0 system prompt |
| USER.md | ✅ 用户画像 | ❌ | 读取并注入 Layer 0 |
| MEMORY.md | ✅ 长期记忆 | ❌ | 读取 + 导入到 Memory Store |
| AGENTS.md | ✅ 行为规则 | ❌ | 读取并注入 Layer 0（规则段） |
| HEARTBEAT.md | ✅ 定时检查 | ❌ | 读取 + 注册为 cron 任务 |
| TOOLS.md | ✅ 工具指南 | ❌ | 读取并注入 Layer 1 |
| IDENTITY.md | ✅ 身份补充 | ❌ | 合并到 SOUL.md 注入 |
| memory/*.md | ✅ 分类记忆 | ❌ | 批量导入到 Memory Store |
| skills/ | ✅ AgentSkills | ✅ SKILL.md | 直接兼容（已实现格式转换） |

**实现方案：** `prompt.py` 的 `build_system_prompt()` 增加 workspace 扫描层：
1. 检测 `~/.openclaw/workspace/` 或 `~/.caveman/workspace/` 或当前目录
2. 按优先级读取 workspace 文件
3. 注入到 system prompt 的对应 Layer

**迁移路径：**
```
已有 OpenClaw 用户:
  ~/.openclaw/workspace/ → Caveman 直接读取（零配置）
  
已有 Hermes 用户:
  caveman import --from hermes  → 导入记忆+技能+轨迹
  
全新用户:
  caveman setup → 创建 ~/.caveman/workspace/ + 引导填写 SOUL.md/USER.md
```

#### 8.8.2 配置格式兼容（P1）

**防止什么：** 用户已有的 OpenClaw/Hermes 配置无法复用，需要重新配置所有 API key、模型偏好、网关设置。

| 配置源 | 路径 | Caveman 读取 |
|--------|------|-------------|
| OpenClaw | `~/.openclaw/config.yaml` | ✅ 提取 providers / gateway / model 配置 |
| Hermes | `~/.hermes/config.yaml` | ✅ 提取 providers / memory / skills 配置 |
| Caveman | `~/.caveman/config.yaml` | ✅ 原生（最高优先级） |
| 环境变量 | `ANTHROPIC_API_KEY` 等 | ✅ 兜底 |

**优先级：** Caveman 原生 > 环境变量 > OpenClaw > Hermes

**实现：** `config/loader.py` 增加 `_discover_external_configs()` 方法，首次启动时自动检测并提示导入。

#### 8.8.3 编码 Agent CLI 调用（P0）

**防止什么：** Caveman 的 Coordinator 无法真正调用外部编码 Agent，只能自己写代码。

| Agent | 调用方式 | 参数 | 状态 |
|-------|---------|------|------|
| Claude Code | `claude --print --permission-mode bypassPermissions -p "task"` | 非交互，一次性输出 | ✅ pipe模式 |
| Codex CLI | PTY 模式，`codex "task"` | 交互式，需 PTY | ✅ PTY模式 |
| Gemini CLI | PTY 模式，`gemini "task"` | 交互式，需 PTY | ✅ PTY模式 |
| OpenClaw ACP | `sessions_spawn(runtime="acp", agentId=...)` | 通过 Bridge | 🔸 骨架已有 |
| Pi (OpenCode) | PTY 模式 | 交互式 | ✅ PTY模式 |

**两种调用模式：**

1. **直接 CLI 调用**（Caveman 自己 spawn 进程）：
   - Claude Code: `--print` 模式，非交互，适合明确任务
   - Codex/Gemini/Pi: PTY 模式，交互式，适合探索性任务
   - 超时控制：🟢简单 300s / 🟡中等 900s / 🔴复杂 1500s

2. **通过 OpenClaw ACP 调用**（委托 OpenClaw 编排）：
   - 适合已有 OpenClaw 实例的用户
   - 利用 OpenClaw 的 thread-bound session、审批流、日志
   - Bridge 层透传 `sessions_spawn` 参数

**实现：** `bridge/cli_agents.py` 新模块：
```python
class CLIAgentRunner:
    """Spawn and manage external coding agent CLI processes."""
    
    async def run_claude_code(self, task: str, cwd: str, timeout: int = 900) -> str:
        """Non-interactive: claude --print --permission-mode bypassPermissions"""
        
    async def run_codex(self, task: str, cwd: str, timeout: int = 900) -> str:
        """PTY mode: codex 'task'"""
        
    async def run_gemini(self, task: str, cwd: str, timeout: int = 900) -> str:
        """PTY mode: gemini 'task'"""
```

#### 8.8.4 记忆格式兼容（P1）

**防止什么：** 用户在 OpenClaw/Hermes 积累的记忆无法迁移到 Caveman，飞轮从零开始。

| 来源 | 格式 | 导入策略 |
|------|------|----------|
| OpenClaw memory/*.md | Markdown 文件 | 解析 → 按日期/项目分类 → 写入 Memory Store |
| OpenClaw MEMORY.md | 结构化 Markdown | 解析段落 → 提取 facts/preferences → 写入 |
| OpenClaw sessions/*.jsonl | JSONL 对话记录 | 三层管线提取知识（见 §8.8.4.1）→ 写入 Memory Store |
| Hermes memories | JSON/SQLite | 读取 → 格式转换 → 写入 Memory Store |
| Codex rollout_summaries/ | Markdown | 解析 → 提取 procedural 记忆 → 写入 |
| Claude Code CLAUDE.md | Markdown | 解析 → 提取项目知识 → 写入 |

**CLI 入口：**
```bash
caveman import --from openclaw          # 导入 OpenClaw workspace + 记忆
caveman import --from openclaw-sessions # 从 session 对话中提取知识（用户 opt-in）
caveman import --from hermes            # 导入 Hermes 记忆 + 技能
caveman import --from codex             # 导入 Codex rollout summaries
caveman import --from claude-code       # 导入 CLAUDE.md 项目知识
caveman import --from directory ./path  # 导入任意 markdown 目录
```

##### 8.8.4.1 Session 知识提取（openclaw-sessions）

**解决什么：** OpenClaw 的 session 对话中包含大量高价值知识（研究结论、根因分析、架构决策、解决方案），但这些知识只存在于 JSONL 对话记录中，不在 memory/ 文件系统里。现有 import 系统无法触及。

**设计原则：**
- **用户自主决定** — opt-in，不自动执行
- **质量优先** — 宁可漏掉，不导入噪音
- **完整溯源** — 每条记忆带 session_id、日期、分类、评分
- **幂等去重** — 基于 content hash，重复导入不产生重复数据
- **LLM 可选** — 规则提取兜底，LLM 增强质量

**三层提取管线：**
1. **解析层**（`session_parser.py`）— JSONL → 结构化 ConversationTurn（user/assistant/toolCall/toolResult）
2. **过滤层**（`session_extractor.py`）— 对每个 turn 打分（正向信号：根因分析/架构/研究/教训/决策；负向信号：简单确认/heartbeat/filler），score ≥ 2.0 才提取
3. **提取层**（`session_extractor.py`）— 聚类相关 turn → 规则提取或 LLM 摘要 → 分类写入 Memory Store

**知识分类 → 记忆类型映射：**
- research / architecture / decision / reference → SEMANTIC
- diagnosis / status → EPISODIC
- solution / lesson → PROCEDURAL

**CLI 选项：**
```bash
caveman import --from openclaw-sessions --dry-run          # 预览
caveman import --from openclaw-sessions --use-llm          # LLM 增强提取
caveman import --from openclaw-sessions --since 2026-04-01 # 日期过滤
caveman import --from openclaw-sessions --session <id>     # 指定 session
caveman import --from openclaw-sessions --min-score 3.0    # 调整阈值
```

**实测数据（2026-04-18）：** 124 个 session → 1342 turn → 810 可提取 → 507 条知识（research 141 / diagnosis 94 / solution 94 / architecture 69 / reference 48 / status 28 / decision 27 / lesson 6）

**实现：** `caveman/import_/session_parser.py` + `session_extractor.py` + `openclaw_sessions.py`，35 个专项测试。

#### 8.8.5 兼容层现状总结

| 兼容维度 | 代码状态 | PRD 优先级 | Round |
|---------|---------|-----------|-------|
| Bridge 骨架（OpenClaw MCP + Hermes REST + ACP） | ✅ 已实现 | — | Phase 1 ✅ |
| 技能格式转换（SKILL.md ↔ Hermes ↔ Caveman） | ✅ 已实现 | — | Phase 1 ✅ |
| Workspace 文件读取 + prompt 注入 | ✅ 已实现 | P0 | Round 8 ✅ |
| CLI Agent 调用（Claude Code/Codex/Gemini） | ✅ 已实现 | P0 | Round 8 ✅ |
| 配置格式兼容 + 自动检测 | ✅ 已实现 | P1 | Round 9 ✅ |
| Session 知识提取（openclaw-sessions） | ✅ 已实现 | P1 | Round 130+ ✅ |


### 8.9 数据模型（Schema 定义）

> **防止什么：** 记忆永续是空话——没有版本化的 schema，升级就是赌博。metadata 是黑盒 dict，任何引擎都能往里塞垃圾，最终污染飞轮。

#### 8.9.1 memories 表（Memory Store 核心）

```sql
-- Schema Version: 1 (v0.4.0)
-- 文件: caveman/memory/sqlite_store.py
CREATE TABLE memories (
    id              TEXT PRIMARY KEY,           -- UUID v4
    content         TEXT NOT NULL,              -- 记忆正文（质量门控：15-3000 字符）
    type            TEXT NOT NULL,              -- MemoryType 枚举值: episodic|semantic|procedural|working
    created_at      TEXT NOT NULL,              -- ISO 8601 时间戳
    metadata_json   TEXT DEFAULT '{}',          -- 结构化元数据（见 §8.9.2）
    trust_score     REAL DEFAULT 0.5,           -- 信任分 [0.0, 1.0]，飞轮核心信号
    retrieval_count INTEGER DEFAULT 0,          -- 被检索次数（复利指标：用得越多→排名越高→被用得更多）
    helpful_count   INTEGER DEFAULT 0,          -- 被标记有用的次数
    entities_json   TEXT DEFAULT '[]'           -- 自动提取的实体列表（人名/路径/IP/项目名）
);

-- FTS5 全文搜索索引（自动同步，触发器维护）
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, content=memories, content_rowid=rowid
);

-- 向量嵌入表（可选，有 embedding_fn 时启用）
CREATE TABLE embeddings (
    memory_id  TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector_json TEXT NOT NULL                   -- JSON 数组，维度取决于 embedding 模型
);
```

**PRAGMA 配置：**
- `journal_mode=WAL` — 读写并发，崩溃安全
- `foreign_keys=ON` — embeddings 级联删除

**trust_score 生命周期：**

| 事件 | delta | 来源 |
|------|-------|------|
| 新建 | 0.5（默认） | `sqlite_store.store()` |
| 被检索（正常） | +0.01 | `sqlite_store.recall()` 微信号 |
| 被检索（复活：trust < 0.3） | +0.05 | `sqlite_store.recall()` 复活机制 |
| 用户标记有用 | +0.08 | `retrieval.adjust_trust()` |
| 用户标记无用 | -0.10 | `retrieval.adjust_trust()` |
| Reflect 确认准确 | +0.20 | `confidence.REFLECT_CONFIRM` |
| Lint 标记过时 | -0.30 | `confidence.LINT_STALE` |
| 下限 | 0.0 | 不删除，只沉底 |
| 上限 | 1.0 | 硬顶 |

#### 8.9.2 metadata_json 规范（Well-Known Keys）

metadata_json 是可扩展的 JSON 对象。以下 key 有明确语义，引擎依赖它们：

| Key | 类型 | 写入者 | 语义 |
|-----|------|--------|------|
| `trust_score` | float | store/recall | 冗余副本（主值在列上），用于 MemoryEntry 传递 |
| `source` | string | Nudge | MemorySource 枚举值: user/feedback/project/reference |
| `last_accessed` | string (ISO 8601) | recall() | 最后被检索的时间，HybridScorer temporal_decay 使用 |
| `retrieval_count` | int | recall() | 冗余副本，用于 MemoryEntry 传递 |
| `related` | list[string] | Ripple | 关联记忆 ID 列表（知识网络的边） |
| `supersedes` | string | Ripple | 被本条取代的旧记忆 ID |
| `conflict_with` | string | Ripple/Drift | 与本条矛盾的记忆 ID |
| `grounding_status` | string | Grounding Gate | verified/unverified/stale |
| `grounding_checked_at` | string (ISO 8601) | Grounding Gate | 最后验证时间 |
| `_fts_rank` | float | recall() 内部 | FTS5 排名分（临时，不持久化） |
| `_vector_sim` | float | recall() 内部 | 向量相似度（临时，不持久化） |
| `nudge_trigger` | string | Nudge | 触发提取的事件: shield_update/tool_error/loop_end |
| `confidence` | float | ConfidenceTracker (legacy) | 旧版信任分，v0.5 移除 |

**扩展规则：** 以 `_` 开头的 key 是临时/内部的，不保证跨版本兼容。其他 key 自由扩展但不保证被引擎读取。

#### 8.9.3 events 表（EventStore 持久化）

```sql
-- 文件: caveman/events_store.py
CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT    NOT NULL,                -- EventType 枚举值（见 §8.10）
    data_json  TEXT    NOT NULL DEFAULT '{}',   -- 事件 payload（见 §8.10）
    timestamp  REAL    NOT NULL,                -- Unix 时间戳（time.time()）
    source     TEXT    NOT NULL DEFAULT '',      -- 发射模块: loop/tool/shield/nudge/skill/memory
    session_id TEXT    NOT NULL DEFAULT '',      -- 所属 session ID
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))  -- SQLite 写入时间
);

CREATE INDEX idx_events_type_ts ON events (event_type, timestamp);
CREATE INDEX idx_events_session ON events (session_id, timestamp);
```

#### 8.9.4 session_essence（Shield 持久化，YAML）

```yaml
# 文件: ~/.caveman/sessions/{session_id}.yaml
# 数据类: caveman.engines.shield.SessionEssence
session_id: "a1b2c3d4e5f6"          # 12 位 hex
decisions:                            # 最近 20 条决策（what + why）
  - "选择 SQLite+FTS5 而非 PostgreSQL — 零依赖，单文件"
progress:                             # 最近 20 条进度
  - "完成 Round 12 安全与交互"
stances:                              # 最近 10 条立场
  - "长期主义+最高复利，摈弃够用就行的垃圾思维"
key_data:                             # 关键数据点（flat dict）
  path_0: "/Users/yeren64g/projects/caveman"
  version_1: "0.4.0"
  tests_passed_2: "1499"
open_todos:                           # 最近 20 条待办
  - "LLM judge 替代 heuristic scorer"
updated_at: "2026-04-18T19:44:27"     # ISO 8601
turn_count: 15                        # assistant 回复轮数
task: "PRD 补强"                      # 当前任务描述
```

#### 8.9.5 project_identity（Shield 子结构，JSON）

```json
// 文件: ~/.caveman/projects/{project_name}.json
// 数据类: caveman.engines.project_identity.ProjectIdentity
{
  "name": "caveman",
  "path": "~/projects/caveman",
  "mission": "做出比 OpenClaw/Hermes 更好用的 Agent",
  "principles": ["长期主义", "写入质量 >> 检索复杂度"],
  "current_phase": "Round 13 — PRD 补强",
  "tech_stack": ["Python 3.12", "SQLite+FTS5", "Anthropic"],
  "key_identifiers": {"repo": "caveman", "loc": "28909"},
  "updated_at": "2026-04-18T19:44:27",
  "session_count": 42
}
```

#### 8.9.6 Schema 版本管理

当前 schema 没有版本号，`migrate_schema()` 通过 `PRAGMA table_info` 检测缺失列并 `ALTER TABLE ADD COLUMN`。这在 v0.4.0 够用，但不可持续。

**v0.5.0 升级计划（见 §8.12）：** 引入 `schema_version` 表 + 编号迁移脚本。

### 8.10 关键接口契约（Event Payload Schema）

> **防止什么：** 引擎之间传什么数据全靠读代码。新引擎接入飞轮时不知道该 emit 什么、该 subscribe 什么。

**设计原则：只定义跨引擎边界的接口，不定义引擎内部实现。**

#### 8.10.1 EventType 枚举（28 种事件）

| 事件 | 发射者 | 消费者 | 语义 |
|------|--------|--------|------|
| `loop.start` | AgentLoop | Metrics, Audit | 任务开始 |
| `loop.end` | AgentLoop | Nudge(LOOP_END触发), Metrics, Trajectory | 任务结束 |
| `iteration.start` | AgentLoop | Metrics | 单轮推理开始 |
| `iteration.end` | AgentLoop | Metrics | 单轮推理结束 |
| `llm.request` | AgentLoop | Metrics(计时开始) | LLM API 调用发出 |
| `llm.response` | AgentLoop | Metrics(计时结束) | LLM API 返回 |
| `llm.stream.delta` | Provider | Display | 流式 token 片段 |
| `llm.error` | Provider | Metrics(错误计数) | LLM 调用失败 |
| `tool.call` | tools_exec | Metrics(计时开始), Audit | 工具调用发出 |
| `tool.result` | tools_exec | Metrics(计时结束), Audit | 工具调用返回 |
| `tool.error` | tools_exec | Nudge(立即提取), Metrics | 工具调用失败 |
| `memory.recall` | AgentLoop | Confidence(反馈), Metrics | 记忆被检索 |
| `memory.store` | MemoryManager | Lint(增量检查), Metrics | 新记忆写入 |
| `memory.nudge` | Nudge | Metrics | Nudge 提取完成 |
| `skill.match` | AgentLoop | Metrics | 技能匹配成功 |
| `skill.nudge` | tools_exec | SkillManager | 技能检测触发 |
| `skill.outcome` | SkillManager | Metrics | 技能执行结果 |
| `context.compress` | Compression | Metrics | 上下文压缩执行 |
| `context.utilization` | AgentLoop | Display | 上下文使用率 |
| `trajectory.turn` | Trajectory | — | 单轮轨迹记录 |
| `trajectory.save` | Trajectory | — | 轨迹文件保存 |
| `shield.update` | AgentLoop | Nudge(节流≥3轮), Metrics | Shield 精华更新 |
| `nudge.extract` | Nudge | Ripple(日志), Metrics | Nudge 提取完成 |
| `ripple.propagate` | Ripple | Metrics | 知识传播执行 |
| `lint.scan` | Lint | Metrics | 审计扫描执行 |
| `permission.check` | Security | Audit | 权限检查 |
| `usage.turn` | AgentLoop | Metrics | 单轮 token 用量统计 |
| `secret.detected` | Security | Audit | 密钥泄露检测 |

#### 8.10.2 关键事件 Payload 定义

**内层飞轮事件链（Shield → Nudge → Ripple → Lint）：**

```python
# shield.update — AgentLoop 每轮结束后发射
{"session_id": str, "turn_count": int}

# nudge.extract — Nudge 提取完成后发射
{"count": int, "types": list[str], "trigger": "shield_update"|"tool_error"|"loop_end"}

# ripple.propagate — Ripple 传播完成后发射
{"source_id": str, "related_count": int, "conflicts": int}

# lint.scan — Lint 扫描完成后发射
{"scanned": int, "stale": int, "contradictions": int, "orphans": int}
```

**工具调用事件：**

```python
# tool.call — 工具调用发出
{"name": str, "call_id": str, "args_keys": list[str]}

# tool.result / tool.error — 工具调用返回
{"name": str, "call_id": str, "is_error": bool, "result_len": int}
```

**记忆事件：**

```python
# memory.recall — 记忆被检索
{"query": str, "results": int, "recalled_ids": list[str], "recall_hit": bool}

# memory.store — 新记忆写入
{"memory_id": str, "type": str, "source": str}  # source: "nudge"|"user"|"import"
```

**生命周期事件：**

```python
# loop.start
{"task": str}

# loop.end
{"task": str, "result_len": int, "iterations": int, "tool_calls": int}

# iteration.start / iteration.end
{"iteration": int, "stop": str, "tool_calls": int, "text_len": int}
```

#### 8.10.3 内层飞轮事件链（wire_inner_flywheel）

```
SHIELD_UPDATE ──→ Nudge.run()（节流：≥3 轮间隔）
                      │
                      ├─ 提取成功 → emit NUDGE_EXTRACT
                      │                  │
                      │                  └─ Ripple 自动触发（via memory.store()）
                      │
TOOL_ERROR ─────→ Nudge.run()（立即，无节流）
                      │
                      └─ emit NUDGE_EXTRACT

MEMORY_STORE ───→ Lint 标记待检查（下次扫描时处理）

LOOP_END ───────→ Nudge.run()（必定提取）+ Shield.save() + Trajectory.save()
```

#### 8.10.4 HybridScorer 接口

```python
# 5 信号加权排名
HybridScorer(
    fts_weight=0.30,      # FTS5 全文匹配
    jaccard_weight=0.25,  # Jaccard 词集相似度
    vector_weight=0.20,   # 向量余弦相似度（可选）
    trust_weight=0.25,    # 信任分（飞轮核心信号）
    temporal_half_life_days=90,  # 时间衰减半衰期
)

# 上下文感知排名
scorer.set_context(query)  # debug→procedural优先, design→semantic优先, coding→episodic优先
```

### 8.11 可观测性架构

> **防止什么：** 飞轮是否在加速？不知道。Agent 性能在退化？不知道。Hermes 和 OpenClaw 都只有"能不能用"的可观测性，没有"飞轮是否在产生复利"的度量。

#### 8.11.1 三层可观测性

| 层 | 做什么 | 实现 | 状态 |
|----|--------|------|------|
| L1 日志 | 结构化事件日志 | `logging_subscriber` → EventBus `on_all` | ✅ |
| L2 指标 | 性能 + 飞轮健康 | `MetricsCollector` + `FlywheelHealth` | ✅ |
| L3 告警 | 关键异常通知 | Gateway 消息推送 | 🔸 待实现 |

#### 8.11.2 L1 结构化日志

```
# 格式: Python logging, DEBUG 级别
event=tool.call source=tool data_keys=['name', 'call_id', 'args_keys']
event=shield.update source=shield data_keys=['session_id', 'turn_count']
```

**日志级别约定：**
- `DEBUG`: 所有 EventBus 事件（通过 `logging_subscriber`）
- `INFO`: 引擎关键动作（Nudge 提取 N 条、Shield 保存、Lint 发现 N 个问题）
- `WARNING`: 可恢复错误（Handler 失败、LLM 重试、Bridge 断开）
- `ERROR`: 不可恢复错误（数据库损坏、配置缺失）

#### 8.11.3 L2 指标体系

**Agent 性能指标（MetricsCollector）：**

| 指标 | 计算方式 | 查询 |
|------|---------|------|
| tool_duration_avg / _p95 | TOOL_CALL → TOOL_RESULT 时间差 | `metrics.snapshot()` |
| llm_latency_avg / _p95 | LLM_REQUEST → LLM_RESPONSE 时间差 | `metrics.snapshot()` |
| total_errors | TOOL_ERROR + LLM_ERROR 计数 | `metrics.snapshot()["counters"]` |
| events_emitted | EventBus 总发射数 | `bus.stats` |
| handlers_failed | Handler 异常数 | `bus.stats` |

**飞轮健康指标（FlywheelHealth）：**

| 指标 | 计算方式 | 健康阈值 |
|------|---------|---------|
| total_memories | `COUNT(*) FROM memories` | > 0 |
| avg_trust | `AVG(trust_score) FROM memories` | ≥ 0.4 |
| trust_distribution | 5 桶直方图 (dead/low/default/good/excellent) | 非全部集中在 default |
| memories_never_recalled | `retrieval_count = 0` 的数量 | < 50% |
| feedback_rate | `helpful_count > 0` 的比例 | ≥ 10% |
| top_recalled | 检索次数最高的 N 条 | 存在即健康 |

**查询入口：**
- CLI: `caveman doctor` → 输出 6 项指标 + 诊断建议
- Agent 自查: `metrics` 工具 → 返回 agent/flywheel/provider 三类指标
- 编程: `FlywheelHealth.diagnose(backend)` / `MetricsCollector.snapshot()`

#### 8.11.4 L3 告警规则（v0.5.0+）

| 告警 | 条件 | 通道 | 优先级 |
|------|------|------|--------|
| 飞轮停转 | 连续 7 天 nudge 提取 0 条 | Gateway 消息 | P0 |
| 记忆库膨胀 | total_memories > 10000 且 avg_trust < 0.3 | Gateway 消息 | P1 |
| LLM 费用异常 | 单日 token 消耗 > 日均 3 倍 | Gateway 消息 | P1 |
| Shield 恢复失败 | Recall 加载 essence 失败 | 日志 WARNING | P0 |
| 进程崩溃 | systemd/launchd watchdog 触发 | Gateway 消息 | P0 |

### 8.12 升级与迁移策略

> **防止什么：** "记忆永续"是 Caveman 的核心承诺。如果升级破坏了用户的记忆数据库，这个承诺就是谎言。Hermes 升级经常破坏记忆格式，OpenClaw 的 compaction safeguard 就是因为升级丢数据才加的。

#### 8.12.1 当前迁移机制（v0.4.0）

```python
# caveman/memory/store_helpers.py
def migrate_schema(conn):
    """检测缺失列，ALTER TABLE ADD COLUMN。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    for col, sql in [
        ("trust_score", "ALTER TABLE memories ADD COLUMN trust_score REAL DEFAULT 0.5"),
        ("retrieval_count", "ALTER TABLE memories ADD COLUMN retrieval_count INTEGER DEFAULT 0"),
        ("helpful_count", "ALTER TABLE memories ADD COLUMN helpful_count INTEGER DEFAULT 0"),
        ("entities_json", "ALTER TABLE memories ADD COLUMN entities_json TEXT DEFAULT '[]'"),
    ]:
        if col not in existing:
            conn.execute(sql)
```

**局限：** 只能加列，不能改列/删列/改类型/改索引。没有版本号，无法知道当前 schema 是哪个版本。

#### 8.12.2 目标迁移框架（v0.5.0）

```
caveman/memory/migrations/
├── __init__.py          # 迁移注册表
├── v001_initial.py      # v0.4.0 基线 schema
├── v002_add_last_accessed.py   # 将 last_accessed 从 metadata 提升为列
├── v003_schema_version.py      # 引入 schema_version 表
└── ...
```

**schema_version 表：**

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT NOT NULL DEFAULT ''
);
```

**迁移脚本规范：**

```python
# caveman/memory/migrations/v002_add_last_accessed.py
VERSION = 2
DESCRIPTION = "将 last_accessed 从 metadata_json 提升为独立列"

def up(conn):
    conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT DEFAULT ''")
    # 从 metadata_json 回填
    rows = conn.execute("SELECT id, metadata_json FROM memories").fetchall()
    for row in rows:
        meta = json.loads(row[1]) if row[1] else {}
        if "last_accessed" in meta:
            conn.execute("UPDATE memories SET last_accessed = ? WHERE id = ?",
                         (meta["last_accessed"], row[0]))

def down(conn):
    # SQLite 不支持 DROP COLUMN（3.35.0 之前），标记为 deprecated
    pass
```

#### 8.12.3 迁移原则

| # | 原则 | 理由 |
|---|------|------|
| 1 | **只加不删** | 旧字段标 deprecated，不删除。SQLite ALTER TABLE 限制 + 回滚安全 |
| 2 | **迁移前自动备份** | `cp memories.db memories.db.bak.{timestamp}` |
| 3 | **迁移失败自动回滚** | 事务包裹，失败时 ROLLBACK + 恢复备份 |
| 4 | **支持 dry-run** | `caveman migrate --dry-run` 只报告变更不执行 |
| 5 | **启动时自动迁移** | `_get_conn()` 检查版本号，执行 pending migrations |
| 6 | **向前兼容** | 新版本代码能读旧版本数据（新列有 DEFAULT） |

#### 8.12.4 跨大版本迁移（v0.x → v1.0）

| 迁移项 | 策略 |
|--------|------|
| memories 表 | 自动迁移脚本链（v001 → v002 → ... → vN） |
| session_essence YAML | 向前兼容（`from_dict` 忽略未知字段，缺失字段用默认值） |
| project_identity JSON | 向前兼容（`from_dict` 同上） |
| events 表 | 不迁移（事件是日志，不是状态） |
| 技能 SKILL.md | 格式转换器（v1 → v2 字段映射） |
| config.yaml | 向前兼容 + `caveman setup --upgrade` 交互式升级 |

#### 8.12.5 数据安全网

```
升级流程:
  1. caveman upgrade --check     → 检测当前版本 + pending migrations
  2. 自动备份 memories.db        → ~/.caveman/backups/memories.db.{timestamp}
  3. 执行迁移脚本链              → 事务包裹，失败回滚
  4. 验证迁移结果                → SELECT COUNT(*) 前后一致
  5. caveman doctor              → 飞轮健康检查
```

---

### 8.13 架构重构记录（v0.4.1, 2026-04-21）

> **背景：** 从 v0.3.0 到 v0.4.0 快速增长（97→399 文件），积累了 v1/v2 并存、重复实现、死代码等技术债。arch/refactor-v1 分支系统性清理。

**关键决策：**

| 问题 | 决策 | 理由 |
|------|------|------|
| split_message 4 个实现 | 统一到 `gateway/message_splitting.py`，其余委托 | fence-parity counting 只维护一处 |
| file_ops v1/v2 并存 | v1 保留（first-wins），v2 死注册删除，保留 file_read_v2 + file_diff | ToolRegistry first-wins 去重，v2 的 file_patch/file_search 永远不会被调用 |
| terminal v1/v2 并存 | v2 安全特性合并到 v1，v2 降为 compat shim | 一个实现，两个入口 |
| web_search v1/v2 并存 | v2 的 Tavily 搜索合并到 v1，v2 删除 | Tavily 是唯一 provider |
| context_refs vs context_references | context_references 瘦身为 compat shim | API 不兼容但都未集成，保留两个入口 |
| prompt.py vs prompt_builder.py | 两者共存 | 不同层次：runtime prompt vs factory config-driven prompt |
| display.py 薄 shim | 保留 | 打破循环依赖的合理设计 |
| estimate_tokens 重复 | 保留两个 | compression/utils 是 higher-level wrapper，已委托到 utils |
| _depth.py 文件 | 保留 + 确保 wired | "预制件仓库"模式，__init__.py re-export |

**质量指标：** 2864 passed / 7 skipped / 9 xfailed / 0 regressions（排除 3 个 pre-existing failures: cron, nfr601, telegram）

## §9 安全架构

### 9.1 安全原则

| 原则 | 描述 |
|------|------|
| 最小权限 | Agent 只拥有完成当前任务所需的最小权限 |
| 分层审批 | 低风险自动，中风险确认，高风险人工审批 |
| Secret 防泄露 | 扫描 URL/LLM输出中的密钥模式 |
| 沙箱隔离 | 代码执行不暴露主机网络/文件系统 |
| 审计追踪 | 所有工具调用完整日志 |
| 人在回路 | 破坏性操作始终需要人工确认 |

### 9.2 权限分级

> **v0.4.1 实现：** 三级权限（AUTO/ASK/DENY），fail-closed 设计（ASK 无 callback → DENY）。

| Level | 枚举值 | 行为 | 示例 |
|-------|--------|------|------|
| AUTO | `auto` | 无需确认，直接执行 | file_read, web_search, bash_safe, memory_read/write, skill_create |
| ASK | `ask` | 需用户批准（无 callback 时 fail-closed → DENY） | file_write/delete, bash_write, http_post, openclaw_spawn, hermes_delegate |
| DENY | `deny` | 始终拒绝 | bash_sudo |

> **设计备注：** 原 PRD v5.0 设计为 L0-L3 四级（含"通知后执行"），实现时简化为三级。14 个预定义 action 覆盖文件/终端/网络/跨 agent 操作。

### 9.3 v0.3.0 安全实现状态

| 安全功能 | 状态 |
|---------|------|
| 12-pattern 密钥扫描器 | ✅ |
| AUTO/ASK/DENY 三级权限 | ✅ |
| 危险命令拦截 | ✅ |
| Secret 防泄露 | ✅ |
| 结构化错误 | ✅ |
| 沙箱代码执行 | ✅ |
| 审计日志 | ✅ JSONL导出 |
| E2E 加密 | ✅ |

### 9.4 技能安全

| 威胁 | 缓解 |
|------|------|
| 自动创建的技能含恶意代码 | 单元测试门控 + 静态分析 + LLM安全审查 |
| 第三方Hub技能不安全 | 沙箱执行 + 权限声明 + 社区评分 + 认证标志 |
| Prompt Injection通过技能 | 技能内容与用户输入分层；输入消毒 |
| 技能改进引入退化 | 改进前后 A/B 对比；版本控制可回滚 |

### 9.5 数据安全

| 数据类型 | 存储位置 | 加密 | 用户控制 |
|---------|---------|------|---------|
| 对话历史 | 本地 SQLite | 可选 AES-256 | 随时删除 |
| 记忆 | 本地文件/SQLite | 可选加密 | 查看/编辑/删除 |
| 技能 | ~/.caveman/skills/ | 明文(需可读) | 完全控制 |
| 轨迹 | 本地 JSONL | 可选加密 | 选择性导出/删除 |
| 模型权重 | 本地文件 | 用户自管 | 完全控制 |
| Cloud备份 | 用户选择的云端 | E2E加密 | 随时撤回 |

---

## §10 训练与数据管道

> **核心认知转变（2026-04-15）：** 训练本地推理模型（SFT Llama 替代 Claude）没有实际价值——云端模型的推理能力远超本地 8B。Caveman 的"学习"通过记忆系统 + 技能进化实现。训练的真正价值在两个方向：① 训练本地 embedding 模型提升记忆检索精度；② 导出轨迹数据供研究者使用。

### 10.1 数据管道

```
日常使用 → 自动产生轨迹 → 质量评分 → 两个出口
                                          │
                              ┌───────────┤
                              ▼           ▼
                     Embedding 训练    数据导出
                     (记忆检索优化)    (ShareGPT/ChatML/OpenAI)
                              │           │
                              ▼           ▼
                     本地 embedding    研究者 SFT/RL
                     (nomic/bge)      (TRL/veRL)
```

### 10.2 训练策略

| 方向 | 目标 | 数据来源 | 优先级 |
|------|------|---------|--------|
| Embedding 微调 | 记忆检索更懂用户语境 | 查询-记忆匹配对 | P1 |
| 数据导出 | 供研究者 SFT/RL | 高质量轨迹 (quality_score ≥0.7) | P1 |
| 技能 RL | 技能路由优化 | 技能使用反馈 | P2 |
| 推理模型 SFT/RL | ~~替代云端模型~~ → 仅作研究用途 | 轨迹 + Verification 对 | P3（降级） |

### 10.3 训练管道代码核验状态（2026-04-24）

| 能力 | 实现证据 | 状态 | 备注 |
|------|----------|------|------|
| 轨迹导出 | `caveman export` + `TrajectoryRecorder.batch_export()`；SFT 数据集支持 ShareGPT/ChatML/OpenAI | ✅ | 符合“供研究者使用”的出口 |
| Embedding 训练对生成 | `RetrievalLog.generate_training_pairs()`、`PairExtractor.extract_from_retrieval_log()`；fallback 到 trajectories；RetrievalLog 已是 SQLite 表并记录 latency/source/adoption | ✅ | 主数据源已从“对话 Q&A”修正为“真实检索日志”，且可被 doctor 统计 |
| Embedding 微调 | `EmbeddingTrainer` 基于 sentence-transformers；默认 nomic，可传 bge 等模型 | 🟡 | 训练代码存在；但 sentence-transformers 未纳入 `embedding` extra，未完成 LOCOMO 门槛验证 |
| 本地 embedding provider | `memory.embedding.get_embedding_fn("local")` 与相关测试 | ✅ | 支持加载本地微调模型 |
| Embedding 评估 | `EmbeddingEvaluator` 支持 Recall@5/10、MRR、HitRate@5、before/after compare；`tests/test_training_pivot_fixes.py` 覆盖 eval set 和 compare 报告 | 🟡 | 模块有；CLI `--eval-only` 尚未接上，也未自动选优 |
| A/B 自动选优 | 未见自动选择更优模型逻辑 | ⏳ | 仍是愿景 |
| 推理模型 SFT/RL | `training.sft` / `training.rl` 以 mock/fallback 和数据导出为主 | ✅（按降级后定位） | 不再作为替代 Claude 的核心目标 |

**文档标记结论：** §10 的方向是对的，检索日志/训练对/本地 embedding provider/评估基础设施均已落地；但“训练依赖产品化 + Embedding 微调 + LOCOMO/等价门槛 + CLI 一键 A/B + 自动选优”还不是闭环产品能力。




---

# 第四部分：决策记录（Decision Log）

## §11 批判性审视记录

### 11.1 从架构审查中砍掉的

> 用「大型项目架构指导框架 v2.0」5层审查 Caveman 后，批判性审视砍掉的条目。

| 原建议 | 为什么砍 |
|--------|---------|
| CODEOWNERS | 1人项目不需要权限控制 |
| ADR 拆成目录 | 7条148行单文件 grep 比7个文件 find 更快 |
| CI/CD | 没有远程仓库，测试0.67秒手动跑无负担 |
| KNOWN_ISSUES.md | 已有 PRD §14 + 日记，第三个信息源 = 维护负担 |
| paths.py 拆3个文件 | 73行常量不是上帝模块 |
| 静态 PROJECT_MAP.md | 应该是 Caveman 自己的 Memory 条目 |

### 11.2 从架构审查中重新定义的

| 原建议 | 重新定义为 | 为什么 |
|--------|-----------|--------|
| "加 Feature Flags" | 引擎模块加载机制（Round 8 前提） | 不是安全网，是内核的 modprobe |
| "EventBus 持久化做可观测性" | 内核事件日志（Round 8） | 不是监控，是 Nudge/Shield/Lint 的输入源 |
| "loop.py 瘦身" | Round 11 再做 | 代码还在变，现在重构 = 重构完又要改 |

### 11.3 审查报告漏掉的关键问题

| 漏掉什么 | 为什么关键 | 纳入 |
|----------|-----------|------|
| 自举验证 | 不能解决自己的问题就无法证明能解决别人的 | Round 11 |
| LLM 调度器 | 多引擎争抢同一 LLM = 竞争条件 | Round 9 |
| 飞轮健康度指标 | 没有度量就没有优化 | Round 9 |

### 11.4 从记忆飞轮研究中砍掉的

| 砍什么 | 为什么砍 |
|--------|---------|
| "每 N 轮提取一次" | 定时轮询 = 够用就行。事件驱动才是长期主义。v0.4.0 已落地：SHIELD_UPDATE/TOOL_ERROR/LOOP_END 三事件触发 + 最小间隔节流 |
| MemPalace AAAK 分类框架 | 过度分类增加写入决策负担。4类型已够 |
| 完整 RAG pipeline | Write quality >> Retrieve sophistication |
| Guardian 独立 Agent | 个人 Agent 安全威胁模型不同于云服务，降级为 P2 |

---

## §12 三巨头基因图谱

> 从 v5.0 的"主角"降为"决策依据"。不是"我们参考了谁"定义我们，而是"我们解决什么问题"定义我们。

### 12.1 各取精华

**From Hermes（学习飞轮）：**
- 技能自动创建（每10轮 nudge 检查）
- 技能自我改进（_spawn_background_review）
- 记忆 Nudge（后台 fork agent 自动写入）
- 轨迹系统（ShareGPT JSONL）
- Atropos RL（强化学习环境）
- Honcho 用户建模
- 7 种记忆 Provider

**From OpenClaw（执行深度）：**
- ACP 协议（编排 Codex/Claude Code/Gemini）
- 设备配对（iOS/Android/macOS）
- 浏览器自动化（Playwright）
- 消息网关（7+平台）
- Canvas（富内容渲染）
- Cron 调度
- ClawHub 生态（5,700+ 技能）

**From Claude Code（工程纪律）：**
- Coordinator 4阶段
- Verification Agent（6借口封堵）
- 3层上下文压缩
- 记忆 4 类型 + "也记确认"
- Memory Drift（用前先验证）
- Prompt Cache + Cache Break Detection
- Hooks 系统（20+ 事件类型）

### 12.2 独有创新（三者都没有的）

| 创新 | 描述 | 学术支撑 |
|------|------|---------|
| Coordinator + 学习闭环 | Worker 完成后自动提取技能 | Memento-Skills |
| Verification + 轨迹数据 | 验证结果嵌入轨迹，可导出供研究 | Agent-R1 |
| RL 技能路由 | 80% vs 语义相似度 50% | Memento-Skills |
| Python-Node.js Bridge | 两个生态统一调用 | — |
| 设备 + 学习 | 物理世界交互纳入飞轮 | — |
| Harness = Skills + Verification | Guide + Sensor 完整工程 | Fowler |

### 12.3 三巨头最新动态（2026年4月）

| 动态 | 对 Caveman 的影响 |
|------|------------------|
| Hermes: hermes-agent-self-evolution (DSPy+GEPA) | 🔴 技能进化必须升级到程序级 |
| OpenClaw: REM Backfill + Dreaming | 🟡 Bridge 需对接 Dreaming 数据 |
| Claude Code: claude-opus-4-6 工具调用提升 | 🟢 模型越强我们越强 |
| Google ADK 层级 Agent 树 | 🟢 验证 Coordinator 方向 |
| "Composable > Framework" 共识 | 🟢 模块化方向正确 |

---

## §13 风险与开放问题

### 13.1 风险矩阵

| # | 风险 | 严重度 | 可能性 | 缓解策略 | 实际经验 |
|---|------|--------|--------|---------|---------|
| R1 | Bridge延迟/复杂度 | 高 | 中 | UDS 2.3μs已验证; 降级纯Python | — |
| R2 | 自动技能质量低 | 高 | 高 | 单元测试门控 + 安全扫描 + LLM judge | — |
| R3 | 上下文窗口压力 | 中 | 高 | 3层压缩已实现✅ | 测试通过 |
| R4 | 训练数据隐私 | 中 | 低 | 本地处理优先 + 导出前脱敏 | — |
| R5 | 工程量/人力 | 中 | 高 | AI辅助>90% | Phase 1: ~2h完成 |
| R6 | 被视为抄袭 | 中 | 中 | 标注来源 + 贡献回上游 | — |
| R7 | 竞品加了我们的创新 | 中 | 中 | 先发飞轮 + 预装经验 | — |
| R11 | **子agent沙箱文件丢失** | **高** | **高** | sandbox:"inherit" 铁律 | ⚠️ 血泪教训 |
| R12 | 第三方API代理不稳定 | 中 | 高 | Credential Pool + 降级Ollama | 模型名必须精确 |
| R13 | Python环境碎片化 | 中 | 高 | pyenv兜底 + Docker镜像 | macOS无brew |

### 13.2 开放问题

| # | 问题 | 决策 | 备注 |
|---|------|------|------|
| Q1 | Bridge 是否值得做？ | ✅ A: Bridge | UDS 2.3μs |
| Q2 | 第一个消息平台？ | ✅ 两个都做了 | Discord + Telegram |
| Q3 | 训练框架？ | C: embedding 微调为主，SFT/RL 仅导出供研究者 | 方向调整 |
| Q4 | 兼容 ClawHub？ | ✅ 是 | AgentSkills 标准 |
| Q5 | GitHub org？ | 待定 | 发布前 |
| Q6 | 安全权限默认级别？ | ✅ CLI默认AUTO(L0), Gateway默认L2 | 已实现 |
| Q8 | Memory 默认后端？ | ✅ JSON→SQLite+FTS5 | 渐进增强 |
| Q9 | 向量搜索？ | ✅ 混合 | OpenAI→Ollama→关键词 |
| Q10 | Trajectory 格式？ | ✅ 扩展ShareGPT | quality_score+duration+tool_calls |
| Q11 | **Compaction 后上下文丢失** | 🔴 **P0 痛点** | Round 8 解决 |

---

## §14 成功指标

### 14.1 核心指标

| 阶段 | 指标 | 底线 | 目标 | 实测(v0.2.0) |
|------|------|------|------|-------------|
| v0.2.0 | 功能完整度 | CLI+3工具+记忆 | +双平台消息 | ✅ 超目标 |
| v0.2.0 | 测试通过率 | ≥90% | 100% | ✅ 378/378 → v0.4.1: 2864/2864 |
| v0.2.0 | 代码量 | ≥50 files | — | ✅ 121 → v0.4.1: 399 files, 72.5K LOC |
| Round 8 | Shield 恢复率 | ≥80% | ≥95% | ✅ 实现 |
| Round 9 | 飞轮转速 | 14天≥2技能 | ≥5个 | — |
| Round 9 | 记忆质量 | LOCOMO≥55% | ≥65% | — |
| Phase 5 | 社区 | ≥500★ | ≥2K★ | — |

### 14.2 开发效率指标

| 指标 | 实测值 |
|------|--------|
| Phase 1 开发时间 | ~2 小时（原计划 8 周） |
| AI 代码占比 | >90% |
| LOC/小时 | ~2,000 |
| API 真实验证 | 通过 |

### 14.3 反指标

| 反指标 | 警戒线 | 意味着 |
|--------|--------|--------|
| 技能删除率 | >30%/月 | 自动创建质量太低 |
| 记忆冲突率 | >10% | Drift检测或GC失效 |
| API 费用/对话 | >$0.50 | 压缩或routing有问题 |
| 安装放弃率 | >30% | 依赖/配置太复杂 |

---

---

# 第五部分：功能需求（Functional Requirements）

> 每个需求有唯一 ID、优先级（P0/P1/P2）、验收标准（Given-When-Then）和所属 Round。
> P0 = 必须有（没有产品不成立）/ P1 = 应该有（显著提升价值）/ P2 = 可以有（锦上添花）

## FR-1xx: Agent OS 内核

### FR-101 Compaction Shield — session 精华维护
- **优先级：** P0
- **描述：** Agent Loop 每轮结束后，自动更新当前 session 的 session_essence（决策/进度/立场/关键数据）
- **验收标准：**
  - Given 一个运行中的 session 已完成 5 轮对话
  - When session 被 compaction
  - Then session_essence 已写入 `memory/sessions/{session_id}.yaml`，包含 decisions/progress/stances/key_data 四个字段
  - And 新 session 启动时能读取该 essence 并恢复上下文
- **Round:** 8

### FR-102 Nudge Phase 1 — 规则提取
- **优先级：** P0
- **描述：** 事件驱动的知识提取。任务完成/新事实发现/用户确认偏好/pre-compaction 时自动触发
- **验收标准：**
  - Given Agent 完成了一个编码任务并发现 "pyenv 在 macOS 不预装"
  - When 任务完成事件触发
  - Then Nudge 自动提取一条 reference 类型记忆写入 Memory Store
  - And 提取延迟 < 2s（Phase 1 规则匹配，零 LLM 成本）
- **Round:** 8

### FR-103 Nudge Phase 2 — LLM 精炼
- **优先级：** P1
- **描述：** 用 LLM 对 Phase 1 提取的原始记忆进行去重、合并、冲突检测
- **验收标准：**
  - Given Phase 1 提取了 10 条原始记忆，其中 3 条重复、1 条与已有记忆矛盾
  - When Phase 2 LLM 精炼运行
  - Then 输出 7 条去重记忆 + 1 条标记为 conflict 的记忆
  - And LLM 调用成本 < $0.01
- **Round:** 9

### FR-104 Recall Engine — 上下文恢复
- **优先级：** P0
- **描述：** 新 session 启动时，自动读取最近 session_essence + 相关记忆，组装并注入 system prompt
- **验收标准：**
  - Given 上一个 session 的 essence 记录了 "正在做 PRD v6.0 重构，Round 8 是下一步"
  - When 新 session 启动
  - Then Agent 的第一轮回复能引用上一个 session 的进度和决策
  - And 恢复延迟 < 500ms
- **Round:** 8

### FR-105 Ripple Engine — 知识传播
- **优先级：** P1
- **描述：** 每次写入新知识时，自动查找相关条目，检查矛盾，更新交叉引用
- **验收标准：**
  - Given Memory Store 中有 "服务器 IP 是 203.0.113.10"
  - When 写入新记忆 "服务器已迁移到 198.51.100.20"
  - Then Ripple 标记旧条目为 stale，新条目关联旧条目
  - And 生成一条 conflict 通知供用户确认
- **Round:** 10

### FR-106 Lint Engine — 知识审计
- **优先级：** P1
- **描述：** 定期扫描 Memory Store，检测过时/矛盾/幻觉/孤岛知识
- **验收标准：**
  - Given Memory Store 中有 100 条记忆，其中 5 条引用了已不存在的文件路径
  - When `caveman doctor` 运行 Lint 检查
  - Then 输出 5 条 stale 警告，建议更新或删除
- **Round:** 9

### FR-107 Feature Flags — 引擎开关
- **优先级：** P0
- **描述：** 每个引擎（Shield/Nudge/Ripple/Lint/Recall）可独立开关，支持配置文件和运行时切换
- **验收标准：**
  - Given config.yaml 中 `engines.nudge.enabled: false`
  - When Agent Loop 运行
  - Then Nudge 相关事件不触发，不消耗 LLM 调用
  - And 日志记录 "Nudge engine disabled by config"
- **Round:** 8

### FR-108 EventBus 持久化
- **优先级：** P0
- **描述：** EventBus 的 28 种事件写入 SQLite events 表，支持回放和审计
- **验收标准：**
  - Given Agent 完成一个任务产生了 task_complete + memory_write + tool_call 三个事件
  - When 查询 events 表
  - Then 三个事件按时间序列存在，包含 event_type/timestamp/payload
  - And 进程重启后事件不丢失
- **Round:** 8

## FR-2xx: 能力层

### FR-201 Workspace 文件读取
- **优先级：** P0
- **描述：** 启动时自动扫描 workspace 目录，读取 SOUL.md/USER.md/MEMORY.md/AGENTS.md 等文件并注入 system prompt
- **验收标准：**
  - Given `~/.openclaw/workspace/SOUL.md` 存在且内容为 "我是豆包"
  - When Caveman 启动
  - Then system prompt Layer 0 包含 "我是豆包"
  - And 支持 `~/.openclaw/workspace/` 和 `~/.caveman/workspace/` 两个路径
- **Round:** 8

### FR-202 CLI Agent 调用 — Claude Code
- **优先级：** P0
- **描述：** Coordinator 可调用 Claude Code CLI 执行编码任务
- **验收标准：**
  - Given 用户请求 "在 ~/projects/myapp 创建一个 Flask hello world"
  - When Coordinator 判断需要编码 Agent
  - Then 调用 `claude --print --permission-mode bypassPermissions -p "task"` 并在 cwd=~/projects/myapp 执行
  - And 超时 900s 后自动终止，返回部分输出
  - And 输出写入轨迹记录
- **Round:** 8

### FR-203 CLI Agent 调用 — Codex/Gemini/Pi
- **优先级：** P1
- **描述：** Coordinator 可调用 Codex CLI / Gemini CLI / Pi (OpenCode) 执行编码任务（PTY 模式）
- **验收标准：**
  - Given 用户请求编码任务且配置了 Codex
  - When Coordinator 选择 Codex
  - Then 通过 PTY spawn `codex "task"`，捕获输出
  - And 支持超时控制和进度回调
- **Round:** 9

### FR-204 记忆导入
- **优先级：** P1
- **描述：** `caveman import` 命令从 OpenClaw/Hermes/Codex/Claude Code 导入已有记忆和技能
- **验收标准：**
  - Given `~/.openclaw/workspace/memory/` 下有 20 个 .md 文件
  - When 运行 `caveman import --from openclaw`
  - Then 20 个文件的内容被解析并写入 Memory Store
  - And 导入报告显示成功/跳过/失败数量
- **Round:** 9

### FR-205 技能自动创建
- **优先级：** P1
- **描述：** 检测重复模式后，用 LLM 分析轨迹提取意图，自动创建 SKILL.md
- **验收标准：**
  - Given 用户在 3 个不同 session 中都执行了 "部署到 VPS" 类似任务
  - When 技能检测触发
  - Then 自动创建 `deploy-to-vps` 技能，包含 triggers/description/instructions
  - And 技能通过单元测试门控 + 安全扫描
- **Round:** 9

### FR-206 技能 RL 路由
- **优先级：** P2
- **描述：** 用 RL 训练的路由器选择最佳技能，替代语义相似度匹配
- **验收标准：**
  - Given 有 20 个已创建技能
  - When 新任务到来
  - Then RL 路由器在 < 50ms 内选出最佳技能
  - And 命中率 ≥ 60%（30 天后）
- **Round:** 10

### FR-207 轨迹自动质量评分
- **优先级：** P1
- **描述：** 每条轨迹自动计算 quality_score（0-1），基于 Verification 结果 + LLM judge
- **验收标准：**
  - Given 一条轨迹包含 10 轮对话，Verification 判定 8 轮 PASS / 2 轮 FAIL
  - When 质量评分运行
  - Then quality_score 在 0.6-0.8 范围内
  - And score < 0.7 的轨迹不进入 SFT 训练集
- **Round:** 9

### FR-208 LLM 调度器
- **优先级：** P1
- **描述：** 多引擎（Nudge/Shield/SmartCompression/Verification）共享 LLM 时的优先级和配额管理
- **验收标准：**
  - Given Shield 和 Nudge 同时请求 LLM
  - When LLM 调度器处理
  - Then Shield（P0）优先执行，Nudge 排队
  - And 每分钟 LLM 调用不超过配额（默认 30 次/分钟）
- **Round:** 9

## FR-3xx: 基础设施

### FR-301 配置格式兼容
- **优先级：** P1
- **描述：** 首次启动时自动检测 OpenClaw/Hermes 配置，提示导入 API key 和模型偏好
- **验收标准：**
  - Given `~/.openclaw/config.yaml` 存在且包含 Anthropic API key
  - When 运行 `caveman setup`
  - Then 提示 "检测到 OpenClaw 配置，是否导入 API key？[Y/n]"
  - And 选择 Y 后 key 写入 `~/.caveman/config.yaml`
- **Round:** 9

### FR-302 飞轮健康度仪表盘
- **优先级：** P1
- **描述：** `caveman doctor` 命令输出飞轮 6 项健康指标 + 诊断建议
- **验收标准：**
  - Given Caveman 已运行 7 天，创建了 2 个技能，记忆 50 条
  - When 运行 `caveman doctor`
  - Then 输出 6 项指标（技能命中率/记忆召回/技能创建速率/轨迹质量/压缩保真/Shield恢复率）
  - And 低于阈值的指标标红并给出改进建议
- **Round:** 9

### FR-303 数据管道
- **优先级：** P2
- **描述：** `caveman train` 命令支持两个方向：① embedding 微调（提升记忆检索精度）；② 轨迹导出为标准格式供研究者 SFT/RL
- **验收标准：**
  - Given 积累了 100 条 quality_score ≥ 0.7 的轨迹
  - When 运行 `caveman train --target embedding`
  - Then 从轨迹中提取查询-记忆匹配对，微调本地 embedding 模型
  - When 运行 `caveman export --format sharegpt`
  - Then 导出标准格式数据集供研究者使用
- **Round:** 11+



---

# 第六部分：非功能需求（Non-Functional Requirements）

## NFR-1: 性能

| ID | 需求 | 指标 | 测量方式 |
|----|------|------|---------|
| NFR-101 | CLI 启动时间 | < 3s（冷启动）/ < 1s（热启动） | `time caveman run "hello"` |
| NFR-102 | 单轮推理延迟 | < 5s（不含 LLM API 时间） | EventBus timestamp 差值 |
| NFR-103 | 记忆搜索延迟 | < 100ms（FTS5）/ < 500ms（混合搜索） | Memory.search() 计时 |
| NFR-104 | Shield 更新延迟 | < 200ms（规则提取）/ < 2s（含 LLM） | Shield.update() 计时 |
| NFR-105 | Recall 恢复延迟 | < 500ms | Recall.restore() 计时 |
| NFR-106 | Bridge UDS 延迟 | < 10μs p50 / < 100μs p99 | UDS roundtrip benchmark |
| NFR-107 | 工具调用延迟 | < 100ms（内部工具）/ < 5s（外部工具） | ToolRegistry 计时 |

## NFR-2: 资源占用

| ID | 需求 | 指标 | 理由 |
|----|------|------|------|
| NFR-201 | 内存占用（空闲） | < 100MB RSS | $5 VPS 只有 1GB |
| NFR-202 | 内存占用（运行中） | < 300MB RSS | 留空间给 LLM 本地推理 |
| NFR-203 | 磁盘占用（基础安装） | < 50MB | pip install 后 |
| NFR-204 | 磁盘占用（运行数据） | < 1GB/月（轨迹+记忆+事件） | 自动 GC 策略 |
| NFR-205 | CPU 空闲占用 | < 1% | 不应在后台消耗资源 |

## NFR-3: 可靠性

| ID | 需求 | 指标 | 验证方式 |
|----|------|------|---------|
| NFR-301 | 进程崩溃恢复 | 崩溃后 < 5s 自动重启 | systemd/launchd watchdog |
| NFR-302 | 数据零丢失 | 已写入 Memory Store 的数据不因崩溃丢失 | SQLite WAL 模式 + fsync |
| NFR-303 | 降级运行 | Bridge 断开后核心功能（CLI+记忆+推理）正常 | 集成测试 |
| NFR-304 | LLM 故障容忍 | 单 provider 不可用时自动切换备用 | Credential Pool 轮转 |
| NFR-305 | 长时间运行稳定 | 连续运行 7 天无内存泄漏（RSS 增长 < 10%） | 压力测试 |

## NFR-4: 安全

| ID | 需求 | 指标 | 验证方式 |
|----|------|------|---------|
| NFR-401 | Secret 零泄露 | LLM 输出/日志/轨迹中不含 API key | 12-pattern 扫描器 |
| NFR-402 | 权限最小化 | 默认 L2（确认后执行），可配置 | 权限系统测试 |
| NFR-403 | 本地数据加密 | 支持 AES-256 加密记忆和轨迹（可选） | 加密/解密测试 |
| NFR-404 | 依赖安全 | 零已知 CVE 的直接依赖 | `pip audit` |

## NFR-5: 可维护性

| ID | 需求 | 指标 | 验证方式 |
|----|------|------|---------|
| NFR-501 | 测试覆盖率 | ≥ 80% 行覆盖率 | pytest --cov |
| NFR-502 | 单文件行数 | ≤ 400 行（核心模块） | lint 检查 |
| NFR-503 | 模块独立性 | 零循环依赖 | import 分析 |
| NFR-504 | 文档同步 | PRD/ARCHITECTURE/ADR 与代码版本一致 | CI 检查 |

## NFR-6: 兼容性

| ID | 需求 | 指标 | 验证方式 |
|----|------|------|---------|
| NFR-601 | Python 版本 | 3.12+ | CI 矩阵 |
| NFR-602 | 操作系统 | macOS 13+ / Ubuntu 22.04+ / Windows 11 (WSL2) | CI 矩阵 |
| NFR-603 | Node.js 版本（Bridge） | 22+（可选，无 Node.js 时降级纯 Python） | 降级测试 |
| NFR-604 | LLM Provider | Anthropic / OpenAI / OpenRouter 400+ / Ollama | Provider 测试 |

---

# 第七部分：用户故事（User Stories）

> 按用户画像分组，每个故事关联功能需求 ID。

## US-1: 技术型个人效率极客（P0 用户）

### US-101 首次安装
- As a 独立开发者
- I want 用 `pip install caveman && caveman setup` 在 3 分钟内完成安装配置
- So that 我不需要折腾复杂的依赖和配置
- **关联 FR:** FR-301（配置兼容）
- **验收:** 从零到第一次对话 < 5 分钟

### US-102 跨 session 记忆
- As a 管理 5 个项目的开发者
- I want Agent 在新 session 中自动记住上次的项目进度和决策
- So that 我不需要每次重新解释上下文
- **关联 FR:** FR-101（Shield）, FR-104（Recall）
- **验收:** compaction 后 Agent 能引用上一个 session 的关键决策

### US-103 自动学习
- As a 重复做部署任务的开发者
- I want Agent 自动从我的操作中学习并创建可复用的技能
- So that 同类任务越做越快
- **关联 FR:** FR-102（Nudge）, FR-205（技能创建）
- **验收:** 第 3 次做类似任务时，Agent 主动使用已学到的技能

### US-104 编码委托
- As a 需要快速实现功能的开发者
- I want 用一句话描述需求，Agent 自动调用 Claude Code/Codex 完成编码
- So that 我只需要审查结果而不是写代码
- **关联 FR:** FR-202（Claude Code）, FR-203（Codex/Gemini）
- **验收:** "在 ~/projects/myapp 加一个 /health 接口" → 代码生成 + 测试通过

### US-105 多平台远程操控
- As a 经常不在电脑前的开发者
- I want 通过 Telegram/Discord 远程给 Agent 下达任务
- So that 我在手机上也能管理项目
- **关联 FR:** Gateway（已实现）
- **验收:** Telegram 发消息 → Agent 执行 → 结果回复到 Telegram

### US-106 知识不过时
- As a 长期使用的用户
- I want Agent 自动检测过时的记忆并提醒我更新
- So that 我不会因为过时信息做出错误决策
- **关联 FR:** FR-106（Lint）, FR-105（Ripple）
- **验收:** 服务器 IP 变更后，Agent 在下次引用时提示 "此信息可能已过时"

## US-2: AI/ML 研究者（P1 用户）

### US-201 轨迹导出
- As a 研究 agent 行为的学者
- I want 导出所有对话轨迹为标准训练格式
- So that 我可以用于微调实验
- **关联 FR:** FR-303（数据管道）
- **验收:** `caveman export --format sharegpt` 输出可直接用于 TRL

### US-202 Embedding 训练
- As a 想让记忆检索更精准的用户
- I want 用积累的查询-记忆对微调本地 embedding 模型
- So that Agent 搜索记忆时更懂我的语境
- **关联 FR:** FR-303（数据管道）
- **验收:** `caveman train --target embedding` → embedding 模型 + 检索精度提升报告

## US-3: 自动化创客（P2 用户）

### US-301 浏览器自动化
- As a 需要抓取数据的创客
- I want Agent 能操控浏览器完成多步骤任务
- So that 我不需要手写爬虫
- **关联 FR:** Browser（已实现 via Bridge）
- **验收:** "登录 XX 网站并下载最新报告" → 文件下载到本地

### US-302 设备控制
- As a 想让 AI 操控手机的用户
- I want Agent 能通过 ADB 操控 Android 设备
- So that 我可以自动化手机上的重复操作
- **关联 FR:** FR-Bridge（Device via OpenClaw Bridge，已实现骨架）
- **验收:** "打开微信发消息给 XX" → 手机执行完成

## US-4: 错误恢复故事

### US-401 Agent 做错事后回滚
- As a 委托 Agent 修改代码的开发者
- I want 当 Agent 改错了文件时，能一键回滚到修改前的状态
- So that 我不会因为 Agent 的错误丢失工作成果
- **关联 FR:** FR-101（Shield 保存状态）
- **验收:** Agent 改错文件 → 用户说"回滚" → Agent 用 git checkout 或备份恢复 → 文件恢复到修改前

### US-402 记忆冲突裁决
- As a 长期使用的用户
- I want 当 Agent 发现两条记忆矛盾时，主动问我哪条是对的
- So that 错误的记忆不会影响后续决策
- **关联 FR:** FR-105（Ripple 矛盾检测）, FR-106（Lint 审计）
- **验收:** 记忆 A 说"服务器 IP 是 X"，记忆 B 说"服务器 IP 是 Y" → Agent 提示冲突 → 用户确认 → 旧记忆标记 stale，trust 降分

### US-403 技能质量差时禁用
- As a 发现自动创建的技能不好用的用户
- I want 能反馈"这个技能不好用"并禁用它
- So that 坏技能不会反复被匹配到
- **关联 FR:** FR-205（技能创建）, FR-206（RL 路由）
- **验收:** 用户说"这个技能不好用" → 技能 quality_score 降分 → 连续 3 次负反馈 → 自动禁用 → 用户可手动重新启用

### US-404 LLM 服务中断时降级
- As a 在远程操控 Agent 的用户
- I want 当 LLM API 不可用时，Agent 告诉我发生了什么而不是静默失败
- So that 我知道该等待还是切换方案
- **关联 FR:** §8.4 异常处理
- **验收:** API 429/500 → Agent 通过 Gateway 发消息"LLM 暂时不可用，重试中(1/3)" → 3 次失败 → "已切换到备用模型" 或 "需要你手动处理"

### US-405 记忆库污染清理
- As a 发现 Agent 记住了错误信息的用户
- I want 能搜索并删除特定记忆
- So that 错误信息不会继续影响 Agent 的判断
- **关联 FR:** Memory Store `forget()`, FR-106（Lint）
- **验收:** 用户说"忘掉关于 XX 的记忆" → Agent 搜索相关记忆 → 列出候选 → 用户确认 → 删除 + Ripple 清理关联引用

## US-5: 边界场景故事

### US-501 大规模记忆库性能
- As a 使用 6 个月、积累 5000+ 条记忆的用户
- I want 记忆搜索依然在 100ms 内返回
- So that Agent 的响应速度不会随使用时间退化
- **关联 NFR:** NFR-103（记忆搜索延迟 < 100ms FTS5）
- **验收:** 5000 条记忆 + FTS5 搜索 < 100ms；混合搜索 < 500ms

### US-502 长时间运行稳定性
- As a 7×24 运行 Agent 的用户
- I want Agent 连续运行 30 天不崩溃、不内存泄漏
- So that 我不需要定期手动重启
- **关联 NFR:** NFR-305（RSS 增长 < 10%）
- **验收:** 30 天压力测试 → RSS 增长 < 10% → 无 OOM → EventBus 无 handler 泄漏

### US-503 离线工作
- As a 在飞机上没有网络的开发者
- I want Agent 的本地工具（文件操作、bash、记忆搜索）依然可用
- So that 我不会因为断网完全失去 Agent 能力
- **关联 FR:** §8.4 异常处理（网络断开 → 本地工具仍可用）
- **验收:** 断网 → LLM 调用失败提示 → 本地工具正常 → 联网后自动恢复

### US-504 Compaction 后无感恢复
- As a 长对话后触发 compaction 的用户
- I want Agent 在 compaction 后依然记得我们在做什么
- So that 我不需要重新解释上下文
- **关联 FR:** FR-101（Shield）, FR-104（Recall）
- **验收:** 50 轮对话 → compaction 触发 → Agent 下一轮回复引用 session_essence 中的决策和进度 → 用户无感

## US-6: 迁移故事

### US-601 从 OpenClaw 迁移
- As a 已经在用 OpenClaw 的开发者
- I want 用一条命令把 OpenClaw 的记忆、技能、配置迁移到 Caveman
- So that 我不需要从零开始积累
- **关联 FR:** FR-204（记忆导入）, §8.8.1（Workspace 兼容）
- **验收:** `caveman import --from openclaw` → 导入 SOUL.md/USER.md/MEMORY.md/memory/*.md/skills/ → 导入报告显示成功数量 → Caveman 启动后能引用导入的记忆

### US-601b 从 OpenClaw Session 提取知识
- As a 在 OpenClaw Discord 上做过大量研究和分析的用户
- I want 从 OpenClaw 的 session 对话中提取高价值知识到 Caveman
- So that 之前在 Discord 上做的研究结论、根因分析、架构决策不会丢失
- **关联 FR:** FR-204, §8.8.4.1（Session 知识提取）
- **验收:** `caveman import --from openclaw-sessions --dry-run` → 显示可提取的知识条数和分类 → `caveman import --from openclaw-sessions` → 知识写入 Memory Store → `memory_search` 能检索到提取的知识 → 重复执行不产生重复数据

### US-602 从 Hermes 迁移
- As a 已经在用 Hermes 的开发者
- I want 把 Hermes 的记忆和技能迁移到 Caveman
- So that 我在 Hermes 积累的经验不会浪费
- **关联 FR:** FR-204, §8.8.4
- **验收:** `caveman import --from hermes` → 导入 memories + skills → 格式转换成功 → Caveman 能检索导入的记忆

### US-603 全新用户 Day 1
- As a 从未用过 AI Agent 框架的开发者
- I want 从安装到第一次有用的对话在 5 分钟内完成
- So that 我不会因为复杂的配置放弃
- **关联 FR:** FR-301（配置兼容）, US-101
- **验收:** `pip install caveman && caveman setup` → 交互式配置（API key + 模型选择）→ `caveman run "hello"` → Agent 回复 → 全程 < 5 分钟


# 附录

## 附录 A：竞品详细分析

### A.1 竞品格局（2026年4月）

| 赛道 | 代表 | 市场规模 | Caveman 定位 |
|------|------|---------|-------------|
| 编码 IDE | Cursor($60B), Windsurf | $6-9.5B | ❌ 不是IDE |
| 编排框架 | LangChain(110K★), CrewAI(44K★) | — | ❌ 不是SDK |
| 编码 Agent | Claude Code, Codex CLI | — | 🔸 编码是能力之一 |
| 个人 Agent | Hermes(32K★), OpenClaw(25K★) | 新兴 | ✅ 我们在这 |

### A.2 个人 Agent 赛道深度对比

| 维度 | Hermes | OpenClaw | Claude Code | Open Interpreter | **Caveman** |
|------|--------|----------|-------------|-----------------|-------------|
| Stars | 32K+ | 25K+ | 闭源 | 55K+ | 新项目 |
| 语言 | Python | TypeScript | TS/Bun | Python | Python+Node.js |
| 自学习 | ✅ 飞轮 | ❌ | 🔸 自动记忆 | ❌ | ✅ 增强版 |
| 训练闭环 | ✅ | ❌ | ❌ | ❌ | 🔸 embedding 训练 + 数据导出 |
| Coordinator | ❌ | 🔸 ACP | ✅ 4阶段 | ❌ | ✅ 内置+外部 |
| Verification | ❌ | ❌ | ✅ | ❌ | ✅ |
| 上下文压缩 | 🔸 | 🔸 | ✅ 3层 | ❌ | ✅ 3层+Memento |
| ACP编排 | ❌ | ✅ | ❌ | ❌ | ✅ |
| 设备控制 | ❌ | ✅ | ❌ | 🔸 | ✅ |
| 浏览器 | 🔸 | ✅ | ❌ | 🔸 | ✅ |
| 消息平台 | ✅ 7+ | ✅ 7+ | ❌ | ❌ | ✅ 7+ |
| 预装经验 | ❌ | ❌ | ❌ | ❌ | ✅ 11K行 |
| Harness | ❌ | 🔸 | ✅ | ❌ | ✅ 全套 |

### A.3 编排框架赛道

| 框架 | Stars | 特点 | 与 Caveman 关系 |
|------|-------|------|----------------|
| LangChain/LangGraph | 110K+ | 状态图编排 | 可作为内部引擎 |
| CrewAI | 44K+ | 角色协作 | 理念类似 Coordinator |
| AutoGen | 54K+ | 多agent对话 | 参考 GroupChat |
| MetaGPT | 48K+ | 软件公司模拟 | 参考角色分工 |
| Google ADK | 新 | 层级agent树 | 参考层级编排 |

### A.4 学术前沿

| 论文/框架 | 时间 | 关键发现 | 对 Caveman 的启示 |
|-----------|------|---------|------------------|
| Memento-Skills | 2026-04 | RL技能重写 80% vs RAG 50% | 技能改进必须用RL |
| LOCOMO (Mem0) | ECAI 2025 | 10种记忆方案横评 | 需要 LOCOMO 基准验证 |
| Agent-R1 | 2026-03 | 端到端RL轨迹训练 | 直接复用其框架 |
| Harness Engineering (Fowler) | 2026-04 | Guide+Sensor | Skills=Guide, Verification=Sensor |
| Self-Evolving Agents (Karpathy) | 2026-03 | AutoResearch: agent自主实验过夜 | 验证学习闭环长期价值 |
| veRL (字节) | 2026 | HybridFlow RLHF | 训练管道直接用 veRL |

### A.5 Memory 框架格局

| 框架 | 方法 | Caveman 采纳 |
|------|------|-------------|
| Mem0 | 智能管道+知识图谱 | ❌ 自建hybrid替代 |
| Cognee | 结构化记忆图谱 | ❌ 过重 |
| LangMem | LangGraph原生 | ❌ 不绑框架 |
| Zep | 时序记忆 | ❌ 需独立部署 |
| Letta (MemGPT) | 虚拟上下文管理 | 🔬 参考压缩策略 |
| SQLite+FTS5 | 全文搜索 | ✅ 默认选择 |

---

## 附录 B：历史路线图

### B.1 Phase 1-10 详细交付

| Phase | 主题 | 关键交付 | commit |
|-------|------|---------|--------|
| 1+2 | 脚手架 | AgentLoop, Providers, Tools, Memory, Skills, Compression, Gateway, Bridge, Security | 多个 |
| 3 | CLI | 8命令 + builtin auto-register | — |
| 4 | 配置 | Config loader + factory + setup wizard | — |
| 5 | 桥接 | OpenClaw MCP + Memory v2 | — |
| 6 | 技能 | Skill v2 + Hermes bridge + prompt builder | — |
| 7 | 压缩 | 3层压缩 + 真实API验证 | — |
| 8-9 | 轨迹+网关 | Trajectory v2 + Discord/Telegram | — |
| 10 | 打包 | README + pyproject + pip installable | — |

### B.2 7轮优化详细

| Round | 主题 | 核心改动 | commit |
|-------|------|---------|--------|
| 1 | Bug修复+DRY | 12bug / 3 DRY | `e93bdef` |
| 2 | 结构性债务 | 循环依赖/并发锁/泄漏 | `8b32745` |
| 3 | 常量统一 | 30+硬编码→paths.py | `d6c67e3` |
| 4 | 事件驱动 | EventBus 22事件 / 复杂度26→7 | `23b086b` |
| 5 | 开发体验 | @tool装饰器 / ConfigValidator | `8260469` |
| 6 | 结缔组织 | ToolResult / ErrorHierarchy / Lifecycle | `c344b8e` |
| 7 | 真流式+声明式 | Provider流式修复 / Browser @tool迁移 | `3e54636` |

---

## 附录 C：商业模式

> v0.2.0 阶段，商业模式是"值得探索但不是焦点"。先做好产品力和用户口碑。

### C.1 定位：开源核心 + 增值服务

```
Tier 0: 开源免费 (MIT)
  核心 Runtime + 6技能 + 3方法论 + CLI + 基础记忆/技能 + 轨迹记录

Tier 1: Caveman Hub (社区驱动)
  免费/付费社区技能 (创作者分成 70/30) + 认证/审核

Tier 2: Caveman Pro ($10-50/月)
  托管网关 + 云端记忆备份 + 高级分析面板

Tier 3: Caveman Data ($按量)
  轨迹导出 + embedding 训练托管 + 数据分析面板
```

### C.2 保守情景

| 情景 | 年1 | 年2 | 前提 |
|------|-----|-----|------|
| 悲观 | $0 | $5K/年 | 200★, 纯兴趣项目 |
| 基准 | $0-2K | $30K/年 | 1K★, 10个Pro订阅 |
| 乐观 | $10K | $120K/年 | 5K★, 100+付费用户 |

**现实预期：** 年 1 做好零收入准备。商业化在 1,000+ 活跃用户后才有意义。

### C.3 开源策略

- 核心永远免费开源（MIT）
- 生态靠社区（创作者 70% 分成）
- 增值不拦路（付费功能是便利性，不是必要性）
- 数据归用户（轨迹/记忆/模型全部本地）

---

## 附录 D：为什么造新项目，而不是 Fork

### D.1 为什么不 Fork Hermes？

| 维度 | Fork Hermes | 新建 Caveman |
|------|------------|-------------|
| 社区 | ❌ fork分裂 | ✅ 干净新社区 |
| Node.js集成 | ❌ 纯Python加Bridge=改架构 | ✅ Day 1双语设计 |
| Coordinator | ❌ 单agent要重写loop | ✅ 架构层面支持 |
| 技术债 | ❌ run_agent.py 10524行 | ✅ 干净无包袱 |

### D.2 为什么不在 OpenClaw 上加学习？

| 维度 | OpenClaw+学习 | 新建 Caveman |
|------|-------------|-------------|
| ML生态 | ❌ TS中做训练几乎不可能 | ✅ Python原生 |
| 架构自由 | ❌ 25K+文件成熟但复杂 | ✅ 轻量起步 |

### D.3 v0.2.0 验证了新建的正确性

~48小时完成全量构建+7轮优化，比 fork + 理解 + 魔改快得多。94文件 / 214测试 / 17模块清晰分离。

---

*Document: docs/PRD.md*
*Architecture: docs/ARCHITECTURE.md (45KB) + docs/ADR.md (148行)*
*Repository: *
*Version: 6.3 | 7 Parts (§1-§14 + FR + NFR + US) + 4 Appendices (A-D) | 一条叙事线*
*Last updated: 2026-04-18 | +§8.9 数据模型 +§8.10 接口契约 +§8.11 可观测性 +§8.12 迁移策略 +Round 13-16 +US-4/5/6*
