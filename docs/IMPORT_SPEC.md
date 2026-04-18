## Caveman Import System — 产品级重写 Spec

### 设计原则
1. **只读源数据** — 绝不修改/删除/移动原始文件
2. **零损害** — 导入的是副本，原软件继续正常运行
3. **全量适配** — 不是"够用就行"，是把每个平台的数据都吃干抹净
4. **幂等** — 重复导入不会产生重复数据（基于 content hash 去重）
5. **可审计** — 每条导入记录都有 source/path/timestamp 元数据
6. **渐进式** — dry-run 先看，confirm 再导，支持 --include/--exclude 过滤

### 数据源全量清单

#### OpenClaw (~/.openclaw/)
```
workspace/
├── SOUL.md, USER.md, MEMORY.md, AGENTS.md    → 导入为 workspace 文件（复制到 ~/.caveman/workspace/）
├── HEARTBEAT.md, TOOLS.md, IDENTITY.md        → 同上
├── agents/*.md                                 → 导入为 workspace/agents/
├── skills/*/SKILL.md                           → 导入为 ~/.caveman/skills/
├── scripts/*.sh                                → 导入为 workspace/scripts/（只复制，不执行）
├── memory/
│   ├── YYYY-MM-DD.md (92个日记)               → 导入为 episodic 记忆
│   ├── 专题.md (credentials.md 等)            → 导入为 semantic 记忆
│   ├── projects/*.md                           → 导入为 semantic/project 记忆
│   ├── seo/*.md                                → 导入为 semantic 记忆
│   ├── archive/*.md                            → 导入为 episodic 记忆
│   ├── studies/*.md                            → 导入为 semantic 记忆
│   ├── lessons/*.md                            → 导入为 procedural 记忆
│   ├── sop-references/*.md                     → 导入为 procedural 记忆
│   └── agent-profiles/*.md                     → 导入为 semantic 记忆
├── .learnings/LEARNINGS.md                     → 导入为 procedural 记忆
├── .learnings/ERRORS.md                        → 导入为 episodic 记忆
└── .agent-state/done/*.json                    → 导入为 episodic 记忆（agent 任务历史）

openclaw.json                                   → 提取 providers/models 配置写入 caveman config
cron/jobs.json                                  → 导入为 ~/.caveman/cron/imported-jobs.json（参考，不自动激活）
memory/main.sqlite                              → 不导入（向量库格式不同，重新 embedding）
```

#### Hermes (~/.hermes/)
```
memories/
├── MEMORY.md                                   → 按 § 分割为独立记忆条目
└── USER.md                                     → 按 § 分割为 working 记忆

skills/*/SKILL.md                               → 导入为 ~/.caveman/skills/
config.yaml                                     → 提取 model/providers 配置

注意：Hermes 记忆用 \n§\n (section sign) 分隔，不是 ## 标题
```

#### Claude Code (~/.claude/)
```
settings.json                                   → 提取 model/env 配置
projects/*/                                     → 扫描目录名提取项目列表
plans/*.md                                      → 导入为 procedural 记忆
```

#### Codex (~/.codex/)
```
MEMORY.md                                       → 导入为 semantic 记忆
rollout_summaries/*.md                          → 导入为 episodic 记忆
```

### 文件结构

重写 `caveman/cli/importer.py`，拆分为模块：

```
caveman/cli/importer.py          → 主入口 + CLI 参数解析（保持）
caveman/import_/                 → 新模块目录
├── __init__.py
├── base.py                      → ImportResult, BaseImporter ABC
├── openclaw.py                  → OpenClawImporter
├── hermes.py                    → HermesImporter  
├── claude_code.py               → ClaudeCodeImporter
├── codex.py                     → CodexImporter
├── directory.py                 → DirectoryImporter（通用）
├── config_merger.py             → 配置合并逻辑
├── dedup.py                     → 基于 content hash 的去重
└── report.py                    → 导入报告生成
```

### BaseImporter 接口

```python
class BaseImporter(ABC):
    """Base class for all importers."""
    
    def __init__(self, caveman_home: Path, dry_run: bool = False):
        self.caveman_home = caveman_home
        self.dry_run = dry_run
        self.result = ImportResult()
    
    @abstractmethod
    def detect(self) -> bool:
        """Check if this source exists on the system."""
    
    @abstractmethod
    def scan(self) -> ImportManifest:
        """Scan source and return what would be imported (no writes)."""
    
    @abstractmethod
    async def execute(self, manifest: ImportManifest) -> ImportResult:
        """Execute the import based on the manifest."""
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source name."""
```

### ImportManifest

```python
@dataclass
class ImportItem:
    source_path: Path           # 原始文件路径
    target_type: str            # "memory" | "workspace" | "skill" | "config" | "script" | "cron"
    memory_type: MemoryType | None  # 如果是 memory 类型
    content_hash: str           # SHA256 of content（去重用）
    size_bytes: int
    preview: str                # 前 100 字符预览
    skip_reason: str | None     # 如果要跳过，原因

@dataclass  
class ImportManifest:
    source: str                 # "openclaw" | "hermes" | "claude-code" | "codex"
    items: list[ImportItem]
    total_size: int
    
    @property
    def summary(self) -> str:
        """Human-readable summary of what will be imported."""
```

### 记忆解析策略

#### OpenClaw Markdown
- 按 `## ` 标题分割为 section
- 每个 section 保留完整上下文（标题 + 内容）
- 超过 4000 字符的 section 按段落再分割
- 保留原始文件路径作为 metadata

#### Hermes § 分隔
- 按 `\n§\n` 分割
- 每条记忆独立存储
- USER.md 的条目标记为 MemoryType.WORKING
- MEMORY.md 的条目按内容推断类型

#### 类型推断增强
```python
def infer_type(text: str, source_path: Path) -> MemoryType:
    """Enhanced type inference based on content + file path."""
    path_str = str(source_path).lower()
    
    # Path-based inference (highest priority)
    if "lessons" in path_str or "learnings" in path_str:
        return MemoryType.PROCEDURAL
    if "projects" in path_str:
        return MemoryType.SEMANTIC
    if "sop" in path_str or "scripts" in path_str:
        return MemoryType.PROCEDURAL
    if "archive" in path_str:
        return MemoryType.EPISODIC
    if re.match(r'.*\d{4}-\d{2}-\d{2}', path_str):
        return MemoryType.EPISODIC
    
    # Content-based inference (fallback)
    # ... existing logic ...
```

### 配置合并

```python
class ConfigMerger:
    """Merge external configs into Caveman config without overwriting."""
    
    def merge_openclaw(self, openclaw_json: dict) -> dict:
        """Extract useful config from openclaw.json.
        
        Extracts:
        - models.providers → caveman providers (API keys, base URLs)
        - channels → reference only (logged, not auto-configured)
        - gateway → reference only
        """
    
    def merge_hermes(self, hermes_yaml: dict) -> dict:
        """Extract useful config from Hermes config.yaml.
        
        Extracts:
        - model → default model
        - providers → API keys
        - terminal → reference only
        """
```

### 去重策略

```python
class ImportDedup:
    """Content-hash based deduplication."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.seen_hashes: set[str] = set()
        self._load_existing_hashes(memory_manager)
    
    def is_duplicate(self, content: str) -> bool:
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
```

### Workspace 文件复制

对于 SOUL.md / USER.md / AGENTS.md 等 workspace 文件：
- 复制到 `~/.caveman/workspace/`
- 如果目标已存在，**不覆盖**，而是创建 `.imported-from-{source}` 备份
- 用户可以手动 diff 和合并

### CLI 接口

```bash
# 自动检测所有可用源
caveman import --detect

# 导入特定源（dry-run）
caveman import --from openclaw --dry-run

# 导入特定源（执行）
caveman import --from openclaw

# 导入所有检测到的源
caveman import --all

# 只导入记忆，不导入配置
caveman import --from openclaw --only memory

# 只导入配置
caveman import --from openclaw --only config

# 排除敏感文件
caveman import --from openclaw --exclude credentials.md
```

### 导入报告

```
🦴 Caveman Import Report
========================
Source: OpenClaw (~/.openclaw/workspace/)
Mode: dry-run

📁 Workspace Files (7)
  ✅ SOUL.md (2.1 KB) → ~/.caveman/workspace/SOUL.md
  ✅ USER.md (0.4 KB) → ~/.caveman/workspace/USER.md
  ⚠️ AGENTS.md (3.7 KB) → exists, will create .imported-from-openclaw backup
  ...

🧠 Memory Entries (847)
  📅 Daily logs: 92 files → 412 entries
  📚 Topic files: 35 files → 198 entries  
  📂 projects/: 15 files → 89 entries
  📂 seo/: 8 files → 48 entries
  ...
  ⏭️ Skipped: 12 (empty or duplicate)

⚙️ Config
  ✅ Will extract: model preferences, provider API keys
  ⏭️ Skipped: gateway, channels (platform-specific)

📊 Total: 847 entries, 1.2 MB
   New: 835 | Duplicate: 12 | Skipped: 0

Run without --dry-run to execute.
```

### 测试要求

1. 单元测试：每个 Importer 的 detect/scan/execute
2. § 分隔解析测试（Hermes 格式）
3. 去重测试（重复导入不产生重复）
4. 配置合并测试（不覆盖已有配置）
5. workspace 文件冲突测试（已存在时创建备份）
6. 深层目录递归测试（memory/projects/seo-matrix/*.md）
7. 大文件处理测试（>10KB 的记忆文件正确分割）
8. 空文件/损坏文件跳过测试
9. dry-run 模式不写任何文件

### 安全

- credentials.md 等敏感文件：导入时标记 `sensitive: true`，不写入 embedding
- 导入前扫描 secret patterns（复用 security/scanner.py）
- 导入日志写入 `~/.caveman/import-log.jsonl`

### 不做的事

- 不导入 OpenClaw 的 SQLite 向量库（格式不同，重新 embedding 更好）
- 不导入 OpenClaw 的 session 数据（太大，且是运行时数据）
- 不导入 Claude Code 的 JSONL 对话历史（太大，价值密度低）
- 不自动激活导入的 cron jobs（只保存为参考）
- 不修改任何源文件
