# Caveman MCP Integration

## Claude Code

Copy `.mcp.json` to your project root, or add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "caveman": {
      "command": "/path/to/caveman/.venv/bin/python",
      "args": ["-m", "caveman.mcp.server"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a memory with tags and importance |
| `memory_search` | Semantic search across memories |
| `memory_recall` | Recall context from previous sessions |
| `shield_save` | Save session state (essence) |
| `shield_load` | Load session essence |
| `reflect` | Extract patterns from task execution |
| `skill_list` | List available skills |
| `skill_get` | Get skill content by name |
| `wiki_compile` | Compile wiki knowledge |
| `wiki_ingest` | Ingest new knowledge into wiki |

## HTTP Transport

```bash
caveman mcp serve --http 8765
```
