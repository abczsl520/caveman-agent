"""Migrate JSON memories to SQLite backend.

Run: python -m caveman.scripts.migrate_json_to_sqlite
"""
import json, asyncio
from pathlib import Path
from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.types import MemoryType
from caveman.paths import MEMORY_DIR, MEMORY_DB_PATH

JSON_FILES = {
    "episodic.json": MemoryType.EPISODIC,
    "semantic.json": MemoryType.SEMANTIC,
    "procedural.json": MemoryType.PROCEDURAL,
    "working.json": MemoryType.WORKING,
}

async def migrate():
    store = SQLiteMemoryStore(str(MEMORY_DB_PATH))
    existing = store.all_entries()
    print(f"SQLite before: {len(existing)} entries")
    total = 0
    for filename, mem_type in JSON_FILES.items():
        path = MEMORY_DIR / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        print(f"  {filename}: {len(data)} entries")
        for d in data:
            content = d.get("content", "")
            if not content or len(content) < 5:
                continue
            await store.store(content, mem_type, metadata=d.get("metadata", {}))
            total += 1
    final = store.all_entries()
    print(f"\nSQLite after: {len(final)} entries")
    print(f"Migrated: {total}")
    store.close()

if __name__ == "__main__":
    asyncio.run(migrate())
