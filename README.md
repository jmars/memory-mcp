# memory-mcp

Persistent knowledge graph MCP server with SQLite backend. Drop-in replacement for the off-the-shelf `mcp-server-memory` (npm) — same tool signatures, but with graph traversal, fuzzy search, temporal queries, and timestamps.

Part of the [Palimpsest](https://github.com/palimpsest-labs) intelligence fusion toolkit.

## Why

The npm `mcp-server-memory` stores everything in a flat JSONL file. Every operation reads the entire file into memory. No timestamps, no graph queries, keyword-only search.

This replaces it with SQLite (WAL mode) while maintaining JSONL export compatibility. Adds:

- **Graph traversal** — `traverse("shen-meta", depth=2)` finds 46 connected nodes across 112 relations
- **Fuzzy search** — trigram similarity for entity names (no more exact-case matching)
- **Temporal queries** — `recent(hours=24)` shows everything added or modified
- **Timestamps** — every entity, observation, and relation has `created_at`/`updated_at`
- **Case-insensitive search** — `search_nodes("giovanni")` matches `Giovanni`

## Tools

### CRUD (backward compatible)
- `create_entities`, `create_relations`, `add_observations`
- `delete_entities`, `delete_observations`, `delete_relations`
- `search_nodes`, `open_nodes`, `read_graph`

### New
- **`traverse(start_node, depth)`** — graph walk from a node, returns all entities within N hops
- **`recent(hours)`** — entities/relations created or updated in the last N hours
- **`search_similar(name, threshold)`** — trigram fuzzy matching for entity names

## Usage

```
pip install -e .
```

Reads from `MEMORY_FILE_PATH` (default: `~/.vibe/memory.jsonl`). Stores in SQLite at `MEMORY_DB_PATH` (default: `~/.vibe/memory.db`). On first run, auto-migrates existing JSONL into SQLite. All writes go to both stores.
