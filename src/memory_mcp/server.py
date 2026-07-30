"""Memory MCP Server — persistent knowledge graph with SQLite backend.

Replaces the off-the-shelf mcp-server-memory (npm) with a Python implementation
that adds timestamps, graph traversal, fuzzy search, and temporal queries while
maintaining backward compatibility with the JSONL file format.

Storage:
  SQLite  at MEMORY_DB_PATH   (default: ~/.vibe/memory.db)
  JSONL   at MEMORY_FILE_PATH (default: ~/.vibe/memory.jsonl)

On first startup, imports existing JSONL into SQLite. All writes go to both.
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", Path.home() / ".vibe" / "memory.db"))
JSONL_PATH = Path(os.environ.get("MEMORY_FILE_PATH", Path.home() / ".vibe" / "memory.jsonl"))

mcp = FastMCP(
    "memory",
    instructions="Persistent knowledge graph with graph traversal, fuzzy search, and temporal queries",
)

# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Get or create the SQLite connection with WAL mode."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.row_factory = sqlite3.Row
        _init_schema()
        _maybe_migrate_from_jsonl()
    return _conn


def _init_schema() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity TEXT NOT NULL,
            to_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(from_entity, to_entity, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity_id);
        CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_entity);
        CREATE INDEX IF NOT EXISTS idx_rel_to ON relations(to_entity);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    """)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSONL migration & sync
# ---------------------------------------------------------------------------

def _maybe_migrate_from_jsonl() -> None:
    """Import existing JSONL into SQLite if the database is empty."""
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if count > 0 or not JSONL_PATH.exists():
        return

    entities_seen: set[str] = set()
    for line in _read_jsonl():
        item = json.loads(line)
        if item.get("type") == "entity":
            name = item["name"]
            if name in entities_seen:
                # Duplicate entity — append observations to existing
                for obs in item.get("observations", []):
                    conn.execute(
                        "INSERT INTO observations (entity_id, content, created_at) "
                        "SELECT id, ?, ? FROM entities WHERE name = ?",
                        (obs, _now(), name),
                    )
                continue
            entities_seen.add(name)
            conn.execute(
                "INSERT INTO entities (name, entity_type, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, item.get("entityType", "unknown"), _now(), _now()),
            )
            for obs in item.get("observations", []):
                conn.execute(
                    "INSERT INTO observations (entity_id, content, created_at) "
                    "SELECT id, ?, ? FROM entities WHERE name = ?",
                    (obs, _now(), name),
                )
        elif item.get("type") == "relation":
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO relations (from_entity, to_entity, relation_type, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (item["from"], item["to"], item.get("relationType", "related_to"), _now()),
                )
            except (KeyError, sqlite3.IntegrityError):
                continue
    conn.commit()


def _read_jsonl() -> list[str]:
    """Read non-empty lines from the JSONL file."""
    if not JSONL_PATH.exists():
        return []
    try:
        text = JSONL_PATH.read_text(encoding="utf-8", errors="replace")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except OSError:
        return []


def _append_jsonl(entry: dict) -> None:
    """Append a single entry to the JSONL file."""
    try:
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _rebuild_jsonl() -> None:
    """Rebuild the entire JSONL file from SQLite."""
    conn = _get_conn()
    entries: list[dict] = []

    for row in conn.execute("SELECT name, entity_type FROM entities ORDER BY id"):
        obs_rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (row["name"],),
        ).fetchall()
        entries.append({
            "type": "entity",
            "name": row["name"],
            "entityType": row["entity_type"],
            "observations": [o["content"] for o in obs_rows],
        })

    for row in conn.execute("SELECT from_entity, to_entity, relation_type FROM relations ORDER BY id"):
        entries.append({
            "type": "relation",
            "from": row["from_entity"],
            "to": row["to_entity"],
            "relationType": row["relation_type"],
        })

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fuzzy search helpers
# ---------------------------------------------------------------------------

def _trigram_similarity(a: str, b: str) -> float:
    """Simple trigram similarity between two strings (0.0 to 1.0)."""
    a = a.lower()
    b = b.lower()
    if a == b:
        return 1.0
    if len(a) < 3 or len(b) < 3:
        # Short strings: direct substring match
        return 0.8 if (a in b or b in a) else 0.0

    def trigrams(s: str) -> set[str]:
        return {s[i:i + 3] for i in range(len(s) - 2)}

    ta = trigrams(a)
    tb = trigrams(b)
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Tools — CRUD (backward compatible)
# ---------------------------------------------------------------------------

@mcp.tool()
def create_entities(entities: list[dict]) -> str:
    """Create multiple new entities in the knowledge graph.

    Each entity must have: name (str), entityType (str), observations (list[str]).
    """
    conn = _get_conn()
    created = 0
    errors: list[str] = []

    for e in entities:
        name = e.get("name")
        etype = e.get("entityType", "unknown")
        observations = e.get("observations", [])

        if not name:
            errors.append("Missing 'name' in entity")
            continue

        try:
            conn.execute(
                "INSERT INTO entities (name, entity_type, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, etype, _now(), _now()),
            )
            for obs in observations:
                conn.execute(
                    "INSERT INTO observations (entity_id, content, created_at) "
                    "SELECT id, ?, ? FROM entities WHERE name = ?",
                    (obs, _now(), name),
                )
            created += 1
            _append_jsonl({
                "type": "entity",
                "name": name,
                "entityType": etype,
                "observations": observations,
            })
        except sqlite3.IntegrityError:
            errors.append(f"Entity already exists: {name}")
        except Exception as ex:
            errors.append(f"Error creating '{name}': {ex}")

    conn.commit()
    msg = f"Created {created} entities."
    if errors:
        msg += f" Errors: {'; '.join(errors)}"
    return msg


@mcp.tool()
def create_relations(relations: list[dict]) -> str:
    """Create multiple new relations between entities in the knowledge graph.

    Each relation must have: from (str), to (str), relationType (str).
    """
    conn = _get_conn()
    created = 0
    errors: list[str] = []

    for r in relations:
        from_e = r.get("from")
        to_e = r.get("to")
        rtype = r.get("relationType", "related_to")

        if not from_e or not to_e:
            errors.append("Missing 'from' or 'to' in relation")
            continue

        try:
            conn.execute(
                "INSERT INTO relations (from_entity, to_entity, relation_type, created_at) VALUES (?, ?, ?, ?)",
                (from_e, to_e, rtype, _now()),
            )
            created += 1
            _append_jsonl({
                "type": "relation",
                "from": from_e,
                "to": to_e,
                "relationType": rtype,
            })
        except sqlite3.IntegrityError:
            errors.append(f"Relation already exists: {from_e} -> {to_e} ({rtype})")
        except Exception as ex:
            errors.append(f"Error creating relation: {ex}")

    conn.commit()
    msg = f"Created {created} relations."
    if errors:
        msg += f" Errors: {'; '.join(errors)}"
    return msg


@mcp.tool()
def add_observations(observations: list[dict]) -> str:
    """Add new observations to existing entities in the knowledge graph.

    Each item must have: entityName (str), contents (list[str]).
    """
    conn = _get_conn()
    added = 0
    errors: list[str] = []

    for item in observations:
        entity_name = item.get("entityName")
        contents = item.get("contents", [])

        if not entity_name:
            errors.append("Missing 'entityName'")
            continue

        row = conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
        if not row:
            errors.append(f"Entity not found: {entity_name}")
            continue

        entity_id = row["id"]
        now = _now()
        for content in contents:
            conn.execute(
                "INSERT INTO observations (entity_id, content, created_at) VALUES (?, ?, ?)",
                (entity_id, content, now),
            )
            added += 1

        conn.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (now, entity_id))

    conn.commit()
    # Rebuild JSONL to reflect added observations
    if added > 0:
        _rebuild_jsonl()

    msg = f"Added {added} observations."
    if errors:
        msg += f" Errors: {'; '.join(errors)}"
    return msg


@mcp.tool()
def delete_entities(entityNames: list[str]) -> str:
    """Delete multiple entities and their associated relations from the knowledge graph."""
    conn = _get_conn()
    deleted = 0

    for name in entityNames:
        cursor = conn.execute("DELETE FROM entities WHERE name = ?", (name,))
        deleted += cursor.rowcount

    conn.commit()
    if deleted > 0:
        _rebuild_jsonl()
    return f"Deleted {deleted} entities."


@mcp.tool()
def delete_observations(deletions: list[dict]) -> str:
    """Delete specific observations from entities in the knowledge graph.

    Each item must have: entityName (str), observations (list[str] — exact content match).
    """
    conn = _get_conn()
    deleted = 0
    errors: list[str] = []

    for item in deletions:
        entity_name = item.get("entityName")
        obs_to_delete = item.get("observations", [])

        if not entity_name:
            errors.append("Missing 'entityName'")
            continue

        row = conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
        if not row:
            errors.append(f"Entity not found: {entity_name}")
            continue

        entity_id = row["id"]
        for obs_content in obs_to_delete:
            cursor = conn.execute(
                "DELETE FROM observations WHERE entity_id = ? AND content = ?",
                (entity_id, obs_content),
            )
            deleted += cursor.rowcount

    conn.commit()
    if deleted > 0:
        _rebuild_jsonl()
    msg = f"Deleted {deleted} observations."
    if errors:
        msg += f" Errors: {'; '.join(errors)}"
    return msg


@mcp.tool()
def delete_relations(relations: list[dict]) -> str:
    """Delete multiple relations from the knowledge graph.

    Each relation must have: from (str), to (str), relationType (str).
    """
    conn = _get_conn()
    deleted = 0

    for r in relations:
        cursor = conn.execute(
            "DELETE FROM relations WHERE from_entity = ? AND to_entity = ? AND relation_type = ?",
            (r.get("from"), r.get("to"), r.get("relationType", "related_to")),
        )
        deleted += cursor.rowcount

    conn.commit()
    if deleted > 0:
        _rebuild_jsonl()
    return f"Deleted {deleted} relations."


# ---------------------------------------------------------------------------
# Tools — Query (improved)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_nodes(query: str) -> str:
    """Search for nodes in the knowledge graph.

    Case-insensitive token match across entity names, types, and observation content.
    Returns matching entities with their observations.
    """
    conn = _get_conn()
    tokens = query.lower().split()

    # Build a query that matches any token against name, type, or observations
    conditions = []
    params: list[str] = []
    for token in tokens:
        like = f"%{token}%"
        conditions.append(
            "(LOWER(e.name) LIKE ? OR LOWER(e.entity_type) LIKE ? "
            "OR e.id IN (SELECT entity_id FROM observations WHERE LOWER(content) LIKE ?))"
        )
        params.extend([like, like, like])

    where = " OR ".join(conditions)
    rows = conn.execute(
        f"SELECT DISTINCT e.name, e.entity_type FROM entities e WHERE {where} ORDER BY e.name",
        params,
    ).fetchall()

    if not rows:
        return json.dumps({"entities": [], "relations": []})

    result_entities = []
    result_relations = []
    names = {r["name"] for r in rows}

    for row in rows:
        obs_rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (row["name"],),
        ).fetchall()
        result_entities.append({
            "name": row["name"],
            "entityType": row["entity_type"],
            "observations": [o["content"] for o in obs_rows],
        })

    # Include relations between matched entities
    rel_rows = conn.execute(
        "SELECT from_entity, to_entity, relation_type FROM relations WHERE from_entity IN ({}) OR to_entity IN ({})".format(
            ",".join("?" * len(names)), ",".join("?" * len(names))
        ),
        list(names) + list(names),
    ).fetchall()

    for rel in rel_rows:
        result_relations.append({
            "from": rel["from_entity"],
            "to": rel["to_entity"],
            "relationType": rel["relation_type"],
        })

    return json.dumps({"entities": result_entities, "relations": result_relations})


@mcp.tool()
def open_nodes(names: list[str]) -> str:
    """Open specific nodes in the knowledge graph by their names.

    Returns full entity details including all observations.
    """
    conn = _get_conn()
    entities = []

    for name in names:
        row = conn.execute(
            "SELECT name, entity_type, created_at, updated_at FROM entities WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            continue

        obs_rows = conn.execute(
            "SELECT content, created_at FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (name,),
        ).fetchall()

        entities.append({
            "name": row["name"],
            "entityType": row["entity_type"],
            "observations": [o["content"] for o in obs_rows],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    return json.dumps({"entities": entities})


@mcp.tool()
def read_graph() -> str:
    """Read the entire knowledge graph.

    Returns all entities with observations and all relations.
    """
    conn = _get_conn()
    entities = []
    relations = []

    for row in conn.execute("SELECT name, entity_type FROM entities ORDER BY name"):
        obs_rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (row["name"],),
        ).fetchall()
        entities.append({
            "name": row["name"],
            "entityType": row["entity_type"],
            "observations": [o["content"] for o in obs_rows],
        })

    for row in conn.execute("SELECT from_entity, to_entity, relation_type FROM relations ORDER BY id"):
        relations.append({
            "from": row["from_entity"],
            "to": row["to_entity"],
            "relationType": row["relation_type"],
        })

    return json.dumps({"entities": entities, "relations": relations})


# ---------------------------------------------------------------------------
# Tools — New (graph traversal, temporal, fuzzy)
# ---------------------------------------------------------------------------

@mcp.tool()
def traverse(start_node: str, depth: int = 1) -> str:
    """Traverse the graph from a starting node, returning all entities within N hops.

    Args:
        start_node: Entity name to start from
        depth:     Number of hops to traverse (default 1, max 3)
    """
    conn = _get_conn()

    # Verify start node exists
    row = conn.execute("SELECT name FROM entities WHERE name = ?", (start_node,)).fetchone()
    if not row:
        return json.dumps({"error": f"Entity not found: {start_node}", "entities": [], "relations": []})

    depth = max(1, min(depth, 3))
    visited: set[str] = {start_node}
    frontier = {start_node}
    all_relations: list[dict] = []

    for _ in range(depth):
        if not frontier:
            break
        next_frontier: set[str] = set()
        placeholders = ",".join("?" * len(frontier))
        frontier_list = list(frontier)

        # Outgoing
        rels = conn.execute(
            f"SELECT from_entity, to_entity, relation_type FROM relations WHERE from_entity IN ({placeholders})",
            frontier_list,
        ).fetchall()
        for r in rels:
            all_relations.append({
                "from": r["from_entity"],
                "to": r["to_entity"],
                "relationType": r["relation_type"],
            })
            if r["to_entity"] not in visited:
                visited.add(r["to_entity"])
                next_frontier.add(r["to_entity"])

        # Incoming
        rels = conn.execute(
            f"SELECT from_entity, to_entity, relation_type FROM relations WHERE to_entity IN ({placeholders})",
            frontier_list,
        ).fetchall()
        for r in rels:
            all_relations.append({
                "from": r["from_entity"],
                "to": r["to_entity"],
                "relationType": r["relation_type"],
            })
            if r["from_entity"] not in visited:
                visited.add(r["from_entity"])
                next_frontier.add(r["from_entity"])

        frontier = next_frontier

    # Fetch entity details for all visited nodes
    entities = []
    for name in sorted(visited):
        erow = conn.execute(
            "SELECT name, entity_type FROM entities WHERE name = ?", (name,),
        ).fetchone()
        if not erow:
            continue
        obs_rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (name,),
        ).fetchall()
        entities.append({
            "name": erow["name"],
            "entityType": erow["entity_type"],
            "observations": [o["content"] for o in obs_rows],
        })

    return json.dumps({
        "entities": entities,
        "relations": all_relations,
        "hops": depth,
        "nodes_found": len(entities),
    })


@mcp.tool()
def recent(hours: int = 24) -> str:
    """Return entities, relations, and observations created or updated in the last N hours.

    Args:
        hours: Look-back window in hours (default 24)
    """
    conn = _get_conn()
    cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
    # SQLite datetime comparison: convert cutoff to ISO format
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Entities created or updated recently
    entities = []
    for row in conn.execute(
        "SELECT name, entity_type, created_at, updated_at FROM entities WHERE updated_at >= ? ORDER BY updated_at DESC",
        (cutoff_iso,),
    ):
        obs_rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
            (row["name"],),
        ).fetchall()
        entities.append({
            "name": row["name"],
            "entityType": row["entity_type"],
            "observations": [o["content"] for o in obs_rows],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    # Relations created recently
    relations = []
    for row in conn.execute(
        "SELECT from_entity, to_entity, relation_type, created_at FROM relations WHERE created_at >= ? ORDER BY created_at DESC",
        (cutoff_iso,),
    ):
        relations.append({
            "from": row["from_entity"],
            "to": row["to_entity"],
            "relationType": row["relation_type"],
            "created_at": row["created_at"],
        })

    return json.dumps({
        "entities": entities,
        "relations": relations,
        "window_hours": hours,
        "cutoff": cutoff_iso,
    })


@mcp.tool()
def search_similar(name: str, threshold: float = 0.3) -> str:
    """Fuzzy search for entity names using trigram similarity.

    Args:
        name:      Name to search for (fuzzy matched)
        threshold: Minimum similarity score 0.0–1.0 (default 0.3)
    """
    conn = _get_conn()
    rows = conn.execute("SELECT name, entity_type FROM entities").fetchall()

    scored = []
    for row in rows:
        score = _trigram_similarity(name, row["name"])
        if score >= threshold:
            scored.append((row["name"], row["entity_type"], score))

    scored.sort(key=lambda x: -x[2])

    return json.dumps({
        "query": name,
        "threshold": threshold,
        "matches": [
            {"name": s[0], "entityType": s[1], "score": round(s[2], 3)}
            for s in scored[:20]
        ],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server."""
    # Ensure DB is initialized on startup
    _get_conn()
    mcp.run(transport="stdio")
