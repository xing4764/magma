"""SQLite-backed graph store for MAGMA knowledge graph."""

import sqlite3
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = os.environ.get("MAGMA_DB_PATH", str(Path(__file__).parent.parent.parent / "data" / "magma.db"))

_conn: Optional[sqlite3.Connection] = None


class SQLiteStore:
    """Simple SQLite graph store with nodes and edges."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self):
        """Create tables if not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP,
                access_count INTEGER NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 0.5,
                ttl_days INTEGER,
                valid_from TIMESTAMP,
                valid_until TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE TABLE IF NOT EXISTS recall_events (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                agent_id TEXT,
                session_key TEXT,
                results TEXT NOT NULL,
                used_node_ids TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS recall_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                signal TEXT NOT NULL,
                delta REAL NOT NULL,
                old_importance REAL,
                new_importance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._migrate()
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_events_session ON recall_events(session_key)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_feedback_node ON recall_feedback(node_id)")
        self._conn.commit()
        return self

    def _migrate(self):
        """Add lifecycle columns to older MAGMA databases."""
        cur = self._conn.execute("PRAGMA table_info(nodes)")
        existing = {row["name"] for row in cur.fetchall()}
        columns = {
            "last_accessed_at": "TIMESTAMP",
            "access_count": "INTEGER NOT NULL DEFAULT 0",
            "importance": "REAL NOT NULL DEFAULT 0.5",
            "ttl_days": "INTEGER",
            "valid_from": "TIMESTAMP",
            "valid_until": "TIMESTAMP",
            "status": "TEXT NOT NULL DEFAULT 'active'",
            "source_agent_id": "TEXT",
            "department": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE nodes ADD COLUMN {name} {definition}")

    def add_node(self, node_id: str, label: str, properties: Dict = None, embedding=None):
        properties = properties or {}
        props = json.dumps(properties, ensure_ascii=False)
        emb = embedding.tobytes() if hasattr(embedding, 'tobytes') else embedding
        importance = float(properties.get("importance", 0.5) or 0.5)
        ttl_days = properties.get("ttl_days")
        valid_from = properties.get("valid_from")
        valid_until = properties.get("valid_until")
        status = properties.get("status", "active")
        source_agent_id = properties.get("source_agent_id")
        department = properties.get("department")
        self._conn.execute(
            """
            INSERT INTO nodes (
                id, label, properties, embedding, updated_at, importance,
                ttl_days, valid_from, valid_until, status,
                source_agent_id, department
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label = excluded.label,
                properties = excluded.properties,
                embedding = excluded.embedding,
                updated_at = CURRENT_TIMESTAMP,
                importance = excluded.importance,
                ttl_days = excluded.ttl_days,
                valid_from = excluded.valid_from,
                valid_until = excluded.valid_until,
                status = excluded.status,
                source_agent_id = excluded.source_agent_id,
                department = excluded.department
            """,
            (node_id, label, props, emb, importance, ttl_days, valid_from, valid_until, status,
             source_agent_id, department)
        )
        self._conn.commit()

    def add_edge(self, source_id: str, target_id: str, relation: str, properties: Dict = None):
        props = json.dumps(properties or {}, ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation, props)
        )
        self._conn.commit()

    def add_edge_once(self, source_id: str, target_id: str, relation: str, properties: Dict = None):
        cur = self._conn.execute(
            """
            SELECT 1 FROM edges
             WHERE source_id = ? AND target_id = ? AND relation = ?
             LIMIT 1
            """,
            (source_id, target_id, relation)
        )
        if cur.fetchone():
            return
        self.add_edge(source_id, target_id, relation, properties)

    def get_node(self, node_id: str) -> Optional[Dict]:
        cur = self._conn.execute(
            """
            SELECT id, label, properties, created_at, updated_at, last_accessed_at,
                   access_count, importance, ttl_days, valid_from, valid_until, status,
                   source_agent_id, department
            FROM nodes WHERE id = ?
            """,
            (node_id,)
        )
        row = cur.fetchone()
        if row:
            return self._row_to_node(row)
        return None

    def query_nodes(self, label: str = None, limit: int = 100, include_archived: bool = False) -> List[Dict]:
        status_clause = "" if include_archived else "status IN ('active', 'stale') AND "
        if label:
            cur = self._conn.execute(
                f"""
                SELECT id, label, properties, created_at, updated_at, last_accessed_at,
                       access_count, importance, ttl_days, valid_from, valid_until, status,
                       source_agent_id, department
                FROM nodes WHERE {status_clause}label = ? LIMIT ?
                """,
                (label, limit)
            )
        else:
            cur = self._conn.execute(
                f"""
                SELECT id, label, properties, created_at, updated_at, last_accessed_at,
                       access_count, importance, ttl_days, valid_from, valid_until, status,
                       source_agent_id, department
                FROM nodes WHERE {status_clause}1 = 1 LIMIT ?
                """,
                (limit,)
            )
        return [self._row_to_node(r) for r in cur.fetchall()]

    def query_nodes_with_embeddings(
        self,
        label: str = None,
        limit: int = 1000,
        include_archived: bool = False,
        property_filters: Dict[str, Any] = None,
    ) -> List[Dict]:
        where = []
        params = []
        if not include_archived:
            where.extend([
                "status = 'active'",
                "(valid_until IS NULL OR datetime(valid_until) >= CURRENT_TIMESTAMP)",
                "(ttl_days IS NULL OR datetime(created_at, '+' || ttl_days || ' days') >= CURRENT_TIMESTAMP)",
            ])
        if label:
            where.append("label = ?")
            params.append(label)
        for key, value in (property_filters or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [item for item in value if item is not None]
                if not values:
                    continue
                placeholders = ", ".join("?" for _ in values)
                where.append(f"json_extract(properties, '$.{key}') IN ({placeholders})")
                params.extend(values)
            else:
                where.append(f"json_extract(properties, '$.{key}') = ?")
                params.append(value)
        where_sql = " AND ".join(where) if where else "1 = 1"
        params.append(limit)
        cur = self._conn.execute(
            f"""
            SELECT id, label, properties, embedding, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
            FROM nodes WHERE {where_sql} LIMIT ?
            """,
            tuple(params)
        )

        nodes = []
        for row in cur.fetchall():
            node = self._row_to_node(row)
            node["embedding"] = row["embedding"]
            nodes.append(node)
        return nodes

    def touch_nodes(self, node_ids: List[str]):
        if not node_ids:
            return
        self._conn.executemany(
            "UPDATE nodes SET last_accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
            [(node_id,) for node_id in node_ids]
        )
        self._conn.commit()

    def record_recall_event(
        self,
        event_id: str,
        query: str,
        agent_id: str = None,
        session_key: str = None,
        results: List[Dict[str, Any]] = None,
        source_agent_id: str = None,
        department: str = None,
    ):
        payload = json.dumps(results or [], ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO recall_events (id, query, agent_id, session_key, results, source_agent_id, department)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                query = excluded.query,
                agent_id = excluded.agent_id,
                session_key = excluded.session_key,
                results = excluded.results,
                source_agent_id = excluded.source_agent_id,
                department = excluded.department
            """,
            (event_id, query, agent_id, session_key, payload, source_agent_id, department)
        )
        self._conn.commit()

    def apply_recall_feedback(
        self,
        event_id: str,
        recalled_node_ids: List[str],
        used_node_ids: List[str],
        positive_delta: float = 0.025,
        unused_delta: float = -0.004,
        source_agent_id: str = None,
        department: str = None,
    ) -> Dict[str, Any]:
        recalled = list(dict.fromkeys(recalled_node_ids or []))
        used = set(used_node_ids or [])
        updates = []
        for node_id in recalled:
            signal = "used" if node_id in used else "unused"
            delta = positive_delta if signal == "used" else unused_delta
            cur = self._conn.execute("SELECT importance FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                continue
            old = float(row["importance"] or 0.5)
            new = min(max(old + delta, 0.05), 1.0)
            self._conn.execute(
                """
                UPDATE nodes
                   SET importance = ?,
                       properties = json_set(properties, '$.importance', ?),
                       updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?
                """,
                (new, new, node_id)
            )
            self._conn.execute(
                """
                INSERT INTO recall_feedback (event_id, node_id, signal, delta, old_importance, new_importance, source_agent_id, department)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, node_id, signal, delta, old, new, source_agent_id, department)
            )
            updates.append({
                "node_id": node_id,
                "signal": signal,
                "delta": round(delta, 6),
                "old_importance": round(old, 6),
                "new_importance": round(new, 6),
            })
        self._conn.execute(
            """
            UPDATE recall_events
               SET used_node_ids = ?,
                   feedback_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (json.dumps(sorted(used), ensure_ascii=False), event_id)
        )
        self._conn.commit()
        return {
            "event_id": event_id,
            "recalled": len(recalled),
            "used": len(used),
            "updates": updates,
        }

    def consolidate(self) -> Dict[str, int]:
        cur = self._conn.execute("""
            DELETE FROM edges WHERE id NOT IN (
                SELECT MIN(id) FROM edges GROUP BY source_id, target_id, relation
            )
        """)
        removed_edges = cur.rowcount

        cur = self._conn.execute("""
            DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes)
               OR target_id NOT IN (SELECT id FROM nodes)
        """)
        orphan_edges = cur.rowcount

        cur = self._conn.execute("""
            UPDATE nodes
               SET status = 'stale'
             WHERE status = 'active'
               AND ttl_days IS NOT NULL
               AND datetime(created_at, '+' || ttl_days || ' days') < CURRENT_TIMESTAMP
        """)
        expired_nodes = cur.rowcount

        cur = self._conn.execute("""
            UPDATE nodes
               SET status = 'stale'
             WHERE status = 'active'
               AND importance < 0.2
               AND access_count = 0
               AND datetime(created_at, '+90 days') < CURRENT_TIMESTAMP
        """)
        low_importance_nodes = cur.rowcount

        cur = self._conn.execute("""
            SELECT properties, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
            FROM nodes WHERE label != 'event' AND status != 'deleted'
            GROUP BY properties HAVING cnt > 1
        """)
        duplicates = cur.fetchall()
        merged = 0
        for row in duplicates:
            ids = row["ids"].split(",")
            keep_id = ids[0]
            for remove_id in ids[1:]:
                self._conn.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, remove_id))
                self._conn.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, remove_id))
                self._add_same_as_edge(remove_id, keep_id, {"reason": "duplicate_properties"})
                self._conn.execute("UPDATE nodes SET status = 'deleted' WHERE id = ?", (remove_id,))
                merged += 1

        cur = self._conn.execute("""
            SELECT
                COALESCE(json_extract(properties, '$.source'), '') as source,
                COALESCE(json_extract(properties, '$.agent_id'), '') as agent_id,
                COALESCE(json_extract(properties, '$.session_key'), '') as session_key,
                COALESCE(json_extract(properties, '$.role'), '') as role,
                COALESCE(json_extract(properties, '$.content'), '') as content,
                COUNT(*) as cnt,
                GROUP_CONCAT(id) as ids
            FROM nodes
            WHERE label = 'event'
              AND status != 'deleted'
              AND json_extract(properties, '$.layer') = 'L0'
              AND COALESCE(json_extract(properties, '$.content'), '') != ''
            GROUP BY source, agent_id, session_key, role, content
            HAVING cnt > 1
        """)
        l0_duplicates = cur.fetchall()
        merged_l0 = 0
        for row in l0_duplicates:
            ids = row["ids"].split(",")
            keep_id = ids[0]
            for remove_id in ids[1:]:
                self._conn.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, remove_id))
                self._conn.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, remove_id))
                self._add_same_as_edge(remove_id, keep_id, {"reason": "duplicate_l0_content"})
                self._conn.execute("UPDATE nodes SET status = 'deleted' WHERE id = ?", (remove_id,))
                merged_l0 += 1

        self._conn.commit()
        return {
            "removed_duplicate_edges": removed_edges,
            "removed_orphan_edges": orphan_edges,
            "expired_nodes": expired_nodes,
            "low_importance_nodes": low_importance_nodes,
            "merged_duplicate_entities": merged,
            "merged_duplicate_l0": merged_l0,
        }

    def _add_same_as_edge(self, source_id: str, target_id: str, properties: Dict[str, Any]):
        cur = self._conn.execute(
            """
            SELECT 1 FROM edges
             WHERE source_id = ? AND target_id = ? AND relation = 'same_as'
             LIMIT 1
            """,
            (source_id, target_id)
        )
        if cur.fetchone():
            return
        self._conn.execute(
            "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, 'same_as', ?)",
            (source_id, target_id, json.dumps(properties or {}, ensure_ascii=False))
        )

    def _row_to_node(self, row: sqlite3.Row) -> Dict:
        properties = json.loads(row["properties"])
        return {
            "id": row["id"],
            "label": row["label"],
            "properties": properties,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_accessed_at": row["last_accessed_at"],
            "access_count": row["access_count"],
            "importance": row["importance"],
            "ttl_days": row["ttl_days"],
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
            "status": row["status"],
            "source_agent_id": row["source_agent_id"],
            "department": row["department"],
        }

    def get_edges(self, node_id: str) -> List[Dict]:
        cur = self._conn.execute(
            "SELECT source_id, target_id, relation, properties FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id)
        )
        return [{"source": r[0], "target": r[1], "relation": r[2], "properties": json.loads(r[3])} for r in cur.fetchall()]

    def get_related_context(self, node_id: str, limit: int = 3, relations: List[str] = None) -> List[Dict]:
        where = ["(e.source_id = ? OR e.target_id = ?)"]
        params = [node_id, node_id]
        if relations:
            placeholders = ", ".join("?" for _ in relations)
            where.append(f"e.relation IN ({placeholders})")
            params.extend(relations)
        params.append(limit)
        cur = self._conn.execute(
            f"""
            SELECT
                e.source_id, e.target_id, e.relation, e.properties,
                n.id as neighbor_id, n.label as neighbor_label, n.properties as neighbor_properties,
                n.status as neighbor_status
            FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE {" AND ".join(where)}
              AND n.status != 'deleted'
            ORDER BY
                CASE e.relation
                    WHEN 'same_as' THEN 0
                    WHEN 'responded_by' THEN 1
                    WHEN 'depends_on' THEN 2
                    WHEN 'caused_by' THEN 3
                    WHEN 'mentions_entity' THEN 4
                    ELSE 5
                END,
                e.id DESC
            LIMIT ?
            """,
            tuple([node_id] + params)
        )
        related = []
        for row in cur.fetchall():
            related.append({
                "source": row["source_id"],
                "target": row["target_id"],
                "relation": row["relation"],
                "properties": json.loads(row["properties"]),
                "neighbor": {
                    "id": row["neighbor_id"],
                    "label": row["neighbor_label"],
                    "properties": json.loads(row["neighbor_properties"]),
                    "status": row["neighbor_status"],
                },
            })
        return related

    def get_version_context(self, node_id: str, limit: int = 3) -> List[Dict]:
        node = self.get_node(node_id)
        if not node:
            return []
        version_key = (node.get("properties") or {}).get("version_key")
        if not version_key:
            return []
        memory_scope = (node.get("properties") or {}).get("memory_scope")
        scope_clause = ""
        params = [node_id, version_key]
        if memory_scope:
            scope_clause = "AND json_extract(properties, '$.memory_scope') = ?"
            params.append(memory_scope)
        params.append(limit)
        cur = self._conn.execute(
            f"""
            SELECT id, label, properties, created_at, updated_at, status
              FROM nodes
             WHERE id != ?
               AND status != 'deleted'
               AND label != 'entity'
               AND json_extract(properties, '$.version_key') = ?
               {scope_clause}
             ORDER BY datetime(updated_at) DESC, datetime(created_at) DESC
             LIMIT ?
            """,
            tuple(params)
        )
        versions = []
        for row in cur.fetchall():
            properties = json.loads(row["properties"])
            versions.append({
                "id": row["id"],
                "label": row["label"],
                "properties": properties,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": row["status"],
                "version_key": version_key,
                "relative": "newer" if str(row["updated_at"]) > str(node.get("updated_at")) else "older_or_peer",
            })
        return versions

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


_store: Optional[SQLiteStore] = None


def get_store(db_path: str = None) -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore(db_path or DB_PATH)
    return _store
