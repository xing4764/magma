"""SQLite-backed graph store for MAGMA knowledge graph."""

import sqlite3
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("magma.graph.sqlite_store")

DB_PATH = os.environ.get("MAGMA_DB_PATH", str(Path(__file__).parent.parent.parent / "data" / "magma.db"))

_conn: Optional[sqlite3.Connection] = None


class SQLiteStore:
    """Simple SQLite graph store with nodes and edges."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()

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
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
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
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label_status ON nodes(label, status)")
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

        # Migrate recall_events table: add source_agent_id / department if missing
        cur = self._conn.execute("PRAGMA table_info(recall_events)")
        recall_existing = {row["name"] for row in cur.fetchall()}
        recall_columns = {
            "source_agent_id": "TEXT",
            "department": "TEXT",
        }
        for name, definition in recall_columns.items():
            if name not in recall_existing:
                self._conn.execute(f"ALTER TABLE recall_events ADD COLUMN {name} {definition}")

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
        with self._write_lock:
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
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, ?, ?)",
                (source_id, target_id, relation, props)
            )
            self._conn.commit()

    def add_edge_once(self, source_id: str, target_id: str, relation: str, properties: Dict = None):
        with self._write_lock:
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
            props = json.dumps(properties or {}, ensure_ascii=False)
            self._conn.execute(
                "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, ?, ?)",
                (source_id, target_id, relation, props)
            )
            self._conn.commit()

    def update_node(self, node_id: str, properties: Dict[str, Any], embedding=None) -> bool:
        """Partial update of node properties. Returns True if node existed.

        Args:
            node_id: Node ID to update.
            properties: Properties to merge/update.
            embedding: Optional new embedding (numpy array or bytes). If provided,
                       updates the embedding column.
        """
        cur = self._conn.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if not row:
            return False
        existing = json.loads(row["properties"])
        existing.update(properties)
        props_json = json.dumps(existing, ensure_ascii=False)
        importance = float(existing.get("importance", 0.5) or 0.5)
        ttl_days = existing.get("ttl_days")
        valid_from = existing.get("valid_from")
        valid_until = existing.get("valid_until")
        status = existing.get("status", "active")
        source_agent_id = existing.get("source_agent_id")
        department = existing.get("department")
        emb = embedding.tobytes() if hasattr(embedding, 'tobytes') else embedding
        with self._write_lock:
            if emb is not None:
                self._conn.execute(
                    """
                    UPDATE nodes
                       SET properties = ?,
                           embedding = ?,
                           importance = ?,
                           ttl_days = ?,
                           valid_from = ?,
                           valid_until = ?,
                           status = ?,
                           source_agent_id = ?,
                           department = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE id = ?
                    """,
                    (props_json, emb, importance, ttl_days, valid_from, valid_until,
                     status, source_agent_id, department, node_id)
                )
            else:
                self._conn.execute(
                    """
                    UPDATE nodes
                       SET properties = ?,
                           importance = ?,
                           ttl_days = ?,
                           valid_from = ?,
                           valid_until = ?,
                           status = ?,
                           source_agent_id = ?,
                           department = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE id = ?
                    """,
                    (props_json, importance, ttl_days, valid_from, valid_until,
                     status, source_agent_id, department, node_id)
                )
            self._conn.commit()
        return True

    def delete_node(self, node_id: str) -> bool:
        """Soft-delete a node by setting status='deleted'. Returns True if node existed."""
        with self._write_lock:
            cur = self._conn.execute(
                "UPDATE nodes SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (node_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

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
            FROM nodes WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params)
        )

        nodes = []
        for row in cur.fetchall():
            node = self._row_to_node(row)
            node["embedding"] = row["embedding"]
            nodes.append(node)
        return nodes

    def query_nodes_properties_only(
        self,
        label: str = None,
        limit: int = 1000,
        include_archived: bool = False,
        property_filters: Dict[str, Any] = None,
    ) -> List[Dict]:
        """Query nodes WITHOUT loading embedding BLOBs (faster when FAISS provides semantics)."""
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
            SELECT id, label, properties, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
            FROM nodes WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params)
        )
        return [self._row_to_node(r) for r in cur.fetchall()]

    def touch_nodes(self, node_ids: List[str]):
        if not node_ids:
            return
        with self._write_lock:
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
        with self._write_lock:
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
        with self._write_lock:
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

    def consolidate(self, semantic_dedup: bool = True, purge_deleted: bool = False) -> Dict[str, int]:
        """Run all consolidation steps including optional semantic dedup.
        
        Args:
            semantic_dedup: If True, also run FAISS-based semantic duplicate detection.
            purge_deleted: If True, physically remove soft-deleted nodes. Default False.
        """
        v2_stats = {}
        if semantic_dedup:
            try:
                from scripts.magma_consolidate_v2 import ConsolidationEngine
                engine = ConsolidationEngine(
                    db_path=self.db_path,
                    semantic_threshold=0.90,
                    entity_threshold=0.50,
                )
                result = engine.apply_merges(dry_run=False)
                engine.close()
                v2_stats = {
                    "v2_semantic_merges": result.get("merged", 0),
                    "v2_same_as_edges": result.get("same_as_edges_added", 0),
                    "v2_errors": len(result.get("errors", [])),
                }
            except Exception as e:
                logger.warning(f"Semantic dedup failed (non-fatal): {e}")
                v2_stats = {"v2_semantic_merges": 0, "v2_error": str(e)}

        with self._write_lock:
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

            # P0-4: Auto-decay importance for long-unused memories
            # Memories not accessed in 30+ days get importance *= 0.95
            cur = self._conn.execute("""
                UPDATE nodes
                   SET importance = MAX(importance * 0.95, 0.05),
                       properties = json_set(properties, '$.importance', MAX(importance * 0.95, 0.05)),
                       updated_at = CURRENT_TIMESTAMP
                 WHERE status = 'active'
                   AND last_accessed_at IS NOT NULL
                   AND datetime(last_accessed_at, '+30 days') < CURRENT_TIMESTAMP
                   AND importance > 0.1
            """)
            decayed_nodes = cur.rowcount

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

        # Physically purge soft-deleted nodes only when explicitly requested
        purged = self.purge_deleted() if purge_deleted else 0

        return {
            "removed_duplicate_edges": removed_edges,
            "removed_orphan_edges": orphan_edges,
            "expired_nodes": expired_nodes,
            "low_importance_nodes": low_importance_nodes,
            "decayed_nodes": decayed_nodes,
            "merged_duplicate_entities": merged,
            "merged_duplicate_l0": merged_l0,
            "purged_deleted_nodes": purged,
            **v2_stats,
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

    def get_neighbor_ids(self, node_ids: List[str]) -> List[str]:
        """Get all neighbor node IDs for a set of source nodes (graph walk helper)."""
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        cur = self._conn.execute(
            f"""
            SELECT source_id, target_id FROM edges
             WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """,
            node_ids + node_ids
        )
        node_set = set(node_ids)
        neighbors = set()
        for row in cur.fetchall():
            src, tgt = row[0], row[1]
            neighbor = tgt if src in node_set else src
            if neighbor:
                neighbors.add(neighbor)
        return list(neighbors)

    def get_neighbor_ids_with_relations(self, node_ids: List[str]) -> List[Dict[str, str]]:
        """Get neighbor node IDs with relation types for weighted graph walk.

        Returns list of {"id": neighbor_id, "relation": relation_type}.
        """
        if not node_ids:
            return []
        placeholders = ", ".join("?" for _ in node_ids)
        cur = self._conn.execute(
            f"""
            SELECT source_id, target_id, relation FROM edges
             WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """,
            node_ids + node_ids
        )
        node_set = set(node_ids)
        neighbors = {}
        for row in cur.fetchall():
            src, tgt, rel = row[0], row[1], row[2]
            neighbor = tgt if src in node_set else src
            if neighbor:
                # Keep best relation weight per neighbor
                if neighbor not in neighbors:
                    neighbors[neighbor] = {"id": neighbor, "relation": rel}
        return list(neighbors.values())

    def get_edges(self, node_id: str) -> List[Dict]:
        cur = self._conn.execute(
            "SELECT source_id, target_id, relation, properties FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id)
        )
        return [{"source": r[0], "target": r[1], "relation": r[2], "properties": json.loads(r[3])} for r in cur.fetchall()]

    def add_causal_edge(self, source_id: str, target_id: str, confidence: float = 0.5,
                        rationale: str = "", inferred_by: str = "llm"):
        """Add a causal edge: source_id caused_by target_id (target caused source).

        Args:
            source_id: The effect node (果).
            target_id: The cause node (因).
            confidence: 0.0-1.0 confidence in this causal relationship.
            rationale: Explanation of why this causal link was inferred.
            inferred_by: What inferred this edge (e.g. 'llm', 'manual').
        """
        props = {
            "confidence": max(0.0, min(1.0, confidence)),
            "rationale": rationale,
            "inferred_by": inferred_by,
        }
        self.add_edge_once(source_id, target_id, "caused_by", props)

    def get_causal_edges(self, node_id: str, direction: str = "both") -> List[Dict]:
        """Get causal edges for a node.

        Args:
            node_id: The node to query.
            direction: 'outgoing' (node caused_by X), 'incoming' (X caused_by node), or 'both'.

        Returns:
            List of causal edge dicts with confidence, rationale.
        """
        if direction == "outgoing":
            cur = self._conn.execute(
                """SELECT source_id, target_id, relation, properties FROM edges
                   WHERE source_id = ? AND relation = 'caused_by'""",
                (node_id,)
            )
        elif direction == "incoming":
            cur = self._conn.execute(
                """SELECT source_id, target_id, relation, properties FROM edges
                   WHERE target_id = ? AND relation = 'caused_by'""",
                (node_id,)
            )
        else:
            cur = self._conn.execute(
                """SELECT source_id, target_id, relation, properties FROM edges
                   WHERE (source_id = ? OR target_id = ?) AND relation = 'caused_by'""",
                (node_id, node_id)
            )
        results = []
        for r in cur.fetchall():
            props = json.loads(r[3])
            results.append({
                "source": r[0],
                "target": r[1],
                "relation": r[2],
                "confidence": props.get("confidence", 0.5),
                "rationale": props.get("rationale", ""),
                "inferred_by": props.get("inferred_by", "unknown"),
            })
        return results

    def get_causal_chain(self, node_id: str, max_depth: int = 5) -> List[List[Dict]]:
        """Walk causal chain backwards from a node to find root causes.

        Returns list of chains, each chain is a list of {node_id, confidence, rationale}.
        """
        chains = []
        visited = set()

        def _walk_back(current_id: str, path: List[Dict], depth: int):
            if depth >= max_depth or current_id in visited:
                if path:
                    chains.append(list(path))
                return
            visited.add(current_id)
            # Find causes of current node (current caused_by cause)
            causal_edges = self.get_causal_edges(current_id, direction="outgoing")
            if not causal_edges:
                if path:
                    chains.append(list(path))
                return
            for edge in causal_edges:
                cause_id = edge["target"]
                path.append({
                    "node_id": cause_id,
                    "confidence": edge["confidence"],
                    "rationale": edge["rationale"],
                })
                _walk_back(cause_id, path, depth + 1)
                path.pop()
            visited.discard(current_id)

        _walk_back(node_id, [], 0)
        return chains

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

    def get_stats(self) -> Dict[str, Any]:
        """Return node/edge counts and layer distribution."""
        node_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM nodes WHERE status != 'deleted'"
        ).fetchone()["cnt"]
        edge_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM edges"
        ).fetchone()["cnt"]
        capture_24h = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM nodes WHERE created_at >= datetime('now', '-1 day')"
        ).fetchone()["cnt"]
        layers = {}
        for row in self._conn.execute(
            "SELECT COALESCE(json_extract(properties, '$.layer'), 'unknown') as layer, COUNT(*) as cnt "
            "FROM nodes WHERE status != 'deleted' GROUP BY layer"
        ).fetchall():
            layers[row["layer"]] = row["cnt"]
        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "capture_24h": capture_24h,
            "layers": layers,
        }

    def get_doctor(self) -> Dict[str, Any]:
        """Enhanced health check returning stats + FAISS status."""
        stats = self.get_stats()
        faiss_status = "unknown"
        try:
            from magma.vector.faiss_index import get_faiss_index
            idx = get_faiss_index(0)
            if idx.is_available:
                faiss_status = f"ok ({idx.count} vectors, dim={idx.dimension})"
            else:
                faiss_status = "not_built"
        except Exception as e:
            faiss_status = f"error: {e}"
        return {
            "status": "ok",
            "service": "magma",
            "version": "0.1.0",
            "db_path": self.db_path,
            "faiss": faiss_status,
            "stats": stats,
        }

    def search_by_entity(self, entity_name: str, entity_type: str = None, limit: int = 20) -> List[Dict]:
        """Search entity nodes by name using json_extract (indexed, avoids O(n) scan)."""
        name_lower = entity_name.lower()
        if entity_type:
            cur = self._conn.execute(
                """
                SELECT id, label, properties, created_at, updated_at, last_accessed_at,
                       access_count, importance, ttl_days, valid_from, valid_until, status,
                       source_agent_id, department
                FROM nodes
                WHERE label = 'entity'
                  AND status != 'deleted'
                  AND LOWER(json_extract(properties, '$.name')) = ?
                  AND json_extract(properties, '$.entity_type') = ?
                LIMIT ?
                """,
                (name_lower, entity_type, limit)
            )
        else:
            cur = self._conn.execute(
                """
                SELECT id, label, properties, created_at, updated_at, last_accessed_at,
                       access_count, importance, ttl_days, valid_from, valid_until, status,
                       source_agent_id, department
                FROM nodes
                WHERE label = 'entity'
                  AND status != 'deleted'
                  AND LOWER(json_extract(properties, '$.name')) = ?
                LIMIT ?
                """,
                (name_lower, limit)
            )
        return [self._row_to_node(r) for r in cur.fetchall()]

    def purge_deleted(self) -> int:
        """Physically remove nodes with status='deleted' and their edges.
        Returns the number of nodes purged."""
        with self._write_lock:
            # First remove edges referencing deleted nodes
            self._conn.execute("""
                DELETE FROM edges WHERE source_id IN (
                    SELECT id FROM nodes WHERE status = 'deleted'
                ) OR target_id IN (
                    SELECT id FROM nodes WHERE status = 'deleted'
                )
            """)
            # Then remove the deleted nodes themselves
            cur = self._conn.execute(
                "DELETE FROM nodes WHERE status = 'deleted'"
            )
            purged = cur.rowcount
            self._conn.commit()
        logger.info(f"Purged {purged} deleted nodes")
        return purged

    def invalidate_old_facts(self, entity_name: str, category: str, exclude_node_id: str = None) -> int:
        """Mark old facts about an entity as superseded (valid_until = now).

        Called when a new fact about the same entity is captured.
        Uses multiple matching strategies:
        1. Entity name in fact_entities JSON array
        2. Entity name substring in content
        3. Category keyword overlap (for phone/email/address etc.)
        Returns the number of facts invalidated.
        """
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name_lower = entity_name.lower()

        # Strategy 1: Entity name in content (substring match)
        # Strategy 2: Same category with overlapping contact keywords
        contact_keywords = ["手机号", "电话", "邮箱", "地址", "微信", "qq"]
        is_contact = any(kw in name_lower for kw in contact_keywords)

        entity_match = "(LOWER(json_extract(properties, '$.fact_entities')) LIKE ? OR LOWER(json_extract(properties, '$.content')) LIKE ?)"
        where_parts = [
            "status = 'active'",
            "label = 'fact'",
            "json_extract(properties, '$.fact_category') = ?",
            entity_match,
        ]
        params = [category, f'%{name_lower}%', f'%{name_lower}%']

        if exclude_node_id:
            where_parts.append("id != ?")
            params.append(exclude_node_id)
        where_parts.append("valid_until IS NULL")
        where_sql = " AND ".join(where_parts)

        with self._write_lock:
            cur = self._conn.execute(f"""
                UPDATE nodes
                   SET valid_until = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE {where_sql}
            """, [now_str] + params)
            invalidated = cur.rowcount

            # Strategy 3: For contact facts, also invalidate same-category facts
            # with overlapping contact keywords in content
            if is_contact and invalidated == 0:
                for kw in contact_keywords:
                    if kw in name_lower:
                        kw_match = "LOWER(json_extract(properties, '$.content')) LIKE ?"
                        where_parts2 = [
                            "status = 'active'",
                            "label = 'fact'",
                            "json_extract(properties, '$.fact_category') = ?",
                            kw_match,
                        ]
                        params2 = [category, f'%{kw}%']
                        if exclude_node_id:
                            where_parts2.append("id != ?")
                            params2.append(exclude_node_id)
                        where_parts2.append("valid_until IS NULL")
                        where_sql2 = " AND ".join(where_parts2)
                        cur2 = self._conn.execute(f"""
                            UPDATE nodes
                               SET valid_until = ?,
                                   updated_at = CURRENT_TIMESTAMP
                             WHERE {where_sql2}
                        """, [now_str] + params2)
                        invalidated += cur2.rowcount

            self._conn.commit()
        return invalidated

    def get_active_facts(self, entity_name: str = None, category: str = None, limit: int = 20) -> List[Dict]:
        """Get currently active facts (valid_until IS NULL)."""
        where = [
            "status = 'active'",
            "label = 'fact'",
            "valid_until IS NULL",
        ]
        params = []
        if entity_name:
            where.append("json_extract(properties, '$.fact_entities') LIKE ?")
            params.append(f'%"{entity_name}"%')
        if category:
            where.append("json_extract(properties, '$.fact_category') = ?")
            params.append(category)
        where_sql = " AND ".join(where)
        params.append(limit)
        cur = self._conn.execute(f"""
            SELECT id, label, properties, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
              FROM nodes
             WHERE {where_sql}
             ORDER BY datetime(created_at) DESC
             LIMIT ?
        """, tuple(params))
        return [self._row_to_node(r) for r in cur.fetchall()]

    def get_fact_timeline(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get time-ordered facts about an entity (current + historical)."""
        cur = self._conn.execute("""
            SELECT id, label, properties, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
              FROM nodes
             WHERE status = 'active'
               AND label = 'fact'
               AND (json_extract(properties, '$.fact_entities') LIKE ?
                    OR LOWER(json_extract(properties, '$.content')) LIKE ?)
             ORDER BY
               CASE WHEN valid_until IS NULL THEN 0 ELSE 1 END,
               datetime(created_at) DESC
             LIMIT ?
        """, (f'%"{entity_name}"%', f'%{entity_name.lower()}%', limit))
        return [self._row_to_node(r) for r in cur.fetchall()]

    def get_recent_conversation(
        self,
        agent_id: str = None,
        session_key: str = None,
        limit: int = 10,
        hours: int = 24,
    ) -> List[Dict]:
        """Get recent conversation nodes for short command resolution.

        Returns the most recent conversation nodes ordered by time,
        filtered by agent_id and/or session_key.
        Used to resolve short commands like "更新", "开始", "继续".
        """
        where = ["status = 'active'"]
        params = []

        # Filter by agent_id if provided
        if agent_id:
            where.append("(source_agent_id = ? OR json_extract(properties, '$.agent_id') = ?)")
            params.extend([agent_id, agent_id])

        # Filter by session_key if provided
        if session_key:
            where.append("json_extract(properties, '$.session_key') = ?")
            params.append(session_key)

        # Time filter (default: last 24 hours)
        if hours:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            where.append("datetime(created_at) >= ?")
            params.append(cutoff)

        where_sql = " AND ".join(where)
        params.append(limit)

        cur = self._conn.execute(f"""
            SELECT id, label, properties, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
              FROM nodes
             WHERE {where_sql}
             ORDER BY datetime(created_at) DESC
             LIMIT ?
        """, tuple(params))

        nodes = [self._row_to_node(r) for r in cur.fetchall()]
        # Return in chronological order (oldest first)
        nodes.reverse()
        return nodes

    def get_pending_decisions(
        self,
        agent_id: str = None,
        session_key: str = None,
        hours: int = 24,
        limit: int = 5,
    ) -> List[Dict]:
        """Get recent L1 decision/task_intent nodes.

        These are the highest-priority targets for short command binding.
        Filtered by agent_id and session_key when provided (same scope as caller).
        """
        where = [
            "status = 'active'",
            "json_extract(properties, '$.layer') = 'L1'",
            "json_extract(properties, '$.kind') IN ('decision', 'task_intent')",
        ]
        params = []

        if agent_id:
            where.append("(source_agent_id = ? OR json_extract(properties, '$.agent_id') = ?)")
            params.extend([agent_id, agent_id])

        if session_key:
            where.append("json_extract(properties, '$.session_key') = ?")
            params.append(session_key)

        if hours:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            where.append("datetime(created_at) >= ?")
            params.append(cutoff)

        where_sql = " AND ".join(where)
        params.append(limit)

        cur = self._conn.execute(f"""
            SELECT id, label, properties, created_at, updated_at,
                   last_accessed_at, access_count, importance, ttl_days,
                   valid_from, valid_until, status, source_agent_id, department
              FROM nodes
             WHERE {where_sql}
             ORDER BY datetime(created_at) DESC
             LIMIT ?
        """, tuple(params))

        return [self._row_to_node(r) for r in cur.fetchall()]

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
