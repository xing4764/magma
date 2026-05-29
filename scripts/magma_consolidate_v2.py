#!/usr/bin/env python
"""MAGMA v2 P1: Memory Consolidation Engine.

Detects and merges duplicate memories using semantic similarity + entity overlap.
Prevents memory bloat by consolidating near-duplicate nodes.

Usage:
    python scripts/magma_consolidate_v2.py --dry-run   # Default: report only
    python scripts/magma_consolidate_v2.py --apply     # Actually merge duplicates
    python scripts/magma_consolidate_v2.py --apply --purge  # Merge + physically purge deleted
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import struct
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("magma.consolidate_v2")

DEFAULT_DB = (
    Path(os.environ["MAGMA_DB_PATH"])
    if "MAGMA_DB_PATH" in os.environ
    else Path(__file__).parent.parent / "data" / "magma.db"
)

# Thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.90
ENTITY_OVERLAP_THRESHOLD = 0.50
FAISS_SEARCH_K = 20  # How many neighbors to check per node


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def decode_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode raw BLOB to float32 numpy array."""
    if not blob:
        return None
    if len(blob) % 4 != 0:
        return None
    count = len(blob) // 4
    if count <= 0:
        return None
    try:
        vec = np.array(struct.unpack("<" + "f" * count, blob), dtype=np.float32)
        return vec
    except struct.error:
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def extract_entities(props: Dict[str, Any]) -> Set[str]:
    """Extract entity set from node properties for overlap calculation."""
    entities = set()

    # Direct entities list
    ent_list = props.get("entities") or props.get("mentions_entities") or []
    if isinstance(ent_list, list):
        for e in ent_list:
            if isinstance(e, str):
                entities.add(e.lower().strip())
            elif isinstance(e, dict):
                name = e.get("name") or e.get("entity") or ""
                if name:
                    entities.add(name.lower().strip())

    # Entity references in content (simple extraction)
    content = props.get("content") or props.get("summary") or ""
    if isinstance(content, str) and len(content) > 0:
        # Store content hash as pseudo-entity for overlap matching
        # This helps detect semantically similar content with same references
        pass

    # Source agent / session as entity proxy
    agent_id = props.get("agent_id") or props.get("source_agent_id") or ""
    if agent_id:
        entities.add(f"agent:{agent_id}")

    # Layer as entity
    layer = props.get("layer") or ""
    if layer:
        entities.add(f"layer:{layer}")

    # Label-based entity
    return entities


def entity_overlap(entities_a: Set[str], entities_b: Set[str]) -> float:
    """Jaccard-like overlap: |intersection| / |union|. Returns 0 if both empty."""
    if not entities_a and not entities_b:
        return 0.0
    if not entities_a or not entities_b:
        return 0.0
    intersection = entities_a & entities_b
    union = entities_a | entities_b
    return len(intersection) / len(union)


def choose_keep_node(node_a: Dict[str, Any], node_b: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Choose which node to keep (newer + higher importance wins). Returns (keep, duplicate)."""
    def sort_key(n: Dict) -> tuple:
        return (
            n.get("access_count", 0) or 0,
            float(n.get("importance", 0.5) or 0.5),
            str(n.get("updated_at", "") or ""),
            str(n.get("created_at", "") or ""),
        )
    if sort_key(node_a) >= sort_key(node_b):
        return node_a, node_b
    return node_b, node_a


def merge_properties(keep_props: Dict, dup_props: Dict) -> Dict:
    """Merge duplicate node properties into keep node. New values override old."""
    merged = dict(keep_props)

    # Merge entity lists (deduplicate)
    keep_entities = set()
    dup_entities = set()

    for field in ("entities", "mentions_entities"):
        lst = merged.get(field)
        if isinstance(lst, list):
            for e in lst:
                if isinstance(e, str):
                    keep_entities.add(e.lower().strip())
        lst = dup_props.get(field)
        if isinstance(lst, list):
            for e in lst:
                if isinstance(e, str):
                    dup_entities.add(e.lower().strip())

    all_entities = sorted(keep_entities | dup_entities)
    if all_entities and "entities" in merged:
        merged["entities"] = all_entities

    # Track merge history
    merge_info = merged.get("_merge_history", [])
    if not isinstance(merge_info, list):
        merge_info = []
    merge_info.append({
        "merged_from": dup_props.get("_original_id", "unknown"),
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    merged["_merge_history"] = merge_info

    # Preserve consolidated count
    prev_count = merged.get("_consolidated_count", 1)
    dup_count = dup_props.get("_consolidated_count", 1)
    merged["_consolidated_count"] = prev_count + dup_count

    return merged


class ConsolidationEngine:
    """FAISS-based semantic duplicate detection and merge engine."""

    def __init__(self, db_path: Path, semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
                 entity_threshold: float = ENTITY_OVERLAP_THRESHOLD):
        self.db_path = db_path
        self.semantic_threshold = semantic_threshold
        self.entity_threshold = entity_threshold
        self.conn = connect(db_path)

    def close(self):
        if self.conn:
            self.conn.close()

    def load_active_nodes_with_embeddings(self) -> List[Dict[str, Any]]:
        """Load all active nodes that have embeddings."""
        cur = self.conn.execute("""
            SELECT id, label, properties, embedding, created_at, updated_at,
                   access_count, importance, status
            FROM nodes
            WHERE status = 'active'
              AND embedding IS NOT NULL
              AND length(embedding) > 0
            ORDER BY created_at
        """)
        nodes = []
        for row in cur.fetchall():
            emb = decode_embedding(row["embedding"])
            if emb is None:
                continue
            props = json.loads(row["properties"] or "{}")
            nodes.append({
                "id": row["id"],
                "label": row["label"],
                "properties": props,
                "embedding": emb,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "access_count": row["access_count"] or 0,
                "importance": float(row["importance"] or 0.5),
            })
        return nodes

    def build_faiss_index(self, nodes: List[Dict[str, Any]]):
        """Build FAISS index from node embeddings."""
        try:
            import faiss
        except ImportError:
            logger.error("faiss-cpu not installed; cannot run semantic dedup")
            return False

        if not nodes:
            logger.warning("No nodes with embeddings to index")
            return False

        vectors = []
        for node in nodes:
            vec = node["embedding"].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        mat = np.array(vectors, dtype=np.float32)
        dim = mat.shape[1]

        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(mat)
        self._faiss_nodes = nodes
        self._faiss_dim = dim

        logger.info(f"FAISS index built: {len(nodes)} vectors, dim={dim}")
        return True

    def find_duplicate_pairs(self) -> List[Dict[str, Any]]:
        """Find all duplicate pairs using FAISS + entity overlap."""
        if not hasattr(self, '_faiss_index') or self._faiss_index is None:
            logger.error("FAISS index not built")
            return []

        nodes = self._faiss_nodes
        n = len(nodes)
        logger.info(f"Scanning {n} nodes for duplicates (thresholds: sim>={self.semantic_threshold}, entity>={self.entity_threshold})")

        # Build vectors matrix
        vectors = np.array([node["embedding"].astype(np.float32) for node in nodes], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-10)

        # Pre-extract entities for all nodes
        node_entities = [extract_entities(node["properties"]) for node in nodes]

        duplicate_pairs = []
        seen_pairs = set()
        batch_size = 500

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_vectors = vectors[batch_start:batch_end]

            # Search all neighbors for each node in batch
            k = min(FAISS_SEARCH_K + 1, n)  # +1 because self-match
            scores, indices = self._faiss_index.search(batch_vectors, k)

            for local_i in range(batch_end - batch_start):
                global_i = batch_start + local_i
                node_a = nodes[global_i]
                entities_a = node_entities[global_i]

                for j in range(1, k):  # Skip self-match at index 0
                    global_j = int(indices[local_i][j])
                    if global_j < 0 or global_j >= n:
                        continue
                    if global_j <= global_i:  # Avoid duplicates
                        continue

                    sim = float(scores[local_i][j])
                    if sim < self.semantic_threshold:
                        continue

                    node_b = nodes[global_j]
                    entities_b = node_entities[global_j]

                    # Entity overlap check
                    e_overlap = entity_overlap(entities_a, entities_b)

                    pair_key = (node_a["id"], node_b["id"])
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    duplicate_pairs.append({
                        "node_a": node_a,
                        "node_b": node_b,
                        "similarity": round(sim, 6),
                        "entity_overlap": round(e_overlap, 6),
                        "meets_entity_threshold": e_overlap >= self.entity_threshold,
                    })

        logger.info(f"Found {len(duplicate_pairs)} candidate pairs with sim >= {self.semantic_threshold}")

        # Filter: must meet both thresholds
        qualified = [p for p in duplicate_pairs if p["meets_entity_threshold"]]
        logger.info(f"After entity overlap filter: {len(qualified)} duplicate pairs")

        return qualified

    def detect_and_report(self) -> Dict[str, Any]:
        """Full detection pipeline. Returns report dict."""
        start_time = time.time()

        nodes = self.load_active_nodes_with_embeddings()
        logger.info(f"Loaded {len(nodes)} active nodes with embeddings")

        if not nodes:
            return {
                "status": "no_data",
                "total_nodes": 0,
                "duplicate_pairs": 0,
                "elapsed_ms": 0,
            }

        if not self.build_faiss_index(nodes):
            return {
                "status": "faiss_unavailable",
                "total_nodes": len(nodes),
                "elapsed_ms": 0,
            }

        pairs = self.find_duplicate_pairs()

        # Deduplicate: each node can only be merged once (greedy matching)
        # Sort by similarity descending to prioritize strongest matches
        pairs.sort(key=lambda p: p["similarity"], reverse=True)

        committed = []
        used_ids = set()
        for pair in pairs:
            a_id = pair["node_a"]["id"]
            b_id = pair["node_b"]["id"]
            if a_id in used_ids or b_id in used_ids:
                continue
            keep, dup = choose_keep_node(pair["node_a"], pair["node_b"])
            committed.append({
                "keep_id": keep["id"],
                "keep_label": keep["label"],
                "keep_updated_at": keep.get("updated_at"),
                "duplicate_id": dup["id"],
                "duplicate_label": dup["label"],
                "duplicate_updated_at": dup.get("updated_at"),
                "similarity": pair["similarity"],
                "entity_overlap": pair["entity_overlap"],
            })
            used_ids.add(a_id)
            used_ids.add(b_id)

        elapsed_ms = int((time.time() - start_time) * 1000)

        report = {
            "status": "ok",
            "mode": "dry-run",
            "total_active_nodes_with_embeddings": len(nodes),
            "candidate_pairs_found": len(pairs),
            "committed_merges": len(commed := committed),
            "estimated_node_reduction": len(committed),
            "thresholds": {
                "semantic_similarity": self.semantic_threshold,
                "entity_overlap": self.entity_threshold,
            },
            "merge_plan": committed[:50],  # Show top 50
            "elapsed_ms": elapsed_ms,
        }

        return report

    def apply_merges(self, dry_run: bool = True) -> Dict[str, Any]:
        """Detect and optionally apply merges."""
        start_time = time.time()

        nodes = self.load_active_nodes_with_embeddings()
        logger.info(f"Loaded {len(nodes)} active nodes with embeddings")

        if not nodes:
            return {"status": "no_data", "total_nodes": 0, "merged": 0, "elapsed_ms": 0}

        if not self.build_faiss_index(nodes):
            return {"status": "faiss_unavailable", "total_nodes": len(nodes), "merged": 0, "elapsed_ms": 0}

        pairs = self.find_duplicate_pairs()

        # Greedy dedup
        pairs.sort(key=lambda p: p["similarity"], reverse=True)
        committed = []
        used_ids = set()
        for pair in pairs:
            a_id = pair["node_a"]["id"]
            b_id = pair["node_b"]["id"]
            if a_id in used_ids or b_id in used_ids:
                continue
            keep, dup = choose_keep_node(pair["node_a"], pair["node_b"])
            committed.append({
                "keep": keep,
                "duplicate": dup,
                "similarity": pair["similarity"],
                "entity_overlap": pair["entity_overlap"],
            })
            used_ids.add(a_id)
            used_ids.add(b_id)

        if dry_run:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "ok",
                "mode": "dry-run",
                "total_nodes_with_embeddings": len(nodes),
                "candidate_pairs": len(pairs),
                "would_merge": len(committed),
                "merges": [
                    {
                        "keep_id": m["keep"]["id"],
                        "duplicate_id": m["duplicate"]["id"],
                        "similarity": m["similarity"],
                        "entity_overlap": m["entity_overlap"],
                    }
                    for m in committed[:50]
                ],
                "elapsed_ms": elapsed_ms,
            }

        # Apply merges
        merged_count = 0
        same_as_count = 0
        errors = []

        for merge_info in committed:
            keep = merge_info["keep"]
            dup = merge_info["duplicate"]
            try:
                self._merge_single(keep, dup, merge_info["similarity"])
                merged_count += 1
                same_as_count += 1
            except Exception as e:
                errors.append({"keep_id": keep["id"], "dup_id": dup["id"], "error": str(e)})
                logger.error(f"Merge failed for {dup['id']} -> {keep['id']}: {e}")

        self.conn.commit()

        # Rebuild FAISS index for consistency (optional, doesn't affect DB)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "ok",
            "mode": "apply",
            "total_nodes_with_embeddings": len(nodes),
            "candidate_pairs": len(pairs),
            "merged": merged_count,
            "same_as_edges_added": same_as_count,
            "errors": errors,
            "elapsed_ms": elapsed_ms,
        }

    def _merge_single(self, keep: Dict, dup: Dict, similarity: float):
        """Merge a single duplicate into the keep node."""
        keep_id = keep["id"]
        dup_id = dup["id"]

        # 1. Add same_as edge from dup -> keep
        self.conn.execute("""
            INSERT INTO edges (source_id, target_id, relation, properties)
            VALUES (?, ?, 'same_as', ?)
        """, (dup_id, keep_id, json.dumps({
            "reason": "semantic_duplicate_v2",
            "similarity": similarity,
            "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False)))

        # 2. Redirect edges from dup to keep
        self.conn.execute("UPDATE edges SET source_id = ? WHERE source_id = ? AND relation != 'same_as'", (keep_id, dup_id))
        self.conn.execute("UPDATE edges SET target_id = ? WHERE target_id = ? AND relation != 'same_as'", (keep_id, dup_id))

        # 3. Merge properties
        keep_props = dict(keep["properties"])
        dup_props = dict(dup["properties"])
        dup_props["_original_id"] = dup_id
        merged_props = merge_properties(keep_props, dup_props)
        merged_props_json = json.dumps(merged_props, ensure_ascii=False)

        # 4. Update keep node with merged properties
        self.conn.execute("""
            UPDATE nodes
               SET properties = ?,
                   updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
        """, (merged_props_json, keep_id))

        # 5. Mark dup as deleted
        self.conn.execute("""
            UPDATE nodes
               SET status = 'deleted',
                   updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
        """, (dup_id,))

        logger.debug(f"Merged {dup_id} -> {keep_id} (sim={similarity:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description="MAGMA v2 P1: Memory Consolidation Engine")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to magma.db")
    parser.add_argument("--apply", action="store_true", help="Apply merges (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Report only (default)")
    parser.add_argument("--purge", action="store_true", help="Physically purge deleted nodes after merge")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--similarity-threshold", type=float, default=SEMANTIC_SIMILARITY_THRESHOLD,
                        help=f"Semantic similarity threshold (default: {SEMANTIC_SIMILARITY_THRESHOLD})")
    parser.add_argument("--entity-threshold", type=float, default=ENTITY_OVERLAP_THRESHOLD,
                        help=f"Entity overlap threshold (default: {ENTITY_OVERLAP_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        return 2

    engine = ConsolidationEngine(
        db_path=args.db,
        semantic_threshold=args.similarity_threshold,
        entity_threshold=args.entity_threshold,
    )

    try:
        result = engine.apply_merges(dry_run=not args.apply)

        # Purge if requested
        if args.apply and args.purge:
            from magma.graph.sqlite_store import SQLiteStore
            store = SQLiteStore(str(args.db))
            store.initialize()
            purged = store.purge_deleted()
            result["purged_deleted_nodes"] = purged
            store.close()

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mode = result.get("mode", "unknown")
            print(f"MAGMA v2 Consolidation Engine ({mode})")
            print(f"DB: {args.db}")
            print(f"Thresholds: similarity >= {args.similarity_threshold}, entity_overlap >= {args.entity_threshold}")
            print()

            if result.get("status") == "no_data":
                print("No active nodes with embeddings found.")
                return 0

            if result.get("status") == "faiss_unavailable":
                print("FAISS unavailable; cannot perform semantic dedup.")
                return 1

            print(f"Nodes with embeddings: {result.get('total_nodes_with_embeddings', 0)}")
            print(f"Candidate pairs found: {result.get('candidate_pairs', 0)}")

            if mode == "dry-run":
                print(f"Would merge: {result.get('would_merge', 0)} pairs")
                merges = result.get("merges", [])
                if merges:
                    print(f"\nTop merges (showing {min(len(merges), 20)}):")
                    for m in merges[:20]:
                        print(f"  {m['duplicate_id']} -> {m['keep_id']}  "
                              f"(sim={m['similarity']:.4f}, entity={m['entity_overlap']:.4f})")
            else:
                print(f"Merged: {result.get('merged', 0)} pairs")
                print(f"Same_as edges added: {result.get('same_as_edges_added', 0)}")
                if result.get("errors"):
                    print(f"Errors: {len(result['errors'])}")
                if result.get("purged_deleted_nodes") is not None:
                    print(f"Purged deleted nodes: {result['purged_deleted_nodes']}")

            print(f"\nElapsed: {result.get('elapsed_ms', 0)}ms")

    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
