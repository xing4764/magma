#!/usr/bin/env python
"""Conservative MAGMA memory governance.

Default mode is a dry run. Apply mode performs only low-risk cleanup:
duplicate/orphan edges are removed, expired/low-value nodes are marked stale,
and duplicate memories are linked with same_as instead of hard-deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DB = (
    Path(os.environ["MAGMA_DB_PATH"])
    if "MAGMA_DB_PATH" in os.environ
    else Path(os.environ.get("MAGMA_DATA_DIR", str(Path(__file__).parent.parent / "data"))) / "magma.db"
)
TEST_SESSION_MARKERS = (
    "magma-memory-eval",
    "magma-capture-test",
    "magma:auto-capture",
    "magma:hook-capture",
    "gateway-fallback",
)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def loads_props(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sample_ids(rows: Iterable[sqlite3.Row], limit: int = 8) -> List[str]:
    return [str(row["id"]) for row in list(rows)[:limit]]


def query_duplicate_edges(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT source_id, target_id, relation, COUNT(*) AS cnt,
               GROUP_CONCAT(id) AS ids
          FROM edges
         GROUP BY source_id, target_id, relation
        HAVING cnt > 1
         ORDER BY cnt DESC
        """
    ).fetchall()


def query_orphan_edges(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT e.id, e.source_id, e.target_id, e.relation
          FROM edges e
          LEFT JOIN nodes s ON s.id = e.source_id
          LEFT JOIN nodes t ON t.id = e.target_id
         WHERE s.id IS NULL OR t.id IS NULL
         ORDER BY e.id
        """
    ).fetchall()


def query_expired_nodes(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, label, created_at, ttl_days, properties
          FROM nodes
         WHERE status = 'active'
           AND COALESCE(ttl_days, json_extract(properties, '$.ttl_days')) IS NOT NULL
           AND datetime(created_at, '+' || COALESCE(ttl_days, json_extract(properties, '$.ttl_days')) || ' days') < CURRENT_TIMESTAMP
         ORDER BY created_at
        """
    ).fetchall()


def query_low_importance_nodes(
    conn: sqlite3.Connection, threshold: float, older_than_days: int
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, label, created_at, importance, access_count, properties
          FROM nodes
         WHERE status = 'active'
           AND importance < ?
           AND access_count = 0
           AND datetime(created_at, '+' || ? || ' days') < CURRENT_TIMESTAMP
         ORDER BY importance ASC, created_at ASC
        """,
        (threshold, older_than_days),
    ).fetchall()


def query_exact_entity_duplicates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT label, properties, COUNT(*) AS cnt, GROUP_CONCAT(id) AS ids
          FROM nodes
         WHERE status = 'active'
           AND label != 'event'
         GROUP BY label, properties
        HAVING cnt > 1
         ORDER BY cnt DESC
        """
    ).fetchall()


def query_exact_l0_duplicates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            COALESCE(source_agent_id, json_extract(properties, '$.agent_id'), '') AS agent_id,
            COALESCE(json_extract(properties, '$.session_key'), '') AS session_key,
            COALESCE(json_extract(properties, '$.role'), '') AS role,
            COALESCE(json_extract(properties, '$.content'), '') AS content,
            COUNT(*) AS cnt,
            GROUP_CONCAT(id) AS ids
          FROM nodes
         WHERE label = 'event'
           AND status = 'active'
           AND json_extract(properties, '$.layer') = 'L0'
           AND COALESCE(json_extract(properties, '$.content'), '') != ''
         GROUP BY agent_id, session_key, role, content
        HAVING cnt > 1
         ORDER BY cnt DESC
        """
    ).fetchall()


def fetch_nodes_by_ids(conn: sqlite3.Connection, ids: List[str]) -> Dict[str, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return {str(row["id"]): row for row in rows}


def choose_keep_id(conn: sqlite3.Connection, ids: List[str]) -> str:
    rows = fetch_nodes_by_ids(conn, ids)
    ranked = sorted(
        ids,
        key=lambda node_id: (
            int(rows[node_id]["access_count"] or 0) if node_id in rows else 0,
            float(rows[node_id]["importance"] or 0.0) if node_id in rows else 0.0,
            str(rows[node_id]["updated_at"] or "") if node_id in rows else "",
            str(rows[node_id]["created_at"] or "") if node_id in rows else "",
        ),
        reverse=True,
    )
    return ranked[0]


def ensure_same_as(
    conn: sqlite3.Connection, source_id: str, target_id: str, reason: str, extra: Optional[Dict[str, Any]] = None
) -> bool:
    if has_same_as(conn, source_id, target_id):
        return False
    props = {"reason": reason, "governed_at": now_iso()}
    if extra:
        props.update(extra)
    conn.execute(
        "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, 'same_as', ?)",
        (source_id, target_id, json.dumps(props, ensure_ascii=False, sort_keys=True)),
    )
    return True


def has_same_as(conn: sqlite3.Connection, source_id: str, target_id: str) -> bool:
    exists = conn.execute(
        """
        SELECT 1 FROM edges
         WHERE source_id = ? AND target_id = ? AND relation = 'same_as'
         LIMIT 1
        """,
        (source_id, target_id),
    ).fetchone()
    return bool(exists)


def mark_stale(conn: sqlite3.Connection, node_ids: List[str], reason: str) -> int:
    changed = 0
    for node_id in node_ids:
        row = conn.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if not row:
            continue
        props = loads_props(row["properties"])
        governance = props.get("governance")
        if not isinstance(governance, dict):
            governance = {}
        governance.update({"status_reason": reason, "updated_at": now_iso()})
        props["governance"] = governance
        conn.execute(
            "UPDATE nodes SET status = 'stale', properties = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'active'",
            (json.dumps(props, ensure_ascii=False, sort_keys=True), node_id),
        )
        changed += conn.total_changes
    return changed


def decode_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    if not blob:
        return None
    if len(blob) % 4 != 0:
        return None
    count = len(blob) // 4
    if count <= 0:
        return None
    try:
        return list(struct.unpack("<" + "f" * count, blob))
    except struct.error:
        return None


def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        aa += x * x
        bb += y * y
    if aa <= 0.0 or bb <= 0.0:
        return 0.0
    return dot / (math.sqrt(aa) * math.sqrt(bb))


def query_l0_for_near_duplicates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, source_agent_id, properties, embedding, importance, access_count, created_at, updated_at
          FROM nodes
         WHERE label = 'event'
           AND status = 'active'
           AND json_extract(properties, '$.layer') = 'L0'
           AND embedding IS NOT NULL
           AND length(COALESCE(json_extract(properties, '$.content'), '')) >= 20
         ORDER BY created_at
        """
    ).fetchall()


def find_near_l0_duplicates(conn: sqlite3.Connection, threshold: float) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in query_l0_for_near_duplicates(conn):
        props = loads_props(row["properties"])
        agent_id = str(row["source_agent_id"] or props.get("agent_id") or "")
        role = str(props.get("role") or "")
        content = str(props.get("content") or "")
        emb = decode_embedding(row["embedding"])
        if not emb:
            continue
        grouped[(agent_id, role)].append(
            {
                "id": str(row["id"]),
                "embedding": emb,
                "content": content,
                "access_count": int(row["access_count"] or 0),
                "importance": float(row["importance"] or 0.0),
                "updated_at": str(row["updated_at"] or ""),
                "created_at": str(row["created_at"] or ""),
            }
        )

    pairs: List[Dict[str, Any]] = []
    for (_agent_id, _role), rows in grouped.items():
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                sim = cosine(left["embedding"], right["embedding"])
                if sim < threshold:
                    continue
                keep, dup = sorted(
                    (left, right),
                    key=lambda item: (
                        item["access_count"],
                        item["importance"],
                        item["updated_at"],
                        item["created_at"],
                    ),
                    reverse=True,
                )
                if has_same_as(conn, dup["id"], keep["id"]):
                    continue
                pairs.append(
                    {
                        "duplicate_id": dup["id"],
                        "keep_id": keep["id"],
                        "similarity": round(sim, 6),
                        "left_preview": left["content"][:120],
                        "right_preview": right["content"][:120],
                    }
                )
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pair in sorted(pairs, key=lambda item: item["similarity"], reverse=True):
        deduped.setdefault((pair["duplicate_id"], pair["keep_id"]), pair)
    return list(deduped.values())


def query_test_noise(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    rows = []
    for row in conn.execute(
        """
        SELECT id, source_agent_id, properties, created_at, status
          FROM nodes
         WHERE status = 'active'
           AND label = 'event'
           AND json_extract(properties, '$.layer') = 'L0'
        """
    ).fetchall():
        props = loads_props(row["properties"])
        session = str(props.get("session_key") or props.get("session_id") or "")
        content = str(props.get("content") or "")
        haystack = f"{session}\n{content}"
        if any(marker in haystack for marker in TEST_SESSION_MARKERS):
            rows.append(row)
    return rows


def build_report(conn: sqlite3.Connection, args: argparse.Namespace) -> Dict[str, Any]:
    duplicate_edges = query_duplicate_edges(conn)
    orphan_edges = query_orphan_edges(conn)
    expired_nodes = query_expired_nodes(conn)
    low_importance = query_low_importance_nodes(conn, args.low_importance_threshold, args.low_importance_days)
    exact_entities = query_exact_entity_duplicates(conn)
    exact_l0 = query_exact_l0_duplicates(conn)
    near_l0 = find_near_l0_duplicates(conn, args.near_duplicate_threshold)
    test_noise = query_test_noise(conn)

    return {
        "db_path": str(args.db),
        "mode": "apply" if args.apply else "dry-run",
        "policy": {
            "low_importance_threshold": args.low_importance_threshold,
            "low_importance_days": args.low_importance_days,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicates_stale": bool(args.stale_near_duplicates),
            "test_session_cleanup": bool(args.include_test_sessions),
        },
        "findings": {
            "duplicate_edge_groups": len(duplicate_edges),
            "duplicate_edges_to_remove": sum(max(0, int(row["cnt"]) - 1) for row in duplicate_edges),
            "orphan_edges": len(orphan_edges),
            "expired_nodes": len(expired_nodes),
            "low_importance_nodes": len(low_importance),
            "exact_duplicate_entity_groups": len(exact_entities),
            "exact_duplicate_l0_groups": len(exact_l0),
            "near_duplicate_l0_pairs": len(near_l0),
            "test_session_noise_nodes": len(test_noise),
        },
        "samples": {
            "orphan_edges": [dict(row) for row in orphan_edges[:5]],
            "expired_nodes": sample_ids(expired_nodes),
            "low_importance_nodes": sample_ids(low_importance),
            "exact_duplicate_entities": [
                {"label": row["label"], "ids": str(row["ids"]).split(",")[:5], "count": row["cnt"]}
                for row in exact_entities[:5]
            ],
            "exact_duplicate_l0": [
                {"agent_id": row["agent_id"], "role": row["role"], "ids": str(row["ids"]).split(",")[:5], "count": row["cnt"]}
                for row in exact_l0[:5]
            ],
            "near_duplicate_l0": near_l0[:5],
            "test_session_noise": sample_ids(test_noise),
        },
        "_rows": {
            "duplicate_edges": duplicate_edges,
            "orphan_edges": orphan_edges,
            "expired_nodes": expired_nodes,
            "low_importance": low_importance,
            "exact_entities": exact_entities,
            "exact_l0": exact_l0,
            "near_l0": near_l0,
            "test_noise": test_noise,
        },
    }


def apply_governance(conn: sqlite3.Connection, report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, int]:
    rows = report["_rows"]
    stats = {
        "duplicate_edges_removed": 0,
        "orphan_edges_removed": 0,
        "expired_nodes_staled": 0,
        "low_importance_nodes_staled": 0,
        "same_as_edges_added": 0,
        "exact_duplicate_nodes_staled": 0,
        "near_duplicate_nodes_staled": 0,
        "test_session_nodes_staled": 0,
    }

    for row in rows["duplicate_edges"]:
        ids = [int(item) for item in str(row["ids"]).split(",") if item]
        remove_ids = sorted(ids)[1:]
        if remove_ids:
            placeholders = ",".join("?" for _ in remove_ids)
            cur = conn.execute(f"DELETE FROM edges WHERE id IN ({placeholders})", tuple(remove_ids))
            stats["duplicate_edges_removed"] += cur.rowcount

    if rows["orphan_edges"]:
        ids = [int(row["id"]) for row in rows["orphan_edges"]]
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(f"DELETE FROM edges WHERE id IN ({placeholders})", tuple(ids))
        stats["orphan_edges_removed"] += cur.rowcount

    expired_ids = [str(row["id"]) for row in rows["expired_nodes"]]
    if expired_ids:
        mark_stale(conn, expired_ids, "expired_ttl")
        stats["expired_nodes_staled"] = len(expired_ids)

    low_ids = [str(row["id"]) for row in rows["low_importance"]]
    if low_ids:
        mark_stale(conn, low_ids, "low_importance_unused")
        stats["low_importance_nodes_staled"] = len(low_ids)

    for row in rows["exact_entities"]:
        ids = str(row["ids"]).split(",")
        keep_id = choose_keep_id(conn, ids)
        duplicate_ids = [node_id for node_id in ids if node_id != keep_id]
        for duplicate_id in duplicate_ids:
            if ensure_same_as(conn, duplicate_id, keep_id, "duplicate_entity_properties"):
                stats["same_as_edges_added"] += 1
        mark_stale(conn, duplicate_ids, "duplicate_entity_properties")
        stats["exact_duplicate_nodes_staled"] += len(duplicate_ids)

    for row in rows["exact_l0"]:
        ids = str(row["ids"]).split(",")
        keep_id = choose_keep_id(conn, ids)
        duplicate_ids = [node_id for node_id in ids if node_id != keep_id]
        for duplicate_id in duplicate_ids:
            if ensure_same_as(conn, duplicate_id, keep_id, "duplicate_l0_content"):
                stats["same_as_edges_added"] += 1
        mark_stale(conn, duplicate_ids, "duplicate_l0_content")
        stats["exact_duplicate_nodes_staled"] += len(duplicate_ids)

    for pair in rows["near_l0"]:
        if ensure_same_as(
            conn,
            pair["duplicate_id"],
            pair["keep_id"],
            "near_duplicate_l0_embedding",
            {"similarity": pair["similarity"]},
        ):
            stats["same_as_edges_added"] += 1
        if args.stale_near_duplicates:
            mark_stale(conn, [pair["duplicate_id"]], "near_duplicate_l0_embedding")
            stats["near_duplicate_nodes_staled"] += 1

    if args.include_test_sessions:
        test_ids = [str(row["id"]) for row in rows["test_noise"]]
        if test_ids:
            mark_stale(conn, test_ids, "test_session_noise")
            stats["test_session_nodes_staled"] = len(test_ids)

    conn.commit()
    return stats


def strip_internal_rows(report: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(report)
    public.pop("_rows", None)
    return public


def print_human(report: Dict[str, Any], apply_stats: Optional[Dict[str, int]] = None) -> None:
    public = strip_internal_rows(report)
    findings = public["findings"]
    print(f"MAGMA Governance ({public['mode']})")
    print(f"DB: {public['db_path']}")
    print("")
    for key, value in findings.items():
        print(f"{key}: {value}")
    print("")
    print("Samples:")
    for key, value in public["samples"].items():
        print(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    if apply_stats is not None:
        print("")
        print("Applied:")
        for key, value in apply_stats.items():
            print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative MAGMA memory governance")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to magma.db")
    parser.add_argument("--apply", action="store_true", help="Apply soft governance actions")
    parser.add_argument("--dry-run", action="store_true", help="Report only; this is the default")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--low-importance-threshold", type=float, default=0.2)
    parser.add_argument("--low-importance-days", type=int, default=90)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.985)
    parser.add_argument("--stale-near-duplicates", action="store_true", help="Also mark near-duplicate L0 nodes stale")
    parser.add_argument("--include-test-sessions", action="store_true", help="Apply stale status to test/eval capture nodes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        return 2

    conn = connect(args.db)
    try:
        report = build_report(conn, args)
        apply_stats = apply_governance(conn, report, args) if args.apply else None
        if args.json:
            payload = strip_internal_rows(report)
            if apply_stats is not None:
                payload["applied"] = apply_stats
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_human(report, apply_stats)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
