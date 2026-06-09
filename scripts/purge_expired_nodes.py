#!/usr/bin/env python3
"""Purge expired MAGMA nodes — mark stale and clean FAISS vectors.

Usage:
  python scripts/purge_expired_nodes.py --dry-run   # Count only, no changes
  python scripts/purge_expired_nodes.py --apply      # Execute the cleanup

Nodes with valid_until < now AND status = 'active' are marked 'stale'.
Their FAISS vectors are removed from the index (if present).
"""

import argparse
import json
import os
import struct
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

DB_PATH = PROJECT_ROOT / "data" / "magma.db"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
ID_MAP_PATH = PROJECT_ROOT / "data" / "id_map.json"
FAISS_META_PATH = PROJECT_ROOT / "data" / "faiss_meta.json"

CST = timezone(timedelta(hours=8))


def get_expired_nodes(conn):
    """Return list of (node_id, valid_until) for expired active nodes."""
    cur = conn.execute(
        "SELECT id, valid_until FROM nodes "
        "WHERE valid_until IS NOT NULL AND valid_until < datetime('now') AND status = 'active'"
    )
    return [(row[0], row[1]) for row in cur]


def load_id_map():
    """Load FAISS id_map.json -> {position_str: node_id}."""
    if not ID_MAP_PATH.exists():
        return {}
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_id_map(id_map):
    """Save FAISS id_map.json."""
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)


def load_faiss_meta():
    """Load faiss_meta.json."""
    if not FAISS_META_PATH.exists():
        return {}
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_faiss_meta(meta):
    """Save faiss_meta.json."""
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def remove_vectors_from_faiss(expired_ids: set, dry_run: bool = True):
    """Remove expired node vectors from FAISS index by rebuilding without them.

    Returns (total_before, removed_count, total_after).
    """
    try:
        import faiss
    except ImportError:
        print("  [WARN] faiss-cpu not installed; skipping FAISS cleanup")
        return 0, 0, 0

    if not FAISS_INDEX_PATH.exists():
        print("  [INFO] No faiss.index file; skipping FAISS cleanup")
        return 0, 0, 0

    # Load existing index
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    id_map = load_id_map()
    total_before = index.ntotal

    if total_before == 0:
        return 0, 0, 0

    # Build position -> node_id mapping
    pos_to_id = {int(k): v for k, v in id_map.items()}

    # Collect vectors to keep
    keep_positions = []
    removed = 0
    for pos in range(total_before):
        node_id = pos_to_id.get(pos)
        if node_id and node_id in expired_ids:
            removed += 1
        else:
            keep_positions.append(pos)

    if removed == 0:
        print(f"  FAISS: {total_before} vectors, 0 to remove (none expired in index)")
        return total_before, 0, total_before

    if dry_run:
        print(f"  FAISS: {total_before} vectors, would remove {removed}, would keep {len(keep_positions)}")
        return total_before, removed, len(keep_positions)

    # Rebuild index without expired vectors
    if keep_positions:
        # Extract vectors to keep using reconstruct_n
        dim = index.d
        all_vecs = index.reconstruct_n(0, total_before)  # numpy array (total_before, dim)
        keep_vecs = all_vecs[keep_positions].astype(np.float32)

        new_index = faiss.IndexFlatIP(dim)
        new_index.add(keep_vecs)

        # Rebuild id_map
        new_id_map = {}
        for new_pos, old_pos in enumerate(keep_positions):
            node_id = pos_to_id.get(old_pos)
            if node_id:
                new_id_map[str(new_pos)] = node_id

        faiss.write_index(new_index, str(FAISS_INDEX_PATH))
        save_id_map(new_id_map)

        # Update meta
        meta = load_faiss_meta()
        meta["node_count"] = len(keep_positions)
        meta["rebuilt_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_faiss_meta(meta)

        print(f"  FAISS: {total_before} -> {len(keep_positions)} vectors (removed {removed})")
        return total_before, removed, len(keep_positions)
    else:
        # All vectors removed — write empty index
        dim = index.d
        new_index = faiss.IndexFlatIP(dim)
        faiss.write_index(new_index, str(FAISS_INDEX_PATH))
        save_id_map({})
        meta = load_faiss_meta()
        meta["node_count"] = 0
        meta["rebuilt_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_faiss_meta(meta)
        print(f"  FAISS: {total_before} -> 0 vectors (all expired)")
        return total_before, removed, 0


def main():
    parser = argparse.ArgumentParser(description="Purge expired MAGMA nodes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Count only, no changes")
    group.add_argument("--apply", action="store_true", help="Execute the cleanup")
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"MAGMA Expired Node Purge [{mode}]  {datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH), timeout=10)

    # Step 1: Find expired active nodes
    expired = get_expired_nodes(conn)
    print(f"\n[1] Expired active nodes (valid_until < now): {len(expired)}")

    if not expired:
        print("  Nothing to purge.")
        conn.close()
        return

    expired_ids = {nid for nid, _ in expired}
    print(f"  Sample IDs: {list(expired_ids)[:5]}")

    # Step 2: Count how many have embeddings
    placeholders = ",".join("?" * len(expired_ids))
    cur = conn.execute(
        f"SELECT COUNT(*) FROM nodes WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
        list(expired_ids),
    )
    with_emb = cur.fetchone()[0]
    print(f"\n[2] Expired nodes with embeddings: {with_emb}")

    # Step 3: Clean FAISS
    print(f"\n[3] FAISS index cleanup:")
    faiss_before, faiss_removed, faiss_after = remove_vectors_from_faiss(
        expired_ids, dry_run=args.dry_run
    )

    # Step 4: Mark nodes as stale
    print(f"\n[4] Node status update:")
    if args.dry_run:
        print(f"  Would mark {len(expired)} nodes as 'stale'")
        # Show breakdown by label
        cur = conn.execute(
            f"SELECT label, COUNT(*) FROM nodes WHERE id IN ({placeholders}) GROUP BY label ORDER BY COUNT(*) DESC",
            list(expired_ids),
        )
        for row in cur:
            print(f"    {row[0]}: {row[1]}")
    else:
        conn.execute(
            f"UPDATE nodes SET status = 'stale', updated_at = CURRENT_TIMESTAMP "
            f"WHERE id IN ({placeholders}) AND status = 'active'",
            list(expired_ids),
        )
        conn.commit()
        # Verify
        cur = conn.execute(
            f"SELECT COUNT(*) FROM nodes WHERE id IN ({placeholders}) AND status = 'stale'",
            list(expired_ids),
        )
        marked = cur.fetchone()[0]
        print(f"  Marked {marked} nodes as 'stale'")

    conn.close()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  Expired active nodes found:  {len(expired)}")
    print(f"  With embeddings:             {with_emb}")
    print(f"  FAISS vectors before:        {faiss_before}")
    print(f"  FAISS vectors removed:       {faiss_removed}")
    print(f"  FAISS vectors after:         {faiss_after}")
    if args.dry_run:
        print(f"\n  [DRY-RUN] No changes made. Run with --apply to execute.")
    else:
        print(f"\n  [APPLY] Cleanup complete.")


if __name__ == "__main__":
    main()
