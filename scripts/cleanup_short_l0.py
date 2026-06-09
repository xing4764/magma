#!/usr/bin/env python3
"""
MAGMA cleanup script: P1-8 short L0 content + P2-10 L1 duplicate detection.

P1-8: Mark L0 nodes with content < 15 chars as 'stale' and remove from FAISS.
P2-10: Detect and deduplicate L1 nodes with identical content.

Usage:
  python scripts/cleanup_short_l0.py --dry-run      # Preview only
  python scripts/cleanup_short_l0.py --apply         # Execute cleanup
  python scripts/cleanup_short_l0.py --dry-run --task P1-8   # Preview P1-8 only
  python scripts/cleanup_short_l0.py --dry-run --task P2-10  # Preview P2-10 only
"""

import sys
import json
import sqlite3
import re
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

# Ensure project root is in path so we can import magma
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows encoding for emoji output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magma.cleanup")

DB_PATH = PROJECT_ROOT / "data" / "magma.db"


def get_db(db_path: str = None) -> sqlite3.Connection:
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_faiss_index():
    """Get the FAISS index singleton, building it if possible."""
    try:
        from magma.vector.faiss_index import get_faiss_index as gfi
        return gfi(0)
    except Exception as e:
        logger.warning(f"FAISS not available: {e}")
        return None


def query_short_l0(db: sqlite3.Connection) -> List[Dict]:
    """Query all active L0 nodes with content < 15 chars."""
    cur = db.execute("""
        SELECT id, label, properties, importance, created_at, status,
               embedding IS NOT NULL AND LENGTH(embedding) > 0 AS has_embedding,
               LENGTH(COALESCE(json_extract(properties, '$.content'), '')) as content_len
        FROM nodes
        WHERE status = 'active'
          AND json_extract(properties, '$.layer') = 'L0'
          AND LENGTH(COALESCE(json_extract(properties, '$.content'), '')) < 15
        ORDER BY content_len, id
    """)
    return [dict(r) for r in cur.fetchall()]


def query_duplicate_l1(db: sqlite3.Connection) -> List[Dict]:
    """
    Find L1 nodes with near-identical content.
    Uses normalized content (strip, lower, collapse whitespace).
    Returns groups of duplicates.
    """
    cur = db.execute("""
        SELECT id, label, properties, importance, created_at, status,
               json_extract(properties, '$.content') as content,
               json_extract(properties, '$.kind') as kind,
               json_extract(properties, '$.source') as source
        FROM nodes
        WHERE status = 'active'
          AND json_extract(properties, '$.layer') = 'L1'
          AND COALESCE(json_extract(properties, '$.content'), '') != ''
        ORDER BY id
    """)
    all_l1 = [dict(r) for r in cur.fetchall()]
    
    # Group by normalized content
    norm_groups = defaultdict(list)
    for n in all_l1:
        content = (n['content'] or '').strip()
        if not content:
            continue
        norm = re.sub(r'\s+', ' ', content.lower())
        norm_groups[norm].append(n)
    
    # Return groups with 2+ nodes
    duplicate_groups = []
    for norm, nodes in norm_groups.items():
        if len(nodes) >= 2:
            # Sort by importance descending so we keep the highest
            nodes_sorted = sorted(nodes, key=lambda x: x['importance'] or 0.0, reverse=True)
            duplicate_groups.append({
                'normalized_content': norm[:200],
                'nodes': nodes_sorted,
            })
    
    return duplicate_groups


def mark_stale_batch(db: sqlite3.Connection, node_ids: List[str]):
    """Mark a batch of nodes as stale."""
    if not node_ids:
        return 0
    
    placeholders = ','.join('?' for _ in node_ids)
    cur = db.execute(
        f"UPDATE nodes SET status = 'stale', updated_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
        node_ids
    )
    db.commit()
    return cur.rowcount


def remove_from_faiss_batch(faiss_idx, node_ids: List[str]) -> int:
    """Remove node IDs from FAISS index. Returns count removed."""
    if faiss_idx is None:
        return 0
    removed = 0
    for nid in node_ids:
        if nid in faiss_idx._id_to_pos:
            faiss_idx.remove(nid)
            removed += 1
    return removed


def add_same_as_edges(db: sqlite3.Connection, groups: List[Dict]):
    """For each duplicate group, add same_as edges from the kept node to removed nodes."""
    from magma.graph.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path=str(DB_PATH))
    store.initialize()
    
    count = 0
    for group in groups:
        nodes = group['nodes']
        keep = nodes[0]  # highest importance
        for dup in nodes[1:]:
            store._add_same_as_edge(dup['id'], keep['id'], {
                'reason': 'duplicate_L1_content',
                'kept_importance': keep['importance'],
                'removed_importance': dup['importance'],
            })
            count += 1
    store.close()
    return count


def run_dry_run(db: sqlite3.Connection, tasks: list):
    """Preview cleanup without making changes."""
    results = {}
    
    if 'P1-8' in tasks:
        short_l0 = query_short_l0(db)
        
        # Content distribution
        len_dist = defaultdict(int)
        has_embed = 0
        no_embed = 0
        for n in short_l0:
            len_dist[n['content_len']] += 1
            if n['has_embedding']:
                has_embed += 1
            else:
                no_embed += 1
        
        results['P1-8'] = {
            'total': len(short_l0),
            'with_embedding': has_embed,
            'without_embedding': no_embed,
            'len_distribution': dict(sorted(len_dist.items())),
            'samples': [
                {'id': n['id'], 'content_len': n['content_len'],
                 'properties': json.loads(n['properties'])}
                for n in short_l0[:5]
            ],
        }
    
    if 'P2-10' in tasks:
        dupes = query_duplicate_l1(db)
        results['P2-10'] = {
            'total_groups': len(dupes),
            'total_duplicate_nodes': sum(len(g['nodes']) - 1 for g in dupes),
            'groups': [
                {
                    'normalized_content': g['normalized_content'][:100],
                    'nodes': [
                        {'id': n['id'], 'importance': n['importance'], 'kind': n['kind']}
                        for n in g['nodes']
                    ],
                }
                for g in dupes
            ],
        }
    
    return results


def run_apply(db: sqlite3.Connection, tasks: list, faiss_idx=None):
    """Execute cleanup."""
    results = {}
    
    if 'P1-8' in tasks:
        short_l0 = query_short_l0(db)
        node_ids = [n['id'] for n in short_l0]
        
        if node_ids:
            stale_count = mark_stale_batch(db, node_ids)
            faiss_removed = 0
            if faiss_idx and faiss_idx.is_available:
                faiss_removed = remove_from_faiss_batch(faiss_idx, node_ids)
            
            results['P1-8'] = {
                'marked_stale': stale_count,
                'faiss_removed': faiss_removed,
                'total_found': len(short_l0),
            }
        else:
            results['P1-8'] = {'marked_stale': 0, 'faiss_removed': 0, 'total_found': 0}
    
    if 'P2-10' in tasks:
        dupes = query_duplicate_l1(db)
        edge_count = add_same_as_edges(db, dupes)
        
        # Mark duplicates as stale (keep the highest importance)
        stale_ids = []
        for g in dupes:
            for dup in g['nodes'][1:]:  # skip the first (keep)
                stale_ids.append(dup['id'])
        
        if stale_ids:
            marked = mark_stale_batch(db, stale_ids)
        else:
            marked = 0
        
        # Also remove from FAISS
        faiss_removed = 0
        if faiss_idx and faiss_idx.is_available and stale_ids:
            faiss_removed = remove_from_faiss_batch(faiss_idx, stale_ids)
        
        results['P2-10'] = {
            'total_groups': len(dupes),
            'marked_stale': marked,
            'same_as_edges_added': edge_count,
            'faiss_removed': faiss_removed,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MAGMA cleanup for P1-8 and P2-10')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no changes')
    parser.add_argument('--apply', action='store_true', help='Execute cleanup')
    parser.add_argument('--task', choices=['P1-8', 'P2-10', 'all'], default='all',
                        help='Which task to run (default: all)')
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        parser.print_help()
        print('\nError: Specify --dry-run or --apply')
        sys.exit(1)
    
    tasks = ['P1-8', 'P2-10'] if args.task == 'all' else [args.task]
    
    db = get_db()
    
    if args.dry_run:
        print('=' * 60)
        print('MAGMA CLEANUP - DRY RUN')
        print('=' * 60)
        
        results = run_dry_run(db, tasks)
        
        for task, data in results.items():
            print(f'\n--- {task} ---')
            if task == 'P1-8':
                print(f'  Short L0 nodes (<15 chars): {data["total"]}')
                print(f'    With embedding: {data["with_embedding"]}')
                print(f'    Without embedding: {data["without_embedding"]}')
                print(f'  Length distribution:')
                for length, count in sorted(data['len_distribution'].items()):
                    print(f'    len={length}: {count} nodes')
                print(f'  Samples:')
                for s in data['samples']:
                    print(f'    {s["id"]}: len={s["content_len"]}')
            elif task == 'P2-10':
                print(f'  Duplicate L1 groups: {data["total_groups"]}')
                print(f'  Duplicate nodes (to remove): {data["total_duplicate_nodes"]}')
                for i, g in enumerate(data['groups']):
                    print(f'  Group {i+1}:')
                    print(f'    Content: {g["normalized_content"]}')
                    print(f'    KEEP: {g["nodes"][0]["id"]} (importance={g["nodes"][0]["importance"]})')
                    for dup in g['nodes'][1:]:
                        print(f'    STALE: {dup["id"]} (importance={dup["importance"]})')
        
        print(f'\n=== SUMMARY ===')
        print(f'P1-8 would mark {results.get("P1-8", {}).get("total", 0)} nodes stale')
        p2 = results.get('P2-10', {})
        print(f'P2-10 would mark {p2.get("total_duplicate_nodes", 0)} nodes stale in {p2.get("total_groups", 0)} groups')
        print('\nRun with --apply to execute.')
    
    elif args.apply:
        print('=' * 60)
        print('MAGMA CLEANUP - APPLY')
        print('=' * 60)
        
        # Try to get FAISS
        faiss_idx = get_faiss_index()
        if faiss_idx and faiss_idx.is_available:
            print(f'FAISS available: {faiss_idx.count} vectors')
        else:
            print('FAISS not available or not built - will skip vector removal')
            faiss_idx = None
        
        results = run_apply(db, tasks, faiss_idx)
        
        for task, data in results.items():
            print(f'\n--- {task} ---')
            if task == 'P1-8':
                print(f'  Nodes marked stale: {data["marked_stale"]}')
                print(f'  FAISS vectors removed: {data["faiss_removed"]}')
            elif task == 'P2-10':
                print(f'  Groups processed: {data["total_groups"]}')
                print(f'  Nodes marked stale: {data["marked_stale"]}')
                print(f'  same_as edges added: {data["same_as_edges_added"]}')
                print(f'  FAISS vectors removed: {data["faiss_removed"]}')
        
        # Verify
        print(f'\n=== POST-CLEANUP VERIFICATION ===')
        cur = db.execute("""
            SELECT COUNT(*) as cnt FROM nodes
            WHERE status = 'active'
              AND json_extract(properties, '$.layer') = 'L0'
              AND LENGTH(COALESCE(json_extract(properties, '$.content'), '')) < 15
        """)
        print(f'Remaining short L0 (should be 0): {cur.fetchone()["cnt"]}')
        
        cur = db.execute("SELECT COUNT(*) as cnt FROM nodes WHERE status = 'stale'")
        print(f'Total stale nodes: {cur.fetchone()["cnt"]}')
        
        print('\nCleanup complete.')
    
    db.close()


if __name__ == '__main__':
    main()
