#!/usr/bin/env python3
"""MAGMA CLI - Command line interface for MAGMA knowledge graph."""

import os
import sys
import json
import re
import argparse
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the official Hub by default. Set HF_ENDPOINT explicitly when a mirror is needed.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

DATA_DIR = Path(project_root) / "data"
DB_PATH = DATA_DIR / "magma.db"
FAISS_PATH = DATA_DIR / "faiss.index"
ID_MAP_PATH = DATA_DIR / "id_map.json"
FAISS_META_PATH = DATA_DIR / "faiss_meta.json"

# Fix Windows encoding for emoji output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("magma.cli")


# ──────────────────────────────────────────
# Markdown Parser
# ──────────────────────────────────────────

def parse_markdown_file(filepath: Path) -> Dict:
    """Parse a markdown file into structured data."""
    content = filepath.read_text(encoding="utf-8")
    filename = filepath.stem
    
    # Extract date from filename if present
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    file_date = date_match.group(1) if date_match else None
    
    # Parse sections
    sections = []
    current_section = {"title": filename, "content": "", "level": 0}
    
    for line in content.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {
                "title": heading_match.group(2).strip(),
                "content": "",
                "level": len(heading_match.group(1)),
            }
        else:
            current_section["content"] += line + "\n"
    
    if current_section["content"].strip():
        sections.append(current_section)
    
    return {
        "filename": filename,
        "date": file_date,
        "sections": sections,
        "full_text": content,
    }


def extract_events_and_entities(parsed: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Extract events and entities from parsed markdown."""
    events = []
    entities = set()
    
    filename = parsed["filename"]
    file_date = parsed["date"]
    
    for section in parsed["sections"]:
        title = section["title"]
        content = section["content"].strip()
        
        if not content:
            continue
        
        # Create event node
        event_id = f"evt:{filename}:{hash(title) & 0xFFFFFFFF:08x}"
        event = {
            "id": event_id,
            "label": "event",
            "properties": {
                "title": title,
                "content": content[:2000],  # Truncate long content
                "source_file": filename,
                "date": file_date,
                "level": section["level"],
            },
        }
        events.append(event)
        
        # Extract entities from content using simple patterns
        # People (names with @ or common patterns)
        for match in re.finditer(r"[@＠]([\u4e00-\u9fff\w]+)", content):
            entities.add(("person", match.group(1)))
        
        # Projects/tools (backtick-quoted)
        for match in re.finditer(r"`([^`]+)`", content):
            name = match.group(1)
            if len(name) > 1 and len(name) < 50:
                entities.add(("tool", name))
        
        # Dates
        for match in re.finditer(r"(\d{4}-\d{2}-\d{2})", content):
            entities.add(("date", match.group(1)))
        
        # Key topics (lines starting with - or *)
        for match in re.finditer(r"^[\s]*[-*]\s+(.+)", content, re.MULTILINE):
            item = match.group(1).strip()
            if len(item) > 2 and len(item) < 100:
                entities.add(("topic", item))
        
        # Technical terms (English words that look like tech)
        for match in re.finditer(r"\b([A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)*)\b", content):
            entities.add(("technology", match.group(1)))
    
    entity_list = []
    for etype, ename in entities:
        entity_id = f"ent:{etype}:{hash(ename) & 0xFFFFFFFF:08x}"
        entity_list.append({
            "id": entity_id,
            "label": etype,
            "properties": {"name": ename},
        })
    
    return events, entity_list


# ──────────────────────────────────────────
# Database Operations
# ──────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
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
    """)
    cur = conn.execute("PRAGMA table_info(nodes)")
    existing = {row[1] for row in cur.fetchall()}
    columns = {
        "last_accessed_at": "TIMESTAMP",
        "access_count": "INTEGER NOT NULL DEFAULT 0",
        "importance": "REAL NOT NULL DEFAULT 0.5",
        "ttl_days": "INTEGER",
        "valid_from": "TIMESTAMP",
        "valid_until": "TIMESTAMP",
        "status": "TEXT NOT NULL DEFAULT 'active'",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE nodes ADD COLUMN {name} {definition}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)")
    conn.commit()
    return conn


def store_nodes(conn: sqlite3.Connection, nodes: List[Dict], embeddings=None):
    """Store nodes in SQLite."""
    for i, node in enumerate(nodes):
        emb = None
        if embeddings is not None:
            import numpy as np
            emb = embeddings[i].tobytes() if hasattr(embeddings[i], 'tobytes') else embeddings[i]
        
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, label, properties, embedding, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (node["id"], node["label"], json.dumps(node["properties"], ensure_ascii=False), emb)
        )
    conn.commit()


def store_edges(conn: sqlite3.Connection, edges: List[Tuple[str, str, str, Dict]]):
    """Store edges in SQLite."""
    for src, tgt, rel, props in edges:
        conn.execute(
            "INSERT INTO edges (source_id, target_id, relation, properties) VALUES (?, ?, ?, ?)",
            (src, tgt, rel, json.dumps(props or {}, ensure_ascii=False))
        )
    conn.commit()


# ──────────────────────────────────────────
# FAISS Index
# ──────────────────────────────────────────

def build_faiss_index(embeddings, ids: List[str], faiss_path: Path, id_map_path: Path):
    """Build and save FAISS index."""
    import faiss
    import numpy as np
    
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, str(faiss_path))
    
    # Save ID mapping
    id_map = {str(i): id_ for i, id_ in enumerate(ids)}
    id_map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    
    logger.info(f"FAISS index saved: {faiss_path} ({index.ntotal} vectors)")
    return index


def node_text_for_embedding(label: str, properties: Dict) -> str:
    """Build a stable text representation for node embedding."""
    parts = [label]
    if label == "event":
        parts.extend([
            str(properties.get("title", "")),
            str(properties.get("content", ""))[:1000],
            str(properties.get("date", "")),
        ])
    else:
        for key in ("name", "title", "content", "description"):
            value = properties.get(key)
            if isinstance(value, str):
                parts.append(value)
        for value in properties.values():
            if isinstance(value, str) and value not in parts:
                parts.append(value)
    return " ".join(part for part in parts if part).strip()


# ──────────────────────────────────────────
# Commands
# ──────────────────────────────────────────

def cmd_import(args):
    """Import markdown files into MAGMA."""
    import_path = Path(args.path)
    
    if not import_path.exists():
        logger.error(f"Path does not exist: {import_path}")
        return 1
    
    # Collect markdown files
    md_files = list(import_path.rglob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found in {import_path}")
        return 1
    
    logger.info(f"Found {len(md_files)} markdown files")
    
    # Parse all files
    all_events = []
    all_entities = []
    all_entity_map = {}  # Deduplicate entities
    
    for md_file in md_files:
        try:
            parsed = parse_markdown_file(md_file)
            events, entities = extract_events_and_entities(parsed)
            all_events.extend(events)
            
            for ent in entities:
                if ent["id"] not in all_entity_map:
                    all_entity_map[ent["id"]] = ent
        except Exception as e:
            logger.warning(f"Failed to parse {md_file}: {e}")
    
    all_entities = list(all_entity_map.values())
    
    logger.info(f"Extracted {len(all_events)} events, {len(all_entities)} entities")
    
    # Initialize DB
    conn = init_db(DB_PATH)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    from magma.vector.encoder import Encoder
    encoder = Encoder()
    
    # Build text for embedding
    all_nodes = all_events + all_entities
    texts = []
    for node in all_nodes:
        props = node["properties"]
        if node["label"] == "event":
            text = f"{props.get('title', '')} {props.get('content', '')[:500]}"
        else:
            text = f"{props.get('name', '')} {node['label']}"
        texts.append(text)
    
    # Encode in batches
    import numpy as np
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = encoder.encode(batch, normalize=True)
        all_embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Encoded {i+len(batch)}/{len(texts)}")
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Store in SQLite
    logger.info("Storing in SQLite...")
    store_nodes(conn, all_nodes, embeddings)
    
    # Create edges: event -> entity relationships
    logger.info("Building edges...")
    edges = []
    for event in all_events:
        event_content = json.dumps(event["properties"], ensure_ascii=False).lower()
        for entity in all_entities:
            entity_name = entity["properties"].get("name", "").lower()
            if entity_name and len(entity_name) > 1 and entity_name in event_content:
                edges.append((event["id"], entity["id"], "mentions", {}))
    
    store_edges(conn, edges)
    logger.info(f"Created {len(edges)} edges")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    node_ids = [n["id"] for n in all_nodes]
    build_faiss_index(embeddings.astype('float32'), node_ids, FAISS_PATH, ID_MAP_PATH)
    
    conn.close()
    
    logger.info(f"✅ Import complete: {len(all_events)} events, {len(all_entities)} entities")
    return 0


def cmd_query(args):
    """Query the knowledge graph."""
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1
    
    import numpy as np
    from magma.vector.encoder import Encoder
    
    encoder = Encoder()
    try:
        query_embedding = encoder.encode(args.query, normalize=True).astype('float32').reshape(1, -1)
    except Exception as e:
        logger.warning(f"Embedding model unavailable, falling back to keyword search: {e}")
        query_embedding = None
    
    results = []
    
    # FAISS search
    if query_embedding is not None and FAISS_PATH.exists() and ID_MAP_PATH.exists():
        import faiss
        index = faiss.read_index(str(FAISS_PATH))
        id_map = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
        if index.d != query_embedding.shape[1]:
            logger.warning(
                "FAISS dimension mismatch: index=%s query=%s. Run `reembed` to rebuild Chinese embeddings.",
                index.d,
                query_embedding.shape[1],
            )
            index = None
        
        if index is not None:
            k = min(args.top_k * 2, index.ntotal)
            scores, indices = index.search(query_embedding, k)
            
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                node_id = id_map.get(str(idx))
                if not node_id:
                    continue
                
                cur = conn.execute("SELECT id, label, properties FROM nodes WHERE id = ?", (node_id,))
                row = cur.fetchone()
                if row:
                    results.append({
                        "id": row["id"],
                        "label": row["label"],
                        "properties": json.loads(row["properties"]),
                        "score": float(score),
                    })
            
            conn.close()
    
    # Fallback: keyword search in SQLite
    if not results:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        query_lower = args.query.lower()
        
        cur = conn.execute("SELECT id, label, properties FROM nodes LIMIT 1000")
        for row in cur.fetchall():
            props_str = json.dumps(json.loads(row["properties"]), ensure_ascii=False).lower()
            if query_lower in props_str:
                results.append({
                    "id": row["id"],
                    "label": row["label"],
                    "properties": json.loads(row["properties"]),
                    "score": 0.5,
                })
        conn.close()
    
    # Sort and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:args.top_k]
    
    # Output
    print(f"\n🔍 Query: {args.query}")
    print(f"📊 Results: {len(results)}\n")
    
    for i, r in enumerate(results, 1):
        print(f"── Result {i} (score: {r['score']:.4f}) ──")
        print(f"  ID: {r['id']}")
        print(f"  Label: {r['label']}")
        props = r['properties']
        if r['label'] == 'event':
            print(f"  Title: {props.get('title', 'N/A')}")
            content = props.get('content', '')
            if content:
                preview = content[:300].replace('\n', ' ')
                print(f"  Content: {preview}...")
            print(f"  Source: {props.get('source_file', 'N/A')}")
            print(f"  Date: {props.get('date', 'N/A')}")
        else:
            print(f"  Name: {props.get('name', 'N/A')}")
        print()
    
    return 0


def cmd_reembed(args):
    """Rebuild SQLite embeddings and FAISS index for existing nodes."""
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    import numpy as np
    from magma.vector.encoder import Encoder

    conn = init_db(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, label, properties FROM nodes WHERE status != 'deleted' ORDER BY id")
    rows = cur.fetchall()
    if not rows:
        logger.warning("No nodes found to re-embed")
        conn.close()
        return 1

    encoder = Encoder(args.model) if args.model else Encoder()
    texts = []
    node_ids = []
    for row in rows:
        props = json.loads(row["properties"])
        texts.append(node_text_for_embedding(row["label"], props))
        node_ids.append(row["id"])

    logger.info(f"Re-embedding {len(texts)} nodes with model={encoder.model_name}")
    batch_size = args.batch_size
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = encoder.encode(batch, normalize=True).astype("float32")
        all_embeddings.append(embeddings)
        logger.info(f"  Encoded {min(i + len(batch), len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings)
    for node_id, embedding in zip(node_ids, embeddings):
        conn.execute(
            "UPDATE nodes SET embedding = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (embedding.tobytes(), node_id),
        )
    conn.commit()
    conn.close()

    build_faiss_index(embeddings.astype("float32"), node_ids, FAISS_PATH, ID_MAP_PATH)
    meta = {
        "model": encoder.model_name,
        "dimension": int(embeddings.shape[1]),
        "node_count": len(node_ids),
        "rebuilt_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    FAISS_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Re-embed complete: {len(node_ids)} nodes, dim={embeddings.shape[1]}")
    return 0


def cmd_stats(args):
    """Show MAGMA statistics."""
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    
    # Count nodes by label
    cur = conn.execute("SELECT label, COUNT(*) FROM nodes GROUP BY label")
    node_counts = cur.fetchall()
    total_nodes = sum(c for _, c in node_counts)
    
    # Count edges
    cur = conn.execute("SELECT COUNT(*) FROM edges")
    total_edges = cur.fetchone()[0]
    
    # Edge relations
    cur = conn.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation")
    edge_counts = cur.fetchall()
    
    # FAISS status
    faiss_status = "✅ exists" if FAISS_PATH.exists() else "❌ missing"
    faiss_size = FAISS_PATH.stat().st_size if FAISS_PATH.exists() else 0
    
    # DB size
    db_size = DB_PATH.stat().st_size
    
    conn.close()
    
    print("\n📊 MAGMA Statistics")
    print("=" * 40)
    print(f"\n📁 Database: {DB_PATH}")
    print(f"   Size: {db_size / 1024:.1f} KB")
    print(f"\n🔍 FAISS Index: {faiss_status}")
    if FAISS_PATH.exists():
        print(f"   Size: {faiss_size / 1024:.1f} KB")
    print(f"\n📌 ID Map: {'✅ exists' if ID_MAP_PATH.exists() else '❌ missing'}")
    
    print(f"\n📈 Nodes: {total_nodes}")
    for label, count in node_counts:
        print(f"   {label}: {count}")
    
    print(f"\n🔗 Edges: {total_edges}")
    for relation, count in edge_counts:
        print(f"   {relation}: {count}")
    
    print()
    return 0


def cmd_consolidate(args):
    """Consolidate knowledge graph (rule-only mode)."""
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    from magma.graph.sqlite_store import SQLiteStore

    store = SQLiteStore(str(DB_PATH)).initialize()
    stats = store.consolidate()
    store.close()

    print("\nConsolidation Complete")
    print(f"   Removed duplicate edges: {stats['removed_duplicate_edges']}")
    print(f"   Removed orphan edges: {stats['removed_orphan_edges']}")
    print(f"   Expired nodes marked stale: {stats['expired_nodes']}")
    print(f"   Merged duplicate entities: {stats['merged_duplicate_entities']}")

    if args.rule_only:
        print("   Mode: rule-only (no re-embedding)")

    return 0
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Rule 1: Remove duplicate edges
    cur = conn.execute("""
        DELETE FROM edges WHERE id NOT IN (
            SELECT MIN(id) FROM edges GROUP BY source_id, target_id, relation
        )
    """)
    removed_edges = cur.rowcount
    
    # Rule 2: Remove orphan edges (referencing non-existent nodes)
    cur = conn.execute("""
        DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes)
           OR target_id NOT IN (SELECT id FROM nodes)
    """)
    orphan_edges = cur.rowcount
    
    # Rule 3: Merge duplicate entities with same name
    cur = conn.execute("""
        SELECT properties, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
        FROM nodes WHERE label != 'event'
        GROUP BY properties HAVING cnt > 1
    """)
    duplicates = cur.fetchall()
    merged = 0
    for props_str, cnt, ids_str in duplicates:
        ids = ids_str.split(",")
        keep_id = ids[0]
        for remove_id in ids[1:]:
            # Update edges to point to kept node
            conn.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, remove_id))
            conn.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, remove_id))
            conn.execute("DELETE FROM nodes WHERE id = ?", (remove_id))
            merged += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n🔧 Consolidation Complete")
    print(f"   Removed duplicate edges: {removed_edges}")
    print(f"   Removed orphan edges: {orphan_edges}")
    print(f"   Merged duplicate entities: {merged}")
    
    if args.rule_only:
        print("   Mode: rule-only (no re-embedding)")
    
    return 0


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAGMA CLI - Memory-Augmented Graph & Multi-modal Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # import
    p_import = subparsers.add_parser("import", help="Import markdown files")
    p_import.add_argument("--path", required=True, help="Path to markdown files directory")
    
    # query
    p_query = subparsers.add_parser("query", help="Query the knowledge graph")
    p_query.add_argument("--query", required=True, help="Query text")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of results")

    # reembed
    p_reembed = subparsers.add_parser("reembed", help="Rebuild embeddings and FAISS index for existing nodes")
    p_reembed.add_argument("--model", default=None, help="Embedding model name (defaults to MAGMA_EMBEDDING_MODEL)")
    p_reembed.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    
    # consolidate
    p_consolidate = subparsers.add_parser("consolidate", help="Consolidate knowledge graph")
    p_consolidate.add_argument("--rule-only", action="store_true", help="Only apply rules, no re-embedding")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "import": cmd_import,
        "query": cmd_query,
        "reembed": cmd_reembed,
        "stats": cmd_stats,
        "consolidate": cmd_consolidate,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
