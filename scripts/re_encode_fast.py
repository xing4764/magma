"""Re-encode all node embeddings using the running MAGMA server's encoder.

Connects to the already-loaded 4B model via the encoder module,
re-encodes all nodes, and rebuilds the FAISS index.
"""
import json
import os
import sys
import time
import sqlite3
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.environ.get("MAGMA_DB_PATH", os.path.join(PROJECT_ROOT, "data", "magma.db"))
BATCH_SIZE = 32  # smaller batches to reduce memory pressure


def node_text(label: str, properties: dict) -> str:
    """Reconstruct the text used for encoding."""
    parts = [label]
    for value in (properties or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def main():
    print("Loading encoder from running MAGMA instance...")
    from magma.vector.encoder import Encoder
    encoder = Encoder()

    # Test encoder to get dimension
    test_vec = encoder.encode(["test"], normalize=True)
    new_dim = test_vec.shape[1]
    print(f"Encoder loaded. Dimension: {new_dim}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Count nodes
    cur = conn.execute("SELECT COUNT(*) FROM nodes")
    total = cur.fetchone()[0]
    print(f"Total nodes: {total}")

    # Fetch all nodes
    cur = conn.execute("SELECT id, label, properties FROM nodes")
    nodes = cur.fetchall()
    print(f"Fetched {len(nodes)} nodes. Starting re-encode...")

    # Process in batches
    updated = 0
    errors = 0
    t0 = time.time()

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i+BATCH_SIZE]
        texts = []
        ids = []
        for node in batch:
            props = json.loads(node["properties"]) if node["properties"] else {}
            text = node_text(node["label"], props)
            texts.append(text)
            ids.append(node["id"])

        try:
            embeddings = encoder.encode(texts, normalize=True)
            for nid, emb in zip(ids, embeddings):
                emb_bytes = emb.astype("float32").tobytes()
                conn.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (emb_bytes, nid))
            updated += len(batch)
        except Exception as e:
            errors += len(batch)
            print(f"  Error at batch {i}: {e}")

        if (i + BATCH_SIZE) % 200 == 0 or i + BATCH_SIZE >= len(nodes):
            elapsed = time.time() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            print(f"  Progress: {updated}/{len(nodes)} ({rate:.1f} nodes/sec, {elapsed:.0f}s elapsed)")

    conn.commit()

    # Rebuild FAISS index
    print("Rebuilding FAISS index...")
    cur = conn.execute("SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    entries = [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]
    
    from magma.vector.faiss_index import FAISSIndex
    faiss_idx = FAISSIndex(dimension=new_dim)
    faiss_idx.build_from_embeddings(entries)
    print(f"FAISS index rebuilt with {len(entries)} vectors, dim={new_dim}")

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone! Re-encoded {updated} nodes in {elapsed:.1f}s ({errors} errors)")
    print(f"Dimension: {new_dim}")


if __name__ == "__main__":
    main()
