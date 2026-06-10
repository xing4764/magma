"""Re-encode all node embeddings from 0.6B (1024d) to 4B (2560d).

Must be run BEFORE restarting the MAGMA server with the 4B model.
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
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen", "Qwen3-Embedding-4B")
BATCH_SIZE = 64


def node_text(label: str, properties: dict) -> str:
    """Reconstruct the text that was used for encoding."""
    parts = [label]
    for value in (properties or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def main():
    print(f"Loading Qwen3-Embedding-4B from {MODEL_PATH}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_PATH)
    new_dim = model.get_embedding_dimension()
    print(f"Model loaded. New dimension: {new_dim}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Count nodes with embeddings
    cur = conn.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
    total = cur.fetchone()[0]
    print(f"Total nodes with embeddings: {total}")

    # Check current dimension
    cur = conn.execute("SELECT LENGTH(embedding) as emb_len FROM nodes WHERE embedding IS NOT NULL LIMIT 1")
    row = cur.fetchone()
    if row:
        old_dim = row["emb_len"] // 4  # float32 = 4 bytes
        print(f"Current embedding dimension: {old_dim}")
        if old_dim == new_dim:
            print("Already at correct dimension. Nothing to do.")
            return

    # Fetch all nodes with embeddings
    cur = conn.execute(
        "SELECT id, label, properties, embedding FROM nodes WHERE embedding IS NOT NULL"
    )
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
            props = json.loads(node["properties"])
            text = node_text(node["label"], props)
            texts.append(text)
            ids.append(node["id"])

        try:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            for nid, emb in zip(ids, embeddings):
                emb_bytes = emb.astype("float32").tobytes()
                conn.execute("UPDATE nodes SET embedding = ? WHERE id = ?", (emb_bytes, nid))
            updated += len(batch)
        except Exception as e:
            errors += len(batch)
            print(f"  Error at batch {i}: {e}")

        if (i + BATCH_SIZE) % 500 == 0 or i + BATCH_SIZE >= len(nodes):
            elapsed = time.time() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            print(f"  Progress: {updated}/{len(nodes)} ({rate:.1f} nodes/sec, {elapsed:.0f}s elapsed)")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone! Re-encoded {updated} nodes in {elapsed:.1f}s ({errors} errors)")
    print(f"New dimension: {new_dim}")


if __name__ == "__main__":
    main()
