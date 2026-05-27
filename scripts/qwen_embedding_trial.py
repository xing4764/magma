"""Prepare and run a Qwen3-Embedding trial copy of MAGMA.

This script does not modify the production MAGMA database. It creates a
consistent SQLite backup, re-embeds that copy with Qwen3-Embedding-0.6B, and
prints the environment needed to launch a trial API on a separate port.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_DB = PROJECT_ROOT / "data" / "magma.db"
DEFAULT_TRIAL_DIR = PROJECT_ROOT / "data" / "qwen_trial"
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "models" / "Qwen" / "Qwen3-Embedding-0___6B"
DEFAULT_REMOTE_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def node_text_for_embedding(label: str, properties: dict) -> str:
    """Build the same stable embedding text used by the MAGMA CLI."""
    parts = [label]
    if label == "event":
        parts.extend(
            [
                str(properties.get("title", "")),
                str(properties.get("content", ""))[:1000],
                str(properties.get("date", "")),
            ]
        )
    else:
        for key in ("name", "title", "content", "description"):
            value = properties.get(key)
            if isinstance(value, str):
                parts.append(value)
        for value in properties.values():
            if isinstance(value, str) and value not in parts:
                parts.append(value)
    return " ".join(part for part in parts if part).strip()


def backup_database(source_db: Path, trial_db: Path) -> None:
    trial_db.parent.mkdir(parents=True, exist_ok=True)
    if trial_db.exists():
        trial_db.unlink()
    with sqlite3.connect(str(source_db)) as src, sqlite3.connect(str(trial_db)) as dst:
        src.backup(dst)


def reembed_database(trial_db: Path, model_name: str, batch_size: int) -> dict:
    import numpy as np
    from magma.vector.encoder import Encoder

    conn = sqlite3.connect(str(trial_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, label, properties FROM nodes WHERE status != 'deleted' ORDER BY id"
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No active nodes found in {trial_db}")

    encoder = Encoder(model_name)
    start = time.perf_counter()
    texts = []
    node_ids = []
    for row in rows:
        props = json.loads(row["properties"] or "{}")
        texts.append(node_text_for_embedding(row["label"], props))
        node_ids.append(row["id"])

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = encoder.encode(batch, normalize=True).astype("float32")
        all_embeddings.append(embeddings)
        print(f"encoded {min(i + len(batch), len(texts))}/{len(texts)}", flush=True)

    embeddings = np.vstack(all_embeddings)
    for node_id, embedding in zip(node_ids, embeddings):
        conn.execute(
            "UPDATE nodes SET embedding = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (embedding.tobytes(), node_id),
        )
    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - start
    return {
        "model": model_name,
        "dimension": int(embeddings.shape[1]),
        "node_count": len(node_ids),
        "elapsed_seconds": round(elapsed, 3),
        "rebuilt_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Qwen3-Embedding MAGMA trial database.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--trial-dir", default=str(DEFAULT_TRIAL_DIR))
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--port", type=int, default=8905)
    args = parser.parse_args()

    source_db = Path(args.source_db)
    trial_dir = Path(args.trial_dir)
    trial_db = trial_dir / "magma.db"
    if not source_db.exists():
        raise SystemExit(f"source database not found: {source_db}")

    model = args.model
    if model is None:
        model = str(DEFAULT_LOCAL_MODEL) if DEFAULT_LOCAL_MODEL.exists() else DEFAULT_REMOTE_MODEL

    print(f"source_db={source_db}")
    print(f"trial_db={trial_db}")
    print(f"model={model}")
    backup_database(source_db, trial_db)
    stats = reembed_database(trial_db, model, args.batch_size)

    meta_path = trial_dir / "qwen_embedding_trial.json"
    meta = {
        **stats,
        "source_db": str(source_db),
        "trial_db": str(trial_db),
        "trial_port": args.port,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("")
    print("Launch trial API with:")
    print(f"$env:MAGMA_DB_PATH='{trial_db}'")
    print(f"$env:MAGMA_EMBEDDING_MODEL='{model}'")
    print(f"$env:MAGMA_API_PORT='{args.port}'")
    print("python -m magma.api.server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
