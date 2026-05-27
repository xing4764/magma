"""Probe Qwen3-Embedding-0.6B for MAGMA without touching production data.

This script loads a candidate embedding model, embeds a sampled copy of MAGMA
nodes in memory, and runs a small retrieval benchmark. It does not write to the
SQLite database and does not change the OpenClaw/MAGMA production API.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = Path(os.environ.get("MAGMA_DB_PATH", str(PROJECT_ROOT / "data" / "magma.db")))
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

CASES = [
    "MAGMA MCP 为什么要改成 8902 主链路薄代理？",
    "recent_capture 变成 yellow 是什么意思，系统坏了吗？",
    "OpenClaw 为什么固定在 2026.5.20，不升级 5.22？",
    "source_agent_id 对跨 agent 记忆有什么作用？",
    "magma_doctor.py 和 magma_ops.py 分别用来做什么？",
    "yunying 运营部的 MAGMA 注入为什么之前是 0？",
    "MAGMA 网关卡顿和 embedding 反复加载有什么关系？",
    "Qwen3 Embedding 是否适合替换 bge-small-zh？",
]


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return None


def node_text(label: str, properties: dict) -> str:
    parts = [label]
    for key in ("title", "name", "content", "summary", "source", "role", "agent_id"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    if len(parts) == 1:
        for value in properties.values():
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return " ".join(parts)


def load_nodes(limit: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, label, properties, importance, access_count, created_at
          FROM nodes
         WHERE status = 'active'
           AND (valid_until IS NULL OR datetime(valid_until) >= CURRENT_TIMESTAMP)
           AND (ttl_days IS NULL OR datetime(created_at, '+' || ttl_days || ' days') >= CURRENT_TIMESTAMP)
         ORDER BY
           CASE WHEN label = 'entity' THEN 1 ELSE 0 END,
           importance DESC,
           datetime(updated_at) DESC
         LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    nodes = []
    for row in rows:
        props = json.loads(row["properties"] or "{}")
        nodes.append({
            "id": row["id"],
            "label": row["label"],
            "properties": props,
            "text": node_text(row["label"], props),
        })
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen embedding model for MAGMA.")
    parser.add_argument("--model", default=os.environ.get("MAGMA_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=1500, help="Max active nodes to embed in memory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    print(f"model={args.model}")
    print(f"db={DB_PATH}")
    before = rss_mb()
    if before is not None:
        print(f"rss_before_mb={before:.1f}")

    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, trust_remote_code=True)
    load_ms = (time.perf_counter() - t0) * 1000
    after_load = rss_mb()
    print(f"load_ms={load_ms:.1f}")
    if after_load is not None:
        print(f"rss_after_load_mb={after_load:.1f}")
        if before is not None:
            print(f"rss_load_delta_mb={after_load - before:.1f}")

    nodes = load_nodes(args.limit)
    texts = [node["text"] for node in nodes]
    print(f"nodes={len(nodes)}")

    t1 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    embed_ms = (time.perf_counter() - t1) * 1000
    print(f"embed_nodes_ms={embed_ms:.1f}")
    print(f"embedding_shape={list(embeddings.shape)}")

    query_times = []
    for query in CASES:
        tq = time.perf_counter()
        qv = model.encode(query, normalize_embeddings=True).astype("float32")
        scores = embeddings @ qv
        top_idx = np.argsort(-scores)[: args.top_k]
        elapsed = (time.perf_counter() - tq) * 1000
        query_times.append(elapsed)
        print("\nQUERY", query)
        print(f"query_ms={elapsed:.1f}")
        for rank, idx in enumerate(top_idx, start=1):
            node = nodes[int(idx)]
            title = node["properties"].get("title") or node["properties"].get("name") or node["properties"].get("content") or node["id"]
            title = str(title).replace("\n", " ")[:120]
            print(f"{rank}. score={scores[idx]:.4f} id={node['id']} label={node['label']} title={title}")

    if query_times:
        arr = sorted(query_times)
        print("\nSUMMARY")
        print(f"query_ms_min={arr[0]:.1f}")
        print(f"query_ms_p50={arr[len(arr)//2]:.1f}")
        print(f"query_ms_max={arr[-1]:.1f}")
    final = rss_mb()
    if final is not None:
        print(f"rss_final_mb={final:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
