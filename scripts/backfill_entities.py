"""Backfill entity anchors and mentions_entity edges for existing MAGMA nodes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from magma.api.server import _attach_entity_anchors, _memory_metadata, _node_text
from magma.graph.sqlite_store import get_store
from magma.vector.encoder import Encoder


def main():
    store = get_store()
    store.initialize()
    encoder = Encoder()
    nodes = store.query_nodes(limit=10000)
    processed = 0
    scoped = 0
    for node in nodes:
        if node["label"] == "entity":
            continue
        text = _node_text(node["label"], node.get("properties"))
        properties = dict(node.get("properties") or {})
        properties.update(_memory_metadata(text))
        store._conn.execute(
            "UPDATE nodes SET properties = ?, importance = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (
                json.dumps(properties, ensure_ascii=False),
                float(properties.get("importance", node.get("importance") or 0.5) or 0.5),
                node["id"],
            ),
        )
        scoped += 1
        before = store._conn.total_changes
        _attach_entity_anchors(
            store,
            encoder,
            node["id"],
            text,
            (node.get("properties") or {}).get("source", "backfill_entities"),
        )
        if store._conn.total_changes > before:
            processed += 1
    store._conn.commit()
    print({"scoped_nodes": scoped, "processed_nodes_with_entities": processed})


if __name__ == "__main__":
    main()
