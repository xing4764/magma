"""Minimal test for created_after/created_before time range filtering."""
import sys
import os
import tempfile
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magma.graph.sqlite_store import SQLiteStore
from magma.search import _time_range_filters, _parse_iso_datetime


def test_parse_iso_datetime():
    assert _parse_iso_datetime("2026-06-10") == "2026-06-10 00:00:00"
    assert _parse_iso_datetime("2026-06-10T12:30:00") == "2026-06-10 12:30:00"
    assert _parse_iso_datetime("") is None
    assert _parse_iso_datetime("invalid") is None
    print("PASS: _parse_iso_datetime")


def test_time_range_filters():
    filters = {"created_after": "2026-06-10", "created_before": "2026-06-11"}
    result = _time_range_filters(filters)
    assert result["created_after"] == "2026-06-10 00:00:00"
    assert result["created_before"] == "2026-06-11 00:00:00"
    print("PASS: _time_range_filters")


def test_query_with_time_filters():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteStore(db_path)
        store.initialize()

        now = datetime.utcnow()
        old_time = (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        new_time = now.strftime("%Y-%m-%d %H:%M:%S")

        node_old = f"node-old-{uuid.uuid4().hex[:8]}"
        node_new = f"node-new-{uuid.uuid4().hex[:8]}"

        store.add_node(node_old, "event", {"content": "old memory"}, None)
        store.add_node(node_new, "event", {"content": "new memory"}, None)

        store._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?", (old_time, node_old)
        )
        store._conn.execute(
            "UPDATE nodes SET created_at = ? WHERE id = ?", (new_time, node_new)
        )
        store._conn.commit()

        # created_after should exclude old node
        results = store.query_nodes_properties_only(
            time_filters={"created_after": new_time}
        )
        ids = [r["id"] for r in results]
        assert node_old not in ids, f"old node should be excluded, got {ids}"
        assert node_new in ids, f"new node should be included, got {ids}"
        print("PASS: created_after filters out old records")

        # created_before should exclude new node
        results = store.query_nodes_properties_only(
            time_filters={"created_before": old_time}
        )
        ids = [r["id"] for r in results]
        assert node_new not in ids, f"new node should be excluded, got {ids}"
        assert node_old in ids, f"old node should be included, got {ids}"
        print("PASS: created_before filters out new records")

        # combined: should return nothing (after new_time AND before old_time = empty)
        results = store.query_nodes_properties_only(
            time_filters={"created_after": new_time, "created_before": old_time}
        )
        assert len(results) == 0, f"combined filter should return empty, got {len(results)}"
        print("PASS: combined created_after + created_before works")

        # test query_nodes_with_embeddings path too
        results = store.query_nodes_with_embeddings(
            time_filters={"created_after": new_time}
        )
        ids = [r["id"] for r in results]
        assert node_old not in ids, f"query_with_embeddings: old node should be excluded"
        assert node_new in ids, f"query_with_embeddings: new node should be included"
        print("PASS: query_nodes_with_embeddings also works with time filters")

    finally:
        try:
            os.unlink(db_path)
        except PermissionError:
            pass


if __name__ == "__main__":
    test_parse_iso_datetime()
    test_time_range_filters()
    test_query_with_time_filters()
    print("\nAll tests passed!")
