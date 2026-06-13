"""Smoke tests for deterministic MAGMA decision particles."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magma.api.server import create_app
from magma.decision_particles import extract_decision_events, summarize_decision_drift
from magma.graph.sqlite_store import SQLiteStore


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_extract_model_switch():
    events = extract_decision_events("别用0.6B了，换4B")
    require(events, "expected at least one decision event")
    event = events[0]
    require(event["decision_key"] == "model", f"unexpected decision_key: {event}")
    require(event["selected"] == "4B", f"unexpected selected: {event}")
    require("0.6B" in event["options"], f"old option missing: {event}")


def test_drift_summary():
    events = [
        {
            "decision_key": "model",
            "selected": "0.6B",
            "tags": ["memory"],
        },
        {
            "decision_key": "model",
            "selected": "4B",
            "tags": ["quality"],
        },
    ]
    summary = summarize_decision_drift(events)
    model = summary["groups"][0]
    require(model["changed"] is True, f"expected changed drift: {model}")
    require(model["path"] == ["4B", "0.6B"] or model["path"] == ["0.6B", "4B"], f"unexpected path: {model}")


def test_store_query():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "magma-test.db")
        store = SQLiteStore(db_path).initialize()
        store.add_node(
            "decision:test",
            "decision_event",
            {
                "kind": "decision_event",
                "decision_key": "model",
                "selected": "4B",
                "options": ["0.6B", "4B"],
                "chain": ["0.6B", "4B"],
                "tags": ["memory"],
                "agent_id": "main",
                "session_key": "agent:main:test",
                "importance": 0.68,
            },
            embedding=None,
        )
        rows = store.get_decision_events(agent_id="main", decision_key="model")
        store.close()
    require(len(rows) == 1, f"expected one decision row, got {len(rows)}")
    require(rows[0]["properties"]["selected"] == "4B", f"unexpected row: {rows[0]}")


class FakeEncoder:
    dimension = 4
    model_name = "fake"

    def encode(self, text):
        return np.array([1.0, 0.0, 0.0, float(len(text) % 7)], dtype=np.float32)


def test_capture_endpoint_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(str(Path(tmp) / "magma-test.db")).initialize()
        app = create_app()
        app.state.store = store
        app.state.encoder = FakeEncoder()
        app.state.faiss_index = None
        client = TestClient(app)
        capture = client.post(
            "/api/v1/capture",
            json={
                "user_text": "别用0.6B了，换4B，内存占用要低",
                "assistant_text": "收到，后续统一使用4B方案。",
                "agent_id": "main",
                "session_key": "agent:main:test",
                "source": "decision-test",
            },
        )
        require(capture.status_code == 200, f"capture failed: {capture.text}")
        capture_json = capture.json()
        require(capture_json["decision_count"] >= 1, f"expected decision write: {capture_json}")

        drift = client.get("/api/v1/decisions/drift", params={"agent_id": "main", "decision_key": "model"})
        store.close()
    require(drift.status_code == 200, f"drift failed: {drift.text}")
    drift_json = drift.json()
    require(drift_json["count"] >= 1, f"expected drift event: {drift_json}")
    require(drift_json["events"][0]["selected"] == "4B", f"unexpected drift event: {drift_json}")


if __name__ == "__main__":
    test_extract_model_switch()
    test_drift_summary()
    test_store_query()
    test_capture_endpoint_smoke()
    print("decision_particles: 4/4 passed")
