"""Test add_node directly to see traceback."""
import sys, json
sys.path.insert(0, "C:/openclaw-magma")

from magma.graph.sqlite_store import get_store
from magma.vector.encoder import Encoder

store = get_store()
encoder = Encoder()

text = "debug test content for API error investigation"
print(f"Encoding text ({len(text)} chars)...")
embedding = encoder.encode(text).astype("float32")
print(f"Embedding shape: {embedding.shape}")

node_id = "l1:test:debug002"
label = "event"
properties = {
    "layer": "L1",
    "kind": "fact",
    "title": "debug test",
    "content": text,
    "importance": 0.8,
    "scope": "system",
}

print(f"Adding node {node_id}...")
try:
    store.add_node(node_id, label, properties, embedding)
    print("OK: node added")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
