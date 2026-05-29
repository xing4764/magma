"""Test update_node on a node WITHOUT importance in JSON properties."""
import urllib.request
import json
import sqlite3

API_BASE = "http://127.0.0.1:8904"

def api(method, path, body=None):
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        err_body = ""
        if hasattr(e, 'read'):
            err_body = e.read().decode()
        print(f"Error: {e}")
        if err_body:
            print(f"Body: {err_body}")
        return None

# Find a node WITHOUT importance in properties JSON
conn = sqlite3.connect(r'C:\openclaw-magma\data\magma.db')
conn.row_factory = sqlite3.Row
cur = conn.execute("SELECT id FROM nodes WHERE json_extract(properties, '$.importance') IS NULL AND status = 'active' LIMIT 1")
row = cur.fetchone()
if not row:
    print("No nodes without importance in properties!")
    exit(1)

node_id = row['id']
print(f"Testing node WITHOUT importance in JSON: {node_id}")

# Try to update this node via the API
print("\n=== Attempting PATCH update ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {"test_field": "test_value"})
print(f"Result: {result}")

# Also test with properties wrapped in a properties key (MCP format)
print("\n=== Attempting PATCH with wrapped properties ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {"properties": {"test_field2": "test_value2"}})
print(f"Result: {result}")

conn.close()
print("\n=== Done ===")
