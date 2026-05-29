"""Test update_node to reproduce the NOT NULL constraint error."""
import urllib.request
import json
import sys

API_BASE = "http://127.0.0.1:8904"  # FastAPI runs on 8904

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
        print(f"Error: {e}")
        if hasattr(e, 'read'):
            print(f"Body: {e.read().decode()}")
        return None

# 1. Get a node to test with
print("=== Step 1: Get a sample node ===")
nodes = api("GET", "/api/v1/nodes?limit=1")
if not nodes or not nodes.get("nodes"):
    print("No nodes found!")
    sys.exit(1)

node_id = nodes["nodes"][0]["id"]
print(f"Testing with node: {node_id}")

# 2. Try updating with just a content field (no importance)
print("\n=== Step 2: Update node with just content ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {"content": "test update"})
print(f"Result: {result}")

# 3. Try updating with empty properties
print("\n=== Step 3: Update node with empty properties ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {})
print(f"Result: {result}")

# 4. Try updating with explicit None importance
print("\n=== Step 4: Update node with importance=None ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {"importance": None})
print(f"Result: {result}")

# 5. Try updating with importance=0
print("\n=== Step 5: Update node with importance=0 ===")
result = api("PATCH", f"/api/v1/nodes/{node_id}", {"importance": 0})
print(f"Result: {result}")

print("\n=== Done ===")
