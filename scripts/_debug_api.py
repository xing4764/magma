"""Debug API add_node 500 error."""
import json, urllib.request, traceback

API = "http://127.0.0.1:8904"

node = {
    "id": "l1:test:debug001",
    "label": "event",
    "properties": {
        "layer": "L1",
        "kind": "fact",
        "title": "debug test",
        "content": "debug test content for API error investigation",
        "importance": 0.8,
        "scope": "system",
    }
}

data = json.dumps(node, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    f"{API}/api/v1/nodes",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as res:
        body = json.loads(res.read().decode("utf-8"))
        print(f"OK: {body}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
    print(f"Headers: {dict(e.headers)}")
    body = e.read().decode("utf-8", errors="replace")
    print(f"Body: {body[:1000]}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
