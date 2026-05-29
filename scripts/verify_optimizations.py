#!/usr/bin/env python3
"""Verify all MAGMA v2 optimizations are working."""
import sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API = "http://127.0.0.1:8904/api/v1"

def api(method, path, body=None):
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

print("=" * 60)
print("MAGMA v2 Optimization Verification")
print("=" * 60)

# 1. Fact Extraction
print("\n[1] Fact Extraction: capture with LLM fact extraction")
r = api("POST", "/capture", {
    "user_text": "我的手机号是 13912345678，偏好浅色模式",
    "agent_id": "jishu",
    "session_key": "agent:jishu:verify:test",
    "source": "verification_test",
})
fact_nodes = [n for n in r.get("written", []) if n.startswith("fact:")]
print(f"  Capture wrote {r['count']} nodes, {len(fact_nodes)} are facts")
for n in r.get("written", []):
    print(f"    {n}")
print(f"  PASS: {len(fact_nodes) >= 2}" if len(fact_nodes) >= 2 else f"  FAIL: expected >= 2 facts")

# 2. Temporal Reasoning
print("\n[2] Temporal Reasoning: capture conflicting fact")
r = api("POST", "/capture", {
    "user_text": "换手机号了，新号码是 13800001111",
    "agent_id": "jishu",
    "session_key": "agent:jishu:verify:test",
    "source": "verification_test",
})
print(f"  Written {r['count']} nodes")

# Check timeline
r = api("GET", "/timeline/%E6%89%8B%E6%9C%BA%E5%8F%B7?limit=5")
historical = [f for f in r.get("facts", []) if f.get("valid_until")]
current = [f for f in r.get("facts", []) if not f.get("valid_until")]
print(f"  Timeline: {len(current)} CURRENT, {len(historical)} HISTORICAL")
print(f"  PASS: {len(historical) >= 1}" if len(historical) >= 1 else "  FAIL: no HISTORICAL facts")

# 3. Active Facts API
print("\n[3] Active Facts API")
r = api("GET", "/facts?limit=5")
print(f"  Active facts: {r.get('count', 0)}")
print(f"  PASS: {r.get('count', 0) >= 1}" if r.get("count", 0) >= 1 else "  FAIL: no active facts")

# 4. Stats
print("\n[4] Stats")
r = api("GET", "/stats")
print(f"  Nodes: {r['total_nodes']}, Edges: {r['total_edges']}, 24h: {r['capture_24h']}")
print(f"  Layers: {r['layers']}")

# 5. Doctor
print("\n[5] Doctor")
r = api("GET", "/doctor")
print(f"  Status: {r['status']}")
print(f"  FAISS: {r['faiss']}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
