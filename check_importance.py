"""Check node importance values in the database."""
import sqlite3

conn = sqlite3.connect(r'C:\openclaw-magma\data\magma.db')
conn.row_factory = sqlite3.Row

# Check if any nodes have NULL importance column
cur = conn.execute("SELECT COUNT(*) as cnt FROM nodes WHERE importance IS NULL")
print(f"Nodes with NULL importance column: {cur.fetchone()['cnt']}")

# Check if any nodes have NULL in properties JSON
cur = conn.execute("SELECT COUNT(*) as cnt FROM nodes WHERE json_extract(properties, '$.importance') IS NULL")
cnt_null_json = cur.fetchone()['cnt']
print(f"Nodes without importance in JSON properties: {cnt_null_json}")

# Get a sample node without importance in properties
if cnt_null_json > 0:
    cur = conn.execute("SELECT id, properties FROM nodes WHERE json_extract(properties, '$.importance') IS NULL LIMIT 3")
    for row in cur.fetchall():
        print(f"  Sample: id={row['id']}, properties={row['properties'][:200]}")

# Try to reproduce the NOT NULL error
print("\n=== Reproducing NOT NULL error ===")
import json
# Find a node where properties doesn't have importance
cur = conn.execute("SELECT id, properties FROM nodes WHERE json_extract(properties, '$.importance') IS NULL LIMIT 1")
row = cur.fetchone()
if row:
    node_id = row['id']
    existing = json.loads(row['properties'])
    print(f"Testing node: {node_id}")
    print(f"  Existing importance in props: {existing.get('importance')}")
    
    # Simulate update_node logic
    existing.update({"test_field": "test_value"})
    importance_from_existing = existing.get("importance")
    print(f"  importance_from_existing: {importance_from_existing}")
    importance_with_fallback = existing.get("importance", 0.5)
    print(f"  importance_with_fallback (0.5): {importance_with_fallback}")
    importance_with_or = existing.get("importance", 0.5) or 0.5
    print(f"  importance_with_or: {importance_with_or}")
    
    # Check what the OLD code (without or 0.5) would do
    old_importance = existing.get("importance")
    print(f"  OLD CODE importance (no fallback): {old_importance}")
    print(f"  OLD CODE would pass NULL to SQL: {old_importance is None}")
else:
    print("All nodes have importance in properties!")

conn.close()
