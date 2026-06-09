"""批量分类 MAGMA unknown 节点"""
import sqlite3
import json
from collections import Counter

DB = r"C:\openclaw-magma\data\magma.db"

def classify(row):
    """根据节点属性推断 layer"""
    node_id, label, props_json = row
    try:
        props = json.loads(props_json) if props_json else {}
    except:
        return "L0"  # 默认

    source = props.get("source", "")
    kind = props.get("kind", "")
    entities = props.get("entities", [])
    block_name = props.get("block_name", "")
    role = props.get("role", "")
    content = props.get("content", "")

    # core_memory
    if block_name or label == "core_memory":
        return "core_memory"

    # ops_anchor
    if kind == "ops_anchor" or label == "ops_anchor":
        return "ops_anchor"

    # entity_anchor
    if label == "entity" or (entities and kind == ""):
        return "entity_anchor"

    # L1 (distilled facts)
    if kind in ("fact", "decision", "lesson", "current_state", "preference"):
        return "L1"
    if props.get("distillation") == "llm":
        return "L1"

    # L0 (auto-captured)
    if source == "openclaw_auto_capture" or role in ("assistant", "user"):
        return "L0"
    if label == "event":
        return "L0"

    # Default
    return "L0"

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Get all unknown nodes
cur.execute("""
    SELECT id, label, properties
    FROM nodes
    WHERE json_extract(properties, '$.layer') = 'unknown'
       OR (json_extract(properties, '$.layer') IS NULL AND status != 'deleted')
""")

rows = cur.fetchall()
print(f"Found {len(rows)} unknown/null layer nodes")

# Classify
classifications = Counter()
updates = []
for row in rows:
    layer = classify(row)
    classifications[layer] += 1
    updates.append((layer, row[0]))

print("\nClassification results:")
for layer, count in classifications.most_common():
    print(f"  {layer}: {count}")

# Apply updates
print(f"\nApplying {len(updates)} updates...")
for layer, node_id in updates:
    cur.execute("""
        UPDATE nodes
        SET properties = json_set(properties, '$.layer', ?)
        WHERE id = ?
    """, (layer, node_id))

conn.commit()
print(f"Done. Updated {len(updates)} nodes.")

# Verify
cur.execute("""
    SELECT json_extract(properties, '$.layer'), COUNT(*)
    FROM nodes WHERE status != 'deleted'
    GROUP BY json_extract(properties, '$.layer')
""")
print("\nPost-update layer distribution:")
for layer, count in cur.fetchall():
    print(f"  {layer}: {count}")

conn.close()
