import sqlite3, json

conn = sqlite3.connect(r'C:\openclaw-magma\data\magma.db')
conn.row_factory = sqlite3.Row

# Check schema
cur = conn.execute('PRAGMA table_info(nodes)')
for row in cur.fetchall():
    print(f"  {row['name']:20s} | notnull={row['notnull']} | default={row['dflt_value']}")

print()

# Check if any nodes have NULL importance
cur = conn.execute("SELECT COUNT(*) as cnt FROM nodes WHERE importance IS NULL")
null_count = cur.fetchone()['cnt']
print(f"Nodes with NULL importance: {null_count}")

# Check total nodes
cur = conn.execute("SELECT COUNT(*) as cnt FROM nodes")
total = cur.fetchone()['cnt']
print(f"Total nodes: {total}")

# Try reproducing the error: update a node and see what happens
# First, get a sample node
cur = conn.execute("SELECT id, importance FROM nodes WHERE status = 'active' LIMIT 1")
row = cur.fetchone()
if row:
    print(f"\nSample node: id={row['id']}, importance={row['importance']}")

conn.close()
