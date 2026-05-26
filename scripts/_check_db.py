import os
import sqlite3
import json
from pathlib import Path

DB_PATH = os.environ.get(
    "MAGMA_DB_PATH",
    str(Path(__file__).parent.parent / "data" / "magma.db"),
)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
print("tables:", [r[0] for r in cur.fetchall()])
for t in ["nodes", "edges", "recall_events", "recall_feedback"]:
    try:
        cur.execute(f'PRAGMA table_info({t})')
        print(f'{t} cols:', [r[1] for r in cur.fetchall()])
    except:
        print(f'{t}: not found')
# sample nodes
cur.execute('SELECT id, label, source_agent_id, department FROM nodes LIMIT 5')
print("sample nodes:", cur.fetchall())
# check if source_agent_id column exists
cur.execute('PRAGMA table_info(nodes)')
cols = [r[1] for r in cur.fetchall()]
print("has source_agent_id:", "source_agent_id" in cols)
print("has department:", "department" in cols)
conn.close()
