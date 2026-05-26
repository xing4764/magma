"""MAGMA DB Migration: Add source_agent_id and department columns.

Run once: python C:\openclaw-magma\scripts\migrate_source_agent.py
"""

import sqlite3
import json
import re
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "magma.db"

DEPT_MAP = {
    "yunying": "运营部",
    "jishu": "技术部",
    "zhuli": "助理",
    "main": "老板",
}


def migrate():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 1. Add columns to nodes (idempotent)
    cur.execute("PRAGMA table_info(nodes)")
    node_cols = {r[1] for r in cur.fetchall()}

    if "source_agent_id" not in node_cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN source_agent_id TEXT")
        print("[nodes] added source_agent_id")
    else:
        print("[nodes] source_agent_id already exists")

    if "department" not in node_cols:
        cur.execute("ALTER TABLE nodes ADD COLUMN department TEXT")
        print("[nodes] added department")
    else:
        print("[nodes] department already exists")

    # 2. Add columns to recall_events
    cur.execute("PRAGMA table_info(recall_events)")
    recall_cols = {r[1] for r in cur.fetchall()}

    if "source_agent_id" not in recall_cols:
        cur.execute("ALTER TABLE recall_events ADD COLUMN source_agent_id TEXT")
        print("[recall_events] added source_agent_id")
    else:
        print("[recall_events] source_agent_id already exists")

    if "department" not in recall_cols:
        cur.execute("ALTER TABLE recall_events ADD COLUMN department TEXT")
        print("[recall_events] added department")
    else:
        print("[recall_events] department already exists")

    # 3. Add columns to recall_feedback
    cur.execute("PRAGMA table_info(recall_feedback)")
    fb_cols = {r[1] for r in cur.fetchall()}

    if "source_agent_id" not in fb_cols:
        cur.execute("ALTER TABLE recall_feedback ADD COLUMN source_agent_id TEXT")
        print("[recall_feedback] added source_agent_id")
    else:
        print("[recall_feedback] source_agent_id already exists")

    if "department" not in fb_cols:
        cur.execute("ALTER TABLE recall_feedback ADD COLUMN department TEXT")
        print("[recall_feedback] added department")
    else:
        print("[recall_feedback] department already exists")

    conn.commit()

    # 4. Backfill nodes from properties JSON
    print("\nBackfilling nodes...")
    cur.execute("SELECT id, properties FROM nodes WHERE source_agent_id IS NULL")
    rows = cur.fetchall()
    updated = 0
    for node_id, props_json in rows:
        agent_id = None
        # Try to extract from properties
        if props_json:
            try:
                props = json.loads(props_json) if isinstance(props_json, str) else props_json
                agent_id = props.get("agent_id")
                if not agent_id:
                    session_key = props.get("session_key", "")
                    m = re.search(r"agent:([^:\s]+)", session_key or "")
                    if m:
                        agent_id = m.group(1)
            except (json.JSONDecodeError, AttributeError):
                pass
        # Fallback: parse from node_id or label
        if not agent_id:
            m = re.search(r"agent:([^:\s]+)", node_id or "")
            if m:
                agent_id = m.group(1)

        if agent_id:
            dept = DEPT_MAP.get(agent_id, "")
            cur.execute(
                "UPDATE nodes SET source_agent_id=?, department=? WHERE id=?",
                (agent_id, dept, node_id),
            )
            updated += 1
    print(f"  backfilled {updated}/{len(rows)} nodes")

    # 5. Backfill recall_events
    print("Backfilling recall_events...")
    cur.execute("SELECT id, agent_id, session_key FROM recall_events WHERE source_agent_id IS NULL")
    rows = cur.fetchall()
    updated = 0
    for evt_id, agent_id, session_key in rows:
        source = agent_id
        if not source and session_key:
            m = re.search(r"agent:([^:\s]+)", session_key or "")
            if m:
                source = m.group(1)
        if source:
            dept = DEPT_MAP.get(source, "")
            cur.execute(
                "UPDATE recall_events SET source_agent_id=?, department=? WHERE id=?",
                (source, dept, evt_id),
            )
            updated += 1
    print(f"  backfilled {updated}/{len(rows)} recall_events")

    # 6. Backfill recall_feedback
    print("Backfilling recall_feedback...")
    cur.execute("""
        SELECT f.id, e.agent_id, e.session_key
        FROM recall_feedback f
        LEFT JOIN recall_events e ON f.event_id = e.id
        WHERE f.source_agent_id IS NULL
    """)
    rows = cur.fetchall()
    updated = 0
    for fb_id, agent_id, session_key in rows:
        source = agent_id
        if not source and session_key:
            m = re.search(r"agent:([^:\s]+)", session_key or "")
            if m:
                source = m.group(1)
        if source:
            dept = DEPT_MAP.get(source, "")
            cur.execute(
                "UPDATE recall_feedback SET source_agent_id=?, department=? WHERE id=?",
                (source, dept, fb_id),
            )
            updated += 1
    print(f"  backfilled {updated}/{len(rows)} recall_feedback")

    conn.commit()

    # 7. Summary
    cur.execute("SELECT source_agent_id, COUNT(*) FROM nodes WHERE source_agent_id IS NOT NULL GROUP BY source_agent_id")
    print(f"\nNode source_agent_id distribution: {dict(cur.fetchall())}")
    cur.execute("SELECT department, COUNT(*) FROM nodes WHERE department IS NOT NULL GROUP BY department")
    print(f"Node department distribution: {dict(cur.fetchall())}")

    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    migrate()
