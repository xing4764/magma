"""MAGMA health board - one-page red/yellow/green operational summary.

This complements magma_doctor.py. Doctor answers "is MAGMA healthy?";
health_board also includes OpenClaw hook timeouts, session locks, L1 artifacts,
and local process memory so day-to-day triage has one entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = Path(os.environ.get("MAGMA_DB_PATH", str(PROJECT_ROOT / "data" / "magma.db")))
OPENCLAW_HOME = Path.home() / ".openclaw"
RECALL_LOG = Path(os.environ.get("MAGMA_RECALL_LOG", str(OPENCLAW_HOME / "logs" / "magma-recall.jsonl")))
CST = timezone(timedelta(hours=8))


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(str(status).lower(), 2)


def _merge_status(*statuses: str) -> str:
    return max((str(s).lower() for s in statuses), key=_status_rank, default="green")


def _parse_ts(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CST)
    except ValueError:
        try:
            return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(CST)
        except ValueError:
            return None


def _load_recent_recall_entries(hours: int = 24) -> list[dict[str, Any]]:
    if not RECALL_LOG.exists():
        return []
    cutoff = datetime.now(CST) - timedelta(hours=hours)
    entries = []
    try:
        lines = RECALL_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    for line in lines[-2000:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _parse_ts(item.get("ts") or item.get("timestamp") or "")
        if ts and ts >= cutoff:
            item["_parsed_ts"] = ts.isoformat()
            entries.append(item)
    return entries


def run_doctor() -> dict[str, Any]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "magma_doctor.py"), "--json"]
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=60, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return {"status": "red", "error": proc.stderr.strip() or proc.stdout.strip()}
        data = json.loads(proc.stdout)
        return {
            "status": str(data.get("overall") or "red").lower(),
            "checks": data.get("checks", {}),
            "warnings": data.get("warnings", []),
            "failures": data.get("failures", []),
            "capture_stats": data.get("capture_stats", {}),
        }
    except Exception as exc:
        return {"status": "red", "error": f"{type(exc).__name__}: {exc}"}


def check_hook_timeouts(hours: int = 24) -> dict[str, Any]:
    entries = _load_recent_recall_entries(hours)
    errors = []
    slow = []
    for item in entries:
        duration = int(item.get("durationMs") or 0)
        error = str(item.get("error") or "")
        if error:
            errors.append({"ts": item.get("_parsed_ts"), "durationMs": duration, "error": error, "queryPreview": item.get("queryPreview", "")[:120]})
        if duration >= 45000:
            slow.append({"ts": item.get("_parsed_ts"), "durationMs": duration, "queryPreview": item.get("queryPreview", "")[:120]})
    status = "green"
    if errors or slow:
        status = "yellow"
    if len(errors) >= 5 or len(slow) >= 2:
        status = "red"
    return {
        "status": status,
        "entries_24h": len(entries),
        "errors_24h": len(errors),
        "slow_45s_24h": len(slow),
        "last_error": errors[-1] if errors else None,
    }


def check_session_locks() -> dict[str, Any]:
    locks = []
    now = datetime.now()
    agents_dir = OPENCLAW_HOME / "agents"
    if agents_dir.exists():
        for path in agents_dir.glob("*/sessions/*.jsonl.lock"):
            try:
                age_s = (now - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
                locks.append({"path": str(path), "age_s": round(age_s, 1), "size": path.stat().st_size})
            except OSError:
                continue
    stale = [item for item in locks if item["age_s"] >= 60]
    status = "green" if not stale else "yellow"
    if any(item["age_s"] >= 300 for item in stale):
        status = "red"
    return {"status": status, "active_locks": len(locks), "stale_locks": len(stale), "locks": stale[:5]}


def check_l1_artifacts() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"status": "red", "error": "database not found"}
    try:
        from magma.l1_distiller import _is_artifact_text

        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, properties
              FROM nodes
             WHERE status != 'deleted'
               AND status != 'suppressed'
               AND json_extract(properties,'$.layer') = 'L1'
            """
        ).fetchall()
        conn.close()
        matches = []
        for row in rows:
            props = json.loads(row["properties"] or "{}")
            text = str(props.get("content") or props.get("title") or "")
            if _is_artifact_text(text):
                matches.append(row["id"])
        status = "green" if not matches else "yellow"
        return {"status": status, "active_artifacts": len(matches), "sample_ids": matches[:10]}
    except Exception as exc:
        return {"status": "red", "error": f"{type(exc).__name__}: {exc}"}


def check_magma_process() -> dict[str, Any]:
    if sys.platform != "win32":
        return {"status": "yellow", "note": "process memory check currently implemented for Windows"}
    ps = r"""
$conn = Get-NetTCPConnection -LocalPort 8904 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $conn) { Write-Output '{"status":"red","error":"port 8904 not listening"}'; exit }
$p = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
if ($null -eq $p) { Write-Output '{"status":"red","error":"owning process missing"}'; exit }
[pscustomobject]@{
  status='green';
  pid=$p.Id;
  process=$p.ProcessName;
  working_set_mb=[math]::Round($p.WorkingSet64/1MB,1);
  private_mb=[math]::Round($p.PrivateMemorySize64/1MB,1)
} | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps], text=True, capture_output=True, timeout=15, encoding="utf-8", errors="replace")
        data = json.loads(proc.stdout.strip())
        private_mb = float(data.get("private_mb") or 0)
        if private_mb >= 12000:
            data["status"] = "red"
        elif private_mb >= 9000:
            data["status"] = "yellow"
        return data
    except Exception as exc:
        return {"status": "red", "error": f"{type(exc).__name__}: {exc}"}


def build_report() -> dict[str, Any]:
    checks = {
        "doctor": run_doctor(),
        "magma_process": check_magma_process(),
        "hook_timeouts": check_hook_timeouts(),
        "session_locks": check_session_locks(),
        "l1_artifacts": check_l1_artifacts(),
    }
    overall = _merge_status(*(item.get("status", "red") for item in checks.values()))
    return {
        "timestamp": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %z"),
        "overall": overall,
        "checks": checks,
    }


def print_human(report: dict[str, Any]) -> None:
    print(f"MAGMA Health Board  {report['timestamp']}")
    print("=" * 60)
    print(f"OVERALL: {report['overall'].upper()}")
    for name, data in report["checks"].items():
        print(f"\n[{data.get('status', 'red').upper()}] {name}")
        for key, value in data.items():
            if key == "status":
                continue
            if isinstance(value, (dict, list)):
                print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGMA consolidated health board")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0 if report["overall"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
