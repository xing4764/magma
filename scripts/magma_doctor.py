"""MAGMA Doctor - Red/Yellow/Green health diagnostics.

Usage:
  python scripts/magma_doctor.py            # human-readable full check
  python scripts/magma_doctor.py --json     # JSON output for automation
  python scripts/magma_doctor.py --quick    # API-only quick check
"""

import json
import os
import re
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent


def _configured_api_base() -> str:
    env_base = os.environ.get("MAGMA_API_BASE")
    if env_base:
        return env_base.rstrip("/")
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'"apiBaseUrl"\s*:\s*"(http://127\.0\.0\.1:\d+)"', text)
        if match:
            return match.group(1).rstrip("/")
    except OSError:
        pass
    return "http://127.0.0.1:8902"


API_BASE = _configured_api_base()
DB_PATH = Path(os.environ.get("MAGMA_DB_PATH", str(PROJECT_ROOT / "data" / "magma.db")))
RECALL_LOG = Path(os.environ.get("MAGMA_RECALL_LOG", str(Path.home() / ".openclaw" / "logs" / "magma-recall.jsonl")))
TIMEOUT_S = 10
CST = timezone(timedelta(hours=8))

DEPT_MAP = {
    "yunying": "运营部",
    "jishu": "技术部",
    "zhuli": "助理",
    "main": "老板",
}


def _now():
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")


def _parse_log_ts(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(CST)
    except ValueError:
        return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=CST)


def _parse_sqlite_utc(value: str):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(CST)


def _parse_source_agent(session_key: str) -> str:
    """Extract agentId from session_key like 'agent:yunying:feishu:...'"""
    if not session_key:
        return ""
    m = re.search(r"agent:([^:\s]+)", session_key)
    return m.group(1) if m else ""


# ─── Individual checks ────────────────────────────────────────────

def check_api():
    """Check configured MAGMA API health."""
    try:
        req = urllib.request.Request(f"{API_BASE}/api/v1/health")
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
        data = json.loads(resp.read())
        return "green", {"api_base": API_BASE, "status": data.get("status"), "version": data.get("version")}
    except urllib.error.URLError as e:
        return "red", {"error": str(e.reason)}
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_mcp_proxy():
    """Check MCP server uses 8902 proxy (not local store)."""
    mcp_server = PROJECT_ROOT / "magma" / "api" / "mcp_server.py"
    if not mcp_server.exists():
        return "red", {"error": "mcp_server.py not found"}
    try:
        content = mcp_server.read_text(encoding="utf-8")
        uses_http_proxy = "API_BASE" in content and "_api_request" in content
        uses_local_store = "MemorySearcher(get_store()" in content
        if uses_http_proxy and not uses_local_store:
            return "green", {"mode": "http_proxy"}
        elif uses_local_store:
            return "red", {"mode": "local_store", "note": "MCP not using 8902 proxy!"}
        else:
            return "yellow", {"mode": "unknown"}
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_recall_active():
    """Check recall is active (events in last 24h)."""
    if not RECALL_LOG.exists():
        return "yellow", {"error": "recall log not found"}
    try:
        with open(RECALL_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return "yellow", {"error": "recall log empty"}
        last_entry = json.loads(lines[-1])
        ts_str = last_entry.get("ts") or last_entry.get("timestamp") or last_entry.get("created_at")
        if not ts_str:
            return "yellow", {"error": "no timestamp in last entry"}
        last_ts = _parse_log_ts(ts_str)
        age_hours = (datetime.now(CST) - last_ts).total_seconds() / 3600
        if age_hours > 24:
            return "yellow", {"last_event_hours_ago": round(age_hours, 1), "total": len(lines)}
        return "green", {"last_event_hours_ago": round(age_hours, 1), "total": len(lines)}
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_feedback_active():
    """Check feedback events exist and are recent."""
    if not DB_PATH.exists():
        return "red", {"error": "database not found"}
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM recall_feedback")
        total = cur.fetchone()[0]
        cur.execute("SELECT MAX(created_at) FROM recall_feedback")
        last = cur.fetchone()[0]
        conn.close()
        if total == 0:
            return "yellow", {"total": 0, "note": "no feedback events"}
        if last:
            last_ts = _parse_sqlite_utc(last)
            age_hours = (datetime.now(CST) - last_ts).total_seconds() / 3600
            if age_hours > 48:
                return "yellow", {"total": total, "last_hours_ago": round(age_hours, 1)}
        return "green", {"total": total, "last": last}
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_embedding_coverage():
    """Check embedding coverage >= 50%."""
    if not DB_PATH.exists():
        return "red", {"error": "database not found"}
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
        with_emb = cur.fetchone()[0]
        conn.close()
        pct = round(with_emb / total * 100, 1) if total > 0 else 0
        if pct < 50:
            return "red", {"pct": pct, "with_embedding": with_emb, "total": total}
        return "green", {"pct": pct, "with_embedding": with_emb, "total": total}
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_recent_capture():
    """Check most recent node capture is within 6 hours."""
    if not DB_PATH.exists():
        return "red", {"error": "database not found"}
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(created_at)
            FROM nodes
            WHERE json_extract(properties, '$.source') = 'openclaw_auto_capture'
        """)
        last = cur.fetchone()[0]
        conn.close()
        if not last:
            return "yellow", {"error": "no auto-captured nodes"}
        last_ts = _parse_sqlite_utc(last)
        age_hours = (datetime.now(CST) - last_ts).total_seconds() / 3600
        detail = {
            "last_capture_hours_ago": round(age_hours, 1),
            "last_capture_time": last_ts.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        if age_hours > 6:
            return "yellow", detail
        return "green", detail
    except Exception as e:
        return "red", {"error": f"{type(e).__name__}: {e}"}


def check_openclaw_version_guardrail():
    """Ensure OpenClaw version/upgrade questions hit the operational anchor first."""
    expected = "ops:openclaw:version-pin-2026-5-20"
    queries = [
        "看一下 OpenClaw 最新测试版能不能升级？",
        "我们用的 5.28？",
        "OpenClaw 6.1 能不能升级？",
    ]
    failures = []
    samples = []
    for query in queries:
        payload = json.dumps({
            "query": query,
            "top_k": 3,
            "filters": {"agent_id": "zhuli"},
        }, ensure_ascii=False).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{API_BASE}/api/v1/query",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            failures.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results = data.get("results") or []
        top_ids = [item.get("id") for item in results[:3]]
        samples.append({"query": query, "top_ids": top_ids})
        if not results or results[0].get("id") != expected:
            failures.append({"query": query, "top_ids": top_ids, "expected_top_id": expected})
    detail = {
        "expected_top_id": expected,
        "samples": samples,
    }
    if failures:
        detail["failures"] = failures
        return "yellow", detail
    return "green", detail


def get_capture_stats():
    """Get capture attempts/success/errors by agent from audit log and DB."""
    result = {
        "attempts_24h": 0,
        "success_24h": 0,
        "errors_24h": 0,
        "last_success": None,
        "last_error": None,
        "by_agent": {},
        "db_last_auto_capture": None,
    }
    cutoff = datetime.now(CST) - timedelta(hours=24)

    if RECALL_LOG.exists():
        try:
            with open(RECALL_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("type") != "capture":
                        continue
                    ts = _parse_log_ts(entry.get("ts") or entry.get("timestamp"))
                    if not ts or ts < cutoff:
                        continue
                    agent_id = entry.get("agentId") or entry.get("agent_id") or "unknown"
                    agent = result["by_agent"].setdefault(agent_id, {
                        "attempts_24h": 0,
                        "success_24h": 0,
                        "errors_24h": 0,
                        "last_success": None,
                        "last_error": None,
                    })
                    result["attempts_24h"] += 1
                    agent["attempts_24h"] += 1
                    if entry.get("error"):
                        result["errors_24h"] += 1
                        agent["errors_24h"] += 1
                        error_info = {"time": ts.isoformat(), "agent": agent_id, "error": entry.get("error")}
                        result["last_error"] = error_info
                        agent["last_error"] = error_info
                    else:
                        result["success_24h"] += 1
                        agent["success_24h"] += 1
                        success_info = {
                            "time": ts.isoformat(),
                            "agent": agent_id,
                            "count": entry.get("count"),
                            "written": entry.get("written", []),
                        }
                        result["last_success"] = success_info
                        agent["last_success"] = success_info
        except Exception as e:
            result["log_error"] = f"{type(e).__name__}: {e}"

    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            cur = conn.cursor()
            cur.execute("""
                SELECT MAX(created_at)
                FROM nodes
                WHERE json_extract(properties, '$.source') = 'openclaw_auto_capture'
            """)
            last = cur.fetchone()[0]
            if last:
                result["db_last_auto_capture"] = _parse_sqlite_utc(last).isoformat()
            conn.close()
        except Exception as e:
            result["db_error"] = f"{type(e).__name__}: {e}"

    return result


def get_agent_injection():
    """Get agent injection stats from DB (source_agent_id) and recall log."""
    result = {"agents": {}, "source_agents": {}}

    # From DB - source_agent_id on nodes
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=5)
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COALESCE(source_agent_id, json_extract(properties, '$.source_agent_id'), json_extract(properties, '$.agent_id')) AS source_agent,
                    COALESCE(department, json_extract(properties, '$.department')) AS dept,
                    COUNT(*),
                    MAX(created_at)
                FROM nodes
                WHERE COALESCE(source_agent_id, json_extract(properties, '$.source_agent_id'), json_extract(properties, '$.agent_id')) IS NOT NULL
                GROUP BY source_agent
            """)
            for agent_id, dept, count, last in cur.fetchall():
                result["source_agents"][agent_id] = {
                    "count": count,
                    "department": dept or DEPT_MAP.get(agent_id, ""),
                    "last_time": _parse_sqlite_utc(last).strftime("%Y-%m-%d %H:%M:%S %z") if last else None,
                }
            conn.close()
        except Exception:
            pass

    # From recall log - agentId field
    if RECALL_LOG.exists():
        try:
            with open(RECALL_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        agent_id = entry.get("agentId") or entry.get("agent_id", "unknown")
                        if agent_id not in result["agents"]:
                            result["agents"][agent_id] = {"count": 0, "last_time": None}
                        result["agents"][agent_id]["count"] += 1
                        ts = entry.get("ts") or entry.get("timestamp")
                        if ts:
                            result["agents"][agent_id]["last_time"] = ts
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    return result


# ─── Overall assessment ───────────────────────────────────────────

def assess_overall(checks: dict) -> tuple:
    """Return (overall_status, failures, warnings)."""
    failures = []
    warnings = []
    for name, (status, detail) in checks.items():
        if status == "red":
            failures.append(f"{name}: {detail.get('error', detail)}")
        elif status == "yellow":
            warnings.append(f"{name}: {detail}")
    if failures:
        return "red", failures, warnings
    if warnings:
        return "yellow", failures, warnings
    return "green", failures, warnings


# ─── Output ───────────────────────────────────────────────────────

def run_json():
    checks = {
        "api": check_api(),
        "mcp_proxy": check_mcp_proxy(),
        "recall_active": check_recall_active(),
        "feedback_active": check_feedback_active(),
        "embedding_coverage": check_embedding_coverage(),
        "recent_capture": check_recent_capture(),
        "openclaw_version_guardrail": check_openclaw_version_guardrail(),
    }
    overall, failures, warnings = assess_overall(checks)
    injection = get_agent_injection()

    output = {
        "timestamp": _now(),
        "overall": overall,
        "failures": failures,
        "warnings": warnings,
        "checks": {name: status for name, (status, _) in checks.items()},
        "check_details": {name: detail for name, (_, detail) in checks.items()},
        "agent_injection": injection,
        "capture_stats": get_capture_stats(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return output


def run_human():
    print(f"{'='*55}")
    print(f"MAGMA Doctor  {_now()}")
    print(f"{'='*55}\n")

    checks = {
        "api": check_api(),
        "mcp_proxy": check_mcp_proxy(),
        "recall_active": check_recall_active(),
        "feedback_active": check_feedback_active(),
        "embedding_coverage": check_embedding_coverage(),
        "recent_capture": check_recent_capture(),
        "openclaw_version_guardrail": check_openclaw_version_guardrail(),
    }

    emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    for name, (status, detail) in checks.items():
        print(f"  {emoji.get(status, '?')} {name}: {status}")
        for k, v in detail.items():
            print(f"      {k}: {v}")

    overall, failures, warnings = assess_overall(checks)
    print(f"\n{'='*55}")
    print(f"  Overall: {emoji.get(overall, '?')} {overall.upper()}")
    if failures:
        print(f"  Failures ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
    if warnings:
        print(f"  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    injection = get_agent_injection()
    sources = injection.get("source_agents", {})
    if sources:
        print(f"\n  Source Agents (from DB):")
        for agent, info in sources.items():
            print(f"    {agent} ({info.get('department', '?')}): {info['count']} nodes, last: {info.get('last_time', '?')}")

    agents = injection.get("agents", {})
    if agents:
        print(f"\n  Recall Agents (from log):")
        for agent, info in agents.items():
            print(f"    {agent}: {info['count']} recalls, last: {info.get('last_time', '?')}")

    capture = get_capture_stats()
    print(f"\n  Capture Stats (last 24h):")
    print(f"    attempts: {capture.get('attempts_24h', 0)}, success: {capture.get('success_24h', 0)}, errors: {capture.get('errors_24h', 0)}")
    if capture.get("last_success"):
        print(f"    last_success: {capture['last_success']}")
    if capture.get("last_error"):
        print(f"    last_error: {capture['last_error']}")

    print(f"{'='*55}")
    return overall


def run_quick():
    status, detail = check_api()
    if status == "green":
        print(f"MAGMA API ok (version={detail.get('version')})")
        return True
    else:
        print(f"MAGMA API down: {detail.get('error')}")
        return False


def run_agent():
    injection = get_agent_injection()
    sources = injection.get("source_agents", {})
    agents = injection.get("agents", {})

    if sources:
        print("Source Agents (from DB):")
        for agent, info in sources.items():
            print(f"  {agent} ({info.get('department', '?')}): {info['count']} nodes, last: {info.get('last_time', '?')}")
    else:
        print("Source Agents (from DB): no records")

    if agents:
        print("\nRecall Agents (from log):")
        for agent, info in agents.items():
            print(f"  {agent}: {info['count']} recalls, last: {info.get('last_time', '?')}")
    else:
        print("\nRecall Agents (from log): no records")


def main():
    if "--json" in sys.argv:
        run_json()
    elif "--agent" in sys.argv:
        run_agent()
    elif "--quick" in sys.argv:
        ok = run_quick()
        sys.exit(0 if ok else 1)
    else:
        overall = run_human()
        sys.exit(0 if overall == "green" else 1)


if __name__ == "__main__":
    main()
