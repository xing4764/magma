"""MAGMA Ops - Self-service status check and safe repair.

Usage:
  python scripts/magma_ops.py status    # One-line health summary
  python scripts/magma_ops.py repair    # Safe self-repair checks and suggestions
"""

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

API_BASE = "http://127.0.0.1:8902"
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "magma.db"
MCP_SERVER = PROJECT_ROOT / "magma" / "api" / "mcp_server.py"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
OPENCLAW_NPM_PACKAGE = Path.home() / ".openclaw" / "npm" / "package.json"
TIMEOUT_S = 5
CST = timezone(timedelta(hours=8))


def _check_port(port: int) -> bool:
    """Check if a TCP port is listening."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def _check_api() -> tuple:
    """Returns (ok, detail_str)."""
    try:
        req = urllib.request.Request(f"{API_BASE}/api/v1/health")
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
        data = json.loads(resp.read())
        return True, f"version={data.get('version')}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_mcp_proxy() -> tuple:
    """Returns (ok, mode_str)."""
    if not MCP_SERVER.exists():
        return False, "mcp_server.py not found"
    content = MCP_SERVER.read_text(encoding="utf-8")
    uses_proxy = "API_BASE" in content and "_api_request" in content
    uses_local = "MemorySearcher(get_store()" in content
    if uses_proxy and not uses_local:
        return True, "http_proxy"
    elif uses_local:
        return False, "local_store (should be http_proxy)"
    return None, "unknown"


def _check_db() -> tuple:
    """Returns (ok, detail_str)."""
    if not DB_PATH.exists():
        return False, "database not found"
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
        with_emb = cur.fetchone()[0]
        conn.close()
        pct = round(with_emb / total * 100, 1) if total > 0 else 0
        return True, f"{total} nodes, {pct}% embedded"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_embedding_model() -> tuple:
    """Check if embedding model is available (quick API call)."""
    try:
        req = urllib.request.Request(
            f"{API_BASE}/api/v1/query",
            data=json.dumps({"query": "test", "top_k": 1}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return True, f"query ok, {data.get('count', 0)} results"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _check_openclaw_version_pin() -> tuple:
    """Check OpenClaw/codex package pins stay on the known-compatible 2026.5.20 line."""
    if not OPENCLAW_NPM_PACKAGE.exists():
        return None, "OpenClaw npm package.json not found"
    try:
        data = json.loads(OPENCLAW_NPM_PACKAGE.read_text(encoding="utf-8"))
        deps = data.get("dependencies", {})
        openclaw_version = deps.get("openclaw")
        codex_version = deps.get("@openclaw/codex")
        expected = "2026.5.20"
        if openclaw_version == expected and codex_version == expected:
            return True, "openclaw/codex pinned to 2026.5.20"
        return False, f"expected openclaw/codex {expected}, got openclaw={openclaw_version}, codex={codex_version}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def cmd_status():
    """One-line status summary."""
    parts = []
    overall = "GREEN"

    # API
    api_ok, api_detail = _check_api()
    parts.append(f"API {'ok' if api_ok else 'DOWN'}")
    if not api_ok:
        overall = "RED"

    # MCP
    mcp_ok, mcp_mode = _check_mcp_proxy()
    parts.append(f"MCP {'proxy' if mcp_ok else mcp_mode}")
    if mcp_ok is False:
        overall = "RED"

    # DB
    db_ok, db_detail = _check_db()
    parts.append(db_detail)
    if not db_ok:
        overall = "RED"

    # Embedding
    if api_ok:
        emb_ok, emb_detail = _check_embedding_model()
        parts.append(f"Embedding {'ok' if emb_ok else 'FAIL'}")
        if not emb_ok:
            overall = "RED"

    status_str = f"MAGMA: {overall} - {', '.join(parts)}"
    print(status_str)
    return overall


def cmd_repair():
    """Safe self-repair checks and suggestions."""
    print(f"MAGMA Repair Check  {datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    issues = []
    actions = []

    # 1. Check 8902 port
    print("\n[1/5] Checking 8902 port...")
    port_ok = _check_port(8902)
    if port_ok:
        print("  Port 8902 is listening")
    else:
        print("  Port 8902 is NOT listening")
        issues.append("8902 API not running")
        actions.append("Start MAGMA API: python -m magma.api.server (port 8902)")

    # 2. Check API health
    print("\n[2/5] Checking API health...")
    api_ok, api_detail = _check_api()
    if api_ok:
        print(f"  API healthy: {api_detail}")
    else:
        print(f"  API error: {api_detail}")
        issues.append(f"API unhealthy: {api_detail}")
        if port_ok:
            actions.append("API port is open but health check fails - check logs")

    # 3. Check MCP proxy config
    print("\n[3/5] Checking MCP proxy mode...")
    mcp_ok, mcp_mode = _check_mcp_proxy()
    if mcp_ok:
        print(f"  MCP mode: {mcp_mode}")
    elif mcp_ok is False:
        print(f"  MCP mode: {mcp_mode}")
        issues.append("MCP not using 8902 proxy")
        actions.append("Check magma/api/mcp_server.py - should use _api_request, not local store")
    else:
        print(f"  MCP mode: {mcp_mode}")

    # 4. Check database
    print("\n[4/5] Checking database...")
    db_ok, db_detail = _check_db()
    if db_ok:
        print(f"  Database: {db_detail}")
    else:
        print(f"  Database: {db_detail}")
        issues.append(f"Database issue: {db_detail}")

    # 5. Check OpenClaw config for MCP and version pins
    print("\n[5/5] Checking OpenClaw MCP config...")
    if OPENCLAW_CONFIG.exists():
        try:
            config_text = OPENCLAW_CONFIG.read_text(encoding="utf-8")
            if "magma-memory" in config_text:
                print("  magma-memory MCP found in OpenClaw config")
                # Check if it points to correct server
                if "magma.api.mcp_server" in config_text or "mcp_server.py" in config_text:
                    print("  MCP points to mcp_server (proxy mode - correct)")
                else:
                    print("  WARNING: MCP config may not point to proxy server")
                    issues.append("MCP config may be misconfigured")
            else:
                print("  WARNING: magma-memory not in OpenClaw config")
                issues.append("MAGMA MCP not registered in OpenClaw")
        except Exception as e:
            print(f"  Could not read config: {e}")
    else:
        print(f"  Config not found: {OPENCLAW_CONFIG}")

    pin_ok, pin_detail = _check_openclaw_version_pin()
    if pin_ok:
        print(f"  Version pins: {pin_detail}")
    elif pin_ok is False:
        print(f"  WARNING: Version pins: {pin_detail}")
        issues.append("OpenClaw/codex version pins drifted")
        actions.append("Pin C:\\Users\\Administrator\\.openclaw\\npm\\package.json openclaw and @openclaw/codex to 2026.5.20")
    else:
        print(f"  Version pins: {pin_detail}")

    # Summary
    print(f"\n{'='*55}")
    if not issues:
        print("All checks passed. No repairs needed.")
    else:
        print(f"Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print(f"\nSuggested actions:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

        # Check if restart might help
        if any("not running" in i or "unhealthy" in i for i in issues):
            print("\n  NOTE: A MAGMA API restart may help.")
            print("  This script will NOT auto-restart. Run manually:")
            print("    python -m magma.api.server")
            print("  Or restart OpenClaw if MCP connection is stale.")

    print(f"\n{'='*55}")
    print("BOUNDARY: This script does NOT auto-modify OpenClaw versions,")
    print("delete data, or clean memories. Manual action required.")
    return len(issues)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "repair"):
        print("Usage:")
        print("  python scripts/magma_ops.py status    # One-line health summary")
        print("  python scripts/magma_ops.py repair    # Safe self-repair checks")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "status":
        overall = cmd_status()
        sys.exit(0 if overall == "GREEN" else 1)
    elif cmd == "repair":
        issues = cmd_repair()
        sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
