"""Run MAGMA L1 distillation through the public API.

Default is dry-run. Use --apply to write L1 nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path


def configured_api_base() -> str:
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


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAGMA L1 distillation.")
    parser.add_argument("--api-base", default=configured_api_base())
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--source-agent-id", default=None, help="Only distill L0 memories from this source agent")
    parser.add_argument("--apply", action="store_true", help="Write L1 nodes. Default is dry-run.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = post_json(
        f"{args.api_base.rstrip('/')}/api/v1/distill_l1",
        {
            "hours": args.hours,
            "limit": args.limit,
            "dry_run": not args.apply,
            "source_agent_id": args.source_agent_id,
        },
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        mode = "apply" if args.apply else "dry-run"
        print(
            f"MAGMA L1 distill {mode}: scanned={result.get('scanned')} "
            f"candidates={result.get('candidate_count')} written={result.get('written_count')}"
        )
        for kind, count in sorted((result.get("by_kind") or {}).items()):
            print(f"- {kind}: {count}")
        for item in result.get("preview", [])[:5]:
            print(f"  {item['kind']} {item['id']}: {item['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
