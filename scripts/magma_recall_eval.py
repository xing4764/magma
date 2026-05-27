"""Small recall-quality eval set for MAGMA auto-recall relevance.

Scores each question by checking whether expected operational facts appear in
the top retrieved memories from the live MAGMA API.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


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


CASES = [
    {
        "id": "mcp_proxy_8902",
        "query": "MAGMA MCP 为什么要改成 8902 主链路薄代理？",
        "expect_any": ["8902", "mcp_proxy", "http_proxy", "薄代理"],
    },
    {
        "id": "recent_capture_yellow",
        "query": "MAGMA doctor 的 recent_capture 变成 yellow 是什么意思？系统坏了吗？",
        "expect_any": ["recent_capture", "6h", "6 小时", "yellow", "最近写入"],
    },
    {
        "id": "version_pin_520",
        "query": "OpenClaw 为什么现在要 pin 在 2026.5.20，不能升 5.22？",
        "expect_any": ["2026.5.20", "5.22", "版本 pin", "codex"],
    },
    {
        "id": "source_agent_id",
        "query": "MAGMA 为什么要把 source_agent_id 正式入库？",
        "expect_any": ["source_agent_id", "department", "跨 agent", "归因"],
    },
    {
        "id": "p0_ops",
        "query": "MAGMA 的 P0 运维化三件套是什么？",
        "expect_any": ["magma_doctor.py", "magma_ops.py", "RUNBOOK.md", "红黄绿"],
    },
    {
        "id": "yunying_injection",
        "query": "为什么之前判断 yunying 没有 MAGMA 注入是不完整判断？",
        "expect_any": ["yunying", "source_agents", "source_agent_id", "subagent"],
    },
]


def post_query(query: str, top_k: int) -> list:
    payload = {
        "query": query,
        "top_k": top_k,
        "filters": {
            "include_related": True,
            "related_limit": 2,
            "include_versions": True,
            "version_limit": 2,
            "pool_size": 5000,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/api/v1/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        body = json.loads(res.read().decode("utf-8"))
    return body.get("results") or []


def direct_result_text(result: dict) -> str:
    chunks = [
        result.get("id"),
        result.get("label"),
        result.get("memory_scope"),
        result.get("source_agent_id"),
        result.get("department"),
        json.dumps(result.get("properties") or {}, ensure_ascii=False),
    ]
    return "\n".join(str(chunk) for chunk in chunks if chunk).lower()


def result_text(result: dict) -> str:
    chunks = [
        direct_result_text(result),
        json.dumps(result.get("related_context") or [], ensure_ascii=False),
        json.dumps(result.get("version_context") or [], ensure_ascii=False),
    ]
    return "\n".join(str(chunk) for chunk in chunks if chunk).lower()


def score_case(case: dict, results: list) -> dict:
    direct_haystack = "\n".join(direct_result_text(item) for item in results)
    haystack = "\n".join(result_text(item) for item in results)
    direct_matched = [term for term in case["expect_any"] if term.lower() in direct_haystack]
    matched = [term for term in case["expect_any"] if term.lower() in haystack]
    top_text = result_text(results[0]) if results else ""
    top_matched = [term for term in case["expect_any"] if term.lower() in top_text]
    if direct_matched:
        score = 2
    elif matched:
        score = 1
    else:
        score = 0
    return {
        "id": case["id"],
        "score": score,
        "matched": matched,
        "direct_matched": direct_matched,
        "top_matched": top_matched,
        "top_ids": [item.get("id") for item in results[:3]],
        "top_scores": [item.get("score") for item in results[:3]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MAGMA recall quality.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reports = []
    for case in CASES:
        try:
            results = post_query(case["query"], args.top_k)
        except (urllib.error.URLError, TimeoutError) as exc:
            reports.append({"id": case["id"], "score": 0, "error": str(exc)})
            continue
        reports.append(score_case(case, results))

    total = sum(item.get("score", 0) for item in reports)
    max_total = len(CASES) * 2
    summary = {
        "total": total,
        "max_total": max_total,
        "pct": round(total / max_total * 100, 1) if max_total else 0,
        "pass": total >= int(max_total * 0.75),
        "cases": reports,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["pass"] else 1

    print(f"MAGMA recall eval: {total}/{max_total} ({summary['pct']}%)")
    for item in reports:
        marker = "OK" if item.get("score") == 2 else "PARTIAL" if item.get("score") == 1 else "MISS"
        print(f"- {marker} {item['id']}: score={item.get('score')} matched={item.get('matched', [])} top={item.get('top_ids', [])}")
        if item.get("error"):
            print(f"  error={item['error']}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
