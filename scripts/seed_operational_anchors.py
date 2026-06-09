"""Seed high-signal MAGMA operational anchor memories.

These anchors make exact operational terms easy to recall even when the
conversation query is broad and semantic neighbors are noisy.
"""

import json
import os
import sys
import urllib.request


API_BASE = os.environ.get("MAGMA_API_BASE", "http://127.0.0.1:8904").rstrip("/")


ANCHORS = [
    {
        "id": "ops:magma:model-runtime-2026-05-26",
        "label": "topic",
        "properties": {
            "title": "MAGMA LLM backend and embedding model are separate",
            "content": (
                "MAGMA slow-path LLM backend is DeepSeek V3 via OpenRouter for relation extraction, "
                "causal reasoning, and consolidation. MAGMA embedding model is local "
                "BAAI/bge-small-zh-v1.5 with 512-dimensional vectors for semantic retrieval. "
                "MiniLM-L6-v2 / 384d was only an early historical state and must not be reported "
                "as current runtime."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 0.96,
            "ttl_days": 3650,
        },
    },
    {
        "id": "ops:magma:health-signals",
        "label": "topic",
        "properties": {
            "title": "MAGMA health signals and doctor yellow meanings",
            "content": (
                "MAGMA doctor red/yellow/green: api checks the configured FastAPI base URL; "
                "mcp_proxy must be http_proxy; recall_active checks recent recall_events; "
                "feedback_active checks recall_feedback; embedding_coverage must stay near 100%; "
                "recent_capture yellow means the last automatic capture is older than the 6 hour "
                "warning threshold. recent_capture yellow is a warning that no recent messages were "
                "captured; it does not mean MAGMA query/recall is broken."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 0.92,
            "ttl_days": 365,
        },
    },
    {
        "id": "ops:magma:mcp-proxy-http",
        "label": "topic",
        "properties": {
            "title": "MAGMA MCP proxy through configured main API",
            "content": (
                "MAGMA MCP was changed to a thin http_proxy to the configured FastAPI main API. "
                "The current production base URL is http://127.0.0.1:8904/api/v1. "
                "The old MCP path directly loaded MemorySearcher, SQLite, and embeddings in stdio, "
                "which could cold-load the model, bypass FastAPI governance, and cause timeouts. "
                "mcp_server.py now proxies magma_query, magma_add_node, magma_add_edge, "
                "magma_list_nodes, and magma_get_node to the FastAPI main chain."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 0.92,
            "ttl_days": 365,
        },
    },
    {
        "id": "ops:openclaw:version-pin-2026-5-20",
        "label": "topic",
        "properties": {
            "title": "OpenClaw version and upgrade policy",
            "content": (
                "OpenClaw version questions are historical + realtime tasks. Always check MAGMA "
                "first for local upgrade history, known bad versions, current pin/decision, and "
                "Gateway restart lessons; then verify npm/GitHub for the latest public version. "
                "Local history: 5.22 had serious runtime/export bugs in this setup; 5.28 was the "
                "stable line after compatibility checks; 6.1 upgrade attempts hit issues and must "
                "be treated cautiously. Upgrade SOP: stop Gateway before npm install, upgrade "
                "plugins only after main package, restart Gateway, then verify Feishu/Codex/MAGMA. "
                "Do not answer OpenClaw latest/version/upgrade questions from npm alone."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 1.0,
            "ttl_days": 3650,
        },
    },
    {
        "id": "ops:magma:p0-ops-suite",
        "label": "topic",
        "properties": {
            "title": "MAGMA P0 operations suite",
            "content": (
                "MAGMA P0 operations suite: source_agent_id formalized in nodes, recall_events, "
                "and recall_feedback; magma_doctor.py outputs red/yellow/green health with --json "
                "and --agent; magma_ops.py provides status and repair; RUNBOOK.md documents SOP, "
                "safe repair boundaries, version pin checks, and source agent attribution."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 0.92,
            "ttl_days": 365,
        },
    },
    {
        "id": "ops:magma:yunying-source-agent",
        "label": "topic",
        "properties": {
            "title": "yunying MAGMA injection interpretation",
            "content": (
                "Earlier yunying MAGMA injection looked like 0 because checking only hook execution "
                "agent_id was incomplete. In subagent flow the hook may execute on the parent side, "
                "while source_agent_id and source_agents show the memory origin. Doctor should inspect "
                "both agent_injection and source_agents to judge yunying recall/capture."
            ),
            "source": "magma_operational_anchor",
            "source_agent_id": "main",
            "department": "老板",
            "memory_scope": "system",
            "layer": "ops_anchor",
            "importance": 0.9,
            "ttl_days": 365,
        },
    },
]


def add_node(anchor: dict) -> None:
    data = json.dumps(anchor, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/api/v1/nodes",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        res.read()


def main() -> int:
    for anchor in ANCHORS:
        add_node(anchor)
        print(f"seeded {anchor['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
