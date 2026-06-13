"""MAGMA MCP Server - Thin proxy to 8902 HTTP API.

All operations route through http://127.0.0.1:8902/api/v1/...
This avoids cold-starting the embedding model in the MCP stdio process
and ensures all queries pass through the FastAPI governance logic.
"""

import json
import os
import sys
import logging
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("magma.mcp")

API_BASE = os.environ.get("MAGMA_API_BASE", "http://127.0.0.1:8902").rstrip("/")
API_TIMEOUT = int(os.environ.get("MAGMA_API_TIMEOUT", "30"))


def _api_request(method: str, path: str, body: dict = None) -> dict:
    """Make an HTTP request to the MAGMA API."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"API {method} {path} returned {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"API {method} {path} unreachable: {e.reason}")


app = Server("magma-memory")


@app.list_tools()
async def list_tools() -> List[Tool]:
    # Core tools (P0): query, add_node, add_edge, list_nodes, get_node,
    #                   feedback, delete_node, update_node, consolidate,
    #                   doctor, stats, search_by_entity
    # Extension tools (P0-3): memory_edit, memory_forget, core_memory_get/set, timeline, distill_l1
    # Extension tools (P1): entity_add, entity_remove, entity_list, explain/mark tools
    return [
        Tool(
            name="magma_query",
            description="Search the MAGMA knowledge graph with a natural language query. Returns matching nodes with relevance scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "description": "Max results to return (default 5)", "default": 5},
                    "filters": {
                        "type": "object",
                        "description": "Optional filters such as agent_id, source, session_key, label, or current_agent_id.",
                        "additionalProperties": True,
                    },
                    "priority": {"type": "string", "enum": ["critical", "normal", "low"], "description": "P2-6: Query priority level. critical=hide token_usage (P0 decisions), normal=default, low=warn at 60% budget", "default": "normal"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="magma_add_node",
            description="Add a node (entity/concept) to the MAGMA knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node ID"},
                    "label": {"type": "string", "description": "Node label/type (e.g. person, project, event)"},
                    "properties": {"type": "object", "description": "Key-value properties for the node", "additionalProperties": True},
                },
                "required": ["id", "label"],
            },
        ),
        Tool(
            name="magma_add_edge",
            description="Add a directed edge (relationship) between two nodes in the knowledge graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "description": "Source node ID"},
                    "target_id": {"type": "string", "description": "Target node ID"},
                    "relation": {"type": "string", "description": "Relationship type (e.g. depends_on, created_by)"},
                    "properties": {"type": "object", "description": "Optional key-value properties", "additionalProperties": True},
                },
                "required": ["source_id", "target_id", "relation"],
            },
        ),
        Tool(
            name="magma_list_nodes",
            description="List nodes in the knowledge graph, optionally filtered by label.",
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Filter by node label (optional)"},
                    "limit": {"type": "integer", "description": "Max nodes to return (default 100)", "default": 100},
                    "importance_label": {"type": "string", "enum": ["critical", "useful", "reference", "noise"], "description": "Filter by importance label (optional)"},
                },
            },
        ),
        Tool(
            name="magma_get_node",
            description="Get a specific node by ID, including its connected edges.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID to look up"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="magma_feedback",
            description="Report which recalled memories were useful after a query. Improves future recall ranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The recall event ID"},
                    "recalled": {"type": "array", "items": {"type": "string"}, "description": "List of recalled node IDs"},
                    "used": {"type": "array", "items": {"type": "string"}, "description": "List of node IDs that were actually useful"},
                    "missed_node_ids": {"type": "array", "items": {"type": "string"}, "description": "List of recalled but unused node IDs that should have been used (protection signal)"},
                    "query": {"type": "string", "description": "Original query (optional)"},
                },
                "required": ["event_id", "recalled", "used"],
            },
        ),
        Tool(
            name="magma_explain_recall",
            description="Explain why a memory was recalled, including score signals, provenance, and related context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "Recall event ID to explain (optional)"},
                    "node_id": {"type": "string", "description": "Specific recalled node ID to explain (optional)"},
                    "query": {"type": "string", "description": "Original query, useful when event_id is unavailable (optional)"},
                },
            },
        ),
        Tool(
            name="magma_mark_wrong",
            description="Mark a recalled memory as wrong or misleading. The memory is suppressed and down-weighted, not hard-deleted.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Memory node ID to mark wrong"},
                    "reason": {"type": "string", "description": "Why this memory is wrong or misleading"},
                    "agent_id": {"type": "string", "description": "Agent applying the correction (optional)"},
                    "session_key": {"type": "string", "description": "Session where the correction happened (optional)"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="magma_mark_important",
            description="Mark a memory as important so future recall ranks it higher.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Memory node ID to promote"},
                    "reason": {"type": "string", "description": "Why this memory is important"},
                    "agent_id": {"type": "string", "description": "Agent applying the mark (optional)"},
                    "session_key": {"type": "string", "description": "Session where the mark happened (optional)"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="magma_suppress_pattern",
            description="Add a governance suppression pattern for recurring noise such as diagnostics, rate-limit fragments, or temporary tests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Literal or regex-like pattern to suppress"},
                    "reason": {"type": "string", "description": "Why this pattern should be suppressed"},
                    "agent_id": {"type": "string", "description": "Agent adding the pattern (optional)"},
                    "ttl_days": {"type": "integer", "description": "Optional time-to-live in days"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="magma_delete_node",
            description="Soft-delete a node by marking it as deleted (status=deleted). Does not remove from disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID to delete"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="magma_update_node",
            description="Partially update a node's properties. Only provided keys are overwritten.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID to update"},
                    "properties": {"type": "object", "description": "Properties to merge/update", "additionalProperties": True},
                    "importance_label": {"type": "string", "enum": ["critical", "useful", "reference", "noise"], "description": "Semantic importance label (optional). Maps to importance value automatically."},
                },
                "required": ["node_id", "properties"],
            },
        ),
        Tool(
            name="magma_consolidate",
            description="Trigger manual cleanup: remove duplicate edges, orphan edges, expire stale nodes.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="magma_distill_l1",
            description="Distill recent L0 auto-captured memories into stable L1 memories (decision, preference, project_state, pending_action, lesson, fact).",
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Lookback window in hours (default 24)", "default": 24},
                    "limit": {"type": "integer", "description": "Max recent L0 nodes to scan (default 200)", "default": 200},
                    "dry_run": {"type": "boolean", "description": "Preview only without writing L1 nodes", "default": False},
                    "source_agent_id": {"type": "string", "description": "Optional source agent filter for scoped distillation"},
                },
            },
        ),
        Tool(
            name="magma_doctor",
            description="Enhanced health check: node/edge counts, FAISS status, recent capture stats.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="magma_stats",
            description="Return knowledge graph statistics: node count, edge count, 24h capture volume, layer distribution.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="magma_search_by_entity",
            description="Find nodes by exact entity name (no semantic search). Useful for precise lookups.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to search for"},
                    "entity_type": {"type": "string", "description": "Optional entity type filter"},
                },
                "required": ["entity_name"],
            },
        ),
        # --- P0-3: Agent self-editing memory tools ---
        Tool(
            name="magma_memory_edit",
            description="Edit an existing memory node's content. Agent can use this to correct or update facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the memory node to edit"},
                    "new_content": {"type": "string", "description": "New content to replace the existing content"},
                    "reason": {"type": "string", "description": "Why this memory is being edited (optional, for audit trail)"},
                },
                "required": ["node_id", "new_content"],
            },
        ),
        Tool(
            name="magma_memory_forget",
            description="Mark a memory as no longer applicable (soft invalidation). Does NOT delete — just sets valid_until=now.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "ID of the memory to forget"},
                    "reason": {"type": "string", "description": "Why this memory is being forgotten (optional)"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="magma_core_memory_get",
            description="Get first-class core memory blocks (persona, preferences, project state, current tasks, operational rules).",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Agent ID to get core memory for (optional, defaults to caller)"},
                    "block_name": {"type": "string", "description": "Optional block name filter"},
                },
            },
        ),
        Tool(
            name="magma_core_memory_set",
            description="Set/update a core memory block. These are the agent's persistent working memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "block_name": {"type": "string", "description": "Name of the memory block (e.g. 'persona', 'human', 'project_status')"},
                    "content": {"type": "string", "description": "Content to store in this block"},
                    "agent_id": {"type": "string", "description": "Agent ID (optional, defaults to caller)"},
                },
                "required": ["block_name", "content"],
            },
        ),
        Tool(
            name="magma_timeline",
            description="Get a time-ordered history of facts about an entity or topic. Shows how knowledge evolved over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to get timeline for"},
                    "limit": {"type": "integer", "description": "Max entries to return (default 10)", "default": 10},
                },
                "required": ["entity_name"],
            },
        ),
        # --- P1-1: Entity management tools ---
        Tool(
            name="magma_entity_add",
            description="Add a custom entity to the entity dictionary. Enables automatic extraction from future captures.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name (e.g. '抖音', '张三', 'iPhone 16')"},
                    "entity_type": {"type": "string", "description": "Type: person/org/product/location/event/tool/platform/custom/selling_point/document/domain"},
                },
                "required": ["name", "entity_type"],
            },
        ),
        Tool(
            name="magma_entity_remove",
            description="Remove a custom entity from the entity dictionary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Entity name to remove"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="magma_entity_list",
            description="List all known entities (built-in + custom) with their types and sources.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="magma_decision_drift",
            description="List recent decision-particle events and summarize whether user choices drifted over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Optional agent/source filter"},
                    "session_key": {"type": "string", "description": "Optional session filter"},
                    "decision_key": {"type": "string", "description": "Optional decision key filter"},
                    "limit": {"type": "integer", "description": "Max events to inspect (default 50)", "default": 50},
                },
            },
        ),
        # --- P1-3: Verify tool ---
        Tool(
            name="magma_verify",
            description="Verify a claim against specified memory nodes using LLM. Returns verdict, evidence, and confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_ids": {"type": "array", "items": {"type": "string"}, "description": "List of node IDs to verify against"},
                    "claim": {"type": "string", "description": "The claim to verify"},
                },
                "required": ["node_ids", "claim"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        if name == "magma_query":
            return await _handle_query(arguments)
        elif name == "magma_add_node":
            return await _handle_add_node(arguments)
        elif name == "magma_add_edge":
            return await _handle_add_edge(arguments)
        elif name == "magma_list_nodes":
            return await _handle_list_nodes(arguments)
        elif name == "magma_get_node":
            return await _handle_get_node(arguments)
        elif name == "magma_feedback":
            return await _handle_feedback(arguments)
        elif name == "magma_explain_recall":
            return await _handle_explain_recall(arguments)
        elif name == "magma_mark_wrong":
            return await _handle_mark_wrong(arguments)
        elif name == "magma_mark_important":
            return await _handle_mark_important(arguments)
        elif name == "magma_suppress_pattern":
            return await _handle_suppress_pattern(arguments)
        elif name == "magma_delete_node":
            return await _handle_delete_node(arguments)
        elif name == "magma_update_node":
            return await _handle_update_node(arguments)
        elif name == "magma_consolidate":
            return await _handle_consolidate(arguments)
        elif name == "magma_distill_l1":
            return await _handle_distill_l1(arguments)
        elif name == "magma_doctor":
            return await _handle_doctor(arguments)
        elif name == "magma_stats":
            return await _handle_stats(arguments)
        elif name == "magma_search_by_entity":
            return await _handle_search_by_entity(arguments)
        elif name == "magma_memory_edit":
            return await _handle_memory_edit(arguments)
        elif name == "magma_memory_forget":
            return await _handle_memory_forget(arguments)
        elif name == "magma_core_memory_get":
            return await _handle_core_memory_get(arguments)
        elif name == "magma_core_memory_set":
            return await _handle_core_memory_set(arguments)
        elif name == "magma_timeline":
            return await _handle_timeline(arguments)
        elif name == "magma_entity_add":
            return await _handle_entity_add(arguments)
        elif name == "magma_entity_remove":
            return await _handle_entity_remove(arguments)
        elif name == "magma_entity_list":
            return await _handle_entity_list(arguments)
        elif name == "magma_decision_drift":
            return await _handle_decision_drift(arguments)
        elif name == "magma_verify":
            return await _handle_verify(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {e}")]


async def _handle_query(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/query"""
    result = _api_request("POST", "/api/v1/query", {
        "query": args["query"],
        "top_k": args.get("top_k", 5),
        "filters": args.get("filters"),
        "priority": args.get("priority", "normal"),
    })
    results = result.get("results", result)
    if not results:
        results = [{"info": f"No results for '{args['query']}'."}]
    # P2: Include narrative and references in MCP output
    output = {
        "results": results,
        "event_id": result.get("event_id"),
        "narrative": result.get("narrative"),
        "references": result.get("references"),
        "token_budget": result.get("token_budget"),
        "tokens_used": result.get("tokens_used"),
        "token_usage_ratio": result.get("token_usage_ratio"),
        "budget_warning": result.get("budget_warning"),  # P2-6
        "backtrack_warning": result.get("backtrack_warning"),  # P2-5
        "backtrack_narrative": result.get("backtrack_narrative"),  # P2-5
        "count": result.get("count", len(results)),
        "intent": result.get("intent"),
        "active_context": result.get("active_context"),
        "bridge_entities": result.get("bridge_entities"),
    }
    return [TextContent(type="text", text=json.dumps(output, ensure_ascii=False, indent=2))]


async def _handle_add_node(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/nodes"""
    _api_request("POST", "/api/v1/nodes", {
        "id": args["id"],
        "label": args["label"],
        "properties": args.get("properties", {}),
    })
    return [TextContent(type="text", text=f"Node '{args['id']}' added successfully.")]


async def _handle_add_edge(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/edges"""
    _api_request("POST", "/api/v1/edges", {
        "source_id": args["source_id"],
        "target_id": args["target_id"],
        "relation": args["relation"],
        "properties": args.get("properties"),
    })
    return [TextContent(type="text", text=f"Edge added: {args['source_id']} --[{args['relation']}]--> {args['target_id']}")]


async def _handle_list_nodes(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to GET /api/v1/nodes"""
    params = []
    if args.get("label"):
        params.append(("label", args["label"]))
    limit = args.get("limit", 100)
    params.append(("limit", str(limit)))
    # P0-2: Pass importance_label filter if provided
    if args.get("importance_label"):
        params.append(("importance_label", args["importance_label"]))
    query_str = urllib.parse.urlencode(params)
    path = f"/api/v1/nodes?{query_str}" if query_str else "/api/v1/nodes"
    result = _api_request("GET", path)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_get_node(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to GET /api/v1/nodes/{node_id}"""
    try:
        node_id = urllib.parse.quote(args["node_id"], safe="")
        result = _api_request("GET", f"/api/v1/nodes/{node_id}")
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Node '{args['node_id']}' not found.")]
        raise


async def _handle_feedback(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/feedback"""
    result = _api_request("POST", "/api/v1/feedback", {
        "event_id": args["event_id"],
        "recalled": args.get("recalled", []),
        "used": args.get("used", []),
        "missed_node_ids": args.get("missed_node_ids", []),
        "query": args.get("query", ""),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_explain_recall(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/recall/explain."""
    result = _api_request("POST", "/api/v1/recall/explain", {
        "event_id": args.get("event_id"),
        "node_id": args.get("node_id"),
        "query": args.get("query"),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_mark_wrong(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/memory/mark_wrong."""
    result = _api_request("POST", "/api/v1/memory/mark_wrong", {
        "node_id": args["node_id"],
        "reason": args.get("reason", ""),
        "agent_id": args.get("agent_id"),
        "session_key": args.get("session_key"),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_mark_important(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/memory/mark_important."""
    result = _api_request("POST", "/api/v1/memory/mark_important", {
        "node_id": args["node_id"],
        "reason": args.get("reason", ""),
        "agent_id": args.get("agent_id"),
        "session_key": args.get("session_key"),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_suppress_pattern(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/memory/suppress_pattern."""
    result = _api_request("POST", "/api/v1/memory/suppress_pattern", {
        "pattern": args["pattern"],
        "reason": args.get("reason", ""),
        "agent_id": args.get("agent_id"),
        "ttl_days": args.get("ttl_days"),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_delete_node(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to DELETE /api/v1/nodes/{node_id}"""
    node_id = urllib.parse.quote(args["node_id"], safe="")
    try:
        result = _api_request("DELETE", f"/api/v1/nodes/{node_id}")
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Node '{args['node_id']}' not found.")]
        raise


async def _handle_update_node(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to PATCH /api/v1/nodes/{node_id}"""
    node_id = urllib.parse.quote(args["node_id"], safe="")
    try:
        body = dict(args.get("properties", {}))
        # P0-2: Pass importance_label if provided at top level
        if "importance_label" in args:
            body["importance_label"] = args["importance_label"]
        result = _api_request("PATCH", f"/api/v1/nodes/{node_id}", body)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Node '{args['node_id']}' not found.")]
        raise


async def _handle_consolidate(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/consolidate"""
    result = _api_request("POST", "/api/v1/consolidate")
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_distill_l1(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/distill_l1."""
    result = _api_request("POST", "/api/v1/distill_l1", {
        "hours": args.get("hours", 24),
        "limit": args.get("limit", 200),
        "dry_run": bool(args.get("dry_run", False)),
        "source_agent_id": args.get("source_agent_id"),
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_doctor(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to GET /api/v1/doctor"""
    result = _api_request("GET", "/api/v1/doctor")
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_stats(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to GET /api/v1/stats"""
    result = _api_request("GET", "/api/v1/stats")
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_search_by_entity(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/search_by_entity (precise entity lookup)."""
    result = _api_request("POST", "/api/v1/search_by_entity", {
        "entity_name": args["entity_name"],
        "entity_type": args.get("entity_type"),
    })
    results = result.get("results", result)
    if not results:
        results = [{"info": f"No entity found for '{args['entity_name']}'."}]
    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]

async def _handle_memory_edit(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to PATCH /api/v1/nodes/{node_id} with content update + re-embedding."""
    node_id = urllib.parse.quote(args["node_id"], safe="")
    new_content = args["new_content"]
    reason = args.get("reason", "")
    # Fetch existing node first
    try:
        existing = _api_request("GET", f"/api/v1/nodes/{node_id}")
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Node '{args['node_id']}' not found.")]
        raise
    # Update with new content
    update_props = {"content": new_content}
    if reason:
        update_props["edit_reason"] = reason
    result = _api_request("PATCH", f"/api/v1/nodes/{node_id}", update_props)
    return [TextContent(type="text", text=f"Memory '{args['node_id']}' updated. {reason if reason else ''}")]


async def _handle_memory_forget(args: Dict[str, Any]) -> List[TextContent]:
    """Mark a memory as no longer valid (set valid_until = now)."""
    node_id = urllib.parse.quote(args["node_id"], safe="")
    from datetime import datetime
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    reason = args.get("reason", "")
    update_props = {"valid_until": now_str}
    if reason:
        update_props["forget_reason"] = reason
    try:
        result = _api_request("PATCH", f"/api/v1/nodes/{node_id}", update_props)
        return [TextContent(type="text", text=f"Memory '{args['node_id']}' marked as forgotten (valid_until={now_str}). {reason if reason else ''}")]
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Node '{args['node_id']}' not found.")]
        raise


async def _handle_core_memory_get(args: Dict[str, Any]) -> List[TextContent]:
    """Get core memory blocks for an agent."""
    params = []
    if args.get("agent_id"):
        params.append(("agent_id", args["agent_id"]))
    if args.get("block_name"):
        params.append(("block_name", args["block_name"]))
    query_str = urllib.parse.urlencode(params)
    path = f"/api/v1/core_memory?{query_str}" if query_str else "/api/v1/core_memory"
    result = _api_request("GET", path)
    if not result.get("blocks"):
        result["blocks"] = [{"info": "No core memory blocks found. Use magma_core_memory_set to create one."}]
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_core_memory_set(args: Dict[str, Any]) -> List[TextContent]:
    """Set/update a core memory block."""
    result = _api_request("PUT", "/api/v1/core_memory", {
        "block_name": args["block_name"],
        "content": args["content"],
        "agent_id": args.get("agent_id"),
        "source": "agent_self_edit",
        "importance": 0.95,
    })
    block = result.get("block") or {}
    return [TextContent(
        type="text",
        text=f"Core memory block '{args['block_name']}' saved as {block.get('id', 'unknown')}."
    )]


async def _handle_timeline(args: Dict[str, Any]) -> List[TextContent]:
    """Get time-ordered facts about an entity."""
    entity_name = args["entity_name"]
    limit = args.get("limit", 10)
    # Query for facts mentioning this entity
    result = _api_request("POST", "/api/v1/query", {
        "query": entity_name,
        "top_k": limit,
        "filters": {"label": "fact"},
    })
    results = result.get("results", result)
    if not isinstance(results, list):
        results = [results]
    # Sort by created_at
    results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    # Add temporal status
    for r in results:
        props = r.get("properties", {}) or {}
        valid_until = props.get("valid_until") or r.get("valid_until")
        if valid_until:
            r["temporal_status"] = "historical"
        else:
            r["temporal_status"] = "current"
    if not results:
        results = [{"info": f"No facts found for '{entity_name}'."}]
    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]


async def _handle_entity_add(args: Dict[str, Any]) -> List[TextContent]:
    """Add a custom entity to the entity dictionary."""
    try:
        result = _api_request("POST", "/api/v1/entities", {
            "name": args["name"],
            "entity_type": args["entity_type"],
        })
        status = result.get("status", "ok")
        name = result.get("name", args["name"])
        entity_type = result.get("entity_type", args["entity_type"])
        if status == "already_exists":
            return [TextContent(type="text", text=f"Entity '{name}' already exists in custom dictionary.")]
        return [TextContent(type="text", text=f"Entity '{name}' ({entity_type}) added to custom dictionary.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error adding entity: {e}")]


async def _handle_entity_remove(args: Dict[str, Any]) -> List[TextContent]:
    """Remove a custom entity from the entity dictionary."""
    try:
        name_encoded = urllib.parse.quote(args["name"], safe="")
        result = _api_request("DELETE", f"/api/v1/entities/{name_encoded}")
        return [TextContent(type="text", text=f"Entity '{args['name']}' removed from custom dictionary.")]
    except RuntimeError as e:
        if "404" in str(e):
            return [TextContent(type="text", text=f"Entity '{args['name']}' not found in custom dictionary.")]
        return [TextContent(type="text", text=f"Error removing entity: {e}")]


async def _handle_entity_list(args: Dict[str, Any]) -> List[TextContent]:
    """List all known entities."""
    try:
        result = _api_request("GET", "/api/v1/entities")
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing entities: {e}")]


async def _handle_decision_drift(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to GET /api/v1/decisions/drift"""
    params = {
        "agent_id": args.get("agent_id"),
        "session_key": args.get("session_key"),
        "decision_key": args.get("decision_key"),
        "limit": args.get("limit", 50),
    }
    params = {k: v for k, v in params.items() if v is not None}
    query_str = urllib.parse.urlencode(params)
    path = f"/api/v1/decisions/drift?{query_str}" if query_str else "/api/v1/decisions/drift"
    result = _api_request("GET", path)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_verify(args: Dict[str, Any]) -> List[TextContent]:
    """Proxy to POST /api/v1/verify"""
    result = _api_request("POST", "/api/v1/verify", {
        "node_ids": args["node_ids"],
        "claim": args["claim"],
    })
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def main():
    logger.info(f"MAGMA MCP Server starting (proxy mode -> {API_BASE})")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
