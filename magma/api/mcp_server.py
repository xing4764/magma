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
    })
    results = result.get("results", result)
    if not results:
        results = [{"info": f"No results for '{args['query']}'."}]
    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]


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


async def main():
    logger.info(f"MAGMA MCP Server starting (proxy mode -> {API_BASE})")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
