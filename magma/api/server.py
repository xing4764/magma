"""MAGMA API Server - FastAPI application."""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

logger = logging.getLogger("magma.api")


DEPT_MAP = {
    "yunying": "运营部",
    "jishu": "技术部",
    "zhuli": "助理",
    "main": "老板",
}


def _parse_source_agent(session_key: str) -> str:
    """Extract agentId from session_key like 'agent:yunying:feishu:...'"""
    if not session_key:
        return ""
    import re as _re
    m = _re.search(r"agent:([^:\s]+)", session_key)
    return m.group(1) if m else ""


def _node_text(label: str, properties: Optional[dict]) -> str:
    text_parts = [label]
    for value in (properties or {}).values():
        if isinstance(value, str):
            text_parts.append(value)
    return " ".join(text_parts)


def _memory_metadata(text: str) -> dict:
    from magma.entities import classify_memory_scope, extract_entities, version_key_for_entities

    entities = extract_entities(text)
    return {
        "entities": entities,
        "memory_scope": classify_memory_scope(entities),
        "version_key": version_key_for_entities(entities),
    }


def _attach_entity_anchors(store, encoder, source_node_id: str, text: str, source: str):
    from magma.entities import classify_memory_scope, extract_entities, version_key_for_entities

    for entity in extract_entities(text):
        properties = {
            "name": entity["name"],
            "entity_type": entity["entity_type"],
            "memory_scope": classify_memory_scope([entity]),
            "version_key": version_key_for_entities([entity]),
            "layer": "entity_anchor",
            "source": "entity_extractor",
            "importance": 0.62,
        }
        embedding = encoder.encode(entity["name"]).astype("float32")
        store.add_node(entity["id"], "entity", properties, embedding)
        store.add_edge_once(source_node_id, entity["id"], "mentions_entity", {
            "source": source,
            "entity_name": entity["name"],
            "entity_type": entity["entity_type"],
        })


def _capture_importance(role: str, text: str) -> float:
    if role == "assistant":
        return 0.42
    question_terms = (
        "?", "\uff1f", "\u4e3a\u4ec0\u4e48", "\u600e\u4e48", "\u5982\u4f55",
        "\u54ea\u4e2a", "\u4ec0\u4e48", "\u662f\u5426", "\u80fd\u4e0d\u80fd",
        "\u53ef\u4ee5\u95ee", "\u6d4b\u8bd5",
    )
    if any(term in (text or "") for term in question_terms):
        return 0.26
    return 0.36


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[dict] = None


class NodeRequest(BaseModel):
    id: str
    label: str
    properties: Optional[dict] = None


class CaptureRequest(BaseModel):
    user_text: str = ""
    assistant_text: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    session_id: Optional[str] = None
    source: str = "openclaw_auto_capture"
    ttl_days: int = 180


class EdgeRequest(BaseModel):
    source_id: str
    target_id: str
    relation: str
    properties: Optional[dict] = None


class QueryResponse(BaseModel):
    query: str
    results: list
    count: int


class FeedbackRequest(BaseModel):
    event_id: str
    query: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    recalled: list = []
    used: list = []
    positive_delta: float = 0.025
    unused_delta: float = -0.004


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="MAGMA API",
        description="Memory-Augmented Graph & Multi-modal Agent API",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup():
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = get_store()
        store.initialize()
        app.state.store = store
        logger.info("Graph store initialized")

        try:
            encoder = Encoder()
            app.state.encoder = encoder
            dim = encoder.dimension
            logger.info(f"Encoder ready, dim={dim}, model={encoder.model_name}")
        except Exception as e:
            app.state.encoder = None
            logger.warning(f"Encoder pre-warm failed: {e}")

        interval = int(os.environ.get("MAGMA_CONSOLIDATE_INTERVAL_SECONDS", "3600"))
        if interval > 0:
            app.state.consolidation_task = asyncio.create_task(_run_consolidation_loop(interval))
            logger.info(f"Consolidation loop scheduled every {interval}s")

        backup_interval = int(os.environ.get("MAGMA_BACKUP_INTERVAL_SECONDS", "86400"))
        if backup_interval > 0:
            app.state.backup_task = asyncio.create_task(_run_backup_loop(backup_interval))
            logger.info(f"Backup loop scheduled every {backup_interval}s")

    @app.on_event("shutdown")
    async def shutdown():
        task = getattr(app.state, "consolidation_task", None)
        if task:
            task.cancel()
        backup_task = getattr(app.state, "backup_task", None)
        if backup_task:
            backup_task.cancel()

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "magma", "version": "0.1.0"}

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        from magma.graph.sqlite_store import get_store
        from magma.search import MemorySearcher
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        results = MemorySearcher(store, encoder).query(req.query, req.top_k, req.filters)
        if not results:
            results = [{
                "id": "system",
                "label": "info",
                "properties": {"message": f"No memory matched '{req.query}'."},
                "score": 0.0,
                "source": "system",
            }]
        return QueryResponse(query=req.query, results=results, count=len(results))

    @app.post("/api/v1/nodes")
    async def add_node(req: NodeRequest):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        properties = dict(req.properties or {})
        text_for_embedding = _node_text(req.label, properties)
        properties.update(_memory_metadata(text_for_embedding))
        embedding = encoder.encode(text_for_embedding).astype("float32")

        store.add_node(req.id, req.label, properties, embedding)
        _attach_entity_anchors(store, encoder, req.id, text_for_embedding, properties.get("source", "api"))
        return {"status": "ok", "id": req.id}

    @app.post("/api/v1/capture")
    async def capture(req: CaptureRequest):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        written = []

        for role, text in (("user", req.user_text), ("assistant", req.assistant_text)):
            cleaned = (text or "").strip()
            if len(cleaned) < 2:
                continue
            digest = hashlib.sha1(
                "\n".join([
                    req.source,
                    req.agent_id or "",
                    req.session_key or "",
                    role,
                    cleaned,
                ]).encode("utf-8")
            ).hexdigest()[:16]
            node_id = f"evt:auto:{digest}"
            properties = {
                "layer": "L0",
                "source": req.source,
                "role": role,
                "content": cleaned,
                "agent_id": req.agent_id,
                "session_key": req.session_key,
                "session_id": req.session_id,
                "ttl_days": req.ttl_days,
                "importance": _capture_importance(role, cleaned),
                "source_agent_id": _parse_source_agent(req.session_key) or req.agent_id,
                "department": DEPT_MAP.get(_parse_source_agent(req.session_key) or req.agent_id or "", ""),
            }
            properties.update(_memory_metadata(cleaned))
            text_for_embedding = f"{role}: {cleaned}"
            embedding = encoder.encode(text_for_embedding).astype("float32")
            store.add_node(node_id, "event", properties, embedding)
            _attach_entity_anchors(store, encoder, node_id, cleaned, req.source)
            written.append(node_id)

        if len(written) == 2:
            store.add_edge(written[0], written[1], "responded_by", {
                "source": req.source,
                "session_key": req.session_key,
            })

        return {"status": "ok", "written": written, "count": len(written)}

    @app.post("/api/v1/edges")
    async def add_edge(req: EdgeRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        store.add_edge(req.source_id, req.target_id, req.relation, req.properties)
        return {"status": "ok"}

    @app.get("/api/v1/nodes")
    async def list_nodes(label: Optional[str] = None, limit: int = 100):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        nodes = store.query_nodes(label=label, limit=limit)
        return {"nodes": nodes, "count": len(nodes)}

    @app.get("/api/v1/nodes/{node_id}")
    async def get_node(node_id: str):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        node = store.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        edges = store.get_edges(node_id)
        return {"node": node, "edges": edges}

    @app.post("/api/v1/consolidate")
    async def consolidate():
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        return {"status": "ok", "stats": store.consolidate()}

    @app.post("/api/v1/backup")
    async def backup():
        from magma.backup import create_backup

        keep_days = int(os.environ.get("MAGMA_BACKUP_KEEP_DAYS", "14"))
        keep_latest = int(os.environ.get("MAGMA_BACKUP_KEEP_LATEST", "7"))
        backup_dir = os.environ.get("MAGMA_BACKUP_DIR")
        return {"status": "ok", "backup": create_backup(
            backup_dir=backup_dir,
            keep_days=keep_days,
            keep_latest=keep_latest,
        )}

    @app.post("/api/v1/feedback")
    async def feedback(req: FeedbackRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        source_agent = _parse_source_agent(req.session_key) or req.agent_id or ""
        dept = DEPT_MAP.get(source_agent, "")
        store.record_recall_event(
            event_id=req.event_id,
            query=req.query,
            agent_id=req.agent_id,
            session_key=req.session_key,
            results=req.recalled,
            source_agent_id=source_agent,
            department=dept,
        )
        recalled_ids = [
            item.get("id") if isinstance(item, dict) else item
            for item in (req.recalled or [])
        ]
        used_ids = [
            item.get("id") if isinstance(item, dict) else item
            for item in (req.used or [])
        ]
        stats = store.apply_recall_feedback(
            event_id=req.event_id,
            recalled_node_ids=[node_id for node_id in recalled_ids if node_id],
            used_node_ids=[node_id for node_id in used_ids if node_id],
            positive_delta=req.positive_delta,
            unused_delta=req.unused_delta,
            source_agent_id=source_agent,
            department=dept,
        )
        return {"status": "ok", "feedback": stats}

    return app


async def _run_consolidation_loop(interval: int):
    from magma.graph.sqlite_store import get_store

    while True:
        await asyncio.sleep(interval)
        try:
            stats = get_store().consolidate()
            logger.info(f"Consolidation complete: {stats}")
        except Exception as e:
            logger.warning(f"Consolidation failed: {e}")


async def _run_backup_loop(interval: int):
    from magma.backup import create_backup

    keep_days = int(os.environ.get("MAGMA_BACKUP_KEEP_DAYS", "14"))
    keep_latest = int(os.environ.get("MAGMA_BACKUP_KEEP_LATEST", "7"))
    backup_dir = os.environ.get("MAGMA_BACKUP_DIR")

    await asyncio.sleep(min(interval, 300))
    while True:
        try:
            stats = create_backup(
                backup_dir=backup_dir,
                keep_days=keep_days,
                keep_latest=keep_latest,
            )
            logger.info(f"Backup complete: {stats}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    host = os.environ.get("MAGMA_API_HOST", "0.0.0.0")
    port = int(os.environ.get("MAGMA_API_PORT", "8901"))
    uvicorn.run(app, host=host, port=port)
