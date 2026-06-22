"""MAGMA API Server - FastAPI application."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

logger = logging.getLogger("magma.api")

# --- P0-2: Importance label helpers ---
IMPORTANCE_LABEL_MAP = {
    "critical": 0.95,
    "useful": 0.75,
    "reference": 0.45,
    "noise": 0.15,
}


# Lazy-import fact extractor to avoid cold-start on import
_fact_extractor = None


def _get_fact_extractor():
    global _fact_extractor
    if _fact_extractor is None:
        try:
            from magma.fact_extractor import extract_facts, is_fact_extraction_available
            _fact_extractor = (extract_facts, is_fact_extraction_available)
        except Exception:
            _fact_extractor = (lambda t, r=None, **kw: [], lambda: False)
    return _fact_extractor


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


def _decision_event_payload(req, event: dict) -> tuple[str, str, dict]:
    content = event.get("content") or ""
    digest = hashlib.sha1(
        "\n".join([
            req.source,
            req.agent_id or "",
            req.session_key or "",
            event.get("decision_key") or "",
            event.get("selected") or "",
            content,
        ]).encode("utf-8")
    ).hexdigest()[:16]
    node_id = f"decision:{digest}"
    text_for_embedding = (
        f"decision {event.get('decision_key')}: selected {event.get('selected')}; "
        f"options {', '.join(event.get('options') or [])}; {content}"
    )
    properties = {
        "layer": "L2",
        "kind": "decision_event",
        "source": "decision_particles",
        "role": "decision",
        "content": content,
        "decision_key": event.get("decision_key"),
        "options": event.get("options") or [],
        "selected": event.get("selected"),
        "chain": event.get("chain") or [],
        "direction": event.get("direction"),
        "confidence": event.get("confidence"),
        "tags": event.get("tags") or [],
        "agent_id": req.agent_id,
        "session_key": req.session_key,
        "session_id": getattr(req, "session_id", None),
        "ttl_days": max(int(getattr(req, "ttl_days", 180) or 180), 180),
        "importance": 0.68,
        "source_agent_id": _parse_source_agent(req.session_key) or req.agent_id,
        "department": DEPT_MAP.get(_parse_source_agent(req.session_key) or req.agent_id or "", ""),
    }
    return node_id, text_for_embedding, properties


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[dict] = None
    priority: str = "normal"  # P2-6: "critical" | "normal" | "low"


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


class EntitySearchRequest(BaseModel):
    entity_name: str
    entity_type: Optional[str] = None


class CoreMemoryRequest(BaseModel):
    block_name: str
    content: str
    agent_id: Optional[str] = None
    source: str = "agent_self_edit"
    importance: float = 0.95


class QueryResponse(BaseModel):
    query: str
    results: list
    count: int
    event_id: Optional[str] = None
    intent: Optional[dict] = None
    narrative: Optional[str] = None
    references: Optional[list] = None
    token_budget: Optional[int] = None
    tokens_used: Optional[int] = None
    token_usage_ratio: Optional[float] = None
    budget_warning: Optional[str] = None  # P2-6: low-priority budget warning
    backtrack_warning: Optional[bool] = None  # P2-5: backtrack detection
    backtrack_narrative: Optional[str] = None  # P2-5: backtrack suggestion
    short_command_resolution: Optional[dict] = None
    active_context: Optional[list] = None
    bridge_entities: Optional[list] = None


class FeedbackRequest(BaseModel):
    event_id: str
    query: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    recalled: list = []
    used: list = []
    missed_node_ids: list = []
    positive_delta: float = 0.05
    unused_delta: float = -0.01


class DecisionDriftResponse(BaseModel):
    events: list
    summary: dict
    count: int


class ExplainRecallRequest(BaseModel):
    query: Optional[str] = None
    event_id: Optional[str] = None
    node_id: Optional[str] = None


class MarkMemoryRequest(BaseModel):
    node_id: str
    reason: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None


class SuppressPatternRequest(BaseModel):
    pattern: str
    reason: str = ""
    agent_id: Optional[str] = None
    ttl_days: Optional[int] = None


class VerifyRequest(BaseModel):
    node_ids: list
    claim: str


class L1DistillRequest(BaseModel):
    hours: int = 24
    limit: int = 200
    dry_run: bool = False
    source_agent_id: Optional[str] = None


def _verify_with_llm(claim: str, node_contents: list) -> dict:
    """Verify claim against node contents using LLM (DeepSeek via OpenRouter)."""
    import httpx

    # Build evidence text
    evidence_parts = []
    for nc in node_contents:
        evidence_parts.append(f"[{nc['id']}] ({nc['label']}): {nc['content']}")
    evidence_text = "\n".join(evidence_parts)

    prompt = f"""You are a fact-checker. Verify whether the following claim is supported by the provided evidence.

Claim: {claim}

Evidence:
{evidence_text}

Respond in JSON format:
{{
  "verdict": "supported" | "unsupported" | "partial",
  "evidence": "<brief explanation of what supports or contradicts the claim>",
  "confidence": <0.0 to 1.0>
}}"""

    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("MAGMA_VERIFY_MODEL", "deepseek/deepseek-chat")

    if not api_key:
        return {"verdict": "unsupported", "evidence": "No API key configured for verification.", "confidence": 0.0}

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            import re as _re
            json_match = _re.search(r'\{[^}]+\}', content, _re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "verdict": result.get("verdict", "unsupported"),
                    "evidence": result.get("evidence", ""),
                    "confidence": float(result.get("confidence", 0.0)),
                }
            return {"verdict": "unsupported", "evidence": "Could not parse LLM response.", "confidence": 0.0}
    except Exception as e:
        return {"verdict": "unsupported", "evidence": f"LLM call error: {e}", "confidence": 0.0}


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
        from magma.vector.encoder import get_encoder
        from magma.vector.faiss_index import get_faiss_index

        store = get_store()
        store.initialize()
        app.state.store = store
        logger.info("Graph store initialized")

        try:
            encoder = get_encoder()
            app.state.encoder = encoder
            dim = encoder.dimension
            logger.info(f"Encoder ready, dim={dim}, model={encoder.model_name}")
        except Exception as e:
            app.state.encoder = None
            logger.warning(f"Encoder pre-warm failed: {e}")

        # Pre-warm jieba to avoid cold-start on first query
        try:
            import jieba
            jieba_userdict = Path(project_root) / "config" / "jieba_userdict.txt"
            if jieba_userdict.exists():
                jieba.load_userdict(str(jieba_userdict))
                logger.info(f"jieba user dict loaded: {jieba_userdict}")
            jieba.lcut("预热")
            logger.info("jieba pre-warmed")
        except Exception:
            pass

        # Build FAISS index at startup
        try:
            faiss_idx = get_faiss_index(getattr(encoder, 'dimension', 0) if app.state.encoder else 0)
            app.state.faiss_index = faiss_idx
            all_nodes = store.query_nodes_with_embeddings(limit=999999, include_archived=True)
            entries = []
            import numpy as _np
            for node in all_nodes:
                blob = node.get("embedding")
                if blob:
                    vec = _np.frombuffer(blob, dtype=_np.float32)
                    if vec.ndim == 1 and vec.shape[0] == getattr(encoder, 'dimension', 0):
                        entries.append((node["id"], vec))
            faiss_idx.build_from_embeddings(entries)
            logger.info(f"FAISS index built with {len(entries)} vectors, dim={faiss_idx.dimension}")
        except Exception as e:
            app.state.faiss_index = None
            logger.warning(f"FAISS index build failed: {e}")

        interval = int(os.environ.get("MAGMA_CONSOLIDATE_INTERVAL_SECONDS", "3600"))
        if interval > 0:
            app.state.consolidation_task = asyncio.create_task(_run_consolidation_loop(interval))
            logger.info(f"Consolidation loop scheduled every {interval}s")

        backup_interval = int(os.environ.get("MAGMA_BACKUP_INTERVAL_SECONDS", "86400"))
        if backup_interval > 0:
            app.state.backup_task = asyncio.create_task(_run_backup_loop(backup_interval))
            logger.info(f"Backup loop scheduled every {backup_interval}s")

        encoder_check_interval = int(os.environ.get("MAGMA_ENCODER_CHECK_INTERVAL", "300"))
        if encoder_check_interval > 0:
            app.state.encoder_check_task = asyncio.create_task(_run_encoder_idle_check(encoder_check_interval))
            logger.info(f"Encoder idle check scheduled every {encoder_check_interval}s")

    @app.on_event("shutdown")
    async def shutdown():
        task = getattr(app.state, "consolidation_task", None)
        if task:
            task.cancel()
        backup_task = getattr(app.state, "backup_task", None)
        if backup_task:
            backup_task.cancel()
        encoder_task = getattr(app.state, "encoder_check_task", None)
        if encoder_task:
            encoder_task.cancel()

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "magma", "version": "0.1.0"}

    @app.get("/api/v1/doctor")
    async def doctor():
        from magma.graph.sqlite_store import get_store
        store = getattr(app.state, "store", None) or get_store()
        return store.get_doctor()

    @app.get("/api/v1/stats")
    async def stats():
        from magma.graph.sqlite_store import get_store
        store = getattr(app.state, "store", None) or get_store()
        return store.get_stats()

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        from magma.graph.sqlite_store import get_store
        from magma.search import MemorySearcher, classify_intent
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        faiss_index = getattr(app.state, "faiss_index", None)
        # P1: classify intent before search
        intent = classify_intent(req.query)
        filters = dict(req.filters or {})
        filters["intent"] = intent
        searcher = MemorySearcher(store, encoder, faiss_index)
        results = await asyncio.to_thread(searcher.query, req.query, req.top_k, filters)
        if not results:
            results = [{
                "id": "system",
                "label": "info",
                "properties": {"message": f"No memory matched '{req.query}'."},
                "score": 0.0,
                "source": "system",
            }]

        # P2: Context synthesis generates a narrative from graph topology.
        narrative_data = None
        try:
            from magma.context_synthesis import synthesize_narrative
            narrative_data = await asyncio.to_thread(
                synthesize_narrative, results, req.query, intent, store, req.priority
            )
        except Exception as e:
            logger.warning(f"Context synthesis failed (non-fatal): {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # Extract short_command_resolution from results if present
        short_cmd_resolution = None
        if results and isinstance(results[0], dict):
            scr = results[0].get("short_command_resolution")
            if scr:
                short_cmd_resolution = scr.get("short_command_resolution")
                # Also add top-level fields for easier access
                if short_cmd_resolution:
                    short_cmd_resolution["drift_warning"] = results[0].get("short_command_drift_warning", False)

        # P1-4: Extract active_context (curated set) from results
        active_context = [
            {"id": r["id"], "label": r.get("label", ""), "score": r.get("score", 0),
             "full_content": r.get("full_content", ""), "tier": r.get("tier", "full")}
            for r in results if r.get("curated")
        ]

        # P1-5: Extract bridge_entities from results
        bridge_entities = []
        if results:
            # Collect bridge entities from score_breakdown of first result
            first_breakdown = (results[0].get("score_breakdown") or {}) if results else {}
            bridge_count = first_breakdown.get("bridge_entity_count", 0)
            if bridge_count > 0:
                # Reconstruct bridge entities from all results
                ent_map = {}
                for r in results:
                    props = r.get("properties") or {}
                    entities = props.get("entities", [])
                    if isinstance(entities, list):
                        for ent in entities:
                            if isinstance(ent, dict) and ent.get("name"):
                                ename = ent["name"].lower()
                                if ename not in ent_map:
                                    ent_map[ename] = set()
                                ent_map[ename].add(r["id"])
                bridge_entities = [
                    {"entity": name, "node_count": len(nids)}
                    for name, nids in ent_map.items() if len(nids) >= 2
                ]

        event_id = f"recall-{uuid.uuid4().hex}"
        try:
            recall_results = [
                {"id": r.get("id"), "score": r.get("score", 0), "label": r.get("label", "")}
                for r in results if r.get("id") != "system"
            ]
            session_key = filters.get("session_key") or filters.get("session_id") or ""
            source_agent = (
                filters.get("current_agent_id")
                or filters.get("agent_id")
                or _parse_source_agent(session_key)
                or ""
            )
            store.record_recall_event(
                event_id=event_id,
                query=req.query,
                agent_id=source_agent,
                session_key=session_key,
                results=recall_results,
                source_agent_id=source_agent,
                department=DEPT_MAP.get(source_agent, ""),
            )
        except Exception as e:
            logger.warning(f"Failed to record recall_event: {e}")

        return QueryResponse(
            query=req.query,
            results=results,
            count=len(results),
            event_id=event_id,
            intent=intent,
            narrative=narrative_data.get("narrative") if narrative_data else None,
            references=narrative_data.get("references") if narrative_data else None,
            token_budget=narrative_data.get("token_budget") if narrative_data else None,
            tokens_used=narrative_data.get("tokens_used") if narrative_data else None,
            token_usage_ratio=narrative_data.get("token_usage_ratio") if narrative_data else None,
            budget_warning=narrative_data.get("budget_warning") if narrative_data else None,
            # P2-5: Backtrack detection from search results
            backtrack_warning=results[0].get("backtrack_warning") if results else None,
            backtrack_narrative=results[0].get("backtrack_narrative") if results else None,
            short_command_resolution=short_cmd_resolution,
            active_context=active_context or None,
            bridge_entities=bridge_entities or None,
        )

    @app.post("/api/v1/nodes")
    async def add_node(req: NodeRequest):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        properties = dict(req.properties or {})
        text_for_embedding = _node_text(req.label, properties)
        properties.update(await asyncio.to_thread(_memory_metadata, text_for_embedding))
        embedding = await asyncio.to_thread(encoder.encode, text_for_embedding)
        embedding = embedding.astype("float32")

        await asyncio.to_thread(store.add_node, req.id, req.label, properties, embedding)
        await asyncio.to_thread(_attach_entity_anchors, store, encoder, req.id, text_for_embedding, properties.get("source", "api"))

        # Incrementally update FAISS index
        faiss_index = getattr(app.state, "faiss_index", None)
        if faiss_index and embedding is not None:
            try:
                await asyncio.to_thread(faiss_index.add, req.id, embedding)
            except Exception as e:
                logger.warning(f"FAISS incremental add failed: {e}")

        return {"status": "ok", "id": req.id}

    @app.post("/api/v1/capture")
    async def capture(req: CaptureRequest):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder
        from magma.capture_policy import classify_capture, deduplicator, MAGMA_FEATURE_CONTENT_DEDUP

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        written = []

        capture_decision = await asyncio.to_thread(
            classify_capture,
            req.user_text,
            req.assistant_text,
            await asyncio.to_thread(store.get_suppression_patterns),
        )
        if not capture_decision.should_capture:
            return {
                "status": "skipped",
                "written": [],
                "count": 0,
                "capture_decision": {
                    "strength": capture_decision.strength,
                    "reasons": capture_decision.reasons,
                },
            }

        # --- P1-1: Content Dedup (MinHash) ---
        # Check if combined content is a near-duplicate of recent nodes
        if MAGMA_FEATURE_CONTENT_DEDUP:
            combined_text = " ".join(t for t in (req.user_text, req.assistant_text) if t and t.strip())
            if combined_text:
                t0 = time.time()
                dup_node_id = deduplicator.is_duplicate(combined_text, store)
                dedup_ms = (time.time() - t0) * 1000
                logger.info(f"Dedup check: {dedup_ms:.1f}ms, duplicate={dup_node_id}")
                if dup_node_id and dedup_ms < 50:
                    # Merge: update content + access_count on existing node
                    try:
                        existing_node = store.get_node(dup_node_id)
                        if existing_node:
                            props = dict(existing_node.get("properties", {}) or {})
                            props["content"] = combined_text
                            props["access_count"] = int(props.get("access_count", 0)) + 1
                            props["last_dedup_merge"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                            store.update_node(dup_node_id, props)
                            return {
                                "status": "merged",
                                "written": [dup_node_id],
                                "count": 1,
                                "dedup": True,
                                "dedup_ms": round(dedup_ms, 1),
                                "capture_decision": {
                                    "strength": capture_decision.strength,
                                    "reasons": capture_decision.reasons,
                                },
                            }
                    except Exception as e:
                        logger.warning(f"Dedup merge failed (falling through to normal capture): {e}")

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
                "capture_strength": capture_decision.strength,
                "capture_reasons": capture_decision.reasons,
            }
            properties.update(await asyncio.to_thread(_memory_metadata, cleaned))
            text_for_embedding = f"{role}: {cleaned}"
            embedding = await asyncio.to_thread(encoder.encode, text_for_embedding)
            embedding = embedding.astype("float32")
            await asyncio.to_thread(store.add_node, node_id, "event", properties, embedding)
            await asyncio.to_thread(_attach_entity_anchors, store, encoder, node_id, cleaned, req.source)

            # Incrementally update FAISS index
            faiss_index = getattr(app.state, "faiss_index", None)
            if faiss_index and embedding is not None:
                try:
                    await asyncio.to_thread(faiss_index.add, node_id, embedding)
                except Exception as e:
                    logger.warning(f"FAISS incremental add failed: {e}")

            written.append(node_id)

        if len(written) == 2:
            await asyncio.to_thread(store.add_edge, written[0], written[1], "responded_by", {
                "source": req.source,
                "session_key": req.session_key,
            })

        decisions_written = []
        try:
            from magma.decision_particles import extract_decision_events

            decision_events = await asyncio.to_thread(
                extract_decision_events,
                req.user_text,
                req.assistant_text,
            )
            faiss_index = getattr(app.state, "faiss_index", None)
            for event in decision_events:
                node_id, text_for_embedding, properties = _decision_event_payload(req, event)
                properties.update(await asyncio.to_thread(_memory_metadata, text_for_embedding))
                embedding = await asyncio.to_thread(encoder.encode, text_for_embedding)
                embedding = embedding.astype("float32")
                await asyncio.to_thread(store.add_node, node_id, "decision_event", properties, embedding)
                for evt_id in written:
                    await asyncio.to_thread(store.add_edge_once, evt_id, node_id, "expresses_decision", {
                        "source": req.source,
                        "decision_key": event.get("decision_key"),
                    })
                if faiss_index and embedding is not None:
                    try:
                        await asyncio.to_thread(faiss_index.add, node_id, embedding)
                    except Exception as e:
                        logger.warning(f"FAISS incremental add for decision event failed: {e}")
                decisions_written.append(node_id)
        except Exception as e:
            logger.warning(f"Decision-particle extraction failed (non-fatal): {e}")

        # --- P0-1: Fact Extraction (background, non-blocking) ---
        # Run fact extraction as background task so capture returns immediately
        extract_facts_fn, is_available_fn = _get_fact_extractor()
        if is_available_fn() and written:
            faiss_index = getattr(app.state, "faiss_index", None)
            asyncio.create_task(_extract_facts_background(
                req, written, encoder, store, faiss_index, extract_facts_fn
            ))

        return {
            "status": "ok",
            "written": written,
            "count": len(written),
            "decisions_written": decisions_written,
            "decision_count": len(decisions_written),
            "capture_decision": {
                "strength": capture_decision.strength,
                "reasons": capture_decision.reasons,
            },
        }

    async def _extract_facts_background(req, written, encoder, store, faiss_index, extract_facts_fn):
        """Background task: extract facts + causal relations without blocking capture response."""
        try:
            from magma.fact_extractor import extract_facts_batch, extract_causal_batch
            facts = await asyncio.to_thread(extract_facts_batch, req.user_text, req.assistant_text)
            for fact_item in facts:
                fact_text = fact_item["fact"]
                category = fact_item.get("category", "fact")
                entities = fact_item.get("entities", [])

                fact_digest = hashlib.sha1(
                    f"{req.source}:{category}:{fact_text}".encode("utf-8")
                ).hexdigest()[:16]
                fact_node_id = f"fact:{category}:{fact_digest}"

                fact_properties = {
                    "layer": "L0",
                    "source": req.source,
                    "role": "fact",
                    "content": fact_text,
                    "fact_category": category,
                    "fact_entities": json.dumps(entities, ensure_ascii=False),
                    "agent_id": req.agent_id,
                    "session_key": req.session_key,
                    "session_id": req.session_id,
                    "ttl_days": 365,
                    "importance": 0.65,
                    "source_agent_id": _parse_source_agent(req.session_key) or req.agent_id,
                    "department": DEPT_MAP.get(_parse_source_agent(req.session_key) or req.agent_id or "", ""),
                }
                fact_properties.update(await asyncio.to_thread(_memory_metadata, fact_text))

                fact_embedding = await asyncio.to_thread(encoder.encode, fact_text)
                fact_embedding = fact_embedding.astype("float32")
                await asyncio.to_thread(store.add_node, fact_node_id, "fact", fact_properties, fact_embedding)

                for evt_id in written:
                    await asyncio.to_thread(store.add_edge_once, evt_id, fact_node_id, "extracted_fact", {
                        "source": req.source,
                        "category": category,
                    })

                for entity_name in entities:
                    try:
                        await asyncio.to_thread(store.invalidate_old_facts,
                            entity_name=entity_name,
                            category=category,
                            exclude_node_id=fact_node_id,
                        )
                    except Exception as e:
                        logger.warning(f"Temporal invalidation failed for {entity_name}: {e}")

                await asyncio.to_thread(_attach_entity_anchors, store, encoder, fact_node_id, fact_text, req.source)

                if faiss_index and fact_embedding is not None:
                    try:
                        await asyncio.to_thread(faiss_index.add, fact_node_id, fact_embedding)
                    except Exception as e:
                        logger.warning(f"FAISS incremental add for fact failed: {e}")

            # --- P0 Causal Extraction (runs after fact extraction) ---
            try:
                causal_relations = await asyncio.to_thread(extract_causal_batch, req.user_text, req.assistant_text)
                for rel in causal_relations:
                    cause_text = rel["cause"]
                    effect_text = rel["effect"]
                    confidence = rel["confidence"]
                    rationale = rel["rationale"]

                    # Create or find cause node
                    cause_digest = hashlib.sha1(
                        f"{req.source}:cause:{cause_text}".encode("utf-8")
                    ).hexdigest()[:16]
                    cause_node_id = f"cause:{cause_digest}"
                    cause_props = {
                        "layer": "L0",
                        "source": req.source,
                        "role": "cause",
                        "content": cause_text,
                        "agent_id": req.agent_id,
                        "session_key": req.session_key,
                        "ttl_days": 365,
                        "importance": 0.6,
                        "source_agent_id": _parse_source_agent(req.session_key) or req.agent_id,
                        "department": DEPT_MAP.get(_parse_source_agent(req.session_key) or req.agent_id or "", ""),
                    }
                    cause_props.update(await asyncio.to_thread(_memory_metadata, cause_text))
                    cause_emb = await asyncio.to_thread(encoder.encode, cause_text)
                    cause_emb = cause_emb.astype("float32")
                    await asyncio.to_thread(store.add_node, cause_node_id, "cause", cause_props, cause_emb)

                    # Create or find effect node
                    effect_digest = hashlib.sha1(
                        f"{req.source}:effect:{effect_text}".encode("utf-8")
                    ).hexdigest()[:16]
                    effect_node_id = f"effect:{effect_digest}"
                    effect_props = {
                        "layer": "L0",
                        "source": req.source,
                        "role": "effect",
                        "content": effect_text,
                        "agent_id": req.agent_id,
                        "session_key": req.session_key,
                        "ttl_days": 365,
                        "importance": 0.6,
                        "source_agent_id": _parse_source_agent(req.session_key) or req.agent_id,
                        "department": DEPT_MAP.get(_parse_source_agent(req.session_key) or req.agent_id or "", ""),
                    }
                    effect_props.update(await asyncio.to_thread(_memory_metadata, effect_text))
                    effect_emb = await asyncio.to_thread(encoder.encode, effect_text)
                    effect_emb = effect_emb.astype("float32")
                    await asyncio.to_thread(store.add_node, effect_node_id, "effect", effect_props, effect_emb)

                    # Add causal edge: effect caused_by cause
                    await asyncio.to_thread(store.add_causal_edge, effect_node_id, cause_node_id, confidence, rationale, "llm")

                    # Link causal nodes to event nodes
                    for evt_id in written:
                        await asyncio.to_thread(store.add_edge_once, evt_id, cause_node_id, "mentions_cause", {
                            "source": req.source,
                        })
                        await asyncio.to_thread(store.add_edge_once, evt_id, effect_node_id, "mentions_effect", {
                            "source": req.source,
                        })

                    # Add to FAISS index
                    if faiss_index:
                        try:
                            await asyncio.to_thread(faiss_index.add, cause_node_id, cause_emb)
                            await asyncio.to_thread(faiss_index.add, effect_node_id, effect_emb)
                        except Exception as e:
                            logger.warning(f"FAISS add for causal nodes failed: {e}")

                    logger.info(f"Causal edge created: {effect_node_id} caused_by {cause_node_id} (conf={confidence:.2f})")

            except Exception as e:
                logger.warning(f"Background causal extraction failed (non-fatal): {e}")

        except Exception as e:
            logger.warning(f"Background fact extraction failed (non-fatal): {e}")

    @app.post("/api/v1/edges")
    async def add_edge(req: EdgeRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        store.add_edge(req.source_id, req.target_id, req.relation, req.properties)
        return {"status": "ok"}

    @app.get("/api/v1/nodes")
    async def list_nodes(label: Optional[str] = None, limit: int = 100, importance_label: Optional[str] = None):
        from magma.graph.sqlite_store import get_store
        from magma.search import importance_label_from_value

        store = getattr(app.state, "store", None) or get_store()
        nodes = store.query_nodes(label=label, limit=limit * 3 if importance_label else limit)
        # P0-2: Filter by importance_label if specified
        if importance_label:
            nodes = [
                n for n in nodes
                if importance_label_from_value(float(n.get("importance") or 0.5)) == importance_label
            ][:limit]
        # Attach importance_label to each node
        for n in nodes:
            n["importance_label"] = importance_label_from_value(float(n.get("importance") or 0.5))
        return {"nodes": nodes, "count": len(nodes)}

    @app.get("/api/v1/decisions/drift", response_model=DecisionDriftResponse)
    async def decision_drift(
        agent_id: Optional[str] = None,
        session_key: Optional[str] = None,
        decision_key: Optional[str] = None,
        limit: int = 50,
    ):
        from magma.decision_particles import summarize_decision_drift
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        nodes = await asyncio.to_thread(
            store.get_decision_events,
            agent_id,
            session_key,
            decision_key,
            limit,
        )
        events = []
        for node in nodes:
            props = node.get("properties", {}) or {}
            event = {
                "id": node.get("id"),
                "created_at": node.get("created_at"),
                "decision_key": props.get("decision_key"),
                "selected": props.get("selected"),
                "options": props.get("options") or [],
                "chain": props.get("chain") or [],
                "direction": props.get("direction"),
                "confidence": props.get("confidence"),
                "tags": props.get("tags") or [],
                "content": props.get("content"),
                "agent_id": props.get("agent_id"),
                "session_key": props.get("session_key"),
            }
            events.append(event)
        return {
            "events": events,
            "summary": summarize_decision_drift(events),
            "count": len(events),
        }

    @app.get("/api/v1/nodes/{node_id}")
    async def get_node(node_id: str):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        node = store.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        edges = store.get_edges(node_id)
        return {"node": node, "edges": edges}

    @app.patch("/api/v1/nodes/{node_id}")
    async def update_node(node_id: str, body: dict):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        # Accept both {"properties": {...}} and flat {...}
        properties = body.get("properties", body)

        # P0-2: Map importance_label to importance value if provided
        importance_label = properties.pop("importance_label", None)
        if importance_label and importance_label in IMPORTANCE_LABEL_MAP:
            properties["importance"] = IMPORTANCE_LABEL_MAP[importance_label]
            properties["importance_label"] = importance_label

        # Check if content-affecting fields changed -> re-encode embedding
        new_embedding = None
        content_keys = {"content", "name", "title", "summary", "message"}
        if content_keys & set(properties.keys()):
            # Fetch existing node to get full context for embedding
            existing_node = store.get_node(node_id)
            if existing_node:
                merged_props = dict(existing_node.get("properties", {}) or {})
                merged_props.update(properties)
                label = existing_node.get("label", "")
                text_for_embedding = _node_text(label, merged_props)
                try:
                    new_embedding = await asyncio.to_thread(encoder.encode, text_for_embedding)
                    new_embedding = new_embedding.astype("float32")
                except Exception as e:
                    logger.warning(f"Failed to re-encode embedding for {node_id}: {e}")

        if not await asyncio.to_thread(store.update_node, node_id, properties, embedding=new_embedding):
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        # Update FAISS index with new embedding
        if new_embedding is not None:
            faiss_index = getattr(app.state, "faiss_index", None)
            if faiss_index:
                try:
                    # Remove old entry and add new one
                    await asyncio.to_thread(faiss_index.remove, node_id)
                    await asyncio.to_thread(faiss_index.add, node_id, new_embedding)
                except Exception as e:
                    logger.warning(f"FAISS update failed for {node_id}: {e}")

        return {"status": "ok", "id": node_id}

    @app.delete("/api/v1/nodes/{node_id}")
    async def delete_node(node_id: str):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        if not store.delete_node(node_id):
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        return {"status": "ok", "id": node_id}

    @app.post("/api/v1/consolidate")
    async def consolidate(purge_deleted: bool = False):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        return {"status": "ok", "stats": store.consolidate(purge_deleted=purge_deleted)}

    @app.post("/api/v1/distill_l1")
    async def distill_l1(req: L1DistillRequest):
        from magma.graph.sqlite_store import get_store
        from magma.l1_distiller import distill_l1 as run_distill_l1
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        result = await asyncio.to_thread(
            run_distill_l1,
            store,
            encoder,
            hours=max(1, req.hours),
            limit=max(1, min(req.limit, 2000)),
            dry_run=req.dry_run,
            source_agent_id=req.source_agent_id,
        )
        faiss_index = getattr(app.state, "faiss_index", None)
        if faiss_index and not req.dry_run:
            for node_id in result.get("written", []):
                node = store.get_node(node_id)
                if not node:
                    continue
                try:
                    text = _node_text(node.get("label", ""), node.get("properties") or {})
                    embedding = await asyncio.to_thread(encoder.encode, text)
                    await asyncio.to_thread(faiss_index.add, node_id, embedding.astype("float32"))
                except Exception as e:
                    logger.warning(f"FAISS L1 incremental add failed for {node_id}: {e}")
        return result

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

    @app.post("/api/v1/search_by_entity")
    async def search_by_entity(req: EntitySearchRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        results = store.search_by_entity(
            entity_name=req.entity_name,
            entity_type=req.entity_type,
        )
        return {"results": results, "count": len(results)}

    @app.get("/api/v1/timeline/{entity_name}")
    async def timeline(entity_name: str, limit: int = 20):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        results = store.get_fact_timeline(entity_name=entity_name, limit=limit)
        return {"entity": entity_name, "facts": results, "count": len(results)}

    @app.get("/api/v1/facts")
    async def list_facts(entity_name: str = None, category: str = None, limit: int = 20):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        results = store.get_active_facts(entity_name=entity_name, category=category, limit=limit)
        return {"facts": results, "count": len(results)}

    @app.get("/api/v1/core_memory")
    async def get_core_memory(agent_id: str = None, block_name: str = None):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        blocks = await asyncio.to_thread(
            store.get_core_memories,
            agent_id=agent_id,
            block_name=block_name,
        )
        return {"blocks": blocks, "count": len(blocks)}

    @app.put("/api/v1/core_memory")
    async def set_core_memory(req: CoreMemoryRequest):
        from magma.graph.sqlite_store import get_store
        from magma.vector.encoder import Encoder

        store = getattr(app.state, "store", None) or get_store()
        encoder = getattr(app.state, "encoder", None) or Encoder()
        text_for_embedding = f"core_memory {req.block_name}: {req.content}"
        embedding = await asyncio.to_thread(encoder.encode, text_for_embedding)
        embedding = embedding.astype("float32")
        node = await asyncio.to_thread(
            store.set_core_memory,
            block_name=req.block_name,
            content=req.content,
            agent_id=req.agent_id,
            source=req.source,
            importance=req.importance,
            embedding=embedding,
        )

        faiss_index = getattr(app.state, "faiss_index", None)
        if faiss_index and embedding is not None:
            try:
                await asyncio.to_thread(faiss_index.add, node["id"], embedding)
            except Exception as e:
                logger.warning(f"FAISS core memory add failed: {e}")

        return {"status": "ok", "block": node}

    # --- P1-1: Entity management endpoints ---
    @app.get("/api/v1/entities")
    async def list_entities():
        from magma.entities import list_entities
        entities = list_entities()
        return {"entities": entities, "count": len(entities)}

    @app.post("/api/v1/entities")
    async def add_entity(body: dict):
        from magma.entities import add_custom_entity
        name = body.get("name")
        entity_type = body.get("entity_type", "custom")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        added = add_custom_entity(name, entity_type)
        return {"status": "ok" if added else "already_exists", "name": name, "entity_type": entity_type}

    @app.delete("/api/v1/entities/{name}")
    async def delete_entity(name: str):
        from magma.entities import remove_custom_entity
        removed = remove_custom_entity(name)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
        return {"status": "ok", "name": name}

    @app.post("/api/v1/feedback")
    async def feedback(req: FeedbackRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        existing_event = store.get_recall_event(req.event_id)
        source_agent = (
            _parse_source_agent(req.session_key)
            or req.agent_id
            or (existing_event or {}).get("source_agent_id")
            or (existing_event or {}).get("agent_id")
            or ""
        )
        dept = DEPT_MAP.get(source_agent, "") or (existing_event or {}).get("department")
        if existing_event is None:
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
            missed_node_ids=req.missed_node_ids,
        )
        return {"status": "ok", "feedback": stats}

    @app.post("/api/v1/recall/explain")
    async def explain_recall(req: ExplainRecallRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        return {
            "status": "ok",
            "explanation": await asyncio.to_thread(
                store.explain_recall,
                query=req.query,
                event_id=req.event_id,
                node_id=req.node_id,
            ),
        }

    @app.post("/api/v1/memory/mark_wrong")
    async def mark_wrong(req: MarkMemoryRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        result = await asyncio.to_thread(
            store.mark_memory_wrong,
            node_id=req.node_id,
            reason=req.reason,
            agent_id=req.agent_id,
            session_key=req.session_key,
        )
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found")
        return result

    @app.post("/api/v1/memory/mark_important")
    async def mark_important(req: MarkMemoryRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        result = await asyncio.to_thread(
            store.mark_memory_important,
            node_id=req.node_id,
            reason=req.reason,
            agent_id=req.agent_id,
            session_key=req.session_key,
        )
        if result.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found")
        return result

    @app.post("/api/v1/memory/suppress_pattern")
    async def suppress_pattern(req: SuppressPatternRequest):
        from magma.graph.sqlite_store import get_store

        store = getattr(app.state, "store", None) or get_store()
        return await asyncio.to_thread(
            store.suppress_pattern,
            pattern=req.pattern,
            reason=req.reason,
            agent_id=req.agent_id,
            ttl_days=req.ttl_days,
        )

    # --- P1-3: Verify endpoint ---
    # Simple LRU cache for verify results (claim+node_ids -> result, 1h TTL)
    _verify_cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
    _VERIFY_CACHE_TTL = 3600  # 1 hour

    @app.post("/api/v1/verify")
    async def verify(req: VerifyRequest):
        """Verify a claim against specified memory nodes using LLM.

        Returns: {verdict, evidence, confidence}
        """
        import hashlib as _hashlib
        import time as _time

        # Cache key: sorted node_ids + claim hash
        cache_key = _hashlib.sha256(
            f"{sorted(req.node_ids)}:{req.claim}".encode()
        ).hexdigest()[:16]

        # Check cache
        cached = _verify_cache.get(cache_key)
        if cached:
            result, ts = cached
            if _time.time() - ts < _VERIFY_CACHE_TTL:
                return {"status": "ok", "cached": True, **result}

        from magma.graph.sqlite_store import get_store
        store = getattr(app.state, "store", None) or get_store()

        # Fetch node contents
        node_contents = []
        for nid in req.node_ids:
            node = store.get_node(nid)
            if node:
                props = node.get("properties", {}) or {}
                content = props.get("content", "") or props.get("summary", "") or ""
                node_contents.append({
                    "id": nid,
                    "label": node.get("label", ""),
                    "content": content[:2000],  # Truncate for LLM context
                })

        if not node_contents:
            return {"status": "ok", "verdict": "unsupported", "evidence": "No valid nodes found.", "confidence": 0.0}

        # Call LLM for verification
        try:
            result = await asyncio.to_thread(_verify_with_llm, req.claim, node_contents)
        except Exception as e:
            logger.warning(f"Verify LLM call failed: {e}")
            result = {"verdict": "unsupported", "evidence": f"LLM verification failed: {e}", "confidence": 0.0}

        # Cache result
        _verify_cache[cache_key] = (result, _time.time())
        # Evict old entries (simple cleanup)
        if len(_verify_cache) > 1000:
            cutoff = _time.time() - _VERIFY_CACHE_TTL
            to_delete = [k for k, (_, ts) in _verify_cache.items() if ts < cutoff]
            for k in to_delete[:500]:
                _verify_cache.pop(k, None)

        return {"status": "ok", "cached": False, **result}

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


async def _run_encoder_idle_check(interval: int):
    """Periodically check if encoder model should be unloaded due to idle timeout."""
    while True:
        await asyncio.sleep(interval)
        try:
            encoder = getattr(app.state, "encoder", None)
            if encoder and hasattr(encoder, "maybe_unload"):
                if encoder.maybe_unload():
                    logger.info("Encoder model unloaded due to idle timeout")
        except Exception as e:
            logger.warning(f"Encoder idle check failed: {e}")


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    host = os.environ.get("MAGMA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("MAGMA_API_PORT", "8902"))
    uvicorn.run(app, host=host, port=port)
