"""Shared retrieval logic for MAGMA API and MCP entrypoints."""

import json
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from magma.entities import classify_memory_scope, extract_entities

INTENT_KEYWORDS = {
    "temporal": (
        "\u4ec0\u4e48\u65f6\u5019", "\u54ea\u5929", "\u51e0\u70b9", "\u65f6\u95f4",
        "\u6700\u8fd1", "\u4e4b\u524d", "\u4e4b\u540e", "\u540e\u6765",
        "\u5148\u540e", "\u987a\u5e8f", "\u5386\u53f2", "\u8fc7\u53bb",
        "\u6628\u5929", "\u4eca\u5929", "\u660e\u5929", "\u4e0a\u6b21",
        "when", "before", "after", "recent", "timeline",
    ),
    "causal": (
        "\u4e3a\u4ec0\u4e48", "\u539f\u56e0", "\u5bfc\u81f4", "\u5f71\u54cd",
        "\u4f9d\u8d56", "\u56e0\u679c", "\u7ed3\u679c", "\u6240\u4ee5",
        "\u600e\u4e48\u56de\u4e8b", "\u4e3a\u4f55", "root cause", "why",
        "because", "cause", "impact", "depend",
    ),
    "entity": (
        "\u8c01", "\u54ea\u4e2a", "\u5546\u54c1", "sku", "\u8d27\u53f7",
        "\u6587\u4ef6", "\u9879\u76ee", "agent", "\u52a9\u7406",
        "\u8fd0\u8425", "\u6280\u672f", "\u8d26\u53f7", "\u5e97\u94fa",
        "\u54c1\u724c", "\u4eba", "\u7ec4\u7ec7", "where", "who", "which",
    ),
}

RELATED_RELATIONS = (
    "same_as",
    "responded_by",
    "depends_on",
    "caused_by",
    "mentions_entity",
    "related_to",
)

OPERATIONAL_KEYWORDS = (
    "8902",
    "2026.5.20",
    "5.20",
    "5.22",
    "magma_doctor.py",
    "magma_ops.py",
    "magma-recall",
    "mcp_proxy",
    "recent_capture",
    "source_agent_id",
    "recall_events",
    "recall_feedback",
    "before_prompt_build",
    "before_message_write",
    "agent_end",
    "http_proxy",
    "runbook.md",
    "bge-small-zh-v1.5",
    "qwen3",
    "reranker",
)

OPERATIONAL_TRIGGER_KEYWORDS = OPERATIONAL_KEYWORDS + (
    "magma",
    "openclaw",
    "embedding",
    "\u7f51\u5173",
    "\u8bb0\u5fc6",
)

HIGH_DENSITY_LAYERS = {"ops_anchor", "L1", "summary", "decision", "fact", "current_state"}


def _is_operational_query(query: str) -> bool:
    query_lower = (query or "").lower()
    return any(keyword.lower() in query_lower for keyword in OPERATIONAL_TRIGGER_KEYWORDS)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def _query_terms(query: str) -> List[str]:
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    terms.extend(
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_./:-]{3,}", query or "")
    )
    try:
        import jieba

        terms.extend(term.lower() for term in jieba.lcut(query) if len(term.strip()) > 1)
    except Exception:
        pass
    if not terms and query:
        terms.append(query.lower())
    return sorted(set(terms), key=len, reverse=True)


def _operational_keyword_score(query: str, searchable: str) -> float:
    query_lower = (query or "").lower()
    score = 0.0
    for keyword in OPERATIONAL_KEYWORDS:
        key = keyword.lower()
        if key in query_lower and key in searchable:
            score += 0.3 if any(ch in key for ch in "._-:") else 0.22
    return min(score, 0.75)


def _node_searchable_text(node: Dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    parts = [str(node.get("id") or ""), str(node.get("label") or "")]
    for key in ("title", "name", "content", "summary", "message", "source_file"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    entities = props.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict) and entity.get("name"):
                parts.append(str(entity["name"]))
    return " ".join(parts).lower()


def _keyword_score(query: str, node: Dict[str, Any]) -> float:
    searchable = _node_searchable_text(node)
    query_lower = query.lower()
    score = 0.0

    if query_lower and query_lower in searchable:
        score += 0.35
    for term in _query_terms(query):
        if term in searchable:
            score += min(0.08 + len(term) / 80.0, 0.18)
    if query_lower and query_lower in node.get("label", "").lower():
        score += 0.2
    score += _operational_keyword_score(query, searchable)
    return min(score, 1.15)


def detect_intent(query: str) -> Dict[str, Any]:
    query_lower = (query or "").lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for keyword in keywords if keyword.lower() in query_lower)
    primary = max(scores, key=scores.get) if any(scores.values()) else "semantic"
    return {
        "primary": primary,
        "scores": scores,
    }


def _intent_multiplier(node: Dict[str, Any], intent: Dict[str, Any]) -> float:
    primary = intent.get("primary") or "semantic"
    props = node.get("properties", {}) or {}
    label = node.get("label") or ""
    if primary == "temporal":
        has_time = any(props.get(key) for key in ("date", "time", "timestamp", "valid_from", "valid_until"))
        if has_time or label == "event":
            return 1.12
        return 0.96
    if primary == "causal":
        searchable = _node_searchable_text(node)
        if any(term in searchable for term in ("\u539f\u56e0", "\u5bfc\u81f4", "\u56e0\u4e3a", "\u4f9d\u8d56", "caused", "cause", "depends")):
            return 1.12
        return 1.0
    if primary == "entity":
        if label != "event" or any(props.get(key) for key in ("name", "title", "source_file", "agent_id", "sku", "brand")):
            return 1.12
        return 0.98
    return 1.0


def _entity_overlap_multiplier(query_entities: List[Dict[str, str]], node: Dict[str, Any]) -> float:
    if not query_entities:
        return 1.0
    searchable = _node_searchable_text(node)
    weights = {
        "sku": 0.28,
        "selling_point": 0.18,
        "document": 0.18,
        "domain": 0.12,
        "system": 0.08,
        "plugin": 0.08,
        "storage": 0.08,
        "api": 0.08,
        "protocol": 0.08,
        "model": 0.08,
    }
    score = 0.0
    matches = 0
    for entity in query_entities:
        if entity["name"].lower() not in searchable:
            continue
        matches += 1
        score += weights.get(entity.get("entity_type"), 0.08)
    if matches == 0:
        return 1.0
    label = node.get("label") or ""
    base = 1.0 + min(score, 0.5)
    if label == "entity":
        base += 0.28
    elif (node.get("properties") or {}).get("layer") == "L0":
        base = min(base, 1.22)
    return min(base, 1.65)


def _memory_quality_multiplier(node: Dict[str, Any]) -> float:
    props = node.get("properties", {}) or {}
    layer = props.get("layer")
    source = props.get("source")
    label = node.get("label") or ""
    if layer in {"L1", "summary", "decision", "fact", "current_state"}:
        return 1.2
    if layer == "ops_anchor" or source == "magma_operational_anchor":
        return 1.08
    if layer == "entity_anchor" or label == "entity":
        return 0.62
    if layer != "L0":
        return 1.03 if label == "topic" else 1.0
    role = props.get("role")
    content = str(props.get("content") or "")
    if role == "assistant":
        multiplier = 0.98
        lower = content.lower()
        if "\u6ca1\u6709\u76f4\u63a5" in content or "not directly" in lower:
            multiplier *= 0.78
        if "\u6839\u636e\u7cfb\u7edf\u81ea\u52a8\u6ce8\u5165" in content:
            multiplier *= 0.9
        if any(term in content for term in ("\u4e09\u4e2a\u95ee\u9898", "\u4ee5\u4e0b\u662f\u4e09\u4e2a", "\u9010\u9879\u56de\u7b54")):
            multiplier *= 0.92
        return multiplier
    if role != "user":
        return 1.0
    question_terms = (
        "?", "\uff1f", "\u4e3a\u4ec0\u4e48", "\u600e\u4e48", "\u5982\u4f55",
        "\u54ea\u4e2a", "\u4ec0\u4e48", "\u662f\u5426", "\u80fd\u4e0d\u80fd",
        "\u53ef\u4ee5\u95ee", "\u6d4b\u8bd5",
    )
    if any(term in content for term in question_terms):
        return 0.62
    return 0.86


def _operational_authority_multiplier(query: str, node: Dict[str, Any], keyword_score: float) -> float:
    if not _is_operational_query(query):
        return 1.0
    props = node.get("properties", {}) or {}
    layer = props.get("layer")
    source = props.get("source")
    memory_scope = props.get("memory_scope")
    label = node.get("label") or ""
    if layer in HIGH_DENSITY_LAYERS or source == "magma_operational_anchor":
        if keyword_score >= 0.3:
            return 1.34
        if keyword_score >= 0.15:
            return 1.16
        return 1.0
    if label == "topic" and memory_scope == "system":
        return 1.1 if keyword_score >= 0.2 else 1.0
    if layer == "L0":
        return 0.92 if keyword_score < 0.15 else 0.98
    return 1.0


def _lifecycle_multiplier(node: Dict[str, Any]) -> float:
    now = datetime.utcnow()
    status = node.get("status") or "active"
    if status == "deleted":
        return 0.0
    multiplier = 1.0 if status == "active" else 0.55

    valid_from = _parse_time(node.get("valid_from"))
    valid_until = _parse_time(node.get("valid_until"))
    if valid_from and valid_from > now:
        multiplier *= 0.4
    if valid_until and valid_until < now:
        multiplier *= 0.35

    created_at = _parse_time(node.get("created_at"))
    ttl_days = node.get("ttl_days")
    if created_at and ttl_days:
        age_days = max((now - created_at).days, 0)
        if age_days > ttl_days:
            multiplier *= 0.4
        else:
            multiplier *= 0.65 + 0.35 * (1.0 - age_days / max(ttl_days, 1))

    access_count = int(node.get("access_count") or 0)
    importance = float(node.get("importance") or 0.5)
    multiplier *= 0.75 + min(max(importance, 0.0), 1.0) * 0.35
    multiplier += min(math.log1p(access_count) * 0.03, 0.12)
    return max(multiplier, 0.0)


def _embedding_from_blob(blob: Optional[bytes], expected_dim: int) -> Optional[np.ndarray]:
    if not blob:
        return None
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.shape[0] != expected_dim:
        return None
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _property_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    supported = {}
    for key in ("agent_id", "source", "session_key", "session_id", "role", "layer", "memory_scope"):
        value = filters.get(key)
        if value is None:
            plural = filters.get(f"{key}s")
            if plural is not None:
                value = plural
        if value is not None:
            supported[key] = value
    return supported


def _provenance(node: Dict[str, Any]) -> Dict[str, Any]:
    props = node.get("properties", {}) or {}
    return {
        "agent_id": node.get("source_agent_id") or props.get("source_agent_id") or props.get("agent_id"),
        "source": props.get("source"),
        "session_key": props.get("session_key"),
        "session_id": props.get("session_id"),
        "role": props.get("role"),
        "layer": props.get("layer"),
    }


def _agent_scope_multiplier(node: Dict[str, Any], filters: Dict[str, Any]) -> float:
    current_agent_id = filters.get("current_agent_id")
    if not current_agent_id:
        return 1.0
    props = node.get("properties") or {}
    source_agent_id = node.get("source_agent_id") or props.get("source_agent_id") or props.get("agent_id")
    if not source_agent_id:
        return 0.98
    if source_agent_id == current_agent_id:
        return float(filters.get("same_agent_boost", 1.05))
    return float(filters.get("cross_agent_penalty", 0.96))


def _memory_scope(node: Dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    scope = props.get("memory_scope")
    if scope:
        return scope
    if node.get("label") == "entity":
        return classify_memory_scope([{
            "entity_type": props.get("entity_type"),
            "name": props.get("name"),
        }])
    entities = props.get("entities") or []
    if isinstance(entities, list):
        return classify_memory_scope(entities)
    return "general"


def _diversify_by_scope(results: List[Dict[str, Any]], top_k: int, query_scope: str) -> List[Dict[str, Any]]:
    if query_scope != "mixed" or top_k < 3:
        return results[:top_k]
    selected: List[Dict[str, Any]] = []
    seen = set()
    for wanted in ("product", "system"):
        candidate = next((item for item in results if item.get("memory_scope") == wanted), None)
        if candidate:
            selected.append(candidate)
            seen.add(candidate["id"])
    for item in results:
        if item["id"] in seen:
            continue
        selected.append(item)
        seen.add(item["id"])
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def _promote_operational_anchor(results: List[Dict[str, Any]], top_k: int, query: str) -> List[Dict[str, Any]]:
    if not _is_operational_query(query) or len(results) <= top_k:
        return results[:top_k]
    selected = results[:top_k]
    selected_ids = {item["id"] for item in selected}
    best_anchor = next(
        (
            item for item in results[: min(len(results), 30)]
            if item["id"] not in selected_ids
            and item.get("keyword_score", 0) >= 0.15
            and (
                (item.get("properties") or {}).get("layer") in HIGH_DENSITY_LAYERS
                or (item.get("properties") or {}).get("source") == "magma_operational_anchor"
            )
        ),
        None,
    )
    if not best_anchor:
        return selected
    weakest_index = min(
        range(len(selected)),
        key=lambda i: (
            0 if (selected[i].get("properties") or {}).get("layer") == "L0" else 1,
            selected[i].get("score", 0.0),
        ),
    )
    weakest = selected[weakest_index]
    if (weakest.get("properties") or {}).get("layer") == "L0" or best_anchor.get("score", 0) >= weakest.get("score", 0) * 0.72:
        selected[weakest_index] = best_anchor
        selected.sort(key=lambda item: item["score"], reverse=True)
    return selected[:top_k]


class MemorySearcher:
    """Semantic + lexical retrieval with lifecycle-aware ranking."""

    def __init__(self, store, encoder):
        self.store = store
        self.encoder = encoder

    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        include_archived = bool(filters.get("include_archived", False))
        label = filters.get("label")
        pool_size = int(filters.get("pool_size", max(top_k * 20, 1000)))
        property_filters = _property_filters(filters)
        intent = filters.get("intent") or detect_intent(query)
        include_related = bool(filters.get("include_related", False))
        related_limit = int(filters.get("related_limit", 3))
        include_versions = bool(filters.get("include_versions", False))
        version_limit = int(filters.get("version_limit", 2))
        query_entities = extract_entities(query)
        query_scope = classify_memory_scope(query_entities)

        query_embedding = None
        expected_dim = 0
        try:
            query_embedding = self.encoder.encode(query, normalize=True).astype("float32")
            expected_dim = int(query_embedding.shape[0])
        except Exception:
            query_embedding = None
        nodes = self.store.query_nodes_with_embeddings(
            label=label,
            limit=pool_size,
            include_archived=include_archived,
            property_filters=property_filters,
        )

        results = []
        for node in nodes:
            semantic_score = 0.0
            embedding = _embedding_from_blob(node.pop("embedding", None), expected_dim)
            if query_embedding is not None and embedding is not None:
                semantic_score = float(np.dot(query_embedding, embedding))
                semantic_score = max(semantic_score, 0.0)

            keyword_score = _keyword_score(query, node)
            if semantic_score <= 0 and keyword_score <= 0:
                continue

            lifecycle = _lifecycle_multiplier(node)
            agent_scope = _agent_scope_multiplier(node, filters)
            intent_scope = _intent_multiplier(node, intent)
            entity_scope = _entity_overlap_multiplier(query_entities, node)
            quality_scope = _memory_quality_multiplier(node)
            authority_scope = _operational_authority_multiplier(query, node, keyword_score)
            combined = (
                (semantic_score * 0.75 + keyword_score * 0.25)
                * lifecycle
                * agent_scope
                * intent_scope
                * entity_scope
                * quality_scope
                * authority_scope
            )
            if combined <= 0:
                continue

            provenance = _provenance(node)
            memory_scope = _memory_scope(node)
            node["score"] = round(float(combined), 6)
            node["semantic_score"] = round(float(semantic_score), 6)
            node["keyword_score"] = round(float(keyword_score), 6)
            node["lifecycle_multiplier"] = round(float(lifecycle), 6)
            node["agent_scope_multiplier"] = round(float(agent_scope), 6)
            node["intent_multiplier"] = round(float(intent_scope), 6)
            node["entity_overlap_multiplier"] = round(float(entity_scope), 6)
            node["memory_quality_multiplier"] = round(float(quality_scope), 6)
            node["operational_authority_multiplier"] = round(float(authority_scope), 6)
            node["query_entities"] = query_entities
            node["query_scope"] = query_scope
            node["query_intent"] = intent
            node["memory_scope"] = memory_scope
            node["retrieval_source"] = "memory"
            node["provenance"] = provenance
            node["source_agent_id"] = provenance.get("agent_id")
            node["memory_source"] = provenance.get("source")
            node["source_session_key"] = provenance.get("session_key")
            results.append(node)

        results.sort(key=lambda item: item["score"], reverse=True)
        results = _promote_operational_anchor(results, top_k, query)
        results = _diversify_by_scope(results, top_k, query_scope)
        if include_related:
            for item in results:
                item["related_context"] = self.store.get_related_context(
                    item["id"],
                    limit=related_limit,
                    relations=list(RELATED_RELATIONS),
                )
        if include_versions:
            for item in results:
                item["version_context"] = self.store.get_version_context(
                    item["id"],
                    limit=version_limit,
                )
        self.store.touch_nodes([item["id"] for item in results])
        return results
