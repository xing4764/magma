"""Shared retrieval logic for MAGMA API and MCP entrypoints."""

import json
import logging
import math
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from magma.entities import classify_memory_scope, extract_entities

logger = logging.getLogger("magma.search")

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


def _keyword_score(query: str, node: Dict[str, Any], searchable_text: str = None, query_terms: List[str] = None) -> float:
    if searchable_text is None:
        searchable = _node_searchable_text(node)
    else:
        searchable = searchable_text
    query_lower = query.lower()
    score = 0.0

    if query_lower and query_lower in searchable:
        score += 0.35
    terms = query_terms if query_terms is not None else _query_terms(query)
    for term in terms:
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
    kind = props.get("kind")
    source = props.get("source")
    label = node.get("label") or ""
    # L1 kind-specific weighting: current_state > decision > fact > L0
    if layer == "L1":
        kind_weights = {
            "current_state": 1.30,
            "decision": 1.25,
            "fact": 1.18,
        }
        return kind_weights.get(kind, 1.20)
    if layer in {"summary", "decision", "fact", "current_state"}:
        return 1.2
    if layer == "ops_anchor" or source == "magma_operational_anchor":
        return 1.08
    if layer == "entity_anchor" or label == "entity":
        return 0.35
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
        return max(multiplier, 0.85)
    if role != "user":
        return 1.0
    question_terms = (
        "?", "\uff1f", "\u4e3a\u4ec0\u4e48", "\u600e\u4e48", "\u5982\u4f55",
        "\u54ea\u4e2a", "\u4ec0\u4e48", "\u662f\u5426", "\u80fd\u4e0d\u80fd",
        "\u53ef\u4ee5\u95ee", "\u6d4b\u8bd5",
    )
    if any(term in content for term in question_terms):
        return 0.85
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


# --- Graph Walk Cache (module-level, 5-min TTL) ---
_graph_walk_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
_GRAPH_WALK_TTL = 300  # 5 minutes
_GRAPH_WALK_MAX_CACHE = 512


def _cache_key(node_ids: List[str], hops: int) -> str:
    return "|".join(sorted(node_ids)) + f":h{hops}"


def _cache_get(key: str) -> Optional[Dict[str, float]]:
    entry = _graph_walk_cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts > _GRAPH_WALK_TTL:
        del _graph_walk_cache[key]
        return None
    return data


def _cache_put(key: str, data: Dict[str, float]):
    if len(_graph_walk_cache) >= _GRAPH_WALK_MAX_CACHE:
        # Evict oldest entries
        oldest_keys = sorted(_graph_walk_cache, key=lambda k: _graph_walk_cache[k][0])[:64]
        for k in oldest_keys:
            _graph_walk_cache.pop(k, None)
    _graph_walk_cache[key] = (time.time(), data)


def graph_walk(store, source_node_ids: List[str], hops: int = 2, max_neighbors_per_hop: int = 15) -> Dict[str, float]:
    """Walk the graph from source nodes, returning {node_id: graph_score}.

    Score: 1-hop = 0.3, 2-hop = 0.15.  Uses SQLite queries (not in-memory graph).
    Results are cached for 5 minutes.
    """
    if not source_node_ids:
        return {}

    cache_key = _cache_key(source_node_ids, hops)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # BFS-style traversal
    visited: Set[str] = set(source_node_ids)
    result: Dict[str, float] = {}  # node_id -> graph score
    current_frontier = set(source_node_ids)
    hop_scores = {1: 0.3, 2: 0.15}
    # Relation-type weights for graph walk scoring
    relation_weights = {
        "depends_on": 1.3,
        "caused_by": 1.3,
        "responded_by": 1.2,
        "same_as": 1.15,
        "mentions_entity": 1.0,
        "extracted_fact": 1.1,
        "related_to": 1.0,
    }

    for hop in range(1, hops + 1):
        if not current_frontier:
            break
        next_frontier: Set[str] = set()
        try:
            neighbors_with_rel = store.get_neighbor_ids_with_relations(list(current_frontier))
            for item in neighbors_with_rel:
                nid = item["id"]
                rel = item["relation"]
                if nid and nid not in visited:
                    visited.add(nid)
                    next_frontier.add(nid)
                    base_score = hop_scores.get(hop, 0.1)
                    rel_weight = relation_weights.get(rel, 1.0)
                    weighted_score = base_score * rel_weight
                    if nid not in result or result[nid] < weighted_score:
                        result[nid] = weighted_score
        except Exception as e:
            logger.warning(f"graph_walk hop {hop} error: {e}")
            break
        current_frontier = next_frontier

    # Only cache non-empty results
    if result:
        _cache_put(cache_key, result)
    return result


class MemorySearcher:
    """Semantic + lexical retrieval with lifecycle-aware ranking."""

    def __init__(self, store, encoder, faiss_index=None):
        self.store = store
        self.encoder = encoder
        self.faiss_index = faiss_index

    def _build_faiss_if_needed(self):
        """Build FAISS index from store if not already built."""
        if self.faiss_index is None:
            return
        if self.faiss_index.is_available:
            return
        try:
            all_nodes = self.store.query_nodes_with_embeddings(limit=999999, include_archived=True)
            entries = []
            for node in all_nodes:
                blob = node.pop("embedding", None)
                if blob:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.ndim == 1 and vec.shape[0] > 0:
                        entries.append((node["id"], vec))
            self.faiss_index.build_from_embeddings(entries)
            logger.info(f"FAISS index auto-built with {len(entries)} vectors")
        except Exception as e:
            logger.warning(f"FAISS auto-build failed: {e}")

    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        include_archived = bool(filters.get("include_archived", False))
        label = filters.get("label")
        pool_size = int(filters.get("pool_size", 99999))
        property_filters = _property_filters(filters)
        intent = filters.get("intent") or detect_intent(query)
        include_related = bool(filters.get("include_related", False))
        related_limit = int(filters.get("related_limit", 3))
        include_versions = bool(filters.get("include_versions", False))
        version_limit = int(filters.get("version_limit", 2))
        query_entities = extract_entities(query)
        query_scope = classify_memory_scope(query_entities)

        # --- Pre-compute shared query data once ---
        query_lower = query.lower()
        query_terms = _query_terms(query)
        query_embedding = None
        expected_dim = 0
        try:
            query_embedding = self.encoder.encode(query, normalize=True).astype("float32")
            expected_dim = int(query_embedding.shape[0])
        except Exception:
            query_embedding = None

        # --- Try FAISS first for semantic scores ---
        faiss_semantic: Dict[str, float] = {}
        faiss_used = False
        if query_embedding is not None and self.faiss_index is not None:
            self._build_faiss_if_needed()
            if self.faiss_index.is_available and self.faiss_index.dimension == expected_dim:
                try:
                    # Get more candidates from FAISS to account for keyword-only matches
                    faiss_top_k = min(max(top_k * 5, 50), self.faiss_index.count)
                    if faiss_top_k > 0:
                        faiss_results = self.faiss_index.search(query_embedding, faiss_top_k)
                        for nid, score in faiss_results:
                            faiss_semantic[nid] = max(score, 0.0)
                        faiss_used = True
                except Exception as e:
                    logger.warning(f"FAISS search failed, falling back to brute force: {e}")
                    faiss_semantic = {}

        if faiss_used:
            # FAISS provides semantic scores - use lighter query without BLOBs
            nodes = self.store.query_nodes_properties_only(
                label=label,
                limit=pool_size,
                include_archived=include_archived,
                property_filters=property_filters,
            )
        else:
            nodes = self.store.query_nodes_with_embeddings(
                label=label,
                limit=pool_size,
                include_archived=include_archived,
                property_filters=property_filters,
            )

        # --- Batch decode embeddings + batch cosine (fallback or complement) ---
        semantic_scores: Dict[str, float] = {}
        if faiss_used:
            # FAISS provides all semantic scores; no BLOBs loaded
            semantic_scores.update(faiss_semantic)
        else:
            # Brute-force fallback (original path)
            if query_embedding is not None and nodes:
                valid_embeddings = []
                valid_ids = []
                for node in nodes:
                    blob = node.pop("embedding", None)
                    if blob:
                        vec = np.frombuffer(blob, dtype=np.float32)
                        if vec.shape[0] == expected_dim:
                            valid_embeddings.append(vec)
                            valid_ids.append(node["id"])
                            continue
                    node.pop("embedding", None)
                if valid_embeddings:
                    emb_matrix = np.array(valid_embeddings, dtype=np.float32)
                    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                    emb_normed = emb_matrix / np.maximum(norms, 1e-10)
                    scores_raw = emb_normed @ query_embedding
                    for nid, sc in zip(valid_ids, scores_raw):
                        semantic_scores[nid] = max(float(sc), 0.0)

        # --- Pre-compute searchable texts ---
        searchable_texts = {}
        for node in nodes:
            searchable_texts[node["id"]] = _node_searchable_text(node)

        # --- Score all nodes ---
        results = []
        for node in nodes:
            nid = node["id"]
            semantic_score = semantic_scores.get(nid, 0.0)
            searchable = searchable_texts.get(nid, "")

            # Fast keyword score using pre-computed data
            keyword_score = _keyword_score(query, node, searchable, query_terms)

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

            # --- P1-A: anti-kill floor ---
            base_score = semantic_score * 0.75 + keyword_score * 0.25
            original_combined = combined
            if base_score >= 0.5 and combined < base_score * 0.6:
                combined = base_score * 0.6

            provenance = _provenance(node)
            memory_scope = _memory_scope(node)
            node["score"] = round(float(combined), 6)
            node["score_breakdown"] = {
                "base": round(base_score, 4),
                "lifecycle": round(float(lifecycle), 4),
                "agent_scope": round(float(agent_scope), 4),
                "intent_scope": round(float(intent_scope), 4),
                "entity_scope": round(float(entity_scope), 4),
                "quality_scope": round(float(quality_scope), 4),
                "authority_scope": round(float(authority_scope), 4),
                "combined_before_floor": round(float(original_combined), 4),
                "floor_applied": round(float(combined), 4) != round(float(original_combined), 4),
            }
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

        # --- P0: Graph Walk Engine (HippoRAG-style) ---
        # Take top-K candidates from FAISS+keyword scoring, walk 2 hops on graph
        if len(results) > 0:
            graph_seed_size = min(max(top_k * 2, 10), len(results))
            seed_ids = [item["id"] for item in results[:graph_seed_size]]
            graph_scores = graph_walk(self.store, seed_ids, hops=2)

            if graph_scores:
                existing_ids = {item["id"] for item in results}
                # Re-score existing results with graph boost
                for item in results:
                    gid = item["id"]
                    if gid in graph_scores:
                        original_score = item["score"]
                        graph_score = graph_scores[gid]
                        fused = original_score * 0.7 + graph_score * 0.3
                        item["score"] = round(fused, 6)
                        item["graph_boost"] = True
                        item["graph_score"] = graph_score
                        item["score_breakdown"]["graph_score"] = round(graph_score, 4)
                        item["score_breakdown"]["fused_score"] = round(fused, 4)

                # Fetch graph-discovered nodes not in original results
                new_node_ids = [nid for nid in graph_scores if nid not in existing_ids]
                if new_node_ids:
                    fetched = []
                    for nid in new_node_ids[:30]:  # Cap fetched neighbors
                        node = self.store.get_node(nid)
                        if node and node.get("status") != "deleted":
                            fetched.append(node)
                    # Score fetched graph neighbors
                    for node in fetched:
                        nid = node["id"]
                        searchable = _node_searchable_text(node)
                        keyword_score = _keyword_score(query, node, searchable, query_terms)
                        semantic_score = faiss_semantic.get(nid, 0.0)
                        base_score = semantic_score * 0.75 + keyword_score * 0.25
                        lifecycle = _lifecycle_multiplier(node)
                        agent_scope = _agent_scope_multiplier(node, filters)
                        intent_scope = _intent_multiplier(node, intent)
                        entity_scope = _entity_overlap_multiplier(query_entities, node)
                        quality_scope = _memory_quality_multiplier(node)
                        authority_scope = _operational_authority_multiplier(query, node, keyword_score)
                        graph_score = graph_scores[nid]
                        original_combined = (
                            base_score
                            * lifecycle * agent_scope * intent_scope
                            * entity_scope * quality_scope * authority_scope
                        )
                        fused = original_combined * 0.7 + graph_score * 0.3
                        if fused <= 0:
                            continue
                        provenance = _provenance(node)
                        memory_scope = _memory_scope(node)
                        node["score"] = round(float(fused), 6)
                        node["score_breakdown"] = {
                            "base": round(base_score, 4),
                            "lifecycle": round(float(lifecycle), 4),
                            "agent_scope": round(float(agent_scope), 4),
                            "intent_scope": round(float(intent_scope), 4),
                            "entity_scope": round(float(entity_scope), 4),
                            "quality_scope": round(float(quality_scope), 4),
                            "authority_scope": round(float(authority_scope), 4),
                            "graph_score": round(graph_score, 4),
                            "fused_score": round(float(fused), 4),
                        }
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
                        node["retrieval_source"] = "graph_walk"
                        node["graph_boost"] = True
                        node["graph_score"] = graph_score
                        node["provenance"] = provenance
                        node["source_agent_id"] = provenance.get("agent_id")
                        node["memory_source"] = provenance.get("source")
                        node["source_session_key"] = provenance.get("session_key")
                        results.append(node)

                # Re-sort with graph-boosted scores
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
