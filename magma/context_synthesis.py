"""P2: Context Synthesis — Transform graph topology into ordered linear narrative.

Implements the context synthesis module from arxiv:2601.03236:
- Topological ordering: causal edges (cause before effect), temporal edges (earlier first)
- Provenance scaffolding: [REF:node_id] references for LLM citation
- Dynamic token budget: allocate tokens based on query complexity
- Narrative generation: ordered, sourced text for LLM consumption
"""

import logging
import re
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("magma.context_synthesis")

# Token budget tiers (approximate: 1 token ≈ 4 chars for English, ~2 chars for Chinese)
TOKEN_BUDGETS = {
    "simple": 500,      # yes/no, factual lookup
    "moderate": 1000,   # single-hop reasoning
    "complex": 2000,    # multi-hop, causal chains, comparisons
}

# Intent → complexity mapping
INTENT_COMPLEXITY = {
    "why": "complex",     # causal reasoning needs more context
    "when": "moderate",   # temporal ordering
    "entity": "moderate", # entity-focused
    "general": "simple",  # general lookup
}

# Result count thresholds for complexity upgrade
COUNT_THRESHOLDS = {
    "simple_to_moderate": 5,
    "moderate_to_complex": 10,
}


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string in common formats."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def estimate_token_budget(intent: Dict[str, Any], result_count: int) -> int:
    """Estimate token budget based on intent complexity and result count.

    Args:
        intent: intent dict with 'primary' key
        result_count: number of search results

    Returns:
        token budget (int)
    """
    primary = intent.get("primary", "general")
    complexity = INTENT_COMPLEXITY.get(primary, "simple")

    # Upgrade complexity based on result count
    if complexity == "simple" and result_count >= COUNT_THRESHOLDS["simple_to_moderate"]:
        complexity = "moderate"
    if complexity == "moderate" and result_count >= COUNT_THRESHOLDS["moderate_to_complex"]:
        complexity = "complex"

    return TOKEN_BUDGETS[complexity]


def topological_sort(results: List[Dict[str, Any]], store=None) -> List[Dict[str, Any]]:
    """Sort results in topological order respecting causal and temporal relations.

    Rules:
    1. Causal edges: cause BEFORE effect (caused_by means effect → cause)
    2. Temporal: earlier events before later events (by created_at)
    3. Default: preserve original score-based order

    Args:
        results: list of result dicts with id, properties, score, etc.
        store: optional graph store for fetching edge information

    Returns:
        topologically sorted list of results
    """
    if len(results) <= 1:
        return list(results)

    # Build index
    result_map = {r["id"]: r for r in results}
    result_ids = set(result_map.keys())

    # --- Build causal ordering constraints ---
    # caused_by edge: target caused_by source → source should come BEFORE target
    # i.e., cause before effect
    causal_before: Dict[str, Set[str]] = defaultdict(set)  # node → set of nodes that must come before it

    # Try to get causal edges from store
    if store:
        for r in results:
            nid = r["id"]
            try:
                causal_edges = store.get_causal_edges(nid, direction="both")
                for edge in causal_edges:
                    src = edge.get("source_id")
                    tgt = edge.get("target_id")
                    rel = edge.get("relation", "")
                    if rel == "caused_by" and src in result_ids and tgt in result_ids:
                        # tgt caused_by src → src (cause) before tgt (effect)
                        causal_before[tgt].add(src)
            except Exception:
                pass

    # Also check edges embedded in result data
    for r in results:
        nid = r["id"]
        # Check beam_path for causal ordering hints
        beam_path = r.get("beam_path", [])
        if beam_path and len(beam_path) >= 2:
            for i in range(len(beam_path) - 1):
                if beam_path[i] in result_ids and beam_path[i + 1] in result_ids:
                    causal_before[beam_path[i + 1]].add(beam_path[i])

        # Check causal_edges field
        causal_edges = r.get("causal_edges", [])
        for edge in causal_edges:
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            if src in result_ids and tgt in result_ids:
                causal_before[tgt].add(src)

        # Check causal_chain field
        causal_chain = r.get("causal_chain", [])
        for chain in causal_chain:
            if isinstance(chain, list):
                for i in range(len(chain) - 1):
                    cause_id = chain[i].get("id") if isinstance(chain[i], dict) else chain[i]
                    effect_id = chain[i + 1].get("id") if isinstance(chain[i + 1], dict) else chain[i + 1]
                    if cause_id in result_ids and effect_id in result_ids:
                        causal_before[effect_id].add(cause_id)

    # --- Build temporal ordering constraints ---
    # Earlier events before later events
    temporal_pairs: List[Tuple[str, str]] = []
    time_sorted = sorted(
        results,
        key=lambda r: _parse_time((r.get("properties") or {}).get("created_at")) or datetime.min
    )
    for i in range(len(time_sorted)):
        for j in range(i + 1, min(i + 3, len(time_sorted))):  # Only adjacent-ish pairs
            earlier_id = time_sorted[i]["id"]
            later_id = time_sorted[j]["id"]
            earlier_time = _parse_time((time_sorted[i].get("properties") or {}).get("created_at"))
            later_time = _parse_time((time_sorted[j].get("properties") or {}).get("created_at"))
            if earlier_time and later_time and earlier_time < later_time:
                # Only add temporal constraint if no conflicting causal constraint
                if later_id not in causal_before.get(earlier_id, set()):
                    causal_before[later_id].add(earlier_id)

    # --- Kahn's algorithm for topological sort ---
    in_degree: Dict[str, int] = {nid: 0 for nid in result_ids}
    for node, predecessors in causal_before.items():
        for pred in predecessors:
            if pred in result_ids:
                in_degree[node] = in_degree.get(node, 0) + 1

    # Queue: nodes with no predecessors, ordered by score (highest first)
    queue = deque()
    zero_in_degree = sorted(
        [nid for nid in result_ids if in_degree.get(nid, 0) == 0],
        key=lambda nid: result_map[nid].get("score", 0),
        reverse=True,
    )
    queue.extend(zero_in_degree)

    sorted_ids: List[str] = []
    while queue:
        nid = queue.popleft()
        sorted_ids.append(nid)
        # Find nodes that depend on this one
        for node, predecessors in causal_before.items():
            if nid in predecessors:
                in_degree[node] -= 1
                if in_degree[node] == 0:
                    queue.append(node)

    # Handle cycles or disconnected nodes (append remaining by score)
    remaining = sorted(
        [nid for nid in result_ids if nid not in sorted_ids],
        key=lambda nid: result_map[nid].get("score", 0),
        reverse=True,
    )
    sorted_ids.extend(remaining)

    return [result_map[nid] for nid in sorted_ids if nid in result_map]


def build_references(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build provenance scaffolding: attach reference_id to each result.

    Format: [REF:<short_id>]

    Args:
        results: list of result dicts

    Returns:
        list of reference dicts with id, ref_id, and source info
    """
    references = []
    for i, r in enumerate(results):
        node_id = r.get("id", f"unknown_{i}")
        # Create a short reference ID
        short_id = node_id if len(node_id) <= 30 else node_id[:28] + ".."
        ref = {
            "index": i + 1,
            "ref_id": f"REF:{short_id}",
            "node_id": node_id,
            "label": r.get("label", ""),
            "score": r.get("score", 0),
            "retrieval_source": r.get("retrieval_source", "memory"),
            "provenance": r.get("provenance", {}),
        }
        references.append(ref)
    return references


def _extract_content_text(result: Dict[str, Any]) -> str:
    """Extract readable text content from a result node."""
    props = result.get("properties", {}) or {}

    # Priority: content > summary > message > title > name
    for key in ("content", "summary", "message", "title", "name"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallback: combine all string properties
    parts = []
    for key, value in props.items():
        if isinstance(value, str) and value.strip() and key not in ("embedding",):
            parts.append(value.strip())
    return " | ".join(parts[:3]) if parts else ""


def _format_node_narrative(result: Dict[str, Any], ref_id: str, token_budget_per_node: int) -> str:
    """Format a single node into narrative text with provenance reference."""
    props = result.get("properties", {}) or {}
    label = result.get("label", "unknown")
    content = _extract_content_text(result)

    # Truncate content to fit budget
    max_chars = token_budget_per_node * 2  # rough: 1 token ≈ 2 chars for Chinese
    if len(content) > max_chars:
        content = content[:max_chars - 3] + "..."

    # Format role/layer info
    role = props.get("role", "")
    layer = props.get("layer", "")
    source = props.get("source", "")
    created_at = props.get("created_at", "")

    meta_parts = []
    if role:
        meta_parts.append(f"role={role}")
    if layer:
        meta_parts.append(f"layer={layer}")
    if source:
        meta_parts.append(f"src={source}")
    if created_at:
        meta_parts.append(f"time={created_at}")
    meta = f" [{', '.join(meta_parts)}]" if meta_parts else ""

    # Add causal info if available
    causal_info = ""
    causal_edges = result.get("causal_edges", [])
    if causal_edges:
        causes = [e.get("rationale", "") for e in causal_edges if e.get("relation") == "caused_by"]
        if causes:
            causal_info = f" (原因: {'; '.join(causes[:2])})"

    return f"{ref_id} [{label}]{meta}: {content}{causal_info}"


def synthesize_narrative(
    results: List[Dict[str, Any]],
    query: str,
    intent: Dict[str, Any],
    store=None,
    priority: str = "normal",
) -> Dict[str, Any]:
    """Synthesize ordered linear narrative from search results.

    Main entry point for P2 context synthesis.

    Args:
        results: search results from MemorySearcher.query()
        query: original query string
        intent: intent dict from classify_intent()
        store: optional graph store for edge queries
        priority: P2-6 priority level ("critical", "normal", "low")

    Returns:
        dict with:
        - narrative: ordered linear narrative text
        - references: list of reference dicts
        - token_budget: allocated token budget
        - tokens_used: estimated tokens used
        - ordering: ordering method applied
    """
    if not results:
        return {
            "narrative": "",
            "references": [],
            "token_budget": 0,
            "tokens_used": 0,
            "ordering": "none",
        }

    # Step 1: Topological sort
    ordered_results = topological_sort(results, store)

    # Step 2: Estimate token budget
    token_budget = estimate_token_budget(intent, len(ordered_results))

    # Step 3: Build references
    references = build_references(ordered_results)

    # Step 4: Generate narrative within token budget
    # Allocate tokens per node (with buffer for header/footer)
    header_reserve = 80
    per_node_budget = max((token_budget - header_reserve) // max(len(ordered_results), 1), 50)

    narrative_lines = []
    tokens_used = header_reserve

    # Header
    primary_intent = intent.get("primary", "general")
    intent_labels = {
        "why": "因果推理",
        "when": "时间线",
        "entity": "实体查询",
        "general": "综合查询",
    }
    header = f"## 记忆检索结果 ({intent_labels.get(primary_intent, '综合查询')})\n"
    narrative_lines.append(header)

    # Body: ordered narrative
    # P1-4/P1-6: Curated/full nodes get full content, summary nodes get truncated
    curated_ids = {r["id"] for r in ordered_results if r.get("curated")}
    has_tiers = any(r.get("tier") for r in ordered_results)

    # P2-4: Sentence compression for summary-tier nodes
    _compress_enabled = False
    try:
        from magma.sentence_compress import is_compress_enabled, apply_compression_to_node
        _compress_enabled = is_compress_enabled()
    except ImportError:
        pass

    for i, (result, ref) in enumerate(zip(ordered_results, references)):
        ref_id = ref["ref_id"]
        tier = result.get("tier", "full")
        is_curated = result.get("id") in curated_ids

        # P2-4: Apply sentence compression to summary-tier nodes
        if _compress_enabled and has_tiers and tier == "summary" and not is_curated:
            result = apply_compression_to_node(result, query=query, top_k=4)

        # For summary-tier or non-curated nodes, use smaller budget
        if has_tiers and tier == "summary" and not is_curated:
            node_budget = per_node_budget // 2
        else:
            node_budget = per_node_budget

        node_text = _format_node_narrative(result, ref_id, node_budget)
        estimated_tokens = len(node_text) // 2  # rough estimate

        if tokens_used + estimated_tokens > token_budget:
            # Truncate or skip
            remaining_budget = token_budget - tokens_used
            if remaining_budget > 50:
                # Truncate
                max_chars = remaining_budget * 2
                node_text = node_text[:max_chars - 3] + "..."
                narrative_lines.append(node_text)
            narrative_lines.append(f"\n... (共 {len(ordered_results)} 条结果，已展示 {i + 1} 条)")
            break

        # P1-4/P1-6: Add tier/curated marker
        marker = ""
        if is_curated:
            marker = " ★"
        elif has_tiers and tier == "summary":
            marker = " ◇"
        narrative_lines.append(node_text + marker)
        tokens_used += estimated_tokens

    # Footer with reference count
    narrative_lines.append(f"\n---\n共 {len(references)} 条记忆证据，token预算: {token_budget}")

    narrative = "\n".join(narrative_lines)

    # P2-6: Token budget priority control
    usage_ratio = round(min(tokens_used, token_budget) / max(token_budget, 1), 4)

    result = {
        "narrative": narrative,
        "references": references,
        "token_budget": token_budget,
        "tokens_used": min(tokens_used, token_budget),
        "ordering": "topological" if len(results) > 1 else "single",
        "complexity": INTENT_COMPLEXITY.get(primary_intent, "simple"),
    }

    if priority == "critical":
        # P2-6: P0 决策场景 — 不暴露 token_usage_ratio，防止 agent 草率决策
        pass  # omit token_usage_ratio
    elif priority == "low":
        # P2-6: 低优先级 — ratio > 0.6 时提示收尾
        result["token_usage_ratio"] = usage_ratio
        if usage_ratio > 0.6:
            result["budget_warning"] = (
                f"⚠️ Token 使用率已达 {usage_ratio:.0%}，建议精简回复内容。"
            )
    else:
        # normal — 正常返回
        result["token_usage_ratio"] = usage_ratio

    return result
