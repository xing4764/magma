"""P2-1: Local Reranker Module.

Uses a lightweight cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
to rerank search results as a complementary signal to RRF fusion.

Feature flag: MAGMA_FEATURE_LOCAL_RERANKER (default OFF — requires model download)
Performance target: rerank 10 results < 100ms
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("magma.reranker")

# Feature flag — default OFF (model must be downloaded first)
MAGMA_FEATURE_LOCAL_RERANKER = os.environ.get("MAGMA_FEATURE_LOCAL_RERANKER", "0") == "1"

# Reranker weights for score fusion
RERANKER_WEIGHT = 0.3
RRF_WEIGHT = 0.7

# Model configuration
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_NAME = os.environ.get("MAGMA_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
RERANKER_MAX_LENGTH = 512  # Max token length for cross-encoder input

# Lazy-loaded model singleton
_reranker_model = None
_reranker_load_attempted = False


def _get_reranker_model():
    """Lazy-load the cross-encoder model on first use."""
    global _reranker_model, _reranker_load_attempted
    if _reranker_load_attempted:
        return _reranker_model
    _reranker_load_attempted = True

    try:
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH)
        load_ms = (time.time() - t0) * 1000
        logger.info(f"Reranker model loaded: {RERANKER_MODEL_NAME} ({load_ms:.0f}ms)")
    except ImportError:
        logger.warning("sentence-transformers not installed; reranker disabled")
        _reranker_model = None
    except Exception as e:
        logger.warning(f"Failed to load reranker model: {e}")
        _reranker_model = None

    return _reranker_model


def _extract_node_text(node: Dict[str, Any]) -> str:
    """Extract searchable text from a result node for reranking."""
    props = node.get("properties", {}) or {}
    parts = []
    for key in ("content", "summary", "message", "title", "name"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts) if parts else str(node.get("id", ""))


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank results using cross-encoder and fuse with RRF scores.

    Args:
        query: original search query
        results: list of result dicts from RRF search
        top_k: optional limit on results to rerank (default: all)

    Returns:
        results with reranker_score added to score_breakdown, re-sorted by fused score.
    """
    if not MAGMA_FEATURE_LOCAL_RERANKER:
        return results
    if not results:
        return results

    model = _get_reranker_model()
    if model is None:
        return results

    # Limit candidates to rerank
    candidates = results[:top_k] if top_k else results
    remainder = results[top_k:] if top_k and len(results) > top_k else []

    # Build query-document pairs
    pairs = []
    for item in candidates:
        doc_text = _extract_node_text(item)
        pairs.append((query, doc_text))

    # Run cross-encoder scoring
    t0 = time.time()
    try:
        raw_scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"Reranker prediction failed: {e}")
        return results
    rerank_ms = (time.time() - t0) * 1000
    logger.info(f"Reranked {len(pairs)} results in {rerank_ms:.1f}ms")

    # Normalize reranker scores to [0, 1]
    import numpy as np
    scores_array = np.array(raw_scores, dtype=np.float32)
    if scores_array.max() > scores_array.min():
        norm_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
    else:
        norm_scores = np.ones_like(scores_array) * 0.5

    # Fuse: final = RERANKER_WEIGHT * reranker + RRF_WEIGHT * original_rrf
    for i, item in enumerate(candidates):
        reranker_score = float(norm_scores[i])
        original_score = item.get("score", 0.0)
        fused = RERANKER_WEIGHT * reranker_score + RRF_WEIGHT * original_score

        item["score"] = round(fused, 6)
        item["score_breakdown"]["reranker_score"] = round(reranker_score, 4)
        item["score_breakdown"]["reranker_fused_score"] = round(fused, 4)
        item["score_breakdown"]["reranker_weight"] = RERANKER_WEIGHT
        item["score_breakdown"]["rrf_weight"] = RRF_WEIGHT
        item["reranker_score"] = round(reranker_score, 6)
        item["reranker_applied"] = True

    # Sort by fused score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Merge remainder (not reranked, appended in original order)
    return candidates + remainder


def is_reranker_available() -> bool:
    """Check if reranker model can be loaded (without actually loading it)."""
    if not MAGMA_FEATURE_LOCAL_RERANKER:
        return False
    try:
        from sentence_transformers import CrossEncoder
        return True
    except ImportError:
        return False


def get_reranker_status() -> Dict[str, Any]:
    """Return reranker module status for diagnostics."""
    return {
        "feature_flag": MAGMA_FEATURE_LOCAL_RERANKER,
        "model_name": RERANKER_MODEL_NAME,
        "model_loaded": _reranker_model is not None,
        "available": is_reranker_available(),
        "weights": {
            "reranker": RERANKER_WEIGHT,
            "rrf": RRF_WEIGHT,
        },
    }
