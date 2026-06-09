"""Lightweight capture filtering for automatic MAGMA writes."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

# --- P1-1: Content Dedup Feature Flag ---
MAGMA_FEATURE_CONTENT_DEDUP = os.environ.get("MAGMA_FEATURE_CONTENT_DEDUP", "1") == "1"
DEDUP_THRESHOLD = float(os.environ.get("MAGMA_DEDUP_THRESHOLD", "0.85"))
DEDUP_NUM_PERM = int(os.environ.get("MAGMA_DEDUP_NUM_PERM", "64"))
DEDUP_WINDOW_HOURS = int(os.environ.get("MAGMA_DEDUP_WINDOW_HOURS", "24"))


NOISE_PATTERNS = (
    r"api rate limit|too many requests|429\b|限流",
    r"failovererror|agent couldn't generate a response",
    r"traceback|stack trace|exception:",
    r"^\s*(ok|好的|收到|嗯|好|可以|done)\s*$",
    r"健康检查：如果你能正常生成回复",
    r"magma-recall\.jsonl|short_command|drift_warning",
    r"验收结果|根因：|结论：|检查项",
)

STRONG_PATTERNS = (
    r"老板偏好|长期要求|以后.*优先|必须|不要|不能",
    r"决策|确认|最终方案|上线|回退|版本|配置|端口|接口变更|API\s*(?:端点|接口|变更|配置)|MCP",
    r"失败根因|根因|修复|故障|卡住|限流",
    r"SKU|尺码|身高|体重|商品|上架|抖音小店|价格|库存",
    r"core memory|core_memory|MAGMA|OpenClaw",
)


@dataclass
class CaptureDecision:
    should_capture: bool
    strength: str
    reasons: list[str]
    user_text: str
    assistant_text: str


def _matches(patterns: Iterable[str], text: str) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def _substantial(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return len(compact) >= 12


def classify_capture(
    user_text: str = "",
    assistant_text: str = "",
    suppression_patterns: Optional[list[str]] = None,
) -> CaptureDecision:
    """Decide whether an automatic capture is worth writing.

    This is intentionally cheap and deterministic. It keeps long-term memory
    cleaner without asking an LLM to judge every turn.
    """
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    combined = "\n".join(part for part in (user_text, assistant_text) if part)

    if not combined:
        return CaptureDecision(False, "none", ["empty"], user_text, assistant_text)

    noise_hits = _matches(NOISE_PATTERNS, combined)
    strong_hits = _matches(STRONG_PATTERNS, combined)
    suppression_hits = _matches(suppression_patterns or [], combined)

    if suppression_hits:
        return CaptureDecision(
            False,
            "suppressed",
            [f"suppression:{hit}" for hit in suppression_hits[:3]],
            user_text,
            assistant_text,
        )

    if noise_hits and not strong_hits:
        return CaptureDecision(False, "noise", [f"noise:{hit}" for hit in noise_hits[:3]], user_text, assistant_text)

    if strong_hits:
        return CaptureDecision(True, "strong", [f"strong:{hit}" for hit in strong_hits[:3]], user_text, assistant_text)

    if _substantial(user_text) or _substantial(assistant_text):
        return CaptureDecision(True, "normal", ["substantial_text"], user_text, assistant_text)

    return CaptureDecision(False, "noise", ["too_short"], user_text, assistant_text)


class MinHashDeduplicator:
    """P1-1: Content-level dedup using MinHash + LSH.

    Compares new content against recent nodes (within DEDUP_WINDOW_HOURS)
    using MinHash signatures with configurable similarity threshold.
    """

    def __init__(self, num_perm: int = DEDUP_NUM_PERM, threshold: float = DEDUP_THRESHOLD):
        self.num_perm = num_perm
        self.threshold = threshold
        self._lsh = None
        self._signatures: Dict[str, Any] = {}  # node_id -> MinHash
        self._last_refresh: float = 0
        self._refresh_interval: float = 300  # 5 min

    def _get_lsh(self):
        """Lazy-init LSH index."""
        if self._lsh is None:
            try:
                from datasketch import MinHashLSH
                self._lsh = MinHashLSH(
                    threshold=self.threshold,
                    num_perm=self.num_perm,
                )
            except ImportError:
                return None
        return self._lsh

    def _compute_minhash(self, text: str):
        """Compute MinHash signature for text."""
        try:
            from datasketch import MinHash
        except ImportError:
            return None
        m = MinHash(num_perm=self.num_perm)
        # Shingling: use character 3-grams for robustness with Chinese text
        text_lower = (text or "").lower()
        if len(text_lower) < 3:
            m.update(text_lower.encode("utf-8"))
            return m
        for i in range(len(text_lower) - 2):
            shingle = text_lower[i:i+3]
            m.update(shingle.encode("utf-8"))
        return m

    def refresh_from_store(self, store) -> int:
        """Load recent node content into LSH index.

        Returns number of nodes loaded.
        """
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return len(self._signatures)

        lsh = self._get_lsh()
        if lsh is None:
            return 0

        # Query recent nodes from store
        try:
            recent_nodes = store.query_nodes_with_embeddings(
                limit=5000,
                include_archived=False,
            )
        except Exception:
            return 0

        # Filter by recency window
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(hours=DEDUP_WINDOW_HOURS)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        loaded = 0
        for node in recent_nodes:
            nid = node.get("id", "")
            created = node.get("created_at", "")
            if created and created < cutoff_str:
                continue
            props = node.get("properties", {}) or {}
            content = props.get("content", "") or ""
            if not content or len(content) < 12:
                continue
            if nid in self._signatures:
                continue
            mh = self._compute_minhash(content)
            if mh is None:
                continue
            try:
                lsh.insert(nid, mh)
                self._signatures[nid] = mh
                loaded += 1
            except ValueError:
                # Already inserted
                self._signatures[nid] = mh

        self._last_refresh = now
        return loaded

    def find_duplicates(self, text: str, store=None) -> list:
        """Find nodes with content similar to text above threshold.

        Returns list of (node_id, estimated_similarity) tuples.
        """
        if not MAGMA_FEATURE_CONTENT_DEDUP:
            return []
        if not text or len(text.strip()) < 12:
            return []

        lsh = self._get_lsh()
        if lsh is None:
            return []

        # Refresh signatures if needed
        if store is not None:
            self.refresh_from_store(store)

        mh = self._compute_minhash(text)
        if mh is None:
            return []

        try:
            candidates = lsh.query(mh)
        except Exception:
            return []

        results = []
        for nid in candidates:
            sig = self._signatures.get(nid)
            if sig is None:
                continue
            est_jaccard = mh.jaccard(sig)
            if est_jaccard >= self.threshold:
                results.append((nid, float(est_jaccard)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def is_duplicate(self, text: str, store=None) -> Optional[str]:
        """Check if text is a duplicate. Returns best-matching node_id or None."""
        dupes = self.find_duplicates(text, store)
        if dupes:
            return dupes[0][0]
        return None


# Module-level singleton
deduplicator = MinHashDeduplicator()
