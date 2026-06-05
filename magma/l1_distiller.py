"""Deterministic L1 distillation for high-value MAGMA memories.

L1 is the stable memory layer. It should contain decisions, preferences,
project state, lessons, pending actions, and concrete facts. It must not absorb
diagnostic fragments, acknowledgements, or transient debug chatter.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from magma.capture_policy import classify_capture


KIND_IMPORTANCE = {
    "decision": 0.88,
    "preference": 0.90,
    "project_state": 0.86,
    "pending_action": 0.84,
    "lesson": 0.86,
    "fact": 0.80,
}

KIND_PATTERNS = {
    "preference": (
        r"老板偏好|长期要求|以后.*优先|不要.*花哨|必须.*真实可用|产品化|要的是.*更强大|不追求.*开源",
    ),
    "decision": (
        r"决定|确认|最终方案|采用|改成|切到|回退|上线|默认|不允许|不能|不要擅自|给你权限|正式|已切换",
    ),
    "project_state": (
        r"已完成|已上线|已推送|已修复|已重启|已经.*(装好|集成|完成|上线)|当前状态|现在是|测试通过|验收通过|GREEN|通过|生产可用",
    ),
    "pending_action": (
        r"下一步|待办|需要继续|准备做|开始做|要做的是|优先做|还需要|需要我.*吗|要不要.*",
    ),
    "lesson": (
        r"教训|根因|原因是|问题在于|不要再|下次.*先|跑偏|卡住|限流|超时|失败",
    ),
    "fact": (
        r"MAGMA|OpenClaw|Gateway|MCP|Core Memory|core_memory|embedding|reranker|SQLite|FAISS|API|端口|版本|SKU|尺码|抖店|上架|agentmemory|MoneyPrinterTurbo|CloakBrowser",
    ),
}

NOISE_MARKERS = (
    "magma_product_benchmark",
    "benchmark_marker=",
    "健康检查：如果你能正常生成回复",
    "分析完了",
    "说实话",
    "我的意思就是",
    "你的意思是",
    "不是让我",
    "不是打了个逗号吗",
    "后面我不是打了个逗号吗",
    "验收结果",
    "检查项",
    "测试结果",
    "API 层",
    "short_command 字段",
    "诊断报告",
    "debug",
    "调试",
    "[Inter-session message]",
)

LOW_VALUE_PATTERNS = (
    r"^(收到|好的|好|嗯|可以|明白|了解|ok|done)[。！!,.，\s✅]*$",
    r"^对[，,。]?(我)?读错了[。！!,.，\s]*$",
    r"^已记录\s*[✅。！!,.，\s]*$",
    r"^全部字段\s*[✅。！!,.，\s]*(验收通过)?[。！!,.，\s]*$",
)

SUBSTANCE_TERMS = (
    "MAGMA",
    "OpenClaw",
    "Gateway",
    "MCP",
    "Core Memory",
    "embedding",
    "reranker",
    "SQLite",
    "FAISS",
    "API",
    "端口",
    "版本",
    "SKU",
    "尺码",
    "抖店",
    "上架",
    "公开仓库",
    "README",
    "agentmemory",
    "MoneyPrinterTurbo",
    "CloakBrowser",
    "doctor",
    "benchmark",
    "recall",
    "L1",
)


@dataclass
class L1Candidate:
    node_id: str
    kind: str
    title: str
    content: str
    source_l0_ids: list[str]
    source_agent_id: str
    department: str
    confidence: float


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _content(node: Dict[str, Any]) -> str:
    props = node.get("properties") or {}
    return str(props.get("content") or props.get("summary") or props.get("message") or "").strip()


def _normalize(text: str) -> str:
    text = re.sub(r"^\[[A-Za-z]{3}\s+\d{4}-\d{2}-\d{2}.*?\]\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_noise(text: str) -> bool:
    if not text or len(text) < 12:
        return True
    if any(marker in text for marker in NOISE_MARKERS):
        return True
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS):
        return True
    return classify_capture(text, "").should_capture is False


def _has_substance(text: str, kind: str) -> bool:
    if any(term.lower() in text.lower() for term in SUBSTANCE_TERMS):
        return True
    if kind in {"preference", "decision", "pending_action", "lesson"} and len(text) >= 28:
        return True
    if kind == "project_state" and len(text) >= 32:
        return True
    return False


def classify_l1_kind(text: str) -> Optional[str]:
    """Classify high-value text into an L1 memory kind."""
    for kind in ("preference", "decision", "project_state", "pending_action", "lesson", "fact"):
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in KIND_PATTERNS[kind]):
            return kind
    return None


def _title(text: str) -> str:
    first = re.split(r"[。！？?!]\s*", text, maxsplit=1)[0].strip()
    return first[:60] if first else text[:60]


def build_l1_candidate(node: Dict[str, Any]) -> Optional[L1Candidate]:
    """Build an L1 candidate from one L0 node, or None for low-value text."""
    props = node.get("properties") or {}
    if props.get("layer") != "L0":
        return None

    text = _normalize(_content(node))
    role = str(props.get("role") or "").lower()
    if role == "assistant" and re.match(r"^(收到|好的|明白|对[，,。]?(我)?读错了)", text, flags=re.IGNORECASE):
        return None
    if _is_noise(text):
        return None

    kind = classify_l1_kind(text)
    if not kind or not _has_substance(text, kind):
        return None
    if kind == "fact":
        if role == "user" and re.search(r"(吗|么|什么|为什么|怎么|如何|是不是|能不能)[？?]?$", text):
            return None
        if role == "fact" and len(text) < 24:
            return None

    source_agent = node.get("source_agent_id") or props.get("source_agent_id") or props.get("agent_id") or ""
    department = node.get("department") or props.get("department") or ""
    digest = _stable_hash(f"{kind}:{source_agent}:{text[:240]}")
    return L1Candidate(
        node_id=f"l1:{kind}:{digest}",
        kind=kind,
        title=_title(text),
        content=text[:500],
        source_l0_ids=[node["id"]],
        source_agent_id=source_agent,
        department=department,
        confidence=0.78 if kind == "fact" else 0.86,
    )


def _recent_l0_nodes(store, hours: int, limit: int, source_agent_id: str = None) -> list[dict[str, Any]]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    nodes = store.query_nodes_properties_only(
        limit=max(limit * 4, limit),
        include_archived=False,
        property_filters={"layer": "L0"},
    )
    recent = []
    for node in nodes:
        ts = node.get("updated_at") or node.get("created_at") or ""
        props = node.get("properties") or {}
        if source_agent_id and (node.get("source_agent_id") or props.get("source_agent_id")) != source_agent_id:
            continue
        if ts[:19] >= cutoff:
            recent.append(node)
        if len(recent) >= limit:
            break
    return recent


def distill_l1(
    store,
    encoder=None,
    *,
    hours: int = 24,
    limit: int = 200,
    dry_run: bool = False,
    source_agent_id: str = None,
) -> dict[str, Any]:
    """Distill recent L0 memories into stable L1 nodes."""
    candidates: list[L1Candidate] = []
    seen_ids = set()
    scanned_nodes = _recent_l0_nodes(store, hours=hours, limit=limit, source_agent_id=source_agent_id)
    for node in scanned_nodes:
        candidate = build_l1_candidate(node)
        if not candidate or candidate.node_id in seen_ids:
            continue
        seen_ids.add(candidate.node_id)
        candidates.append(candidate)

    written = []
    if not dry_run:
        for candidate in candidates:
            props = {
                "layer": "L1",
                "kind": candidate.kind,
                "title": candidate.title,
                "content": candidate.content,
                "source_l0_ids": candidate.source_l0_ids,
                "source": "l1_distiller",
                "source_agent_id": candidate.source_agent_id,
                "department": candidate.department,
                "importance": KIND_IMPORTANCE.get(candidate.kind, 0.8),
                "confidence": candidate.confidence,
                "ttl_days": 365,
                "memory_scope": "system",
            }
            embedding = None
            if encoder is not None:
                embedding = encoder.encode(f"{candidate.kind}: {candidate.content}").astype("float32")
            store.add_node(candidate.node_id, "event", props, embedding)
            for source_id in candidate.source_l0_ids:
                store.add_edge_once(candidate.node_id, source_id, "distilled_from", {
                    "source": "l1_distiller",
                    "kind": candidate.kind,
                    "confidence": candidate.confidence,
                })
            written.append(candidate.node_id)

    by_kind: dict[str, int] = {}
    for candidate in candidates:
        by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1

    return {
        "status": "dry_run" if dry_run else "ok",
        "scanned": len(scanned_nodes),
        "source_agent_id": source_agent_id,
        "candidate_count": len(candidates),
        "written_count": len(written),
        "written": written,
        "by_kind": by_kind,
        "preview": [
            {
                "id": item.node_id,
                "kind": item.kind,
                "title": item.title,
                "source_l0_ids": item.source_l0_ids,
            }
            for item in candidates[:10]
        ],
    }
