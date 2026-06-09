"""Deterministic L1 distillation for high-value MAGMA memories.

L1 is the stable memory layer. It should contain decisions, preferences,
project state, lessons, pending actions, and concrete facts. It must not absorb
diagnostic fragments, acknowledgements, or transient debug chatter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set

from magma.capture_policy import classify_capture


KIND_IMPORTANCE = {
    "decision": 0.88,
    "preference": 0.90,
    "project_state": 0.86,
    "pending_action": 0.84,
    "lesson": 0.86,
    "fact": 0.80,
}

# --- P2-3: Auto-labeling rules for importance_label ---
STATUS_CHANGE_KEYWORDS = (
    "已完成", "已切换", "已更新", "已修复", "已上线", "已部署",
    "已迁移", "已生效", "已确认", "已合并", "已发布", "已搞定",
    "修好了", "搞定了", "测试通过", "验收通过", "验证通过",
    "已切好", "已接入", "已经完成", "已重启", "已推送",
    "done", "completed", "fixed", "deployed", "merged", "released",
)

CONSTRAINT_KEYWORDS = (
    "不要", "不应该", "只用于", "默认", "不允许", "不能", "必须",
    "禁止", "不要擅自", "不要用", "优先", "慎重", "谨慎",
    "should not", "must not", "do not", "only for", "default to",
)

OPS_KEYWORDS = (
    "端口", "模型", "版本", "状态", "服务", "配置", "环境",
    "端口号", "API", "MCP", "Gateway", "proxy", "server",
    "port", "model", "version", "status", "config", "env",
)


def auto_label_importance(kind: str, content: str) -> str:
    """P2-3: Auto-label importance_label based on content analysis.

    Rules:
    - Status change keywords -> useful
    - Constraint keywords -> useful
    - Ops keywords -> reference
    - fact/decision kinds -> useful (default)
    - Everything else -> reference (default)
    """
    content_lower = content.lower() if content else ""

    # Status change -> useful
    if any(kw in content for kw in STATUS_CHANGE_KEYWORDS):
        return "useful"

    # Constraint -> useful
    if any(kw in content for kw in CONSTRAINT_KEYWORDS):
        return "useful"

    # Ops keywords -> reference
    if any(kw.lower() in content_lower for kw in OPS_KEYWORDS):
        return "reference"

    # L1 distill fact/decision -> useful by default
    if kind in ("fact", "decision"):
        return "useful"

    # Default
    return "reference"

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
    if not text:
        return True
    # Chinese chars count as 2 (higher info density)
    effective_len = len(text) + len(re.findall(r'[\u4e00-\u9fff]', text))
    if effective_len < 24:
        return True
    if any(marker in text for marker in NOISE_MARKERS):
        return True
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS):
        return True
    # Pure punctuation / whitespace / emoji only
    stripped = re.sub(r'[\s\W\d_]+', '', text)
    if len(stripped) < 8:
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


def _load_openclaw_config() -> Optional[Dict[str, Any]]:
    """Load openclaw.json for config validation."""
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_agent_configs(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract agent model configs from openclaw.json.
    Returns {agent_id: primary_model} mapping."""
    agents_cfg = config.get("agents", {})
    defaults_model = (agents_cfg.get("defaults", {}).get("model", {}).get("primary", ""))
    result: Dict[str, str] = {}
    for agent in agents_cfg.get("list", []):
        agent_id = agent.get("id", "")
        primary = (agent.get("model", {}).get("primary") or defaults_model)
        if agent_id and primary:
            result[agent_id] = primary
    return result


def _extract_plugin_versions(config: Dict[str, Any]) -> Dict[str, bool]:
    """Extract plugin enabled status from openclaw.json."""
    plugins = config.get("plugins", {}).get("entries", {})
    return {name: entry.get("enabled", False) for name, entry in plugins.items()}


def _validate_against_config(candidate: L1Candidate, agent_configs: Dict[str, str], plugin_states: Dict[str, bool]) -> Optional[float]:
    """Validate current_state/project_state L1 candidate against live config.
    Returns adjusted importance if config mismatch detected, else None."""
    content_lower = candidate.content.lower()

    # Check agent model claims
    for agent_id, actual_model in agent_configs.items():
        # Pattern: "<agent> 用/使用/模型是 <model>"
        agent_pattern = re.compile(
            rf"{re.escape(agent_id)}.*?(?:用|使用|模型是|model.*?is|切换到|切到|改成|pin在|pinned)\s*[:：]?\s*(\S+)",
            re.IGNORECASE
        )
        for match in agent_pattern.finditer(candidate.content):
            claimed_model = match.group(1).strip().rstrip("。，、")
            if claimed_model and claimed_model.lower() != actual_model.lower():
                # Check the claimed model doesn't appear in actual config at all
                all_models = set(m.lower() for m in agent_configs.values())
                if claimed_model.lower() not in all_models:
                    return 0.35  # Strong mismatch: model not in any config
                return 0.55  # Weak mismatch: model exists but wrong agent

    # Check reversed pattern: "<model> 是/给 <agent> 的模型"
    for agent_id, actual_model in agent_configs.items():
        if actual_model.lower() in content_lower:
            # Content mentions the correct model - OK
            pass
        # Check if content claims a different model for a known agent
        model_claim = re.compile(
            r"(\S+?)(?:是|给)" + re.escape(agent_id) + r".*?(?:的模型|模型|默认模型)",
            re.IGNORECASE
        )
        for match in model_claim.finditer(candidate.content):
            claimed = match.group(1).strip()
            if claimed.lower() != actual_model.lower() and len(claimed) > 3:
                return 0.40

    return None


def _detect_state_reversals(candidates: list[L1Candidate]) -> dict[str, float]:
    """Detect pending_action nodes that are superseded by completion events.
    Returns {node_id: adjusted_importance} for stale pending_actions."""
    adjustments: dict[str, float] = {}

    # Separate pending_actions from completion events
    pending = [(i, c) for i, c in enumerate(candidates) if c.kind == "pending_action"]
    completions = [c for c in candidates if c.kind in {"project_state", "decision", "lesson"}]

    completion_keywords = re.compile(
        r"已完成|已解决|已修复|已上线|已部署|已迁移|已生效|已确认|已合并|已发布|"
        r"已搞定|修好了|搞定了|测试通过|验收通过|验证通过|已切好|已接入|已经完成",
        re.IGNORECASE
    )

    for idx, pend in pending:
        pend_tokens = _extract_topic_tokens(pend.content)
        if not pend_tokens:
            continue

        for comp in completions:
            if not completion_keywords.search(comp.content):
                continue
            comp_tokens = _extract_topic_tokens(comp.content)
            if not comp_tokens:
                continue

            # Compute entity/topic overlap
            overlap = pend_tokens & comp_tokens
            # Use absolute overlap: 2+ shared meaningful terms indicates same topic
            # (ratio-based thresholds are unreliable with fine-grained Chinese tokenization)
            if len(overlap) >= 2:
                adjustments[pend.node_id] = min(adjustments.get(pend.node_id, 0.84), 0.28)
                break  # One match is enough

    return adjustments


def _extract_topic_tokens(text: str) -> Set[str]:
    """Extract meaningful topic tokens for overlap comparison."""
    # Chinese: extract 2-4 char segments around key terms
    tokens = set()
    # Extract technical terms (alphanumeric + dots)
    for match in re.finditer(r'[A-Za-z][A-Za-z0-9_.\-]{2,}', text):
        tokens.add(match.group().lower())
    # Extract Chinese noun phrases (2-4 chars) excluding common stopwords
    stopwords = {"已经", "完成", "需要", "可以", "应该", "目前", "现在", "之后", "然后", "但是", "不过", "所以", "因为", "如果", "虽然", "这个", "那个", "什么", "怎么", "如何", "为什么", "一下", "一些", "一个"}
    for match in re.finditer(r'[\u4e00-\u9fff]{2,4}', text):
        word = match.group()
        if word not in stopwords and len(word) >= 2:
            tokens.add(word)
    return tokens


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

    # --- P0-2: Config validation for project_state nodes ---
    config = _load_openclaw_config()
    config_adjustments: dict[str, float] = {}
    if config:
        agent_configs = _extract_agent_configs(config)
        plugin_states = _extract_plugin_versions(config)
        for candidate in candidates:
            if candidate.kind == "project_state":
                adjusted = _validate_against_config(candidate, agent_configs, plugin_states)
                if adjusted is not None:
                    config_adjustments[candidate.node_id] = adjusted

    # --- P1-3: State reversal detection ---
    reversal_adjustments = _detect_state_reversals(candidates)

    written = []
    if not dry_run:
        for candidate in candidates:
            base_importance = KIND_IMPORTANCE.get(candidate.kind, 0.8)
            final_importance = base_importance
            flags = []

            # Apply config validation adjustment
            if candidate.node_id in config_adjustments:
                final_importance = config_adjustments[candidate.node_id]
                flags.append("needs_verification")

            # Apply reversal adjustment (take lower)
            if candidate.node_id in reversal_adjustments:
                final_importance = min(final_importance, reversal_adjustments[candidate.node_id])
                flags.append("stale_pending")

            props = {
                "layer": "L1",
                "kind": candidate.kind,
                "title": candidate.title,
                "content": candidate.content,
                "source_l0_ids": candidate.source_l0_ids,
                "source": "l1_distiller",
                "source_agent_id": candidate.source_agent_id,
                "department": candidate.department,
                "importance": round(final_importance, 3),
                "confidence": candidate.confidence,
                "ttl_days": 365,
                "memory_scope": "system",
                # P2-3: Auto-label importance_label
                "importance_label": auto_label_importance(candidate.kind, candidate.content),
            }
            if flags:
                props["flags"] = flags

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
    else:
        # Dry-run: also show adjustments
        for candidate in candidates:
            if candidate.node_id in config_adjustments:
                candidate._dry_run_note = f"CONFIG MISMATCH -> importance {config_adjustments[candidate.node_id]}"
            if candidate.node_id in reversal_adjustments:
                candidate._dry_run_note = f"STATE REVERSAL -> importance {reversal_adjustments[candidate.node_id]}"

    by_kind: dict[str, int] = {}
    for candidate in candidates:
        by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1

    preview_items = []
    for item in candidates[:10]:
        entry = {
            "id": item.node_id,
            "kind": item.kind,
            "title": item.title,
            "source_l0_ids": item.source_l0_ids,
            # P2-3: Show auto-label in dry-run preview
            "importance_label": auto_label_importance(item.kind, item.content),
        }
        if hasattr(item, "_dry_run_note"):
            entry["note"] = item._dry_run_note
        preview_items.append(entry)

    return {
        "status": "dry_run" if dry_run else "ok",
        "scanned": len(scanned_nodes),
        "source_agent_id": source_agent_id,
        "candidate_count": len(candidates),
        "written_count": len(written),
        "written": written,
        "by_kind": by_kind,
        "config_validations": len(config_adjustments),
        "reversal_detections": len(reversal_adjustments),
        "preview": preview_items,
    }
