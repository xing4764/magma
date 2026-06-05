"""Short command detection and resolution for MAGMA.

Short confirmations such as "更新", "继续", "开始", "可以", "1", or "ok"
are not meaningful semantic queries by themselves. MAGMA resolves them against
the most recent scoped conversation context, preferring explicit L1 decisions,
assistant questions, and fact:action nodes.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("magma.short_command")

MAX_SHORT_COMMAND_LENGTH = 6

CONFIRMATION_WORDS = {
    "好",
    "行",
    "可以",
    "没问题",
    "确定",
    "确认",
    "对",
    "嗯",
    "搞吧",
    "弄吧",
    "干吧",
    "更新",
    "开始",
    "继续",
    "执行",
    "运行",
    "启动",
    "完成",
    "ok",
    "yes",
    "y",
    "go",
    "run",
    "start",
    "continue",
    "update",
    "done",
    "finish",
    "1",
}

REJECTION_WORDS = {
    "不",
    "不要",
    "算了",
    "取消",
    "停止",
    "跳过",
    "no",
    "n",
    "stop",
    "cancel",
    "skip",
}

ACTION_WORDS = {
    "更新",
    "开始",
    "继续",
    "执行",
    "运行",
    "启动",
    "重试",
    "结束",
    "完成",
    "update",
    "start",
    "continue",
    "run",
    "retry",
    "finish",
}

DIAGNOSTIC_MARKERS = (
    "验收结果",
    "检查项",
    "根因",
    "结论",
    "发现问题",
    "问题链",
    "short_command",
    "extract_pending_action",
    "drift_warning",
    "magma-recall.jsonl",
    "API 返回",
    "Codex 已",
)

QUESTION_INDICATORS = (
    "需要我",
    "要我",
    "要不",
    "是否",
    "你想",
    "你要",
    "帮你",
    "吗？",
    "吗",
    "?",
    "？",
)

QUESTION_ACTION_RE = re.compile(
    r"(?:需要我|要我|帮你|是否)(.*?)(?:吗|么|\?|？|$)"
)


def normalize_short_command_query(query: str) -> str:
    """Strip OpenClaw runtime wrappers before short-command detection."""
    if not query:
        return ""

    q = str(query).strip()
    if not q:
        return ""

    lines = [line.strip() for line in q.splitlines() if line.strip()]
    while lines and lines[0].startswith("Note:"):
        lines.pop(0)
    if not lines:
        return ""

    # Use the first user line. Later lines are often test instructions, not
    # the short command itself.
    q = lines[0]

    # OpenClaw prompt text commonly arrives as:
    # [Fri 2026-06-05 19:53 GMT+8] 更新
    m = re.match(r"^\[[^\]]+\]\s*(.+)$", q)
    if m:
        q = m.group(1).strip()

    return q


def is_short_command(query: str) -> bool:
    """Return True when a query should be resolved against recent context."""
    q = normalize_short_command_query(query)
    if not q or len(q) > MAX_SHORT_COMMAND_LENGTH:
        return False
    lowered = q.lower()
    if lowered in CONFIRMATION_WORDS or lowered in REJECTION_WORDS or lowered in ACTION_WORDS:
        return True
    if re.fullmatch(r"[1-9]\d{0,2}", q):
        return True
    if re.fullmatch(r"[yYnN]", q):
        return True
    # Very short Chinese replies are usually confirmations in this route.
    return len(q) <= 2


def _content(node: Dict[str, Any]) -> str:
    props = node.get("properties") or {}
    return str(props.get("content") or props.get("summary") or props.get("message") or "")


def _is_diagnostic(content: str) -> bool:
    return any(marker in content for marker in DIAGNOSTIC_MARKERS)


def extract_pending_action(recent_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the best pending action while ignoring diagnostic fragments."""
    if not recent_nodes:
        return None

    newest_first = list(reversed(recent_nodes))

    for node in newest_first:
        props = node.get("properties") or {}
        content = _content(node)
        if _is_diagnostic(content):
            continue
        if props.get("layer") == "L1" and props.get("kind") in ("decision", "task_intent"):
            return {
                "node": node,
                "source": "l1_decision",
                "confidence": 0.95,
                "action_hint": content,
            }

    event_nodes = [
        node for node in newest_first
        if node.get("label") == "event"
        and (node.get("properties") or {}).get("role") in ("assistant", "user")
    ]

    for node in event_nodes:
        props = node.get("properties") or {}
        content = _content(node)
        if _is_diagnostic(content):
            continue
        if props.get("role") == "assistant" and any(ind in content for ind in QUESTION_INDICATORS):
            match = QUESTION_ACTION_RE.search(content)
            return {
                "node": node,
                "source": "assistant_question",
                "confidence": 0.85,
                "action_hint": match.group(1).strip() if match else content[:100],
            }

    for node in newest_first:
        props = node.get("properties") or {}
        content = _content(node)
        node_id = str(node.get("id") or "")
        if _is_diagnostic(content):
            continue
        if props.get("layer") == "L0" and props.get("role") == "fact" and node_id.startswith("fact:action:"):
            return {
                "node": node,
                "source": "fact_action",
                "confidence": 0.75,
                "action_hint": content[:100],
            }

    pending_keywords = ("需要", "准备", "打算", "计划", "待", "pending", "todo")
    for node in event_nodes[:10]:
        props = node.get("properties") or {}
        content = _content(node)
        if _is_diagnostic(content):
            continue
        if props.get("layer") == "L0" and any(kw in content.lower() for kw in pending_keywords):
            return {
                "node": node,
                "source": "l0_pending",
                "confidence": 0.6,
                "action_hint": content[:100],
            }

    return None


def resolve_short_command(
    query: str,
    recent_nodes: List[Dict[str, Any]],
    store=None,
) -> Optional[Dict[str, Any]]:
    """Resolve a short command against scoped recent conversation context."""
    command = normalize_short_command_query(query)
    if not is_short_command(command):
        return None

    pending = extract_pending_action(recent_nodes)
    if not pending:
        logger.info("Short command '%s' detected but no pending action found", command)
        return None

    q = command.strip().lower()
    is_reject = q in REJECTION_WORDS
    is_confirm = not is_reject
    context_nodes = [pending["node"]]

    if store and pending["node"].get("id"):
        try:
            edges = store.get_edges(pending["node"]["id"])
            for edge in edges:
                if edge.get("relation") != "responded_by":
                    continue
                source_id = edge.get("source_id") or edge.get("target_id")
                if source_id and source_id != pending["node"]["id"]:
                    evidence_node = store.get_node(source_id)
                    if evidence_node:
                        context_nodes.append(evidence_node)
        except Exception as exc:
            logger.warning("Failed to fetch short-command evidence chain: %s", exc)

    return {
        "resolved": True,
        "is_confirmation": is_confirm,
        "is_rejection": is_reject,
        "pending_action": pending,
        "context_nodes": context_nodes,
        "confidence": pending["confidence"],
        "suggested_action": pending["action_hint"] if is_confirm else None,
        "query": command,
        "raw_query": query,
    }


def build_short_command_response(resolution: Dict[str, Any]) -> Dict[str, Any]:
    """Build a serializable explanation of a short-command resolution."""
    if not resolution:
        return {}

    pending = resolution.get("pending_action") or {}
    return {
        "short_command_resolution": {
            "original_query": resolution.get("query"),
            "resolved_as": "confirmation" if resolution.get("is_confirmation") else "rejection",
            "pending_action_source": pending.get("source"),
            "action_hint": pending.get("action_hint"),
            "confidence": resolution.get("confidence"),
            "context_node_ids": [n.get("id") for n in resolution.get("context_nodes", [])],
            "evidence_chain": [
                {
                    "id": n.get("id"),
                    "label": n.get("label"),
                    "content": (n.get("properties") or {}).get("content", "")[:200],
                    "layer": (n.get("properties") or {}).get("layer"),
                    "role": (n.get("properties") or {}).get("role"),
                }
                for n in resolution.get("context_nodes", [])
            ],
        }
    }
