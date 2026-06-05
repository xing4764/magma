"""Short command detection and resolution for MAGMA.

When user sends a short message like "更新", "开始", "继续", "1", "可以",
it should be resolved against recent conversation context (pending decisions,
open questions) rather than treated as a standalone semantic query.

Expected flow:
  short command → recent context anchor → pending decision/task_intent →
  L1 decision node priority → L0 evidence supplement → execute action
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("magma.short_command")

# Short command patterns (Chinese + English)
SHORT_COMMAND_PATTERNS = [
    # Confirmation/agreement
    r"^(好|行|可以|没问题|ok|yes|确认|确定|对|嗯|搞吧|弄吧|上|干吧|整)$",
    r"^(好的|行的|可以的|没问题的|对的)$",
    # Action triggers
    r"^(更新|开始|继续|执行|运行|启动|停止|取消|重试|跳过|结束|完成)$",
    r"^(update|start|continue|run|go|stop|cancel|retry|skip|done|finish)$",
    # Numeric selection
    r"^[1-9]\d{0,2}$",
    # Single character confirmations
    r"^[yYnN]$",
    # Short phrases (<=3 chars that are likely confirmations)
    r"^.{1,2}$",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SHORT_COMMAND_PATTERNS]

# Maximum length to be considered a short command
MAX_SHORT_COMMAND_LENGTH = 6


def is_short_command(query: str) -> bool:
    """Detect if a query is a short command that needs context resolution.

    A short command is:
    - Very short (<= MAX_SHORT_COMMAND_LENGTH chars)
    - Matches known short command patterns
    - Not a meaningful semantic query on its own
    """
    if not query or not query.strip():
        return False

    q = query.strip()

    # Too long to be a short command
    if len(q) > MAX_SHORT_COMMAND_LENGTH:
        return False

    # Check against patterns
    for pattern in _COMPILED_PATTERNS:
        if pattern.match(q):
            return True

    return False


def extract_pending_action(recent_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the most recent pending action/question from recent conversation.

    Looks for:
    1. L1 decision/task_intent nodes
    2. Assistant questions (ending with ?/？)
    3. Recent L0 events that suggest a pending action

    Returns the best candidate pending action node, or None.
    """
    if not recent_nodes:
        return None

    # Priority 1: L1 decision/task_intent nodes (most reliable)
    for node in recent_nodes:
        props = node.get("properties") or {}
        layer = props.get("layer")
        kind = props.get("kind")
        if layer == "L1" and kind in ("decision", "task_intent"):
            return {
                "node": node,
                "source": "l1_decision",
                "confidence": 0.95,
                "action_hint": props.get("content") or props.get("summary") or "",
            }

    # Priority 2: Assistant messages with questions (pending question)
    for node in recent_nodes:
        props = node.get("properties") or {}
        role = props.get("role")
        content = str(props.get("content") or props.get("message") or "")
        if role == "assistant" and content:
            # Check if it contains a question directed at the user
            question_indicators = ("需要我", "要我", "要不", "是否", "你想", "你要", "帮你", "吗？", "吗?", "?", "？")
            if any(indicator in content for indicator in question_indicators):
                # Extract the action from the question
                action_match = re.search(
                    r"(?:需要我|要我|帮你|是否)(.*?)(?:吗|吧|\?|？|$)",
                    content
                )
                action_hint = action_match.group(1).strip() if action_match else content[:100]
                return {
                    "node": node,
                    "source": "assistant_question",
                    "confidence": 0.85,
                    "action_hint": action_hint,
                }

    # Priority 3: Recent L0 events mentioning pending actions
    for node in recent_nodes[:3]:  # Only check most recent 3
        props = node.get("properties") or {}
        content = str(props.get("content") or props.get("message") or "")
        layer = props.get("layer")
        if layer == "L0" and content:
            # Check if the content suggests a pending action
            pending_keywords = ("需要", "准备", "打算", "计划", "待", "pending", "todo")
            if any(kw in content.lower() for kw in pending_keywords):
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
    """Resolve a short command against recent conversation context.

    Returns:
        Dict with resolution result, or None if no context found.
        {
            "resolved": True,
            "pending_action": {...},  # The pending action being confirmed
            "context_nodes": [...],   # Relevant context nodes
            "confidence": float,      # How confident we are in the resolution
            "suggested_action": str,  # What action should be taken
        }
    """
    if not is_short_command(query):
        return None

    pending = extract_pending_action(recent_nodes)
    if not pending:
        logger.info(f"Short command '{query}' detected but no pending action found in recent context")
        return None

    # Determine if the short command is a confirmation or rejection
    q = query.strip().lower()
    confirmation_words = {
        "好", "行", "可以", "没问题", "ok", "yes", "确认", "确定", "对", "嗯",
        "搞吧", "弄吧", "上", "干吧", "整", "好的", "行的", "可以的",
        "y", "1", "继续", "更新", "开始", "执行", "运行", "启动",
    }
    rejection_words = {
        "不", "不要", "算了", "取消", "no", "n", "停", "停止", "跳过",
    }

    is_confirm = q in confirmation_words
    is_reject = q in rejection_words

    if not is_confirm and not is_reject:
        # Default to confirmation for ambiguous short commands
        is_confirm = True

    # Build context nodes list (the pending action + its evidence chain)
    context_nodes = [pending["node"]]

    # Try to find the L0 evidence that led to this pending action
    if store and pending["node"].get("id"):
        try:
            edges = store.get_edges(pending["node"]["id"])
            for edge in edges:
                if edge.get("relation") == "responded_by":
                    source_id = edge.get("source_id") or edge.get("target_id")
                    if source_id and source_id != pending["node"]["id"]:
                        evidence_node = store.get_node(source_id)
                        if evidence_node:
                            context_nodes.append(evidence_node)
        except Exception as e:
            logger.warning(f"Failed to fetch evidence chain: {e}")

    return {
        "resolved": True,
        "is_confirmation": is_confirm,
        "is_rejection": is_reject,
        "pending_action": pending,
        "context_nodes": context_nodes,
        "confidence": pending["confidence"],
        "suggested_action": pending["action_hint"] if is_confirm else None,
        "query": query,
    }


def build_short_command_response(resolution: Dict[str, Any]) -> Dict[str, Any]:
    """Build a response structure for short command resolution.

    This is used to provide context to the caller about what was resolved.
    """
    if not resolution:
        return {}

    pending = resolution.get("pending_action") or {}
    node = pending.get("node") or {}

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
