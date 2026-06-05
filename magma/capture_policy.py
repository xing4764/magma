"""Lightweight capture filtering for automatic MAGMA writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


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
