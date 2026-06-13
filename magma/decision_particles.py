"""Deterministic decision-particle extraction for MAGMA.

This module intentionally avoids LLM calls. It captures small but useful
signals such as option switches, explicit selections, and preference tags.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional


_TAG_KEYWORDS = {
    "cost": ("贵", "便宜", "成本", "省钱", "预算", "划算", "price", "cost", "cheap", "expensive"),
    "speed": ("快", "慢", "速度", "延迟", "耗时", "timeout", "latency", "fast", "slow"),
    "stability": ("稳定", "蓝屏", "崩", "挂", "失败", "可靠", "风险", "stable", "crash", "fail"),
    "quality": ("质量", "效果", "精度", "准确", "能力", "更好", "quality", "accurate"),
    "memory": ("内存", "显存", "占用", "ram", "vram", "memory"),
    "local": ("本地", "离线", "隐私", "local", "offline", "privacy"),
}

_STOP_WORDS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "这个",
    "那个",
    "现在",
    "先",
    "直接",
    "还是",
    "或者",
    "不要",
    "别",
    "用",
    "换",
    "换成",
    "改成",
    "选择",
    "决定",
    "保留",
    "使用",
    "the",
    "a",
    "an",
    "use",
    "choose",
    "switch",
    "keep",
}


@dataclass
class DecisionEvent:
    decision_key: str
    options: List[str]
    selected: Optional[str]
    chain: List[str]
    direction: str
    confidence: float
    tags: List[str]
    content: str
    source: str = "decision_particles"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_token(value: str) -> str:
    value = re.sub(r"^[\s:：,，。.!！?？;；\"'`]+|[\s:：,，。.!！?？;；\"'`]+$", "", value or "")
    value = re.sub(r"(?<=[A-Za-z0-9])(?:了|啊|呢|吧|吗)$", "", value)
    return value[:80]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = _clean_token(value)
        key = cleaned.lower()
        if cleaned and key not in seen and cleaned not in _STOP_WORDS:
            seen.add(key)
            result.append(cleaned)
    return result


def _tags_for_text(text: str) -> List[str]:
    lower = text.lower()
    tags = []
    for tag, keywords in _TAG_KEYWORDS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            tags.append(tag)
    return tags


def _decision_key(text: str, options: List[str], selected: Optional[str]) -> str:
    lower = text.lower()
    if re.search(r"\d+(?:\.\d+)?b", lower):
        return "model"
    for keyword, key in (
        ("embedding", "embedding_model"),
        ("模型", "model"),
        ("网关", "gateway"),
        ("openclaw", "openclaw"),
        ("magma", "magma"),
        ("mimo", "mimo_code"),
        ("内存", "memory_budget"),
        ("显存", "memory_budget"),
    ):
        if keyword in lower:
            return key
    base = selected or (options[0] if options else "")
    base = re.sub(r"\W+", "_", base.lower()).strip("_")
    return base[:40] or "general_decision"


def _append_event(events: List[DecisionEvent], text: str, options: List[str], selected: Optional[str], direction: str, confidence: float):
    options = _dedupe(options)
    selected = _clean_token(selected or "") or None
    chain = _dedupe([*options, selected] if selected else options)
    if not selected and len(chain) == 1:
        selected = chain[0]
    if not chain and not selected:
        return
    events.append(
        DecisionEvent(
            decision_key=_decision_key(text, chain, selected),
            options=chain,
            selected=selected,
            chain=chain,
            direction=direction,
            confidence=confidence,
            tags=_tags_for_text(text),
            content=text.strip()[:500],
        )
    )


def extract_decision_events(user_text: str = "", assistant_text: str = "") -> List[Dict[str, Any]]:
    """Extract explicit user-facing decision events from a conversation turn."""

    text = (user_text or "").strip() or (assistant_text or "").strip()
    if not text:
        return []

    events: List[DecisionEvent] = []
    short_text = text[:1200]

    for match in re.finditer(r"([\w.\-_/\\\u4e00-\u9fff]{1,40})\s*(?:还是|或者|or)\s*([\w.\-_/\\\u4e00-\u9fff]{1,40})", short_text, re.IGNORECASE):
        _append_event(events, short_text, [match.group(1), match.group(2)], None, "compare_options", 0.58)

    for match in re.finditer(r"(?:从|由|把)?\s*([\w.\-_/\\\u4e00-\u9fff]{1,40})\s*(?:改到|换到|换成|改成|升级到|切到|switch(?:ed)?\s+to)\s*([\w.\-_/\\\u4e00-\u9fff]{1,40})", short_text, re.IGNORECASE):
        _append_event(events, short_text, [match.group(1), match.group(2)], match.group(2), "switch", 0.82)

    for match in re.finditer(r"(?:选|选择|决定用|就用|使用|保留|keep|choose|use)\s*([\w.\-_/\\\u4e00-\u9fff]{1,60})", short_text, re.IGNORECASE):
        _append_event(events, short_text, [match.group(1)], match.group(1), "select", 0.74)

    for match in re.finditer(r"(?:不要|别用|不用|别)\s*([\w.\-_/\\\u4e00-\u9fff]{1,40}).{0,24}?(?:用|换|换成|改成)\s*([\w.\-_/\\\u4e00-\u9fff]{1,40})", short_text, re.IGNORECASE):
        _append_event(events, short_text, [match.group(1), match.group(2)], match.group(2), "reject_then_select", 0.86)

    unique = {}
    for event in events:
        key = (event.decision_key, event.selected, tuple(event.chain), event.direction)
        unique[key] = event
    return [event.to_dict() for event in unique.values()]


def summarize_decision_drift(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize selected-option and tag drift from newest/oldest event rows."""

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = event.get("decision_key") or "general_decision"
        grouped[key].append(event)

    drift = []
    for key, group in grouped.items():
        ordered = list(reversed(group))
        selections = [item.get("selected") for item in ordered if item.get("selected")]
        tag_sets = [set(item.get("tags") or []) for item in ordered]
        unique_selections = []
        for selected in selections:
            if selected not in unique_selections:
                unique_selections.append(selected)
        added_tags = sorted(set.union(*tag_sets) if tag_sets else set())
        drift.append(
            {
                "decision_key": key,
                "events": len(group),
                "current": selections[-1] if selections else None,
                "first": selections[0] if selections else None,
                "changed": len(unique_selections) > 1,
                "path": unique_selections,
                "tags": added_tags,
            }
        )

    drift.sort(key=lambda item: (not item["changed"], -item["events"], item["decision_key"]))
    return {"count": len(events), "groups": drift}
