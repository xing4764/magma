"""LLM-based atomic fact extraction for MAGMA.

Extracts structured atomic facts from conversation text.
Each fact is a single, verifiable, self-contained statement.

Example:
  Input: "我用的手机号是 13800138000，邮箱 test@example.com，偏好深色模式"
  Output: [
    {"fact": "用户手机号是 13800138000", "entities": ["手机号"], "category": "contact"},
    {"fact": "用户邮箱是 test@example.com", "entities": ["邮箱"], "category": "contact"},
    {"fact": "用户偏好深色模式", "entities": ["深色模式"], "category": "preference"},
  ]
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Dict, List, Optional

logger = logging.getLogger("magma.fact_extractor")

# LLM config from environment or defaults
LLM_BASE_URL = os.environ.get("MAGMA_LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
LLM_API_KEY = os.environ.get("MAGMA_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("MAGMA_LLM_MODEL", "mimo-v2.5")
LLM_TIMEOUT = int(os.environ.get("MAGMA_LLM_TIMEOUT", "60"))

EXTRACTION_PROMPT = """你是一个精确的事实提取器。从以下对话文本中提取原子事实。

规则：
1. 每个事实必须是单一、可验证、自包含的陈述
2. 不要提取观点、情绪、或模糊表述
3. 保留原始数字、代码、专有名词（不要改写）
4. 如果文本中没有值得记忆的事实，返回空列表
5. 事实用中文表达，即使原文混合了中英文
6. entities 字段必须包含【类型名】而非【具体值】
   例如："手机号是13800138000" → entities: ["手机号"]（不是 ["13800138000"]）
         "邮箱是test@example.com" → entities: ["邮箱"]（不是 ["test@example.com"]）
         "品牌色改成#1a1a2e" → entities: ["品牌色"]

分类：
- preference: 偏好、习惯、喜好
- contact: 联系方式、账号、地址
- identity: 身份、角色、关系
- fact: 客观事实、状态、配置
- action: 已完成的操作、决策
- plan: 计划、安排、待办

文本：
{text}

输出 JSON 数组（不要输出其他内容）：
[{{"fact": "...", "category": "...", "entities": ["..."]}}]
"""


def _call_llm(prompt: str, api_key: str = None, base_url: str = None, model: str = None) -> str:
    """Call OpenAI-compatible LLM API."""
    api_key = api_key or LLM_API_KEY
    base_url = (base_url or LLM_BASE_URL).rstrip("/")
    model = model or LLM_MODEL

    if not api_key:
        logger.warning("No LLM API key configured, skipping fact extraction")
        return ""

    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return ""


def _parse_facts(llm_output: str) -> List[Dict[str, any]]:
    """Parse LLM JSON output into fact list."""
    if not llm_output:
        return []

    # Strip markdown code fences if present
    text = llm_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        facts = json.loads(text)
        if not isinstance(facts, list):
            return []
        # Validate structure
        valid = []
        for item in facts:
            if isinstance(item, dict) and "fact" in item:
                valid.append({
                    "fact": str(item["fact"]).strip(),
                    "category": str(item.get("category", "fact")).strip(),
                    "entities": [str(e).strip() for e in item.get("entities", []) if e],
                })
        return valid
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse LLM output as JSON: {e}")
        return []


def extract_facts(text: str, role: str = "user", api_key: str = None) -> List[Dict[str, any]]:
    """Extract atomic facts from text using LLM.

    Args:
        text: Input text to extract facts from.
        role: "user" or "assistant" — affects extraction strategy.
        api_key: Optional API key override.

    Returns:
        List of fact dicts with keys: fact, category, entities.
    """
    cleaned = (text or "").strip()
    if len(cleaned) < 5:
        return []

    # Skip extraction for obvious non-fact content
    skip_patterns = (
        "你好", "谢谢", "好的", "收到", "嗯", "ok", "OK", "hi", "hello",
        "测试", "test", "请问", "可以问",
    )
    if cleaned in skip_patterns or len(cleaned) < 8:
        return []

    # For assistant responses, extract only definitive statements
    if role == "assistant":
        # Skip if response is mostly questions or hedging
        question_count = sum(1 for c in cleaned if c in "？?")
        if question_count > len(cleaned) / 20:
            return []

    prompt = EXTRACTION_PROMPT.format(text=cleaned)
    llm_output = _call_llm(prompt, api_key=api_key)


def extract_facts_batch(user_text: str, assistant_text: str, api_key: str = None) -> List[Dict[str, any]]:
    """Extract atomic facts from both user and assistant text in a single LLM call.

    Combines both texts into one prompt to reduce LLM calls from 2 to 1.
    """
    parts = []
    for role, text in (("user", user_text), ("assistant", assistant_text)):
        cleaned = (text or "").strip()
        if len(cleaned) < 20:
            continue
        skip_patterns = (
            "你好", "谢谢", "好的", "收到", "嗯", "ok", "OK", "hi", "hello",
            "测试", "test", "请问", "可以问",
        )
        if cleaned in skip_patterns:
            continue
        parts.append(f"【{role}】{cleaned}")

    if not parts:
        return []

    combined_text = "\n".join(parts)
    prompt = EXTRACTION_PROMPT.format(text=combined_text)
    llm_output = _call_llm(prompt, api_key=api_key)
    facts = _parse_facts(llm_output)

    # Post-filter: deduplicate and limit
    seen = set()
    unique = []
    for fact in facts:
        key = fact["fact"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(fact)

    # Normalize entity names: remove common suffixes for better matching
    for fact in unique:
        normalized = []
        for entity in fact["entities"]:
            # Strip common suffixes for temporal matching
            for suffix in ["号码", "号", "地址", "账号", "邮箱地址", "手机号码"]:
                if entity.endswith(suffix) and len(entity) > len(suffix):
                    shortened = entity[:-len(suffix)]
                    if len(shortened) >= 2:
                        normalized.append(shortened)
            normalized.append(entity)
        fact["entities"] = list(dict.fromkeys(normalized))  # dedupe preserving order

    return unique[:5]  # Max 5 facts per capture


def is_fact_extraction_available() -> bool:
    """Check if LLM is configured for fact extraction."""
    return bool(LLM_API_KEY)
