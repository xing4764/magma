"""Lightweight entity extraction for MAGMA anchors."""

import hashlib
import re
from typing import Dict, List

PRODUCT_ENTITY_TYPES = {"sku", "selling_point", "document", "domain"}
SYSTEM_ENTITY_TYPES = {"system", "plugin", "storage", "api", "protocol", "model"}

KNOWN_ENTITIES = {
    "MAGMA": "system",
    "GBrain": "system",
    "OpenClaw": "system",
    "MCP": "protocol",
    "FAISS": "storage",
    "SQLite": "storage",
    "FastAPI": "api",
    "magma-recall": "plugin",
    "memory-core": "plugin",
    "\u817e\u8baf\u63d2\u4ef6": "plugin",
    "\u51b0\u4e1d\u51c9\u611f": "selling_point",
    "\u51c9\u723d\u900f\u6c14": "selling_point",
    "\u6296\u5e97\u8349\u7a3f": "document",
    "\u6296\u97f3\u7535\u5546": "domain",
}


PATTERNS = (
    ("sku", re.compile(r"\b[A-Z]{1,4}\d{2,6}\b")),
    ("model", re.compile(r"\b(?:bge-small-zh-v1\.5|MiniLM-L6-v2|text2vec-base-chinese)\b", re.I)),
)


def entity_id(name: str) -> str:
    digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"ent:anchor:{digest}"


def extract_entities(text: str) -> List[Dict[str, str]]:
    value = text or ""
    found: Dict[str, Dict[str, str]] = {}
    for name, entity_type in KNOWN_ENTITIES.items():
        if name in value:
            found[name.lower()] = {"id": entity_id(name), "name": name, "entity_type": entity_type}
    for entity_type, pattern in PATTERNS:
        for match in pattern.findall(value):
            name = match.strip()
            found[name.lower()] = {"id": entity_id(name), "name": name, "entity_type": entity_type}
    return sorted(found.values(), key=lambda item: item["name"].lower())


def classify_memory_scope(entities: List[Dict[str, str]]) -> str:
    types = {entity.get("entity_type") for entity in entities}
    has_product = bool(types & PRODUCT_ENTITY_TYPES)
    has_system = bool(types & SYSTEM_ENTITY_TYPES)
    if has_product and has_system:
        return "mixed"
    if has_product:
        return "product"
    if has_system:
        return "system"
    return "general"


def version_key_for_entities(entities: List[Dict[str, str]]) -> str:
    if not entities:
        return ""
    priority = {
        "sku": 0,
        "system": 1,
        "plugin": 2,
        "document": 3,
        "selling_point": 4,
        "model": 5,
        "storage": 6,
        "api": 7,
        "protocol": 8,
        "domain": 9,
    }
    ordered = sorted(
        entities,
        key=lambda item: (priority.get(item.get("entity_type"), 99), item.get("name", "").lower()),
    )
    entity = ordered[0]
    return f"{entity.get('entity_type', 'entity')}:{entity.get('name', '').lower()}"
