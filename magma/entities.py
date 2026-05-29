"""Dynamic entity extraction for MAGMA anchors.

Supports three sources:
1. Built-in config (config/entities.json) — system entities + patterns
2. User custom config (config/custom_entities.json) — user-defined entities
3. LLM-based extraction (optional) — for entities not in any dictionary
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("magma.entities")

PRODUCT_ENTITY_TYPES = {"sku", "selling_point", "document", "domain"}
SYSTEM_ENTITY_TYPES = {"system", "plugin", "storage", "api", "protocol", "model", "tool", "platform"}
PERSON_ENTITY_TYPES = {"person", "org"}

_config_dir = Path(__file__).parent.parent / "config"
_entities_config: Optional[Dict] = None
_custom_entities: Optional[Dict] = None
_compiled_patterns: Optional[List] = None


def _load_entities_config() -> Dict:
    """Load entities from config/entities.json."""
    global _entities_config
    if _entities_config is not None:
        return _entities_config
    config_path = _config_dir / "entities.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                _entities_config = json.load(f)
            logger.info(f"Loaded {len(_entities_config.get('entities', {}))} entities from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load entities config: {e}")
            _entities_config = {"entities": {}, "patterns": {}}
    else:
        _entities_config = {"entities": {}, "patterns": {}}
    return _entities_config


def _load_custom_entities() -> Dict:
    """Load user-defined entities from config/custom_entities.json."""
    global _custom_entities
    if _custom_entities is not None:
        return _custom_entities
    config = _load_entities_config()
    custom_path = config.get("custom_entities_file", "config/custom_entities.json")
    # Resolve relative to project root
    if not os.path.isabs(custom_path):
        custom_path = str(_config_dir.parent / custom_path)
    custom_file = Path(custom_path)
    if custom_file.exists():
        try:
            with open(custom_file, encoding="utf-8") as f:
                data = json.load(f)
            _custom_entities = data.get("entities", {})
            logger.info(f"Loaded {len(_custom_entities)} custom entities from {custom_file}")
        except Exception as e:
            logger.warning(f"Failed to load custom entities: {e}")
            _custom_entities = {}
    else:
        _custom_entities = {}
    return _custom_entities


def _load_patterns() -> List:
    """Load and compile regex patterns from config."""
    global _compiled_patterns
    if _compiled_patterns is not None:
        return _compiled_patterns
    config = _load_entities_config()
    patterns_data = config.get("patterns", {})
    _compiled_patterns = []
    for entity_type, pattern_str in patterns_data.items():
        try:
            _compiled_patterns.append((entity_type, re.compile(pattern_str, re.I)))
        except re.error as e:
            logger.warning(f"Invalid pattern for {entity_type}: {e}")
    return _compiled_patterns


def reload_configs():
    """Force reload all entity configs (call after editing config files)."""
    global _entities_config, _custom_entities, _compiled_patterns
    _entities_config = None
    _custom_entities = None
    _compiled_patterns = None
    _load_entities_config()
    _load_custom_entities()
    _load_patterns()


def entity_id(name: str) -> str:
    digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"ent:anchor:{digest}"


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities from text using dictionary + pattern matching.

    Sources (in order):
    1. Built-in config entities
    2. User custom entities
    3. Regex patterns (SKU, model names)
    """
    value = text or ""
    found: Dict[str, Dict[str, str]] = {}

    # 1. Config entities
    config = _load_entities_config()
    for name, entity_type in config.get("entities", {}).items():
        if name in value:
            found[name.lower()] = {"id": entity_id(name), "name": name, "entity_type": entity_type}

    # 2. Custom entities
    custom = _load_custom_entities()
    for name, entity_type in custom.items():
        if name in value:
            found[name.lower()] = {"id": entity_id(name), "name": name, "entity_type": entity_type}

    # 3. Pattern matching
    for entity_type, pattern in _load_patterns():
        for match in pattern.findall(value):
            name = match.strip()
            if name:
                found[name.lower()] = {"id": entity_id(name), "name": name, "entity_type": entity_type}

    return sorted(found.values(), key=lambda item: item["name"].lower())


def add_custom_entity(name: str, entity_type: str) -> bool:
    """Add a new custom entity to config/custom_entities.json.

    Returns True if added, False if already exists.
    """
    custom = _load_custom_entities()
    if name in custom:
        return False
    custom[name] = entity_type

    custom_path = _config_dir / "custom_entities.json"
    try:
        # Read existing file to preserve comments
        data = {"_comment": "", "entities": custom}
        if custom_path.exists():
            with open(custom_path, encoding="utf-8") as f:
                data = json.load(f)
            data["entities"] = custom
        with open(custom_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        reload_configs()
        logger.info(f"Added custom entity: {name} ({entity_type})")
        return True
    except Exception as e:
        logger.warning(f"Failed to save custom entity: {e}")
        return False


def remove_custom_entity(name: str) -> bool:
    """Remove a custom entity from config/custom_entities.json."""
    custom = _load_custom_entities()
    if name not in custom:
        return False
    del custom[name]

    custom_path = _config_dir / "custom_entities.json"
    try:
        data = {"_comment": "", "entities": custom}
        if custom_path.exists():
            with open(custom_path, encoding="utf-8") as f:
                data = json.load(f)
            data["entities"] = custom
        with open(custom_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        reload_configs()
        logger.info(f"Removed custom entity: {name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to remove custom entity: {e}")
        return False


def list_entities() -> Dict[str, Dict[str, str]]:
    """List all known entities (config + custom)."""
    config = _load_entities_config()
    custom = _load_custom_entities()
    all_entities = {}
    for name, etype in config.get("entities", {}).items():
        all_entities[name] = {"type": etype, "source": "config"}
    for name, etype in custom.items():
        all_entities[name] = {"type": etype, "source": "custom"}
    return all_entities


def classify_memory_scope(entities: List[Dict[str, str]]) -> str:
    types = {entity.get("entity_type") for entity in entities}
    has_product = bool(types & PRODUCT_ENTITY_TYPES)
    has_system = bool(types & SYSTEM_ENTITY_TYPES)
    has_person = bool(types & PERSON_ENTITY_TYPES)
    if has_product and has_system:
        return "mixed"
    if has_product:
        return "product"
    if has_system:
        return "system"
    if has_person:
        return "personal"
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
        "tool": 10,
        "platform": 11,
        "person": 12,
        "org": 13,
    }
    ordered = sorted(
        entities,
        key=lambda item: (priority.get(item.get("entity_type"), 99), item.get("name", "").lower()),
    )
    entity = ordered[0]
    return f"{entity.get('entity_type', 'entity')}:{entity.get('name', '').lower()}"


# Auto-load configs on import
_load_entities_config()
_load_custom_entities()
_load_patterns()
