import json
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)

# Find magma-recall plugin config
plugins = cfg.get("plugins", {})
magma_cfg = plugins.get("magma-recall", {})
print("Current magma-recall config:")
print(json.dumps(magma_cfg, indent=2, ensure_ascii=False))
