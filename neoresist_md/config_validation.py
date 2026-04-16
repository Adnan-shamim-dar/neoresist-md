from __future__ import annotations

import json
from pathlib import Path

import yaml


def validate_config_contract(config_dir: str | Path) -> None:
    cfg = Path(config_dir)
    req = ["module_weights.yaml", "tool_registry.yaml", "strategy_library.json", "canonical_schema_v1.yaml"]
    missing = [name for name in req if not (cfg / name).is_file()]
    if missing:
        raise ValueError(f"Missing required config files: {', '.join(missing)}")

    weights = yaml.safe_load((cfg / "module_weights.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(weights, dict) or "immunogenicity" not in weights or "resistance" not in weights:
        raise ValueError("module_weights.yaml must contain top-level 'immunogenicity' and 'resistance' sections.")

    registry = yaml.safe_load((cfg / "tool_registry.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(registry, dict) or not registry:
        raise ValueError("tool_registry.yaml must be a non-empty mapping.")

    library = json.loads((cfg / "strategy_library.json").read_text(encoding="utf-8"))
    if not isinstance(library, list):
        raise ValueError("strategy_library.json must be a JSON array.")

