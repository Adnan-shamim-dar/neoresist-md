from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neoresist.paths import config_dir


@dataclass(frozen=True)
class ScoringProfile:
    profile_id: str
    display_name: str
    version: str
    description: str
    higher_is_better: bool
    primary_score_column: str
    expression_tpm_cap: float
    blend_expression: float
    blend_ccf: float
    weights: dict[str, float]
    escape_penalty_weight: float


@dataclass(frozen=True)
class RuleProfile:
    profile_id: str
    display_name: str
    version: str
    tier1_above: float
    tier2_above: float


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _float_value(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key)
    return float(default if value is None else value)


def _scoring_profile_path(profile_id: str) -> Path:
    for subdir in ("scoring_profiles", "profiles"):
        path = config_dir() / subdir / f"{profile_id}.yaml"
        if path.is_file():
            return path
    return config_dir() / "scoring_profiles" / f"{profile_id}.yaml"


@lru_cache(maxsize=8)
def load_scoring_profile(profile_id: str) -> ScoringProfile:
    path = _scoring_profile_path(profile_id)
    raw = _load_yaml(path)
    blend = raw.get("blend") or {}
    w = raw.get("weights") or {}
    return ScoringProfile(
        profile_id=str(raw.get("profile_id") or profile_id),
        display_name=str(raw.get("display_name") or profile_id),
        version=str(raw.get("version") or "0"),
        description=str(raw.get("description") or "").strip(),
        higher_is_better=bool(raw.get("higher_is_better", True)),
        primary_score_column=str(raw.get("primary_score_column") or "rl_priority"),
        expression_tpm_cap=_float_value(raw, "expression_tpm_cap", 1000.0),
        blend_expression=_float_value(blend, "real_expression_weight", 0.85),
        blend_ccf=_float_value(blend, "real_ccf_weight", 0.85),
        weights={
            "expression_norm": _float_value(w, "expression_norm", 0.2),
            "presentation": _float_value(w, "presentation", 0.3),
            "ccf": _float_value(w, "ccf", 0.3),
            "self_dissimilarity": _float_value(w, "self_dissimilarity", 0.1),
        },
        escape_penalty_weight=_float_value(raw, "escape_penalty_weight", -0.2),
    )


@lru_cache(maxsize=4)
def load_rule_profile(profile_id: str) -> RuleProfile:
    path = config_dir() / "rule_profiles" / f"{profile_id}.yaml"
    raw = _load_yaml(path)
    th = raw.get("tier_thresholds") or {}
    return RuleProfile(
        profile_id=str(raw.get("profile_id") or profile_id),
        display_name=str(raw.get("display_name") or profile_id),
        version=str(raw.get("version") or "0"),
        tier1_above=float(th.get("tier1_above") or 0.7),
        tier2_above=float(th.get("tier2_above") or 0.4),
    )
