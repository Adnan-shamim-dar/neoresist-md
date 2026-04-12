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


@lru_cache(maxsize=8)
def load_scoring_profile(profile_id: str) -> ScoringProfile:
    path = config_dir() / "scoring_profiles" / f"{profile_id}.yaml"
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
        expression_tpm_cap=float(raw.get("expression_tpm_cap") or 1000.0),
        blend_expression=float(blend.get("real_expression_weight") or 0.85),
        blend_ccf=float(blend.get("real_ccf_weight") or 0.85),
        weights={
            "expression_norm": float(w.get("expression_norm") or 0.2),
            "presentation": float(w.get("presentation") or 0.3),
            "ccf": float(w.get("ccf") or 0.3),
            "self_dissimilarity": float(w.get("self_dissimilarity") or 0.1),
        },
        escape_penalty_weight=float(raw.get("escape_penalty_weight") or -0.2),
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
