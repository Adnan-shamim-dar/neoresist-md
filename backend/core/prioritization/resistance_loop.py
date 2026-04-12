from __future__ import annotations

"""
ResistanceLoop scoring: delegates to ``neoresist.scoring`` (YAML-driven RL v1).

Re-export helpers used elsewhere / tests.
"""

import math

import pandas as pd

from neoresist.config import get_app_config
from neoresist.scoring import apply_resistance_loop_engine, expression_norm_from_tpm

EXPRESSION_TPM_CAP = 1000.0
W_BLEND_EXPRESSION_REAL = 0.85
W_BLEND_CCF_REAL = 0.85


def tier_from_rl(rl: float) -> int:
    from neoresist.rules import tier_from_score

    rid = get_app_config().defaults.rule_profile_id
    return tier_from_score(rl, rule_profile_id=rid)


def apply_resistance_loop(
    df: pd.DataFrame,
    *,
    prefer_real_evidence: bool = False,
    scoring_profile_id: str | None = None,
    rule_profile_id: str | None = None,
) -> pd.DataFrame:
    cfg = get_app_config()
    return apply_resistance_loop_engine(
        df,
        prefer_real_evidence=prefer_real_evidence,
        profile_id=scoring_profile_id or cfg.defaults.scoring_profile_id,
        rule_profile_id=rule_profile_id or cfg.defaults.rule_profile_id,
    )


def _self_dissimilarity_fill(peptide: str, gene: str, existing: object) -> float:
    if existing is not None and pd.notna(existing):
        try:
            v = float(existing)
            if not math.isnan(v):
                return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            pass
    h = hash((gene or "", peptide or "")) % (2**32)
    return h / (2**32 - 1)


__all__ = [
    "EXPRESSION_TPM_CAP",
    "W_BLEND_EXPRESSION_REAL",
    "W_BLEND_CCF_REAL",
    "expression_norm_from_tpm",
    "tier_from_rl",
    "apply_resistance_loop",
    "_self_dissimilarity_fill",
]
