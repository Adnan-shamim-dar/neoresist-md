"""
Config-driven ResistanceLoop scoring (RL v1). No eval(); weights from YAML.
"""

from __future__ import annotations

import math

import pandas as pd

from neoresist.profiles import load_scoring_profile
from neoresist.rules import tier_from_score


def expression_norm_from_tpm(tpm: float, *, cap: float) -> float:
    t = max(float(tpm), 0.0)
    return max(0.0, min(1.0, math.log1p(t) / math.log1p(cap)))


def _self_dissimilarity_fill(peptide: str, gene: str, existing: object) -> float:
    if existing is not None and pd.notna(existing):
        try:
            v = float(existing)
            if not math.isnan(v):
                return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            pass
    return float("nan")


def _float_or(row: pd.Series, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        if v is None or pd.isna(v):
            return default
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _expression_norm_for_row(
    row: pd.Series, *, prefer_real: bool, sp: object
) -> tuple[float, str]:
    stub_tpm = _float_or(row, "expression_tpm", 0.0)
    n_stub = expression_norm_from_tpm(stub_tpm, cap=sp.expression_tpm_cap)
    if not prefer_real:
        return n_stub, "stub"

    rt = row.get("real_expression_tpm")
    if rt is None or pd.isna(rt):
        return n_stub, "stub"
    n_real = expression_norm_from_tpm(float(rt), cap=sp.expression_tpm_cap)
    w = sp.blend_expression
    blend = w * n_real + (1.0 - w) * n_stub
    return max(0.0, min(1.0, blend)), "blended_real"


def _ccf_for_row(row: pd.Series, *, prefer_real: bool, sp: object) -> tuple[float, str]:
    stub_c = max(0.0, min(1.0, _float_or(row, "ccf", 0.0)))
    if not prefer_real:
        return stub_c, "stub"

    rc = row.get("real_ccf")
    if rc is None or pd.isna(rc):
        return stub_c, "stub"
    n_real = max(0.0, min(1.0, float(rc)))
    w = sp.blend_ccf
    blend = w * n_real + (1.0 - w) * stub_c
    return max(0.0, min(1.0, blend)), "blended_real"


def apply_resistance_loop_engine(
    df: pd.DataFrame,
    *,
    prefer_real_evidence: bool = False,
    profile_id: str = "rl_v1",
    rule_profile_id: str = "default_rules",
) -> pd.DataFrame:
    # Canonical RL formula: immunogenicity_blend × (1 − resistance_penalty). See docs/architecture.md
    sp = load_scoring_profile(profile_id)
    out = df.copy()
    if out.empty:
        return out

    use_real = bool(prefer_real_evidence)
    if use_real and "real_expression_tpm" not in out.columns and "real_ccf" not in out.columns:
        use_real = False

    if "escape_penalty" not in out.columns:
        out["escape_penalty"] = 0.0

    expr_src: list[str] = []
    ccf_src: list[str] = []
    expr_norm: list[float] = []
    pres: list[float] = []
    ccf: list[float] = []
    sd: list[float] = []
    esc: list[float] = []

    for _, row in out.iterrows():
        en, esrc = _expression_norm_for_row(row, prefer_real=use_real, sp=sp)
        expr_norm.append(en)
        expr_src.append(esrc)

        ccf_b, csrc = _ccf_for_row(row, prefer_real=use_real, sp=sp)
        ccf.append(ccf_b)
        ccf_src.append(csrc)

        ps_f = max(0.0, min(1.0, _float_or(row, "presentation_score", 0.0)))
        pres.append(ps_f)

        pep = str(row.get("mutant_peptide", ""))
        gen = str(row.get("gene", row.get("gene_name", "")))
        ex_sd = row.get("self_dissimilarity")
        sd.append(_self_dissimilarity_fill(pep, gen, ex_sd))

        esc.append(max(0.0, min(1.0, _float_or(row, "escape_penalty", 0.0))))

    out["expression_norm"] = expr_norm
    out["evidence_expression_source"] = expr_src
    out["evidence_ccf_source"] = ccf_src
    out["self_dissimilarity"] = sd
    out["self_dissimilarity_confidence"] = [
        "UNAVAILABLE" if pd.isna(v) else "HIGH" for v in sd
    ]

    w = sp.weights
    w_expr = max(0.0, float(w.get("expression_norm", 0.0)))
    w_pres = max(0.0, float(w.get("presentation", 0.0)))
    w_ccf = max(0.0, float(w.get("ccf", 0.0)))
    w_sd = max(0.0, float(w.get("self_dissimilarity", 0.0)))
    w_sum = max(w_expr + w_pres + w_ccf + w_sd, 1e-12)
    w_expr, w_pres, w_ccf, w_sd = (
        w_expr / w_sum,
        w_pres / w_sum,
        w_ccf / w_sum,
        w_sd / w_sum,
    )
    resistance_weight = max(0.0, min(1.0, abs(float(sp.escape_penalty_weight))))

    rl = []
    for e_n, p, c, s, e_p in zip(expr_norm, pres, ccf, sd, esc, strict=True):
        s_term = 0.0 if pd.isna(s) else float(s)
        immunogenicity_blend = (
            w_expr * e_n + w_pres * p + w_ccf * c + w_sd * s_term
        )
        resistance_penalty = max(0.0, min(1.0, float(e_p) * resistance_weight))
        score = immunogenicity_blend * (1.0 - resistance_penalty)
        rl.append(max(0.0, min(1.0, score)))
    out["rl_priority"] = rl
    out["tier"] = [tier_from_score(x, rule_profile_id=rule_profile_id) for x in rl]

    out["scoring_profile"] = sp.profile_id
    out["scoring_version"] = sp.version
    out["rule_profile"] = rule_profile_id

    return out


def compute_rl_priority(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return apply_resistance_loop_engine(df, **kwargs)
