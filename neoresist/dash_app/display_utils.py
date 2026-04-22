from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from neoresist.paths import config_dir
from neoresist.profiles import load_scoring_profile
from neoresist.scoring import expression_norm_from_tpm

logger = logging.getLogger(__name__)

# Canonical component names used in the RL formula (scoring.py):
#   expression, presentation, ccf, self_dissimilarity, escape
# Display names used in all UI charts and labels:
#   Expression, Presentation / Binding, Clonality (CCF), Foreignness,
#   Escape resistance
# These must stay synchronized with scoring.py and the paper.
COMPONENT_DISPLAY_NAMES = {
    "expression": "Expression",
    "presentation": "Presentation / Binding",
    "ccf": "Clonality (CCF)",
    "self_dissimilarity": "Foreignness",
    "escape": "Escape resistance",
}
COMPONENT_DISPLAY_ORDER = (
    "expression",
    "presentation",
    "ccf",
    "self_dissimilarity",
    "escape",
)

DISPLAY_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "gene": ("gene", "gene_name", "Hugo_Symbol", "GENE", "Gene"),
    "mutant_peptide": ("mutant_peptide", "peptide", "Peptide", "MT_peptide", "mut_peptide", "HGVSp_Short"),
    "protein_change": ("protein_change", "source_hgvsp_short", "HGVSp_Short", "HGVSp", "hgvsp_short"),
    "binding_rank": ("binding_rank", "percentile_rank", "presentation_percentile"),
    "binding_affinity": ("binding_affinity", "affinity_nm", "affinity"),
    "presentation_composite": ("presentation_composite", "presentation_component", "presentation_score"),
}

DOT_COLORS = {
    "green": "#4CAF50",
    "yellow": "#FFC107",
    "red": "#F44336",
    "grey": "#9E9E9E",
}

_MISSING_TEXT = {"", "NA", "N/A", "NAN", "NONE", "NULL", "PENDING"}


def _is_missing_scalar(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().upper() in _MISSING_TEXT


def display_text(value: Any, default: str = "Pending") -> str:
    if _is_missing_scalar(value):
        return default
    return str(value)


def normalize_hla_display(hla_value: Any) -> str:
    text = "" if hla_value is None else str(hla_value).strip()
    if text.lower() in {"hla_test", "test", "unknown", "unk", "na", "n/a", "none", "null", ""}:
        return "HLA-UNK"
    return text


def is_unknown_hla(value: Any) -> bool:
    text = normalize_hla_display(value).upper()
    return text in {"", "HLA-UNK", "UNK", "UNKNOWN", "N/A", "NA", "NONE", "NULL"} or "*" not in text


def _series_missing(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.upper()
    return text.isin(_MISSING_TEXT)


def _coalesce_columns(df: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> pd.Series:
    result = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for name in aliases:
        if name not in df.columns:
            continue
        source = df[name]
        fill_mask = _series_missing(result) if result.dtype == "object" else result.isna()
        source_mask = ~_series_missing(source)
        result = result.mask(fill_mask & source_mask, source)
    return result


def _numeric_series(df: pd.DataFrame, column: str, fallback: str | None = None) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    if fallback and fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float")


def ensure_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for target, aliases in DISPLAY_COLUMN_ALIASES.items():
        current_missing = target not in out.columns or _series_missing(out[target]).all()
        if current_missing:
            out[target] = _coalesce_columns(out, target, aliases)
    if "hla_allele" in out.columns:
        out["hla_allele"] = out["hla_allele"].map(normalize_hla_display)
    if "resistance_composite" not in out.columns:
        out["resistance_composite"] = _numeric_series(out, "escape_penalty")
    if "expression_flag" not in out.columns:
        expr_tpm = _numeric_series(out, "expression_tpm")
        expr_score = _numeric_series(out, "expression_norm")
        flag = pd.Series(["UNKNOWN"] * len(out), index=out.index, dtype="object")
        flag = flag.mask(expr_tpm.ge(10) | expr_score.gt(0.7), "EXPRESSED")
        flag = flag.mask((expr_tpm.ge(1) & expr_tpm.lt(10)) | (expr_score.ge(0.3) & expr_score.le(0.7)), "LOW")
        flag = flag.mask((expr_tpm.ge(0) & expr_tpm.lt(1)) | (expr_score.ge(0) & expr_score.lt(0.3)), "ABSENT")
        out["expression_flag"] = flag
    if "clonality_class" not in out.columns:
        ccf = _numeric_series(out, "ccf")
        label = pd.Series(["UNKNOWN"] * len(out), index=out.index, dtype="object")
        label = label.mask(ccf.gt(0.8), "CLONAL")
        label = label.mask(ccf.ge(0.3) & ccf.le(0.8), "SUBCLONAL")
        label = label.mask(ccf.ge(0) & ccf.lt(0.3), "LOW_FREQUENCY")
        out["clonality_class"] = label
    if "hla_confidence" not in out.columns:
        out["hla_confidence"] = out.get("hla_allele", pd.Series(index=out.index, dtype="object")).map(
            lambda v: "pan-allele estimate" if is_unknown_hla(v) else "allele-specific"
        )
    available = pd.DataFrame(
        {
            "presentation": _numeric_series(out, "presentation_score", "presentation_composite"),
            "expression": _numeric_series(out, "expression_tpm", "expression_norm"),
            "clonality": _numeric_series(out, "ccf"),
            "recognition": _numeric_series(out, "self_dissimilarity"),
        }
    ).notna().sum(axis=1)
    out["score_confidence"] = pd.Series("minimal", index=out.index, dtype="object")
    out.loc[available.ge(2), "score_confidence"] = "partial"
    out.loc[available.ge(4), "score_confidence"] = "full"
    out["marker_opacity"] = out["score_confidence"].map({"full": 1.0, "partial": 0.6, "minimal": 0.3}).fillna(0.6)
    out["data_quality_zero"] = _numeric_series(out, "rl_priority").fillna(0.0).eq(0.0) & available.le(1)
    return out


def load_exclusion_rules(strategy_id: str) -> dict[str, float]:
    for candidate_id in (strategy_id, "rl_v1"):
        path = config_dir() / "scoring_profiles" / f"{candidate_id}.yaml"
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        rules = data.get("exclusion_rules") or {}
        if not isinstance(rules, dict):
            continue
        return {
            "min_expression_tpm": float(rules.get("min_expression_tpm", 1.0)),
            "max_escape_penalty": float(rules.get("max_escape_penalty", 0.5)),
            "min_self_dissimilarity": float(rules.get("min_self_dissimilarity", 0.1)),
        }
    return {}


def apply_display_policy(
    df: pd.DataFrame,
    *,
    tier1_above: float,
    tier2_above: float,
    exclusion_rules: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = ensure_display_columns(df)
    if out.empty:
        return out
    exclusion_rules = exclusion_rules or {}
    score = pd.to_numeric(out.get("composite_priority", out.get("rl_priority")), errors="coerce").fillna(0.0).clip(0, 1)
    tier = pd.Series(3, index=out.index, dtype="object")
    tier = tier.mask(score >= float(tier2_above), 2)
    tier = tier.mask(score >= float(tier1_above), 1)
    out["tier_note"] = ""
    unknown_hla = out["hla_confidence"].eq("pan-allele estimate")
    confirm_mask = unknown_hla & tier.eq(1)
    tier = tier.mask(confirm_mask, 2)
    out.loc[confirm_mask, "tier_note"] = "Tier 1 eligible — requires HLA typing to confirm"

    if exclusion_rules:
        expr = _numeric_series(out, "expression_tpm")
        escape = _numeric_series(out, "escape_penalty", "resistance_composite")
        self_d = _numeric_series(out, "self_dissimilarity")
        excluded = (
            expr.lt(float(exclusion_rules.get("min_expression_tpm", 1.0))).fillna(False)
            | escape.gt(float(exclusion_rules.get("max_escape_penalty", 0.5))).fillna(False)
            | self_d.lt(float(exclusion_rules.get("min_self_dissimilarity", 0.1))).fillna(False)
        )
        tier = tier.astype("object")
        tier = tier.mask(excluded, "EXCLUDED")
        out.loc[excluded, "tier_note"] = out.loc[excluded, "tier_note"].mask(
            out.loc[excluded, "tier_note"].eq(""),
            "Excluded by reviewer-facing default filters",
        )
    out["tier"] = tier
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _resolved_profile_id(row: pd.Series) -> str:
    for key in ("scoring_profile", "strategy_id"):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        return str(value)
    return "rl_v1"


def _normalized_strategy_weights(profile_id: str) -> tuple[float, float, float, float, float]:
    sp = load_scoring_profile(profile_id)
    w_expr = max(0.0, float(sp.weights.get("expression_norm", 0.0)))
    w_pres = max(0.0, float(sp.weights.get("presentation", 0.0)))
    w_ccf = max(0.0, float(sp.weights.get("ccf", 0.0)))
    w_sd = max(0.0, float(sp.weights.get("self_dissimilarity", 0.0)))
    w_sum = max(w_expr + w_pres + w_ccf + w_sd, 1e-12)
    resistance_weight = _clip01(abs(float(sp.escape_penalty_weight)))
    return w_expr / w_sum, w_pres / w_sum, w_ccf / w_sum, w_sd / w_sum, resistance_weight


def _expression_term(row: pd.Series, profile_id: str) -> float:
    existing = _float_or_none(row.get("expression_norm"))
    if existing is not None:
        return _clip01(existing)
    sp = load_scoring_profile(profile_id)
    stub_tpm = _float_or_none(row.get("expression_tpm")) or 0.0
    stub_norm = expression_norm_from_tpm(stub_tpm, cap=sp.expression_tpm_cap)
    real_tpm = _float_or_none(row.get("real_expression_tpm"))
    if real_tpm is None:
        return _clip01(stub_norm)
    real_norm = expression_norm_from_tpm(real_tpm, cap=sp.expression_tpm_cap)
    return _clip01(sp.blend_expression * real_norm + (1.0 - sp.blend_expression) * stub_norm)


def _ccf_term(row: pd.Series, profile_id: str) -> float:
    existing = _float_or_none(row.get("strategy_ccf"))
    if existing is not None:
        return _clip01(existing)
    stub_ccf = _clip01(_float_or_none(row.get("ccf")) or 0.0)
    real_ccf = _float_or_none(row.get("real_ccf"))
    if real_ccf is None:
        return stub_ccf
    sp = load_scoring_profile(profile_id)
    return _clip01(sp.blend_ccf * real_ccf + (1.0 - sp.blend_ccf) * stub_ccf)


def evidence_summary_binding(row: pd.Series) -> tuple[str, str]:
    score = _float_or_none(row.get("presentation_score", row.get("presentation_composite")))
    if score is None:
        return "Binding: pending", DOT_COLORS["grey"]
    if score > 0.8:
        return "Strong binder", DOT_COLORS["green"]
    if score >= 0.5:
        return "Moderate binder", DOT_COLORS["yellow"]
    return "Weak binder", DOT_COLORS["red"]


def evidence_summary_expression(row: pd.Series) -> tuple[str, str]:
    tpm = _float_or_none(row.get("expression_tpm"))
    score = _float_or_none(row.get("expression_norm"))
    if tpm is None and score is None:
        return "Expression: pending", DOT_COLORS["grey"]
    if (tpm is not None and tpm >= 10) or (score is not None and score > 0.7):
        return "Expressed", DOT_COLORS["green"]
    if (tpm is not None and 1 <= tpm < 10) or (score is not None and 0.3 <= score <= 0.7):
        return "Low expression", DOT_COLORS["yellow"]
    return "Not expressed", DOT_COLORS["red"]


def evidence_summary_clonality(row: pd.Series) -> tuple[str, str]:
    ccf = _float_or_none(row.get("ccf", row.get("ccf_score")))
    if ccf is None:
        return "Clonality: pending", DOT_COLORS["grey"]
    if ccf > 0.8:
        return "Clonal", DOT_COLORS["green"]
    if ccf >= 0.3:
        return "Subclonal", DOT_COLORS["yellow"]
    return "Low frequency", DOT_COLORS["red"]


def escape_risk_summary(row: pd.Series) -> tuple[str, str]:
    value = _float_or_none(row.get("escape_penalty", row.get("resistance_composite")))
    if value is None:
        return "Escape risk: unknown", DOT_COLORS["grey"]
    if value > 0.5:
        return "High escape risk", DOT_COLORS["red"]
    if value > 0.1:
        return "Moderate escape risk", DOT_COLORS["yellow"]
    return "Low escape risk", DOT_COLORS["green"]


def display_score_terms(row: pd.Series) -> tuple[float, float, float]:
    profile_id = _resolved_profile_id(row)
    w_expr, w_pres, w_ccf, w_sd, resistance_weight = _normalized_strategy_weights(profile_id)
    expression = _expression_term(row, profile_id)
    presentation = _clip01(_float_or_none(row.get("presentation_score", row.get("presentation_composite"))))
    ccf = _ccf_term(row, profile_id)
    self_dissimilarity = _clip01(_float_or_none(row.get("self_dissimilarity")))
    raw_escape = _float_or_none(row.get("escape_penalty"))

    immunogenicity = _clip01(
        w_expr * expression
        + w_pres * presentation
        + w_ccf * ccf
        + w_sd * self_dissimilarity
    )
    if raw_escape is None:
        resistance = _clip01(_float_or_none(row.get("resistance_penalty")))
    else:
        resistance = _clip01(raw_escape * resistance_weight)
    displayed_score = _clip01(immunogenicity * (1.0 - resistance))
    stored_score = _clip01(_float_or_none(row.get("composite_priority", row.get("rl_priority"))))
    if abs(displayed_score - stored_score) > 0.01:
        logger.warning(
            "Evidence math mismatch: %.3f x (1-%.3f) = %.3f but rl_priority = %.3f",
            immunogenicity,
            resistance,
            displayed_score,
            stored_score,
        )
    return immunogenicity, resistance, displayed_score


def tier_badge_text(value: Any) -> str:
    text = str(value).upper()
    if text in {"1", "TIER_1"}:
        return "TIER_1"
    if text in {"2", "TIER_2"}:
        return "TIER_2"
    if text in {"3", "TIER_3"}:
        return "TIER_3"
    return "EXCLUDED"
