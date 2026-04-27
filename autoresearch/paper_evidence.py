"""
Paper evidence pipeline for the cross-context thesis.

This script builds four paper-facing outputs:
  1. Context winner table
  2. Transfer matrix
  3. Selector-vs-fixed comparison
  4. Supplement package (leaderboard + exclusions)

It uses:
  - Exact TESLA-style TTIF / FR / AUPRC / TFA-mean on TESLA-like tasks
  - Patient-aware ranking metrics (mean patient AUC primary) on reranking tasks

Run:
    py -3.11 backend/strategy_engine/paper_evidence.py
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetEligibility:
    dataset: str
    context: str
    task_type: str
    eligible: bool
    leakage_clean: bool
    auditable_features: bool
    labels_real: bool
    main_table_allowed: bool
    reason: str


@dataclass(frozen=True)
class ContextSpec:
    context: str
    task_type: str
    discovery_datasets: tuple[str, ...]
    validation_datasets: tuple[str, ...]
    held_out_available: bool
    notes: str = ""


@dataclass
class StrategySpec:
    strategy_name: str
    display_name: str
    strategy_origin: str
    origin_dataset: str | None
    family: str
    discovery_kind: str
    components: list[tuple[str, float, str]]
    is_existing: bool
    fixed_universal_candidate: bool
    eligible_for_context_table: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_name": self.strategy_name,
            "display_name": self.display_name,
            "strategy_origin": self.strategy_origin,
            "origin_dataset": self.origin_dataset,
            "family": self.family,
            "discovery_kind": self.discovery_kind,
            "components": self.components,
            "is_existing": self.is_existing,
            "fixed_universal_candidate": self.fixed_universal_candidate,
            "eligible_for_context_table": self.eligible_for_context_table,
            "notes": self.notes,
        }


@dataclass
class StrategyEvaluation:
    context: str
    dataset: str
    task_type: str
    strategy_name: str
    display_name: str
    strategy_origin: str
    origin_dataset: str | None
    selection_status: str
    validation_status: str
    eligibility_status: str
    primary_metric_name: str
    primary_metric: float | None
    secondary_metrics: dict[str, float | None]
    bootstrap_ci: dict[str, float | None]
    feature_coverage_notes: dict[str, object]
    notes: str = ""
    components: list[tuple[str, float, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "strategy_name": self.strategy_name,
            "display_name": self.display_name,
            "strategy_origin": self.strategy_origin,
            "origin_dataset": self.origin_dataset,
            "selection_status": self.selection_status,
            "validation_status": self.validation_status,
            "eligibility_status": self.eligibility_status,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "bootstrap_ci": self.bootstrap_ci,
            "feature_coverage_notes": self.feature_coverage_notes,
            "notes": self.notes,
            "components": self.components,
        }


CONTEXT_SPECS: dict[str, ContextSpec] = {
    "melanoma": ContextSpec(
        context="melanoma",
        task_type="reranking",
        discovery_datasets=("ott_2017",),
        validation_datasets=("sahin_2017",),
        held_out_available=True,
        notes="Discover on Ott, validate on Sahin.",
    ),
    "gbm": ContextSpec(
        context="gbm",
        task_type="reranking",
        discovery_datasets=("hilf_2019",),
        validation_datasets=(),
        held_out_available=False,
        notes="Hilf is eligible; Keskin is excluded from main evidence due insufficient negatives.",
    ),
    "pancreatic": ContextSpec(
        context="pancreatic",
        task_type="reranking",
        discovery_datasets=("rojas_2023",),
        validation_datasets=(),
        held_out_available=False,
        notes="Single-cohort context; winner is provisional.",
    ),
    "mixed_tesla": ContextSpec(
        context="mixed_tesla",
        task_type="tesla_like",
        discovery_datasets=("tesla_2020",),
        validation_datasets=(),
        held_out_available=False,
        notes="Use exact TESLA-style metrics; winner is provisional without a held-out cohort.",
    ),
    "muller_nci": ContextSpec(
        context="muller_nci",
        task_type="reranking",
        discovery_datasets=("muller_nci",),
        validation_datasets=(),
        held_out_available=False,
        notes="NCI multi-cancer cohort; used as transfer-only dataset for cross-cancer evaluation.",
    ),
    "bladder_support": ContextSpec(
        context="bladder_support",
        task_type="reranking",
        discovery_datasets=("harbst2022_bladder",),
        validation_datasets=(),
        held_out_available=False,
        notes="Curated Harbst 2022 bladder slice from the Borch/IMPROVE source family; support-only context for heterogeneity analysis.",
    ),
}

DATASET_CONTEXT: dict[str, str] = {
    "ott_2017": "melanoma",
    "sahin_2017": "melanoma",
    "hilf_2019": "gbm",
    "keskin_2019": "gbm",
    "rojas_2023": "pancreatic",
    "tesla_2020": "mixed_tesla",
    "muller_nci": "muller_nci",
    "harbst2022_bladder": "bladder_support",
}

DATASET_ELIGIBILITY: dict[str, DatasetEligibility] = {
    "ott_2017": DatasetEligibility(
        dataset="ott_2017",
        context="melanoma",
        task_type="reranking",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=True,
        reason="Real labels, audited feature table, used only as melanoma discovery cohort.",
    ),
    "sahin_2017": DatasetEligibility(
        dataset="sahin_2017",
        context="melanoma",
        task_type="reranking",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=True,
        reason="Real labels, audited feature table, valid held-out melanoma validation cohort.",
    ),
    "hilf_2019": DatasetEligibility(
        dataset="hilf_2019",
        context="gbm",
        task_type="reranking",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=True,
        reason="Real labels, auditable features, enough patients for LOPO ranking evaluation.",
    ),
    "keskin_2019": DatasetEligibility(
        dataset="keskin_2019",
        context="gbm",
        task_type="reranking",
        eligible=False,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=False,
        reason="Excluded: only two patients and insufficient non-immunogenic rows for stable LOPO evidence.",
    ),
    "rojas_2023": DatasetEligibility(
        dataset="rojas_2023",
        context="pancreatic",
        task_type="reranking",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=True,
        reason="Real labels, auditable features, enough patients for LOPO ranking evaluation; no held-out partner available.",
    ),
    "tesla_2020": DatasetEligibility(
        dataset="tesla_2020",
        context="mixed_tesla",
        task_type="tesla_like",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=True,
        reason="Real TESLA tested candidates with exact TESLA-style metrics; no held-out partner available.",
    ),
    "harbst2022_bladder": DatasetEligibility(
        dataset="harbst2022_bladder",
        context="bladder_support",
        task_type="reranking",
        eligible=True,
        leakage_clean=True,
        auditable_features=True,
        labels_real=True,
        main_table_allowed=False,
        reason="Curated bladder slice from the Borch/IMPROVE source family; support-only heterogeneity analysis, not independent validation.",
    ),
}

FAMILY_FEATURES: dict[str, list[str]] = {
    "binding_presentation": ["bind_log50k", "presentation_score", "binding_sigmoid"],
    "expression": ["expression_log2", "expression_zscore"],
    "tcr": ["tcr_volume_mean", "tcr_charge_sum", "tcr_hydro_mean", "tcr_volume_diff", "tcr_charge_diff", "tcr_hydro_diff"],
    "sequence": ["self_dissimilarity", "hamming_distance", "blosum62_score", "blosum62_at_mutation", "calis_simplified", "aliphatic_index"],
}

EXCLUDED_STRATEGIES: list[dict[str, object]] = [
    {
        "strategy_name": "melanoma_ml_v1",
        "reason": "Excluded from this pipeline because the saved model was trained on Ott+Sahin jointly; the paper-evidence pipeline requires split-safe retraining or fixed formulas.",
    },
    {
        "strategy_name": "gbm_ml_v1",
        "reason": "Excluded from this pipeline because the saved model is a full-context artifact and does not satisfy the strict nested selection rule.",
    },
    {
        "strategy_name": "pancreatic_ml_v1",
        "reason": "Excluded from this pipeline because the saved model is a full-context artifact and does not satisfy the strict nested selection rule.",
    },
    {
        "strategy_name": "mixed_ml_v1",
        "reason": "Excluded from this pipeline because the saved model is a full-context artifact and does not satisfy the strict nested selection rule.",
    },
]


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pick_eval_dataset(row: dict[str, object]) -> str:
    validation = row.get("validation_dataset")
    if validation is not None and not pd.isna(validation):
        return str(validation)
    discovery = row.get("discovery_dataset")
    if discovery is None or pd.isna(discovery):
        raise ValueError(f"Row missing discovery dataset: {row}")
    return str(discovery)


def make_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None:
        return None
    try:
        if pd.isna(obj):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return obj


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if mask.sum() < 2:
        return None
    yy = y_true[mask]
    ss = y_score[mask]
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, ss))


def bootstrap_mean_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float | None]:
    if not values:
        return {"ci_lo": None, "ci_hi": None, "n": 0}
    rng = np.random.RandomState(seed)
    vals = np.array(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(float(np.mean(sample)))
    return {
        "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
        "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
        "n": int(len(vals)),
    }


def normalize_hla(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    s = str(value).strip().upper()
    for old in ["HLA-", "*", ":", "-"]:
        s = s.replace(old, "")
    return s or "UNKNOWN"


def load_dataset(dataset: str) -> pd.DataFrame:
    path = ARTIFACTS / f"{dataset}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "immunogenic" not in df.columns:
        raise ValueError(f"{dataset} missing immunogenic column")
    df = df[df["immunogenic"].isin([0, 1])].copy()
    if dataset == "tesla_2020" and "usable_for_training" in df.columns:
        mask = df["usable_for_training"].fillna(False).astype(bool)
        if mask.sum() > 0:
            df = df[mask].copy()
    return ensure_derived_columns(df)


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "presentation_score_el" not in out.columns:
        if "netmhcpan_score_el" in out.columns:
            out["presentation_score_el"] = pd.to_numeric(out["netmhcpan_score_el"], errors="coerce")
        elif "presentation_score" in out.columns:
            out["presentation_score_el"] = pd.to_numeric(out["presentation_score"], errors="coerce")
        else:
            out["presentation_score_el"] = np.nan
    if "binding_log50k" not in out.columns:
        nm_source = out.get("binding_nm")
        nm = pd.to_numeric(nm_source, errors="coerce") if nm_source is not None else pd.Series(np.nan, index=out.index)
        if nm.notna().any():
            out["binding_log50k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
        else:
            ba_rank = pd.to_numeric(out.get("ba_rank"), errors="coerce") if "ba_rank" in out.columns else pd.Series(np.nan, index=out.index)
            el_rank = pd.to_numeric(out.get("el_rank"), errors="coerce") if "el_rank" in out.columns else pd.Series(np.nan, index=out.index)
            fallback_rank = ba_rank.where(ba_rank.notna(), el_rank)
            out["binding_log50k"] = (1.0 / (1.0 + fallback_rank)).clip(0, 1)
    if "bind_log50k" not in out.columns:
        out["bind_log50k"] = pd.to_numeric(out.get("binding_log50k"), errors="coerce")
    if "binding_stability" not in out.columns:
        out["binding_stability"] = np.nan
    if "expression_log2" not in out.columns:
        expr_source = out.get("expression_tpm")
        if expr_source is None and "tpm" in out.columns:
            expr_source = out["tpm"]
        expr = pd.Series(pd.to_numeric(expr_source, errors="coerce"), index=out.index)
        out["expression_log2"] = np.log2(expr.clip(lower=0) + 1)
    if "dai_agretopicity" not in out.columns:
        if "dai" in out.columns:
            out["dai_agretopicity"] = pd.to_numeric(out["dai"], errors="coerce")
        else:
            out["dai_agretopicity"] = np.nan
    if "presentation_score" not in out.columns:
        if "el_rank" in out.columns:
            out["presentation_score"] = 1.0 / (1.0 + pd.to_numeric(out["el_rank"], errors="coerce"))
        else:
            out["presentation_score"] = np.nan
    if "calis_simplified" not in out.columns:
        out["calis_simplified"] = np.nan
    if "self_dissimilarity" not in out.columns:
        out["self_dissimilarity"] = np.nan
    return out


def patient_percentile_score(series: pd.Series, higher_is_better: bool, groups: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        vals = -vals

    def _rank(block: pd.Series) -> pd.Series:
        valid = pd.to_numeric(block, errors="coerce")
        ranked = valid.rank(pct=True, method="average")
        return ranked

    return vals.groupby(groups).transform(_rank)


def score_strategy(df: pd.DataFrame, strategy: StrategySpec) -> tuple[pd.Series, dict[str, object]]:
    parts: list[pd.Series] = []
    used_components: list[dict[str, object]] = []
    for feature, weight, direction in strategy.components:
        if feature not in df.columns:
            continue
        raw = pd.to_numeric(df[feature], errors="coerce")
        non_null = int(raw.notna().sum())
        if non_null < 5:
            continue
        higher_is_better = direction == "desc"
        part = patient_percentile_score(raw, higher_is_better=higher_is_better, groups=df["patient_id"])
        if part.notna().sum() < 5:
            continue
        parts.append(part * weight)
        used_components.append(
            {
                "feature": feature,
                "weight": weight,
                "direction": direction,
                "coverage": non_null,
            }
        )

    if not parts:
        return pd.Series(np.nan, index=df.index), {"used_components": [], "status": "no_valid_components"}

    total_weight = float(sum(x["weight"] for x in used_components))
    score = sum(parts) / total_weight
    return score, {"used_components": used_components, "status": "ok"}


def evaluate_reranking(df: pd.DataFrame, score: pd.Series) -> dict[str, object]:
    work = df.copy()
    work["_score"] = pd.to_numeric(score, errors="coerce")
    patient_metrics: list[dict[str, float | str | None]] = []
    pooled_y: list[int] = []
    pooled_s: list[float] = []

    for patient_id, grp in work.groupby("patient_id"):
        y = pd.to_numeric(grp["immunogenic"], errors="coerce").values.astype(float)
        s = pd.to_numeric(grp["_score"], errors="coerce").values.astype(float)
        mask = np.isfinite(y) & np.isfinite(s)
        y = y[mask]
        s = s[mask]
        if len(y) < 2 or len(np.unique(y)) < 2:
            continue
        order = np.argsort(-s)
        y_sorted = y[order]
        n_pos = int(y.sum())
        auc = roc_auc_score(y, s)
        recall10 = float(y_sorted[:10].sum() / max(n_pos, 1))
        recall20 = float(y_sorted[:20].sum() / max(n_pos, 1))
        patient_metrics.append(
            {
                "patient_id": str(patient_id),
                "auc": float(auc),
                "recall_at_10": recall10,
                "recall_at_20": recall20,
            }
        )
        pooled_y.extend(y.astype(int).tolist())
        pooled_s.extend(s.tolist())

    aucs = [safe_float(p["auc"]) for p in patient_metrics if safe_float(p["auc"]) is not None]
    r10 = [safe_float(p["recall_at_10"]) for p in patient_metrics if safe_float(p["recall_at_10"]) is not None]
    r20 = [safe_float(p["recall_at_20"]) for p in patient_metrics if safe_float(p["recall_at_20"]) is not None]
    mean_patient_auc = float(np.mean(aucs)) if aucs else None
    pooled_auc = safe_auc(np.array(pooled_y, dtype=float), np.array(pooled_s, dtype=float)) if pooled_y else None

    return {
        "primary_metric_name": "mean_patient_auc",
        "primary_metric": mean_patient_auc,
        "secondary_metrics": {
            "pooled_auc": pooled_auc,
            "mean_recall_at_10": float(np.mean(r10)) if r10 else None,
            "mean_recall_at_20": float(np.mean(r20)) if r20 else None,
        },
        "bootstrap_ci": bootstrap_mean_ci([x for x in aucs if x is not None]),
        "patient_metrics": patient_metrics,
    }


def tesla_patient_metrics(grp: pd.DataFrame) -> dict[str, float | None]:
    work = grp.copy()
    work = work[np.isfinite(pd.to_numeric(work["_score"], errors="coerce"))].copy()
    if work.empty:
        return {"TTIF": None, "FR": None, "AUPRC": None, "TFA_mean": None}
    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    work["rank"] = np.arange(1, len(work) + 1)
    validated = work["immunogenic"].astype(int).values
    n_true = int(validated.sum())
    if n_true == 0:
        return {"TTIF": 0.0, "FR": 0.0, "AUPRC": 0.0, "TFA_mean": 0.0}

    top_twenty = work[work["rank"] < 20]
    ttif = float(top_twenty["immunogenic"].sum() / len(top_twenty)) if len(top_twenty) else None
    top_hundred = work[work["rank"] <= 100]
    fr = float(top_hundred["immunogenic"].sum() / n_true) if len(top_hundred) else None
    auprc = float(average_precision_score(validated, work["_score"].astype(float).values))
    metrics = [m for m in [ttif, fr, auprc] if m is not None]
    tfa_mean = float(np.mean(metrics)) if metrics else None
    return {"TTIF": ttif, "FR": fr, "AUPRC": auprc, "TFA_mean": tfa_mean}


def evaluate_tesla(df: pd.DataFrame, score: pd.Series) -> dict[str, object]:
    work = df.copy()
    work["_score"] = pd.to_numeric(score, errors="coerce")
    patient_metrics: list[dict[str, object]] = []
    for patient_id, grp in work.groupby("patient_id"):
        metrics = tesla_patient_metrics(grp)
        metrics["patient_id"] = str(patient_id)
        patient_metrics.append(metrics)

    def _values(key: str) -> list[float]:
        return [safe_float(x.get(key)) for x in patient_metrics if safe_float(x.get(key)) is not None]

    tfa_values = _values("TFA_mean")
    return {
        "primary_metric_name": "TFA_mean",
        "primary_metric": float(np.mean(tfa_values)) if tfa_values else None,
        "secondary_metrics": {
            "TTIF": float(np.mean(_values("TTIF"))) if _values("TTIF") else None,
            "FR": float(np.mean(_values("FR"))) if _values("FR") else None,
            "AUPRC": float(np.mean(_values("AUPRC"))) if _values("AUPRC") else None,
        },
        "bootstrap_ci": bootstrap_mean_ci([x for x in tfa_values if x is not None]),
        "patient_metrics": patient_metrics,
    }


def evaluate_dataset(df: pd.DataFrame, strategy: StrategySpec, task_type: str) -> tuple[dict[str, object], dict[str, object]]:
    score, coverage = score_strategy(df, strategy)
    if coverage.get("status") != "ok":
        return {
            "primary_metric_name": "invalid",
            "primary_metric": None,
            "secondary_metrics": {},
            "bootstrap_ci": {"ci_lo": None, "ci_hi": None, "n": 0},
            "patient_metrics": [],
        }, coverage
    if task_type == "tesla_like":
        return evaluate_tesla(df, score), coverage
    return evaluate_reranking(df, score), coverage


def feature_orientation(df: pd.DataFrame, feature: str, task_type: str) -> tuple[str, float | None]:
    strategy = StrategySpec(
        strategy_name=f"single_{feature}",
        display_name=f"Single {feature}",
        strategy_origin="discovered",
        origin_dataset=None,
        family="single_feature",
        discovery_kind="single_feature",
        components=[(feature, 1.0, "desc")],
        is_existing=False,
        fixed_universal_candidate=False,
    )
    desc_metrics, _ = evaluate_dataset(df, strategy, task_type)
    desc_score = safe_float(desc_metrics["primary_metric"])

    strategy.components = [(feature, 1.0, "asc")]
    asc_metrics, _ = evaluate_dataset(df, strategy, task_type)
    asc_score = safe_float(asc_metrics["primary_metric"])

    if asc_score is None and desc_score is None:
        return "desc", None
    if asc_score is None:
        return "desc", desc_score
    if desc_score is None:
        return "asc", asc_score
    if desc_score >= asc_score:
        return "desc", desc_score
    return "asc", asc_score


def existing_library() -> list[StrategySpec]:
    return [
        StrategySpec(
            strategy_name="binding_only",
            display_name="Binding Only",
            strategy_origin="pre_specified",
            origin_dataset=None,
            family="binding",
            discovery_kind="fixed_existing",
            components=[("bind_log50k", 1.0, "desc")],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Universal baseline.",
        ),
        StrategySpec(
            strategy_name="rl_v1_original",
            display_name="RL v1 Original",
            strategy_origin="pre_specified",
            origin_dataset=None,
            family="mixed",
            discovery_kind="fixed_existing",
            components=[
                ("presentation_score", 0.50, "desc"),
                ("expression_log2", 0.33, "desc"),
                ("self_dissimilarity", 0.17, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Original RL v1 style blend.",
        ),
        StrategySpec(
            strategy_name="binding_plus_calis",
            display_name="Binding + Calis",
            strategy_origin="pre_specified",
            origin_dataset=None,
            family="sequence",
            discovery_kind="fixed_existing",
            components=[("bind_log50k", 0.6, "desc"), ("calis_simplified", 0.4, "desc")],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Published-style binding plus sequence prior.",
        ),
        StrategySpec(
            strategy_name="rl_tcr_v1_from_ott_locked",
            display_name="rl_tcr_v1 (Ott-locked)",
            strategy_origin="discovered_existing",
            origin_dataset="ott_2017",
            family="tcr",
            discovery_kind="locked_existing",
            components=[
                ("bind_log50k", 0.4, "desc"),
                ("tcr_volume_mean", 0.2, "desc"),
                ("tcr_charge_diff", 0.2, "desc"),
                ("tcr_hydro_mean", 0.2, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=False,
            notes="Existing Ott-locked TCR blend; discovery cohort is Ott.",
        ),
        StrategySpec(
            strategy_name="rl_expression_v1_from_tesla_locked",
            display_name="rl_expression_v1 (TESLA-locked)",
            strategy_origin="discovered_existing",
            origin_dataset="tesla_2020",
            family="expression",
            discovery_kind="locked_existing",
            components=[("bind_log50k", 0.75, "desc"), ("expression_log2", 0.25, "desc")],
            is_existing=True,
            fixed_universal_candidate=False,
            notes="Existing TESLA-locked expression blend; discovery cohort is TESLA.",
        ),
        StrategySpec(
            strategy_name="neoguider_v1_recipe",
            display_name="NeoGuider v1 recipe",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[
                ("presentation_score_el", 0.20, "desc"),
                ("bind_log50k", 0.20, "desc"),
                ("binding_stability", 0.20, "desc"),
                ("expression_log2", 0.20, "desc"),
                ("dai_agretopicity", 0.20, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Fixed NeoGuider-inspired five-feature recipe using the local feature map, without using the pre-trained global LR model.",
        ),
        StrategySpec(
            strategy_name="neoguider_v2_tesla_adapted",
            display_name="NeoGuider v2 TESLA Adapted",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[
                ("bind_log50k", 0.40, "desc"),
                ("expression_log2", 0.30, "desc"),
                ("calis_simplified", 0.30, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Repo-local NeoGuider-inspired bind+expr+calis variant from configs/strategies_extra/neoguider_v2_tessla_adapted.yaml.",
        ),
        StrategySpec(
            strategy_name="neoguider_v2_melanoma",
            display_name="NeoGuider v2 Melanoma",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[
                ("bind_log50k", 0.40, "desc"),
                ("tcr_hydro_mean", 0.20, "desc"),
                ("tcr_volume_mean", 0.20, "desc"),
                ("tcr_charge_sum", 0.20, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Repo-local NeoGuider-inspired melanoma/TCR chemistry variant from configs/strategies_extra/neoguider_v2_melanoma.yaml.",
        ),
        StrategySpec(
            strategy_name="neoguider_bind_expr",
            display_name="NeoGuider bind+expr",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[("bind_log50k", 0.60, "desc"), ("expression_log2", 0.40, "desc")],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="Coverage-friendly NeoGuider-style ablation emphasizing abundance over agretopicity/stability.",
        ),
        StrategySpec(
            strategy_name="neoguider_bind_expr_calis",
            display_name="NeoGuider bind+expr+calis",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[
                ("bind_log50k", 0.45, "desc"),
                ("expression_log2", 0.25, "desc"),
                ("calis_simplified", 0.30, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="NeoGuider-style adaptation that swaps agretopicity for Calis when coverage is broader.",
        ),
        StrategySpec(
            strategy_name="neoguider_bind_tcrchem",
            display_name="NeoGuider bind+TCR chemistry",
            strategy_origin="pre_specified_neoguider_family",
            origin_dataset=None,
            family="neoguider",
            discovery_kind="fixed_existing",
            components=[
                ("bind_log50k", 0.40, "desc"),
                ("tcr_hydro_mean", 0.20, "desc"),
                ("tcr_volume_mean", 0.20, "desc"),
                ("tcr_charge_sum", 0.20, "desc"),
            ],
            is_existing=True,
            fixed_universal_candidate=True,
            notes="NeoGuider-guided recognition variant inspired by the local variant-screen family.",
        ),
    ]


def build_discovered_candidates(df: pd.DataFrame, context: str, origin_dataset: str, task_type: str) -> list[StrategySpec]:
    best_family: dict[str, tuple[str, str, float | None]] = {}
    for family, features in FAMILY_FEATURES.items():
        candidates = []
        for feature in features:
            if feature not in df.columns:
                continue
            direction, metric = feature_orientation(df, feature, task_type)
            if metric is None:
                continue
            candidates.append((feature, direction, metric))
        if candidates:
            candidates.sort(key=lambda x: x[2] if x[2] is not None else -1, reverse=True)
            best_family[family] = candidates[0]

    discovered: list[StrategySpec] = []
    for family, candidate in best_family.items():
        feat, direction, _ = candidate
        discovered.append(
            StrategySpec(
                strategy_name=f"{context}_{family}_single",
                display_name=f"{context} {family} single",
                strategy_origin="discovered_training_only",
                origin_dataset=origin_dataset,
                family=family,
                discovery_kind="family_single",
                components=[(feat, 1.0, direction)],
                is_existing=False,
                fixed_universal_candidate=False,
                notes=f"Top {family} feature selected on {origin_dataset}.",
            )
        )

    binding = best_family.get("binding_presentation")
    expression = best_family.get("expression")
    tcr = best_family.get("tcr")
    sequence = best_family.get("sequence")

    def add_mix(name: str, display: str, components: list[tuple[str, float, str]], family: str) -> None:
        discovered.append(
            StrategySpec(
                strategy_name=name,
                display_name=display,
                strategy_origin="discovered_training_only",
                origin_dataset=origin_dataset,
                family=family,
                discovery_kind="bounded_grid_search",
                components=components,
                is_existing=False,
                fixed_universal_candidate=False,
                notes=f"Bounded interpretable mixture discovered on {origin_dataset}.",
            )
        )

    if binding and expression:
        bf, bd, _ = binding
        ef, ed, _ = expression
        add_mix(f"{context}_bind_expr_80_20", f"{context} bind+expr 80/20", [(bf, 0.8, bd), (ef, 0.2, ed)], "binding_expression")
        add_mix(f"{context}_bind_expr_60_40", f"{context} bind+expr 60/40", [(bf, 0.6, bd), (ef, 0.4, ed)], "binding_expression")
    if binding and tcr:
        bf, bd, _ = binding
        tf, td, _ = tcr
        add_mix(f"{context}_bind_tcr_80_20", f"{context} bind+tcr 80/20", [(bf, 0.8, bd), (tf, 0.2, td)], "binding_tcr")
        add_mix(f"{context}_bind_tcr_60_40", f"{context} bind+tcr 60/40", [(bf, 0.6, bd), (tf, 0.4, td)], "binding_tcr")
    if binding and sequence:
        bf, bd, _ = binding
        sf, sd, _ = sequence
        add_mix(f"{context}_bind_seq_80_20", f"{context} bind+seq 80/20", [(bf, 0.8, bd), (sf, 0.2, sd)], "binding_sequence")
        add_mix(f"{context}_bind_seq_60_40", f"{context} bind+seq 60/40", [(bf, 0.6, bd), (sf, 0.4, sd)], "binding_sequence")
    if binding and expression and tcr:
        bf, bd, _ = binding
        ef, ed, _ = expression
        tf, td, _ = tcr
        add_mix(
            f"{context}_bind_expr_tcr",
            f"{context} bind+expr+tcr",
            [(bf, 0.5, bd), (ef, 0.25, ed), (tf, 0.25, td)],
            "hybrid",
        )
    if binding and expression and tcr and sequence:
        bf, bd, _ = binding
        ef, ed, _ = expression
        tf, td, _ = tcr
        sf, sd, _ = sequence
        add_mix(
            f"{context}_all_signals",
            f"{context} all-signal mix",
            [(bf, 0.4, bd), (ef, 0.2, ed), (tf, 0.2, td), (sf, 0.2, sd)],
            "hybrid",
        )

    neoguider_features = [
        "presentation_score_el",
        "bind_log50k",
        "binding_stability",
        "expression_log2",
        "dai_agretopicity",
        "calis_simplified",
        "tcr_hydro_mean",
        "tcr_volume_mean",
        "tcr_charge_sum",
    ]
    neo_best: dict[str, tuple[str, str, float | None]] = {}
    for feature in neoguider_features:
        if feature not in df.columns:
            continue
        direction, metric = feature_orientation(df, feature, task_type)
        if metric is None:
            continue
        neo_best[feature] = (feature, direction, metric)

    def neo_component(feature: str, default_direction: str = "desc") -> tuple[str, str] | None:
        picked = neo_best.get(feature)
        if picked is None:
            return None
        return picked[0], picked[1] or default_direction

    bind = neo_component("bind_log50k")
    scoreel = neo_component("presentation_score_el")
    bindstab = neo_component("binding_stability")
    expr = neo_component("expression_log2")
    agret = neo_component("dai_agretopicity")
    calis = neo_component("calis_simplified")
    hydro = neo_component("tcr_hydro_mean")
    vol = neo_component("tcr_volume_mean")
    charge = neo_component("tcr_charge_sum")

    def add_neo(name: str, display: str, comps: list[tuple[str, float, str]]) -> None:
        discovered.append(
            StrategySpec(
                strategy_name=name,
                display_name=display,
                strategy_origin="discovered_training_only_neoguider_guided",
                origin_dataset=origin_dataset,
                family="neoguider_guided",
                discovery_kind="bounded_neoguider_guided",
                components=comps,
                is_existing=False,
                fixed_universal_candidate=False,
                notes=f"NeoGuider-guided bounded variant selected on {origin_dataset}.",
            )
        )

    if bind and expr:
        add_neo(f"{context}_neo_bind_expr", f"{context} neo bind+expr", [(bind[0], 0.6, bind[1]), (expr[0], 0.4, expr[1])])
    if bind and expr and calis:
        add_neo(
            f"{context}_neo_bind_expr_calis",
            f"{context} neo bind+expr+calis",
            [(bind[0], 0.45, bind[1]), (expr[0], 0.25, expr[1]), (calis[0], 0.30, calis[1])],
        )
    if scoreel and bind and expr and agret:
        add_neo(
            f"{context}_neo_recipe_core",
            f"{context} neo recipe core",
            [(scoreel[0], 0.25, scoreel[1]), (bind[0], 0.25, bind[1]), (expr[0], 0.25, expr[1]), (agret[0], 0.25, agret[1])],
        )
    if bind and hydro and vol and charge:
        add_neo(
            f"{context}_neo_bind_tcrchem",
            f"{context} neo bind+tcrchem",
            [(bind[0], 0.4, bind[1]), (hydro[0], 0.2, hydro[1]), (vol[0], 0.2, vol[1]), (charge[0], 0.2, charge[1])],
        )
    if scoreel and bind and bindstab and expr:
        add_neo(
            f"{context}_neo_scoreel_stab_expr",
            f"{context} neo scoreel+stab+expr",
            [(scoreel[0], 0.30, scoreel[1]), (bind[0], 0.25, bind[1]), (bindstab[0], 0.20, bindstab[1]), (expr[0], 0.25, expr[1])],
        )
    return discovered


def discovery_primary_dataset(spec: ContextSpec) -> str:
    return spec.discovery_datasets[0]


def evaluate_strategies_for_dataset(
    df: pd.DataFrame,
    strategies: list[StrategySpec],
    context: str,
    dataset: str,
    task_type: str,
    selection_status: str,
    validation_status: str,
) -> list[StrategyEvaluation]:
    evaluations: list[StrategyEvaluation] = []
    for strategy in strategies:
        metrics, coverage = evaluate_dataset(df, strategy, task_type)
        status = "eligible" if metrics["primary_metric"] is not None else "invalid"
        evaluations.append(
            StrategyEvaluation(
                context=context,
                dataset=dataset,
                task_type=task_type,
                strategy_name=strategy.strategy_name,
                display_name=strategy.display_name,
                strategy_origin=strategy.strategy_origin,
                origin_dataset=strategy.origin_dataset,
                selection_status=selection_status,
                validation_status=validation_status,
                eligibility_status=status,
                primary_metric_name=str(metrics["primary_metric_name"]),
                primary_metric=safe_float(metrics["primary_metric"]),
                secondary_metrics={k: safe_float(v) for k, v in metrics["secondary_metrics"].items()},
                bootstrap_ci={k: safe_float(v) for k, v in metrics["bootstrap_ci"].items()},
                feature_coverage_notes=coverage,
                components=list(strategy.components),
                notes=strategy.notes,
            )
        )
    return evaluations


def select_winner(evaluations: list[StrategyEvaluation]) -> StrategyEvaluation | None:
    eligible = [e for e in evaluations if e.eligibility_status == "eligible" and e.primary_metric is not None]
    if not eligible:
        return None
    eligible.sort(key=lambda e: e.primary_metric if e.primary_metric is not None else -1, reverse=True)
    return eligible[0]


def find_strategy(name: str, strategies: list[StrategySpec]) -> StrategySpec:
    for strategy in strategies:
        if strategy.strategy_name == name:
            return strategy
    raise KeyError(name)


def context_pipeline(context: str, spec: ContextSpec) -> dict[str, object]:
    discovery_dataset = discovery_primary_dataset(spec)
    discovery_df = load_dataset(discovery_dataset)
    existing = existing_library()
    discovered = build_discovered_candidates(discovery_df, context=context, origin_dataset=discovery_dataset, task_type=spec.task_type)
    strategy_library = existing + discovered

    discovery_evaluations = evaluate_strategies_for_dataset(
        discovery_df,
        strategy_library,
        context=context,
        dataset=discovery_dataset,
        task_type=spec.task_type,
        selection_status="context_discovery",
        validation_status="discovery",
    )
    winner_on_discovery = select_winner(discovery_evaluations)

    validation_rows: list[dict[str, object]] = []
    validation_evaluations: list[StrategyEvaluation] = []

    if spec.validation_datasets and winner_on_discovery is not None:
        winner_strategy = find_strategy(winner_on_discovery.strategy_name, strategy_library)
        comparator_names = {
            "binding_only",
            "rl_v1_original",
            "binding_plus_calis",
            winner_strategy.strategy_name,
        }
        compare_strategies = [s for s in strategy_library if s.strategy_name in comparator_names]
        for dataset in spec.validation_datasets:
            df = load_dataset(dataset)
            rows = evaluate_strategies_for_dataset(
                df,
                compare_strategies,
                context=context,
                dataset=dataset,
                task_type=spec.task_type,
                selection_status="locked_after_discovery",
                validation_status="held_out_validation",
            )
            validation_evaluations.extend(rows)
            winner_row = next((r for r in rows if r.strategy_name == winner_strategy.strategy_name), None)
            binding_row = next((r for r in rows if r.strategy_name == "binding_only"), None)
            if winner_row is not None:
                validation_rows.append(
                    {
                        "dataset": dataset,
                        "winner_metric": winner_row.primary_metric,
                        "binding_metric": binding_row.primary_metric if binding_row else None,
                        "delta_vs_binding": (
                            winner_row.primary_metric - binding_row.primary_metric
                            if winner_row.primary_metric is not None and binding_row and binding_row.primary_metric is not None
                            else None
                        ),
                    }
                )
    elif winner_on_discovery is not None:
        validation_rows.append(
            {
                "dataset": discovery_dataset,
                "winner_metric": winner_on_discovery.primary_metric,
                "binding_metric": next(
                    (r.primary_metric for r in discovery_evaluations if r.strategy_name == "binding_only"),
                    None,
                ),
                "delta_vs_binding": None,
            }
        )

    return {
        "context": context,
        "spec": spec,
        "strategy_library": [s.to_dict() for s in strategy_library],
        "discovery_evaluations": [e.to_dict() for e in discovery_evaluations],
        "validation_evaluations": [e.to_dict() for e in validation_evaluations],
        "winner_on_discovery": winner_on_discovery.to_dict() if winner_on_discovery else None,
        "validation_rows": validation_rows,
    }


def choose_context_row(context_result: dict[str, object]) -> dict[str, object] | None:
    winner = context_result.get("winner_on_discovery")
    if winner is None:
        return None
    spec: ContextSpec = context_result["spec"]
    discovery_rows = context_result["discovery_evaluations"]
    winner_name = winner["strategy_name"]
    binding_discovery = next((r for r in discovery_rows if r["strategy_name"] == "binding_only"), None)

    if spec.validation_datasets:
        validation_rows = context_result["validation_evaluations"]
        target_rows = [r for r in validation_rows if r["strategy_name"] == winner_name]
        if not target_rows:
            return None
        main_row = target_rows[0]
        binding_row = next((r for r in validation_rows if r["strategy_name"] == "binding_only"), None)
        return {
            "context": spec.context,
            "task_type": spec.task_type,
            "winner_strategy": winner_name,
            "winner_display_name": main_row["display_name"],
            "winner_origin": winner["strategy_origin"],
            "winner_origin_dataset": winner["origin_dataset"],
            "winner_metric_name": main_row["primary_metric_name"],
            "winner_metric": main_row["primary_metric"],
            "winner_ci_lo": main_row["bootstrap_ci"].get("ci_lo"),
            "winner_ci_hi": main_row["bootstrap_ci"].get("ci_hi"),
            "binding_only_metric": binding_row["primary_metric"] if binding_row else None,
            "delta_vs_binding_only": (
                main_row["primary_metric"] - binding_row["primary_metric"]
                if binding_row and main_row["primary_metric"] is not None and binding_row["primary_metric"] is not None
                else None
            ),
            "validation_status": "held_out_validated",
            "eligibility_status": "main_table",
            "discovery_dataset": discovery_primary_dataset(spec),
            "validation_dataset": spec.validation_datasets[0],
        }

    return {
        "context": spec.context,
        "task_type": spec.task_type,
        "winner_strategy": winner_name,
        "winner_display_name": winner["display_name"],
        "winner_origin": winner["strategy_origin"],
        "winner_origin_dataset": winner["origin_dataset"],
        "winner_metric_name": winner["primary_metric_name"],
        "winner_metric": winner["primary_metric"],
        "winner_ci_lo": winner["bootstrap_ci"].get("ci_lo"),
        "winner_ci_hi": winner["bootstrap_ci"].get("ci_hi"),
        "binding_only_metric": binding_discovery["primary_metric"] if binding_discovery else None,
        "delta_vs_binding_only": (
            winner["primary_metric"] - binding_discovery["primary_metric"]
            if binding_discovery and winner["primary_metric"] is not None and binding_discovery["primary_metric"] is not None
            else None
        ),
        "validation_status": "provisional_single_cohort",
        "eligibility_status": "provisional",
        "discovery_dataset": discovery_primary_dataset(spec),
        "validation_dataset": None,
    }


def transfer_matrix_rows(context_rows: list[dict[str, object]], context_results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    winner_map = {row["context"]: row["winner_strategy"] for row in context_rows}
    for home_context, strategy_name in winner_map.items():
        for dataset, dataset_context in DATASET_CONTEXT.items():
            eligibility = DATASET_ELIGIBILITY.get(dataset)
            if eligibility is None or not eligibility.eligible:
                continue
            spec = CONTEXT_SPECS[dataset_context]
            df = load_dataset(dataset)
            strategy_library = [find_strategy(strategy_name, [StrategySpec(**{
                "strategy_name": s["strategy_name"],
                "display_name": s["display_name"],
                "strategy_origin": s["strategy_origin"],
                "origin_dataset": s["origin_dataset"],
                "family": s["family"],
                "discovery_kind": s["discovery_kind"],
                "components": s["components"],
                "is_existing": s["is_existing"],
                "fixed_universal_candidate": s["fixed_universal_candidate"],
                "eligible_for_context_table": s["eligible_for_context_table"],
                "notes": s["notes"],
            }) for s in context_results[home_context]["strategy_library"]])]
            result_rows = evaluate_strategies_for_dataset(
                df,
                strategy_library,
                context=dataset_context,
                dataset=dataset,
                task_type=spec.task_type,
                selection_status="locked_context_winner_transfer",
                validation_status="home_context_transfer" if dataset_context == home_context else "out_of_domain_transfer",
            )
            row = result_rows[0]
            rows.append(
                {
                    "home_context": home_context,
                    "strategy_name": strategy_name,
                    "test_dataset": dataset,
                    "test_context": dataset_context,
                    "task_type": spec.task_type,
                    "primary_metric_name": row.primary_metric_name,
                    "primary_metric": row.primary_metric,
                    "ci_lo": row.bootstrap_ci.get("ci_lo"),
                    "ci_hi": row.bootstrap_ci.get("ci_hi"),
                    "validation_status": row.validation_status,
                    "eligibility_status": row.eligibility_status,
                }
            )
    return rows


def choose_best_global_fixed(context_rows: list[dict[str, object]], context_results: dict[str, dict[str, object]]) -> tuple[str | None, float | None]:
    candidate_scores: dict[str, list[float]] = {}
    for row in context_rows:
        context = row["context"]
        dataset = pick_eval_dataset(row)
        evaluations = (
            context_results[context]["validation_evaluations"]
            if row.get("validation_dataset") is not None and not pd.isna(row.get("validation_dataset"))
            else context_results[context]["discovery_evaluations"]
        )
        for ev in evaluations:
            if ev["dataset"] != dataset:
                continue
            if ev["primary_metric"] is None:
                continue
            if ev["strategy_origin"] != "pre_specified":
                continue
            candidate_scores.setdefault(ev["strategy_name"], []).append(float(ev["primary_metric"]))
    if not candidate_scores:
        return None, None
    best_name, best_values = max(candidate_scores.items(), key=lambda item: float(np.mean(item[1])))
    return best_name, float(np.mean(best_values))


def build_context_winner_table(context_results: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    base_rows = [choose_context_row(result) for result in context_results.values()]
    context_rows = [r for r in base_rows if r is not None]

    best_global_fixed, best_global_fixed_mean = choose_best_global_fixed(context_rows, context_results)
    transfer_rows = transfer_matrix_rows(context_rows, context_results)
    transfer_df = pd.DataFrame(transfer_rows)

    enriched_rows: list[dict[str, object]] = []
    for row in context_rows:
        context = row["context"]
        dataset = pick_eval_dataset(row)
        evaluations = (
            context_results[context]["validation_evaluations"]
            if row.get("validation_dataset") is not None and not pd.isna(row.get("validation_dataset"))
            else context_results[context]["discovery_evaluations"]
        )
        fixed_row = next(
            (e for e in evaluations if e["dataset"] == dataset and e["strategy_name"] == best_global_fixed),
            None,
        )
        other_winners = transfer_df[(transfer_df["test_dataset"] == dataset) & (transfer_df["home_context"] != context)].copy()
        wrong = None
        if not other_winners.empty:
            other_winners = other_winners[other_winners["primary_metric"].notna()]
            if not other_winners.empty:
                wrong = other_winners.sort_values("primary_metric").iloc[0].to_dict()
        row = dict(row)
        row["best_fixed_universal_strategy"] = best_global_fixed
        row["best_fixed_universal_metric"] = fixed_row["primary_metric"] if fixed_row else best_global_fixed_mean
        row["wrong_context_strategy"] = wrong["strategy_name"] if wrong else None
        row["wrong_context_home"] = wrong["home_context"] if wrong else None
        row["wrong_context_metric"] = wrong["primary_metric"] if wrong else None
        enriched_rows.append(row)
    return pd.DataFrame(enriched_rows), transfer_rows


def dataset_centroid(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    centroid: dict[str, float] = {}
    for feature in features:
        if feature not in df.columns:
            continue
        vals = pd.to_numeric(df[feature], errors="coerce")
        if vals.notna().sum() < 5:
            continue
        centroid[feature] = float(vals.median())
    return centroid


AUTO_SELECTOR_FEATURES = [
    "bind_log50k",
    "expression_log2",
    "tcr_volume_mean",
    "tcr_charge_sum",
    "tcr_hydro_mean",
    "self_dissimilarity",
    "calis_simplified",
]


def centroid_distance(a: dict[str, float], b: dict[str, float]) -> float:
    shared = sorted(set(a).intersection(b))
    if not shared:
        return float("inf")
    diffs = [(a[k] - b[k]) ** 2 for k in shared]
    return float(np.sqrt(np.mean(diffs)))


def build_auto_selector_centroids(context_results: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    centroids = {}
    for context, spec in CONTEXT_SPECS.items():
        df = load_dataset(discovery_primary_dataset(spec))
        centroids[context] = dataset_centroid(df, AUTO_SELECTOR_FEATURES)
    return centroids


def selector_vs_fixed_table(context_winner_df: pd.DataFrame, context_results: dict[str, dict[str, object]]) -> pd.DataFrame:
    best_global_fixed, _ = choose_best_global_fixed(context_winner_df.to_dict("records"), context_results)
    centroids = build_auto_selector_centroids(context_results)
    rows: list[dict[str, object]] = []
    primary_scores: list[float] = []
    auto_scores: list[float] = []
    fixed_scores: list[float] = []

    for row in context_winner_df.to_dict("records"):
        context = row["context"]
        dataset = pick_eval_dataset(row)
        task_type = row["task_type"]
        df = load_dataset(dataset)
        eval_rows = (
            context_results[context]["validation_evaluations"]
            if row.get("validation_dataset") is not None and not pd.isna(row.get("validation_dataset"))
            else context_results[context]["discovery_evaluations"]
        )

        labeled_metric = row["winner_metric"]
        primary_scores.append(labeled_metric)

        fixed_row = next((e for e in eval_rows if e["dataset"] == dataset and e["strategy_name"] == best_global_fixed), None)
        fixed_metric = fixed_row["primary_metric"] if fixed_row else None
        if fixed_metric is not None:
            fixed_scores.append(float(fixed_metric))

        target_centroid = dataset_centroid(df, AUTO_SELECTOR_FEATURES)
        chosen_context = None
        chosen_distance = float("inf")
        for candidate_context, centroid in centroids.items():
            distance = centroid_distance(target_centroid, centroid)
            if distance < chosen_distance:
                chosen_context = candidate_context
                chosen_distance = distance
        auto_strategy_name = None
        auto_metric = None
        if chosen_context is not None:
            auto_strategy_name = next(
                (x["winner_strategy"] for x in context_winner_df.to_dict("records") if x["context"] == chosen_context),
                None,
            )
            if auto_strategy_name is not None:
                strategy_dicts = context_results[chosen_context]["strategy_library"]
                strategy_obj = find_strategy(
                    auto_strategy_name,
                    [
                        StrategySpec(
                            strategy_name=s["strategy_name"],
                            display_name=s["display_name"],
                            strategy_origin=s["strategy_origin"],
                            origin_dataset=s["origin_dataset"],
                            family=s["family"],
                            discovery_kind=s["discovery_kind"],
                            components=s["components"],
                            is_existing=s["is_existing"],
                            fixed_universal_candidate=s["fixed_universal_candidate"],
                            eligible_for_context_table=s["eligible_for_context_table"],
                            notes=s["notes"],
                        )
                        for s in strategy_dicts
                    ],
                )
                metrics, _ = evaluate_dataset(df, strategy_obj, task_type)
                auto_metric = safe_float(metrics["primary_metric"])
                if auto_metric is not None:
                    auto_scores.append(auto_metric)

        rows.append(
            {
                "context": context,
                "dataset": dataset,
                "task_type": task_type,
                "metric_name": row["winner_metric_name"],
                "cancer_labeled_selector_strategy": row["winner_strategy"],
                "cancer_labeled_selector_metric": labeled_metric,
                "auto_selector_context": chosen_context,
                "auto_selector_strategy": auto_strategy_name,
                "auto_selector_metric": auto_metric,
                "best_global_fixed_strategy": best_global_fixed,
                "best_global_fixed_metric": fixed_metric,
            }
        )

    summary = {
        "context": "ALL",
        "dataset": "aggregate_mean",
        "task_type": "mixed",
        "metric_name": "mean_primary_metric",
        "cancer_labeled_selector_strategy": "context_labeled_selector",
        "cancer_labeled_selector_metric": float(np.mean(primary_scores)) if primary_scores else None,
        "auto_selector_context": "auto_feature_selector",
        "auto_selector_strategy": "auto_feature_selector",
        "auto_selector_metric": float(np.mean(auto_scores)) if auto_scores else None,
        "best_global_fixed_strategy": best_global_fixed,
        "best_global_fixed_metric": float(np.mean(fixed_scores)) if fixed_scores else None,
    }
    rows.append(summary)
    return pd.DataFrame(rows)


def write_outputs(
    context_results: dict[str, dict[str, object]],
    context_winner_df: pd.DataFrame,
    transfer_rows: list[dict[str, object]],
    selector_df: pd.DataFrame,
) -> None:
    leaderboard_rows: list[dict[str, object]] = []
    exclusions = [x.copy() for x in EXCLUDED_STRATEGIES]
    for dataset, eligibility in DATASET_ELIGIBILITY.items():
        if not eligibility.eligible:
            exclusions.append(
                {
                    "strategy_name": None,
                    "dataset": dataset,
                    "reason": eligibility.reason,
                }
            )

    for result in context_results.values():
        leaderboard_rows.extend(result["discovery_evaluations"])
        leaderboard_rows.extend(result["validation_evaluations"])

    context_winner_df.to_csv(ARTIFACTS / "paper_context_winner_table.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(ARTIFACTS / "paper_transfer_matrix.csv", index=False)
    selector_df.to_csv(ARTIFACTS / "paper_selector_vs_fixed.csv", index=False)
    pd.DataFrame(leaderboard_rows).to_csv(ARTIFACTS / "paper_strategy_leaderboard.csv", index=False)

    aggregate_selector_row = selector_df[selector_df["context"] == "ALL"]
    selector_metric = None
    fixed_metric = None
    if not aggregate_selector_row.empty:
        selector_metric = safe_float(aggregate_selector_row.iloc[0]["cancer_labeled_selector_metric"])
        fixed_metric = safe_float(aggregate_selector_row.iloc[0]["best_global_fixed_metric"])

    acceptance_checks = {
        "distinct_context_winners_ge_2": int(context_winner_df["winner_strategy"].nunique()) >= 2,
        "winner_trails_complete": bool(
            not context_winner_df["winner_origin"].isna().any()
            and not context_winner_df["discovery_dataset"].isna().any()
        ),
        "selector_beats_or_matches_best_fixed": (
            selector_metric is not None and fixed_metric is not None and selector_metric >= fixed_metric
        ),
        "single_cohort_contexts_demoted_to_provisional": bool(
            all(
                (pd.notna(row.validation_dataset) and row.eligibility_status == "main_table")
                or (pd.isna(row.validation_dataset) and row.eligibility_status == "provisional")
                for row in context_winner_df.itertuples()
            )
        ),
    }

    summary = {
        "context_winner_table": context_winner_df.to_dict(orient="records"),
        "transfer_matrix": transfer_rows,
        "selector_vs_fixed": selector_df.to_dict(orient="records"),
        "acceptance_checks": acceptance_checks,
        "supplement": {
            "leaderboard": leaderboard_rows,
            "dataset_eligibility": [vars(x) for x in DATASET_ELIGIBILITY.values()],
            "excluded_items": exclusions,
        },
    }
    summary = make_jsonable(summary)
    (ARTIFACTS / "paper_context_winner_table.json").write_text(
        json.dumps(summary["context_winner_table"], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "paper_transfer_matrix.json").write_text(
        json.dumps(summary["transfer_matrix"], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "paper_selector_vs_fixed.json").write_text(
        json.dumps(summary["selector_vs_fixed"], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "paper_supplement_package.json").write_text(
        json.dumps(summary["supplement"], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "paper_evidence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    print("=" * 72)
    print("PAPER EVIDENCE PIPELINE — CROSS-CONTEXT THESIS")
    print("=" * 72)

    context_results = {
        context: context_pipeline(context, spec)
        for context, spec in CONTEXT_SPECS.items()
    }
    context_winner_df, transfer_rows = build_context_winner_table(context_results)
    selector_df = selector_vs_fixed_table(context_winner_df, context_results)
    write_outputs(context_results, context_winner_df, transfer_rows, selector_df)

    print("\nContext winner table:")
    print(context_winner_df.to_string(index=False))
    print("\nSelector vs fixed:")
    print(selector_df.to_string(index=False))
    print(f"\nArtifacts saved under: {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
