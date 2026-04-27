from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.strategy_engine.paper_evidence import (  # noqa: E402
    CONTEXT_SPECS,
    DATASET_CONTEXT,
    StrategyEvaluation,
    StrategySpec,
    build_discovered_candidates,
    evaluate_strategies_for_dataset,
    existing_library,
    feature_orientation,
    find_strategy,
    load_dataset,
    make_jsonable,
    safe_float,
    select_winner,
)
from backend.strategy_engine.autoresearch_registry import AutoresearchRegistry  # noqa: E402
from neoresist.external_tools import load_dtu_wsl_config  # noqa: E402

ARTIFACTS = ROOT / "backend" / "strategy_engine" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
CONFIGS = ROOT / "configs"

DEFAULT_ACTIVE_DATASETS = ("ott_2017", "sahin_2017", "hilf_2019", "rojas_2023", "tesla_2020")
EXTENDED_DATASETS = DEFAULT_ACTIVE_DATASETS + ("muller_nci", "harbst2022_bladder")

BASE_FEATURES = (
    "bind_log50k",
    "presentation_score",
    "expression_log2",
    "self_dissimilarity",
    "tcr_volume_mean",
    "tcr_charge_diff",
    "calis_simplified",
)

PAIRWISE_INTERACTIONS = (
    ("bind_log50k", "expression_log2", "bind_x_expr"),
    ("bind_log50k", "tcr_volume_mean", "bind_x_tcr_volume"),
    ("bind_log50k", "calis_simplified", "bind_x_calis"),
    ("bind_log50k", "self_dissimilarity", "bind_x_self_dissimilarity"),
    ("presentation_score", "tcr_volume_mean", "presentation_x_tcr_volume"),
    ("expression_log2", "tcr_volume_mean", "expression_x_tcr_volume"),
)

ICEFIRE_FEATURES = (
    "bind_log50k",
    "presentation_score",
    "binding_stability",
    "expression_log2",
    "self_dissimilarity",
    "calis_simplified",
    "tcr_volume_mean",
    "tcr_charge_diff",
    "tcr_hydro_mean",
)

LEAKAGE_FEATURE_NAMES = {
    "immunogenic",
    "immunogenic_label",
    "label",
    "labels",
    "target",
    "targets",
    "outcome",
    "outcomes",
    "response",
    "response_type",
    "validated",
    "validation",
    "heldout",
    "transfer",
    "train",
    "test",
    "fold",
    "split",
    "truth",
    "ground_truth",
    "winner",
    "selected",
    "case_id",
    "patient_id",
    "sample_id",
    "cohort",
    "paper_source",
}

DISCOVERY_KIND_TO_OPERATOR: dict[str, str] = {
    "mutated": "mutation",
    "recombined": "recombination",
    "random_blend": "random_blend",
    "ml_importance": "ml_importance",
    "toolstack": "toolstack",
    "tool_adapter": "icefire",
    "gated_ranking": "gated_ranking",
    "seed_from_previous_round": "existing",
}

LEAKAGE_NAME_TOKENS = (
    "immunogenic",
    "label",
    "target",
    "outcome",
    "response_type",
    "validated",
    "heldout",
    "transfer",
    "validation",
    "train",
    "test",
    "fold",
    "split",
    "truth",
    "winner",
    "selected",
    "patient_id",
    "sample_id",
    "case_id",
    "cohort",
    "paper_source",
)


@dataclass(frozen=True)
class OperatorFamilySpec:
    family: str
    generator: str
    enabled_from_phase: int = 1
    base_weight: float = 1.0
    min_candidates: int = 0
    max_candidates: int = 4
    requires_tool: str | None = None


PHASE3_EXTRA_FEATURES = (
    "bind_x_expr_x_tcr",
    "presentation_x_self_dissimilarity",
    "tcr_composite",
    "sequence_composite",
)

FEATURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "bind_x_expr": ("bind_log50k", "expression_log2"),
    "bind_x_tcr_volume": ("bind_log50k", "tcr_volume_mean"),
    "bind_x_calis": ("bind_log50k", "calis_simplified"),
    "bind_x_self_dissimilarity": ("bind_log50k", "self_dissimilarity"),
    "presentation_x_tcr_volume": ("presentation_score", "tcr_volume_mean"),
    "expression_x_tcr_volume": ("expression_log2", "tcr_volume_mean"),
    "bind_x_expr_x_tcr": ("bind_log50k", "expression_log2", "tcr_volume_mean"),
    "presentation_x_self_dissimilarity": ("presentation_score", "self_dissimilarity"),
    "tcr_composite": ("tcr_volume_mean", "tcr_charge_diff", "tcr_hydro_mean"),
    "sequence_composite": ("self_dissimilarity", "calis_simplified"),
}

HOME_PROMOTION_OVERFIT_AUC_CEILING = 0.95
SEED_STABILITY_RUNS = 3
SEED_STABILITY_MIN_VOTES = 2
SPECIFICITY_FLATTEN_EPS = 0.01


def _engineer_phase3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add phase-3 derived columns if the base columns exist. Non-destructive."""
    df = df.copy()
    cols = set(df.columns)
    if {"bind_log50k", "expression_log2", "tcr_volume_mean"} <= cols:
        df["bind_x_expr_x_tcr"] = df["bind_log50k"] * df["expression_log2"] * df["tcr_volume_mean"]
    if {"presentation_score", "self_dissimilarity"} <= cols:
        df["presentation_x_self_dissimilarity"] = df["presentation_score"] * df["self_dissimilarity"]
    if {"tcr_volume_mean", "tcr_charge_diff", "tcr_hydro_mean"} <= cols:
        df["tcr_composite"] = (df["tcr_volume_mean"] + df["tcr_charge_diff"] + df["tcr_hydro_mean"]) / 3.0
    if {"self_dissimilarity", "calis_simplified"} <= cols:
        df["sequence_composite"] = (df["self_dissimilarity"] + df["calis_simplified"]) / 2.0
    return df


def _default_feature_pools() -> dict[int, tuple[str, ...]]:
    phase2 = (
        BASE_FEATURES
        + tuple(target for _, _, target in PAIRWISE_INTERACTIONS)
        + ("binding_sigmoid", "expression_zscore", "tcr_hydro_mean", "tcr_volume_diff", "tcr_hydro_diff", "binding_stability")
    )
    return {
        1: BASE_FEATURES,
        2: phase2,
        3: phase2
        + ICEFIRE_FEATURES
        + PHASE3_EXTRA_FEATURES
        + ("ba_rank", "el_rank", "germline_el_rank", "agretopicity", "tpm"),
    }


def _default_operator_families() -> tuple[OperatorFamilySpec, ...]:
    return (
        OperatorFamilySpec(family="existing", generator="existing", enabled_from_phase=1, base_weight=1.3, min_candidates=2, max_candidates=4),
        OperatorFamilySpec(family="mutation", generator="mutation", enabled_from_phase=1, base_weight=1.2, min_candidates=2, max_candidates=5),
        OperatorFamilySpec(family="recombination", generator="recombination", enabled_from_phase=1, base_weight=0.9, min_candidates=1, max_candidates=3),
        OperatorFamilySpec(family="random_blend", generator="random_blend", enabled_from_phase=1, base_weight=1.0, min_candidates=1, max_candidates=4),
        OperatorFamilySpec(family="toolstack", generator="toolstack", enabled_from_phase=1, base_weight=1.15, min_candidates=1, max_candidates=4),
        OperatorFamilySpec(family="ml_importance", generator="ml_importance", enabled_from_phase=2, base_weight=1.0, min_candidates=1, max_candidates=3),
        OperatorFamilySpec(family="gated_ranking", generator="gated_ranking", enabled_from_phase=1, base_weight=1.1, min_candidates=1, max_candidates=2),
        OperatorFamilySpec(family="icefire", generator="icefire", enabled_from_phase=3, base_weight=0.7, min_candidates=0, max_candidates=2, requires_tool="icefire"),
    )


def _default_phase_datasets() -> dict[int, tuple[str, ...]]:
    return {
        1: DEFAULT_ACTIVE_DATASETS,
        2: EXTENDED_DATASETS,
        3: EXTENDED_DATASETS,
    }


def _default_feature_pool_lookup() -> dict[int, tuple[str, ...]]:
    return _default_feature_pools()


def _resolve_relative_path(base: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = Path(str(raw))
    if value.is_absolute():
        return value
    return (base.parent if base.is_file() else base) / value


def _load_yaml_like(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    try:
        import yaml
    except Exception:
        print(f"[autoresearch] WARNING: pyyaml not installed — config file {path} ignored.", file=sys.stderr)
        return {}
    try:
        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except Exception as exc:
        print(f"[autoresearch] WARNING: could not parse {path} ({exc}) — using defaults.", file=sys.stderr)
        return {}


def _is_leakage_feature_name(feature: str) -> bool:
    lowered = str(feature).strip().lower()
    if not lowered:
        return True
    if lowered in LEAKAGE_FEATURE_NAMES:
        return True
    return any(token in lowered for token in LEAKAGE_NAME_TOKENS)


def _safe_feature_candidates(frame: pd.DataFrame, phase: int, feature_pools: dict[int, tuple[str, ...]] | None = None) -> list[str]:
    return [feature for feature in _feature_space(frame, phase, feature_pools) if not _is_leakage_feature_name(feature)]


def _strategy_uses_safe_features(strategy: StrategySpec, allowed_features: set[str]) -> bool:
    if not strategy.components:
        return False
    return all(feature in allowed_features and not _is_leakage_feature_name(feature) for feature, _weight, _direction in strategy.components)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _standardised_contributions(
    strategy: "StrategySpec",
    df: pd.DataFrame,
) -> dict[str, float | None]:
    """Return each feature's standardised contribution = |weight| * feature_std / total_score_std.

    This makes weights comparable across features regardless of their dynamic range.
    A contribution near 1.0 means that feature dominates the ranking variance.
    Returns an empty dict if the score has zero variance (degenerate case).
    """
    score = pd.Series(0.0, index=df.index)
    feature_stds: dict[str, float] = {}
    for feat, weight, direction in strategy.components:
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
        signed_col = col if direction == "desc" else -col
        score = score + float(weight) * signed_col
        std = float(col.std())
        feature_stds[feat] = std if math.isfinite(std) and std > 0 else 0.0
    score_std = float(score.std())
    if not math.isfinite(score_std) or score_std <= 0:
        return {}
    result: dict[str, float | None] = {}
    for feat, weight, _direction in strategy.components:
        fstd = feature_stds.get(feat, 0.0)
        contrib = abs(float(weight)) * fstd / score_std
        result[f"contrib_{feat}"] = round(contrib, 4) if math.isfinite(contrib) else None
    return result


def _normalize_weights(values: list[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        if not values:
            return []
        return [1.0 / len(values)] * len(values)
    return [float(v) / total for v in values]


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sort_val(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    """Always returns a sortable float — never None — even when CSV NaN is present."""
    v = _as_float(row.get(key))
    return v if v is not None else default


def _top_n_recall_per_patient(
    frame: pd.DataFrame,
    score_col: str,
    n_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | None]:
    """Mean per-patient recall@N — the clinically relevant metric (how many
    immunogenic peptides land in the top-N ranked candidates per patient)."""
    result: dict[str, float | None] = {}
    if score_col not in frame.columns or "immunogenic" not in frame.columns:
        return {f"recall_at_{n}": None for n in n_values}
    patient_col = "patient_id" if "patient_id" in frame.columns else None
    for n in n_values:
        try:
            if patient_col:
                groups = frame.groupby(patient_col, sort=False)
                recalls: list[float] = []
                for _pid, grp in groups:
                    if int(grp["immunogenic"].sum()) == 0:
                        continue
                    ranked = grp.sort_values(score_col, ascending=False)
                    topn = ranked.head(n)
                    recalls.append(float(topn["immunogenic"].sum()) / float(grp["immunogenic"].sum()))
                result[f"recall_at_{n}"] = float(np.mean(recalls)) if recalls else None
            else:
                total_pos = int(frame["immunogenic"].sum())
                if total_pos == 0:
                    result[f"recall_at_{n}"] = None
                    continue
                ranked = frame.sort_values(score_col, ascending=False)
                result[f"recall_at_{n}"] = float(ranked.head(n)["immunogenic"].sum()) / total_pos
        except Exception:
            result[f"recall_at_{n}"] = None
    return result


def _feature_space(df: pd.DataFrame, phase: int, feature_pools: dict[int, tuple[str, ...]] | None = None, min_coverage: float = 0.5) -> list[str]:
    pools = feature_pools or _default_feature_pool_lookup()
    active_phase = phase
    if active_phase not in pools:
        active_phase = max((key for key in pools if key <= phase), default=1)
    candidates = list(pools.get(active_phase, pools.get(1, ())))
    n = len(df)
    return [
        feature
        for feature in dict.fromkeys(candidates)
        if feature in df.columns
        and not _is_leakage_feature_name(feature)
        and _feature_is_deployable(feature)
        and (n == 0 or df[feature].notna().sum() / n >= min_coverage)
    ]


def _feature_is_deployable(feature: str) -> bool:
    deps = FEATURE_DEPENDENCIES.get(str(feature), ())
    return len(deps) <= 2


def _strategy_from_payload(payload: dict[str, Any]) -> StrategySpec:
    data = dict(payload or {})
    allowed_keys = {
        "strategy_name",
        "display_name",
        "strategy_origin",
        "origin_dataset",
        "family",
        "discovery_kind",
        "components",
        "is_existing",
        "fixed_universal_candidate",
        "eligible_for_context_table",
        "notes",
    }
    data = {key: value for key, value in data.items() if key in allowed_keys}
    data["components"] = [
        (str(feature), float(weight), str(direction))
        for feature, weight, direction in (data.get("components") or [])
    ]
    return StrategySpec(**data)


def _strategy_stability_signature(strategy: StrategySpec) -> tuple[object, ...]:
    return (
        strategy.family,
        strategy.discovery_kind,
        tuple((feat, direction) for feat, _weight, direction in strategy.components),
    )


def _strategy_payload(strategy: StrategySpec) -> dict[str, Any]:
    return dict(strategy.to_dict())


def _candidate_signature(strategy: StrategySpec) -> tuple[object, ...]:
    return (
        strategy.strategy_name,
        strategy.family,
        strategy.discovery_kind,
        tuple((feat, round(float(weight), 4), direction) for feat, weight, direction in strategy.components),
    )


def _strategy_bio_score(strategy: StrategySpec) -> float:
    domains = {
        "binding": 0.0,
        "presentation": 0.0,
        "expression": 0.0,
        "sequence": 0.0,
        "tcr": 0.0,
    }
    components = list(strategy.components)
    feature_names = {feature for feature, _weight, _direction in components}

    for feature, weight, _direction in components:
        magnitude = abs(float(weight))
        if feature in {"bind_log50k", "binding_stability", "binding_sigmoid"}:
            domains["binding"] += magnitude
        if feature in {"presentation_score", "binding_stability", "binding_sigmoid", "calis_simplified"}:
            domains["presentation"] += magnitude * 0.8
        if feature in {"expression_log2", "expression_zscore"} or feature.startswith("expression_"):
            domains["expression"] += magnitude
        if feature in {"self_dissimilarity", "calis_simplified", "sequence_composite"} or feature.endswith("_self_dissimilarity"):
            domains["sequence"] += magnitude
        if feature.startswith("tcr_") or feature.endswith("_tcr") or "tcr" in feature:
            domains["tcr"] += magnitude

    grounding = sum(domains.values())

    # Reward biologically coherent combinations rather than just isolated signals.
    coherent_pairs = (
        {"bind_log50k", "presentation_score"},
        {"bind_log50k", "expression_log2"},
        {"presentation_score", "self_dissimilarity"},
        {"binding_stability", "self_dissimilarity"},
        {"tcr_volume_mean", "tcr_charge_diff"},
        {"tcr_volume_mean", "tcr_hydro_mean"},
        {"tcr_volume_mean", "binding_stability"},
        {"bind_log50k", "self_dissimilarity"},
    )
    for pair in coherent_pairs:
        if pair.issubset(feature_names):
            grounding += 0.35

    active_domains = sum(1 for value in domains.values() if value > 0)
    if active_domains >= 2:
        grounding += 0.25 * (active_domains - 1)

    if {"bind_log50k", "presentation_score", "expression_log2"} <= feature_names:
        grounding += 0.5
    if {"tcr_volume_mean", "tcr_charge_diff", "tcr_hydro_mean"} <= feature_names:
        grounding += 0.4
    if {"self_dissimilarity", "calis_simplified"} <= feature_names:
        grounding += 0.25

    return grounding


def _strategy_bio_domains(strategy: StrategySpec) -> tuple[str, ...]:
    domains: set[str] = set()
    for feature, _weight, _direction in strategy.components:
        if feature in {"bind_log50k", "binding_stability", "binding_sigmoid"}:
            domains.add("binding")
        if feature in {"presentation_score", "binding_stability", "binding_sigmoid"}:
            domains.add("presentation")
        if feature in {"expression_log2", "expression_zscore"} or feature.startswith("expression_"):
            domains.add("expression")
        if feature in {"self_dissimilarity", "calis_simplified", "sequence_composite"} or feature.endswith("_self_dissimilarity"):
            domains.add("sequence")
        if feature.startswith("tcr_") or feature.endswith("_tcr") or "tcr" in feature:
            domains.add("tcr")
    return tuple(sorted(domains))


def _structure_signature_text(strategy: StrategySpec) -> str:
    return json.dumps(_strategy_stability_signature(strategy), default=str, sort_keys=True)


def _robust_validation_value(
    heldout_delta: float | None,
    transfer_delta: float | None,
    discovery_delta: float | None = None,
) -> tuple[str, float | None]:
    if heldout_delta is not None and transfer_delta is not None:
        # Reward the worst supported cohort, not the best spike.
        return "robust_transfer_delta_vs_binding", min(float(heldout_delta), float(transfer_delta))
    if heldout_delta is not None:
        return "heldout_delta_vs_binding", float(heldout_delta)
    if transfer_delta is not None:
        return "transfer_delta_vs_binding", float(transfer_delta)
    if discovery_delta is not None:
        return "discovery_delta_vs_binding", float(discovery_delta)
    return "delta_vs_binding", None


def _support_tier(row: dict[str, Any]) -> int:
    """Transfer-supported rows should outrank heldout-only rows, which should outrank discovery-only rows."""
    heldout = _as_float(row.get("heldout_delta_vs_binding"))
    transfer = _as_float(row.get("transfer_delta_vs_binding"))
    if heldout is not None and transfer is not None:
        return 0
    if heldout is not None or transfer is not None:
        return 1
    return 2


def _novelty_bonus(row: dict[str, Any], bonus: float) -> float:
    """Boost rows whose structures were not already seen before this round."""
    structure_seen_before = row.get("structure_seen_before")
    if structure_seen_before is None or bool(structure_seen_before):
        return 0.0
    return float(bonus)


def _strategy_complexity(strategy: StrategySpec) -> int:
    return len(strategy.components)


def _task_spec(context: str) -> dict[str, Any]:
    spec = CONTEXT_SPECS[context]
    return {
        "context": context,
        "task_type": spec.task_type,
        "discovery_dataset": spec.discovery_datasets[0],
        "validation_datasets": tuple(spec.validation_datasets),
        "held_out_available": bool(spec.validation_datasets),
    }


def _active_datasets(phase: int) -> tuple[str, ...]:
    return EXTENDED_DATASETS if phase >= 2 else DEFAULT_ACTIVE_DATASETS


def _dataset_context(dataset: str) -> str:
    return DATASET_CONTEXT.get(dataset, "unknown")


def _best_feature_direction(df: pd.DataFrame, feature: str, task_type: str) -> str:
    direction, _metric = feature_orientation(df, feature, task_type)
    return direction


def _build_weighted_strategy(
    *,
    strategy_name: str,
    display_name: str,
    family: str,
    discovery_kind: str,
    origin_dataset: str | None,
    components: list[tuple[str, float, str]],
    notes: str,
    strategy_origin: str = "autoresearch_training_only",
) -> StrategySpec:
    return StrategySpec(
        strategy_name=strategy_name,
        display_name=display_name,
        strategy_origin=strategy_origin,
        origin_dataset=origin_dataset,
        family=family,
        discovery_kind=discovery_kind,
        components=components,
        is_existing=False,
        fixed_universal_candidate=False,
        notes=notes,
    )


def _top_existing_seeds(discovery_df: pd.DataFrame, context: str, origin_dataset: str, task_type: str) -> list[StrategySpec]:
    library = existing_library() + build_discovered_candidates(discovery_df, context=context, origin_dataset=origin_dataset, task_type=task_type)
    return library


def _mutate_strategy(
    strategy: StrategySpec,
    frame: pd.DataFrame,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
) -> StrategySpec:
    components = list(strategy.components)
    if not components:
        fallback = "bind_log50k" if "bind_log50k" in frame.columns else frame.columns[0]
        components = [(fallback, 1.0, _best_feature_direction(frame, fallback, task_type))]
    feature_space = _feature_space(frame, phase, feature_pools)
    if not feature_space:
        feature_space = [feat for feat, _weight, _direction in components]
    action = rng.choice(["flip", "swap", "weight", "add", "drop"])
    if action == "flip":
        idx = int(rng.integers(0, len(components)))
        feat, weight, direction = components[idx]
        components[idx] = (feat, weight, "asc" if direction == "desc" else "desc")
    elif action == "swap":
        idx = int(rng.integers(0, len(components)))
        feat = str(rng.choice(feature_space))
        components[idx] = (feat, components[idx][1], _best_feature_direction(frame, feat, task_type))
    elif action == "weight":
        weights = _normalize_weights(rng.dirichlet(np.ones(len(components))).tolist())
        components = [(feat, weights[i], direction) for i, (feat, _weight, direction) in enumerate(components)]
    elif action == "add" and len(components) < min(5, len(feature_space)):
        feat = str(rng.choice([feat for feat in feature_space if feat not in {item[0] for item in components}] or feature_space))
        components.append((feat, 1.0 / (len(components) + 1), _best_feature_direction(frame, feat, task_type)))
        weights = _normalize_weights(rng.dirichlet(np.ones(len(components))).tolist())
        components = [(feat, weights[i], direction) for i, (feat, _weight, direction) in enumerate(components)]
    elif action == "drop" and len(components) > 1:
        idx = int(rng.integers(0, len(components)))
        components.pop(idx)
        weights = _normalize_weights(rng.dirichlet(np.ones(len(components))).tolist())
        components = [(feat, weights[i], direction) for i, (feat, _weight, direction) in enumerate(components)]

    return _build_weighted_strategy(
        strategy_name=f"{strategy.strategy_name}_mut_{int(rng.integers(1_000_000))}",
        display_name=f"{strategy.display_name} (mut)",
        family=f"{strategy.family}_mut",
        discovery_kind="mutated",
        origin_dataset=strategy.origin_dataset,
        components=components,
        notes=f"Mutated from {strategy.strategy_name}.",
        strategy_origin=strategy.strategy_origin,
    )


def _recombine_strategies(
    lhs: StrategySpec,
    rhs: StrategySpec,
    frame: pd.DataFrame,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
) -> StrategySpec:
    left = list(lhs.components)
    right = list(rhs.components)
    # Build a direction map from each parent; randomly pick whose direction wins for shared features.
    right_direction: dict[str, str] = {f: d for f, _w, d in right}
    merged: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for feature, weight, direction in left + right:
        if feature in seen:
            continue
        seen.add(feature)
        # For features present in both parents, randomly inherit direction.
        if feature in right_direction and bool(rng.integers(0, 2)):
            direction = right_direction[feature]
        merged.append((feature, float(weight), direction))
    if not merged:
        fallback = _feature_space(frame, phase, feature_pools)[:2] or ["bind_log50k"]
        merged = [(feature, 1.0, _best_feature_direction(frame, feature, task_type)) for feature in fallback]
    weights = _normalize_weights(rng.dirichlet(np.ones(len(merged))).tolist())
    merged = [(feature, weights[i], direction) for i, (feature, _weight, direction) in enumerate(merged)]
    return _build_weighted_strategy(
        strategy_name=f"{lhs.strategy_name}_x_{rhs.strategy_name}_{int(rng.integers(1_000_000))}",
        display_name=f"{lhs.display_name} x {rhs.display_name}",
        family="recombined",
        discovery_kind="recombined",
        origin_dataset=lhs.origin_dataset or rhs.origin_dataset,
        components=merged[: min(4, len(merged))],
        notes="Recombined from two surviving candidates.",
        strategy_origin="autoresearch_training_only",
    )


def _random_strategy(
    frame: pd.DataFrame,
    context: str,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    family: str = "random_blend",
    feature_pools: dict[int, tuple[str, ...]] | None = None,
) -> StrategySpec:
    features = _feature_space(frame, phase, feature_pools)
    if not features:
        features = ["bind_log50k"] if "bind_log50k" in frame.columns else [str(frame.columns[0])]
    low = 2 if len(features) >= 2 else 1
    high = min(5, len(features))
    count = int(rng.integers(low, high + 1))
    picks = list(rng.choice(features, size=count, replace=False))
    weights = _normalize_weights(rng.dirichlet(np.ones(count)).tolist())
    components = [(feature, weights[i], _best_feature_direction(frame, feature, task_type)) for i, feature in enumerate(picks)]
    return _build_weighted_strategy(
        strategy_name=f"{context}_{family}_{int(rng.integers(1_000_000))}",
        display_name=f"{context} {family}",
        family=family,
        discovery_kind=family,
        origin_dataset=None,
        components=components,
        notes="Randomly sampled interpretable blend.",
    )


def _ml_importance_candidates(
    frame: pd.DataFrame,
    context: str,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
) -> list[StrategySpec]:
    features = _safe_feature_candidates(frame, phase, feature_pools)
    numeric_features = []
    for feature in features:
        series = pd.to_numeric(frame.get(feature), errors="coerce")
        if int(series.notna().sum()) >= 5:
            numeric_features.append(feature)
    if len(numeric_features) < 2 or "immunogenic" not in frame.columns:
        return []
    y = pd.to_numeric(frame["immunogenic"], errors="coerce")
    if int(y.nunique(dropna=True)) < 2:
        return []
    X = pd.DataFrame({feature: pd.to_numeric(frame[feature], errors="coerce") for feature in numeric_features})
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(0).astype(int)
    if len(np.unique(y)) < 2:
        return []

    groups = None
    if "patient_id" in frame.columns:
        groups = pd.Series(frame["patient_id"], copy=False).astype(str)
    y_series = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    if groups is not None:
        groups = groups.reset_index(drop=True)

    def _fold_iter() -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is not None and int(groups.nunique()) >= 2:
            splitter = GroupKFold(n_splits=min(5, int(groups.nunique())))
            return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(X, y_series, groups)]
        unique_classes = int(y_series.nunique(dropna=True))
        if unique_classes < 2:
            return []
        min_class = int(y_series.value_counts().min())
        n_splits = min(5, max(2, min_class))
        if n_splits < 2:
            return []
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(1_000_000)))
        return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(X, y_series)]

    folds = _fold_iter()
    if len(folds) < 2:
        return []

    candidates: list[StrategySpec] = []
    models: list[tuple[str, Any]] = [
        (
            "logreg",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=int(rng.integers(1_000_000)))),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=int(rng.integers(1_000_000)),
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=int(rng.integers(1_000_000))),
        ),
    ]

    for model_name, model in models:
        fold_importances: list[np.ndarray] = []
        fold_signs: list[np.ndarray] = []
        for train_idx, test_idx in folds:
            train_X = X.iloc[train_idx].copy()
            train_y = y_series.iloc[train_idx].copy()
            if int(train_y.nunique(dropna=True)) < 2:
                continue
            fold_model = clone(model)
            try:
                fold_model.fit(train_X, train_y)
            except Exception:
                continue
            # Validate on held-out fold: only accept importances when fold AUC > chance.
            test_X = X.iloc[test_idx]
            test_y = y_series.iloc[test_idx]
            try:
                if int(test_y.nunique(dropna=True)) >= 2 and hasattr(fold_model, "predict_proba"):
                    test_proba = fold_model.predict_proba(test_X)[:, 1]
                    from sklearn.metrics import roc_auc_score as _roc_auc
                    fold_auc = float(_roc_auc(test_y, test_proba))
                    if fold_auc < 0.50:
                        continue
            except Exception:
                pass
            if model_name == "logreg":
                classifier = fold_model.named_steps["clf"]
                fold_importances.append(np.asarray(np.abs(classifier.coef_[0]), dtype=float))
                fold_signs.append(np.asarray(np.sign(classifier.coef_[0]), dtype=float))
            else:
                estimator = fold_model
                fold_importances.append(np.asarray(getattr(estimator, "feature_importances_", np.zeros(len(numeric_features))), dtype=float))
                fold_signs.append(np.asarray([0.0] * len(numeric_features), dtype=float))
        if len(fold_importances) < 2:
            continue
        importances = np.asarray(np.mean(fold_importances, axis=0), dtype=float)
        if not np.isfinite(importances).any() or float(importances.sum()) <= 0:
            continue
        support = np.asarray(np.mean([imp > 0 for imp in fold_importances], axis=0), dtype=float)
        if model_name == "logreg":
            sign_consistency = np.asarray(np.mean([sign > 0 for sign in fold_signs], axis=0), dtype=float)
        else:
            sign_consistency = np.ones(len(numeric_features), dtype=float)
        stable_mask = (support >= 0.6) & (importances > 0)
        if model_name == "logreg":
            stable_mask &= (np.maximum(sign_consistency, 1.0 - sign_consistency) >= 0.6)
        stable_idx = np.where(stable_mask)[0]
        if len(stable_idx) < 2:
            continue
        order = stable_idx[np.argsort(importances[stable_idx])[::-1]][: min(4, len(stable_idx))]
        selected = [numeric_features[i] for i in order[: max(2, min(4, len(order)))]]
        top_importances = [float(importances[i]) for i in order[: len(selected)]]
        weights = _normalize_weights(top_importances)
        components = []
        for i, feature_idx in enumerate(order[: len(selected)]):
            feature = numeric_features[feature_idx]
            if model_name == "logreg":
                direction = "desc" if float(sign_consistency[feature_idx]) >= 0.5 else "asc"
            else:
                direction = _best_feature_direction(frame, feature, task_type)
            components.append((feature, weights[i], direction))
        candidates.append(
            _build_weighted_strategy(
                strategy_name=f"{context}_{model_name}_{int(rng.integers(1_000_000))}",
                display_name=f"{context} {model_name}",
                family=f"ml_{model_name}",
                discovery_kind="ml_importance",
                origin_dataset=None,
                components=components,
                notes=f"ML-importance blend induced by {model_name}.",
            )
        )
    return candidates


def _icefire_candidates(
    frame: pd.DataFrame,
    context: str,
    task_type: str,
    rng: np.random.Generator,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
) -> list[StrategySpec]:
    config = load_dtu_wsl_config()
    tool = config.tools.get("icefire")
    if tool is None:
        return []
    if tool.status not in {"ready", "installed_not_wired", "installed"}:
        return []

    phase = 3
    features = [feature for feature in _feature_space(frame, phase, feature_pools) if feature in ICEFIRE_FEATURES]
    if len(features) < 2:
        return []
    seed = ["bind_log50k", "presentation_score", "expression_log2", "self_dissimilarity", "calis_simplified"]
    selected = [feature for feature in seed if feature in features][:4]
    if len(selected) < 2:
        selected = features[: min(4, len(features))]
    weights = _normalize_weights([0.35, 0.25, 0.20, 0.20][: len(selected)])
    components = [(feature, weights[i], _best_feature_direction(frame, feature, task_type)) for i, feature in enumerate(selected)]
    return [
        _build_weighted_strategy(
            strategy_name=f"{context}_icefire_{int(rng.integers(1_000_000))}",
            display_name=f"{context} icefire blend",
            family="icefire",
            discovery_kind="tool_adapter",
            origin_dataset=None,
            components=components,
            notes=f"Icefire adapter available via {tool.logical_name}; used as a proposal source, not a frozen scorer.",
        )
    ]


def _toolstack_candidates(
    frame: pd.DataFrame,
    context: str,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    *,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
    tool_plans: list[dict[str, Any]] | None = None,
) -> list[StrategySpec]:
    if not tool_plans:
        return []
    allowed = set(_safe_feature_candidates(frame, phase, feature_pools))
    if not allowed:
        return []
    candidates: list[StrategySpec] = []
    for plan in tool_plans:
        feature_sets = list(plan.get("feature_sets") or [])
        display_name = str(plan.get("display_name") or plan.get("family") or "toolstack")
        family = str(plan.get("family") or plan.get("template_id") or "toolstack")
        for feature_set in feature_sets:
            features = [str(feature) for feature in (feature_set or []) if str(feature) in allowed]
            if len(features) < 2:
                continue
            weights = _normalize_weights(rng.dirichlet(np.ones(len(features))).tolist())
            components = [
                (feature, weights[i], _best_feature_direction(frame, feature, task_type))
                for i, feature in enumerate(features)
            ]
            candidates.append(
                _build_weighted_strategy(
                    strategy_name=f"{context}_{plan.get('template_id') or family}_{int(rng.integers(1_000_000))}",
                    display_name=f"{context} {display_name}",
                    family=family,
                    discovery_kind="toolstack",
                    origin_dataset=None,
                    components=components,
                    notes=str(plan.get("notes") or "Tool stack template."),
                )
            )
    return candidates


def _existing_candidates(
    *_args: Any,
    seeds: list[StrategySpec] | None = None,
    budget: int = 0,
    **_kwargs: Any,
) -> list[StrategySpec]:
    if not seeds or budget <= 0:
        return []
    return list(seeds[:budget])


def _gated_ranking_candidates(
    frame: pd.DataFrame,
    context: str,
    task_type: str,
    rng: np.random.Generator,
    phase: int,
    feature_pools: dict[int, tuple[str, ...]] | None = None,
    budget: int = 2,
    **_kwargs: Any,
) -> list[StrategySpec]:
    """DeepNeo-inspired two-stage: binding gate × TCR/immunogenicity reranker.

    Generates linear strategies where binding features dominate (>= 0.6 weight)
    and a secondary TCR/sequence feature provides the discriminating rerank.
    This encodes the biological prior that binding is necessary but TCR recognition
    determines the final immunogenic hit.
    """
    available = set(_safe_feature_candidates(frame, phase, feature_pools))
    binding_features = [f for f in ["bind_log50k", "presentation_score", "binding_sigmoid", "binding_stability"] if f in available]
    tcr_features = [f for f in ["tcr_volume_mean", "tcr_charge_diff", "tcr_hydro_mean", "self_dissimilarity", "calis_simplified"] if f in available]
    if not binding_features or not tcr_features:
        return []
    candidates: list[StrategySpec] = []
    for _ in range(budget):
        gate_feat = str(rng.choice(binding_features))
        rerank_feat = str(rng.choice(tcr_features))
        gate_weight = float(rng.uniform(0.6, 0.85))
        rerank_weight = 1.0 - gate_weight
        components = [
            (gate_feat, round(gate_weight, 4), _best_feature_direction(frame, gate_feat, task_type)),
            (rerank_feat, round(rerank_weight, 4), _best_feature_direction(frame, rerank_feat, task_type)),
        ]
        candidates.append(
            _build_weighted_strategy(
                strategy_name=f"{context}_gated_{gate_feat}x{rerank_feat}_{int(rng.integers(1_000_000))}",
                display_name=f"{context} gate({gate_feat})×rank({rerank_feat})",
                family="gated_ranking",
                discovery_kind="gated_ranking",
                origin_dataset=None,
                components=components,
                notes="DeepNeo-inspired binding-gated TCR reranker.",
            )
        )
    return candidates


OPERATOR_GENERATORS: dict[str, Callable[..., list[StrategySpec]]] = {
    "existing": lambda *, seeds=None, budget=0, **kwargs: _existing_candidates(seeds=seeds, budget=budget, **kwargs),
    "mutation": lambda *, discovery_df=None, task_type=None, rng=None, phase=None, feature_pools=None, seed_pool=None, budget=0, **kwargs: [
        _mutate_strategy(rng.choice(seed_pool), discovery_df, task_type, rng, phase, feature_pools)  # type: ignore[arg-type]
        for _ in range(budget)
        if seed_pool
    ],
    "recombination": lambda *, discovery_df=None, task_type=None, rng=None, phase=None, feature_pools=None, seed_pool=None, budget=0, **kwargs: [
        _recombine_strategies(*(rng.choice(seed_pool, size=2, replace=False)), discovery_df, task_type, rng, phase, feature_pools)  # type: ignore[arg-type]
        for _ in range(budget)
        if seed_pool and len(seed_pool) >= 2
    ],
    "random_blend": lambda *, discovery_df=None, context=None, task_type=None, rng=None, phase=None, feature_pools=None, budget=0, **kwargs: [
        _random_strategy(discovery_df, context, task_type, rng, phase, feature_pools=feature_pools)  # type: ignore[arg-type]
        for _ in range(budget)
    ],
    "toolstack": lambda *, discovery_df=None, context=None, task_type=None, rng=None, phase=None, feature_pools=None, budget=0, tool_plans=None, **kwargs: (
        _toolstack_candidates(discovery_df, context, task_type, rng, phase, feature_pools=feature_pools, tool_plans=tool_plans)[:budget]  # type: ignore[arg-type]
    ),
    "ml_importance": lambda *, discovery_df=None, context=None, task_type=None, rng=None, phase=None, feature_pools=None, budget=0, **kwargs: (
        _ml_importance_candidates(discovery_df, context, task_type, rng, phase, feature_pools)[:budget]  # type: ignore[arg-type]
    ),
    "icefire": lambda *, discovery_df=None, context=None, task_type=None, rng=None, feature_pools=None, budget=0, **kwargs: (
        _icefire_candidates(discovery_df, context, task_type, rng, feature_pools)[:budget]  # type: ignore[arg-type]
    ),
    "gated_ranking": lambda *, discovery_df=None, context=None, task_type=None, rng=None, phase=None, feature_pools=None, budget=0, **kwargs: (
        _gated_ranking_candidates(discovery_df, context, task_type, rng, phase, feature_pools, budget=budget)  # type: ignore[arg-type]
    ),
}


def _select_binding_baseline() -> StrategySpec:
    return find_strategy("binding_only", existing_library())


def _sort_evaluations(evaluations: list[StrategyEvaluation]) -> list[StrategyEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (
            _as_float(item.primary_metric) or float("-inf"),
            _as_float(item.secondary_metrics.get("pooled_auc")) or float("-inf"),
            _as_float(item.bootstrap_ci.get("ci_lo")) or float("-inf"),
            -len(item.components),
        ),
        reverse=True,
    )


def _pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    points: list[tuple[float, float, float, float]] = []
    for row in records:
        points.append(
            (
                _as_float(row.get("heldout_metric")) or float("-inf"),
                _as_float(row.get("transfer_metric")) or float("-inf"),
                _as_float(row.get("stability")) or float("-inf"),
                _as_float(row.get("bio_score")) or float("-inf"),
            )
        )
    keep: list[dict[str, Any]] = []
    for i, lhs in enumerate(points):
        dominated = False
        for j, rhs in enumerate(points):
            if i == j:
                continue
            if all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs)):
                dominated = True
                break
        if not dominated:
            keep.append(records[i])
    return keep


@dataclass
class AutoResearchConfig:
    state_path: Path = ARTIFACTS / "autoresearch_loop_state.json"
    ledger_path: Path = ARTIFACTS / "autoresearch_loop_ledger.csv"
    ranked_path: Path = ARTIFACTS / "autoresearch_loop_ranked_winners.csv"
    pareto_path: Path = ARTIFACTS / "autoresearch_loop_pareto_front.csv"
    progress_path: Path = ARTIFACTS / "autoresearch_loop_progress.json"
    summary_path: Path = ARTIFACTS / "autoresearch_loop_summary.json"
    runlog_path: Path = ARTIFACTS / "autoresearch_loop_runlog.md"
    max_rounds: int = 1
    stall_patience: int = 3
    promote_delta: float = 0.01
    substantial_delta: float = 0.02
    ci_margin: float = 0.0
    elite_bank_size: int = 6
    max_candidates_per_task: int = 14
    seed: int = 42
    daemon_sleep_seconds: float = 0.0
    operator_priority: tuple[str, ...] = ("existing", "mutation", "recombination", "toolstack", "ml_importance", "random_blend", "icefire")
    operator_families: tuple[OperatorFamilySpec, ...] = field(default_factory=_default_operator_families)
    feature_pools: dict[int, tuple[str, ...]] = field(default_factory=_default_feature_pool_lookup)
    datasets_manifest: Path | None = CONFIGS / "autoresearch_datasets.yaml"
    operators_manifest: Path | None = CONFIGS / "autoresearch_operators.yaml"
    tool_registry_manifest: Path | None = CONFIGS / "autoresearch_tools.yaml"
    dataset_capabilities_manifest: Path | None = CONFIGS / "autoresearch_dataset_capabilities.yaml"
    phase_datasets: dict[int, tuple[str, ...]] = field(
        default_factory=_default_phase_datasets
    )
    require_transfer_support_when_available: bool = True
    require_heldout_support_when_available: bool = True
    overfit_auc_ceiling: float = HOME_PROMOTION_OVERFIT_AUC_CEILING
    seed_stability_runs: int = SEED_STABILITY_RUNS
    seed_stability_min_votes: int = SEED_STABILITY_MIN_VOTES
    seed_stability_strong_delta_bypass: float = 0.08
    # Minimum number of contexts that must promote before the phase advances.
    min_contexts_for_phase_advance: int = 2
    # Delta required for soft promotion (no CI gate). Must exceed substantial_delta * 1.5.
    soft_promote_delta: float = 0.03
    # Force a pivot once the same context keeps surfacing the same family without real improvement.
    pivot_patience: int = 2
    pivot_metric_epsilon: float = 0.005
    pivot_exploration_boost: float = 0.25
    pivot_family_penalty: float = 0.45
    pivot_elite_retention: int = 2
    transfer_floor: float = 0.0
    novelty_bonus: float = 0.03


@dataclass
class LoopState:
    iteration: int = 0
    phase: int = 1
    stall_count: int = 0
    exploration_pressure: float = 0.0
    best_by_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    family_stats: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {"tried": 0.0, "kept": 0.0, "promoted": 0.0, "weight": 1.0}))  # type: ignore[assignment]
    seen_signatures: list[str] = field(default_factory=list)
    seen_structures: list[str] = field(default_factory=list)
    substantial_promotions: list[dict[str, Any]] = field(default_factory=list)
    elite_bank: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    crystallized_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_pivot_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "phase": self.phase,
            "stall_count": self.stall_count,
            "exploration_pressure": self.exploration_pressure,
            "best_by_context": self.best_by_context,
            "family_stats": {key: dict(value) for key, value in self.family_stats.items()},
            "seen_signatures": self.seen_signatures,
            "seen_structures": self.seen_structures,
            "substantial_promotions": self.substantial_promotions,
            "elite_bank": self.elite_bank,
            "crystallized_contexts": self.crystallized_contexts,
            "context_pivot_state": self.context_pivot_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoopState":
        state = cls(
            iteration=int(payload.get("iteration") or 0),
            phase=int(payload.get("phase") or 1),
            stall_count=int(payload.get("stall_count") or 0),
            exploration_pressure=float(payload.get("exploration_pressure") or 0.0),
            best_by_context=dict(payload.get("best_by_context") or {}),
            seen_signatures=list(payload.get("seen_signatures") or []),
            seen_structures=list(payload.get("seen_structures") or []),
            substantial_promotions=list(payload.get("substantial_promotions") or []),
            elite_bank={str(key): list(value or []) for key, value in dict(payload.get("elite_bank") or {}).items()},
            crystallized_contexts={str(key): dict(value or {}) for key, value in dict(payload.get("crystallized_contexts") or {}).items()},
        )
        family_stats = dict(payload.get("family_stats") or {})
        if family_stats:
            state.family_stats = defaultdict(
                lambda: {"tried": 0.0, "kept": 0.0, "promoted": 0.0, "weight": 1.0},
                {key: {"tried": float(value.get("tried") or 0.0), "kept": float(value.get("kept") or 0.0), "promoted": float(value.get("promoted") or 0.0), "weight": float(value.get("weight") or 1.0)} for key, value in family_stats.items()},
            )
        state.context_pivot_state = {
            str(key): dict(value or {})
            for key, value in dict(payload.get("context_pivot_state") or {}).items()
        }
        return state


class ToolRegistry:
    def __init__(self) -> None:
        self.config = load_dtu_wsl_config()

    def available(self, tool_name: str) -> bool:
        tool = self.config.tools.get(tool_name)
        if tool is None:
            return False
        return tool.status in {"ready", "installed", "installed_not_wired"}

    def status(self, tool_name: str) -> dict[str, Any]:
        tool = self.config.tools.get(tool_name)
        if tool is None:
            return {"tool_id": tool_name, "available": False, "status": "not_configured"}
        return {
            "tool_id": tool.tool_id,
            "logical_name": tool.logical_name,
            "actual_tool": tool.actual_tool,
            "executable": tool.executable,
            "status": tool.status,
            "available": self.available(tool_name),
        }


class AutoResearchLoop:
    def __init__(
        self,
        config: AutoResearchConfig | None = None,
        *,
        dataset_loader: Callable[[str], pd.DataFrame] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config or AutoResearchConfig()
        self.dataset_loader = dataset_loader or load_dataset
        self.tool_registry = tool_registry or ToolRegistry()
        self.research_registry = AutoresearchRegistry.load(
            tools_manifest=self.config.tool_registry_manifest,
            dataset_capabilities_manifest=self.config.dataset_capabilities_manifest,
            tool_available=self.tool_registry.available,
        )
        self.state = self._load_state()

    def _load_state(self) -> LoopState:
        if self.config.state_path.exists():
            try:
                payload = json.loads(self.config.state_path.read_text(encoding="utf-8"))
                state = LoopState.from_dict(dict(payload or {}))
                self._sanitize_loaded_state(state)
                return state
            except Exception as exc:
                print(f"[autoresearch] WARNING: state file corrupt or unreadable ({exc}); starting fresh.", file=sys.stderr)
        return LoopState(phase=1)

    def _sanitize_loaded_state(self, state: LoopState) -> None:
        allowed = set(self.config.feature_pools.get(max(state.phase, 1), self.config.feature_pools.get(1, ())))
        sanitized_bank: dict[str, list[dict[str, Any]]] = {}
        for context, payloads in state.elite_bank.items():
            kept: list[dict[str, Any]] = []
            for payload in payloads:
                try:
                    strategy = _strategy_from_payload(payload)
                except Exception:
                    continue
                if _strategy_uses_safe_features(strategy, allowed):
                    kept.append(payload)
            if kept:
                sanitized_bank[context] = kept
        state.elite_bank = sanitized_bank

    def _is_registry_context(self, context: str) -> bool:
        spec = CONTEXT_SPECS.get(context)
        if spec is None or not spec.held_out_available:
            return False
        discovery_dataset = spec.discovery_datasets[0]
        return not self.research_registry.support_only(discovery_dataset)

    def _context_spec(self, context: str):
        return CONTEXT_SPECS[context]

    def _context_heldout_datasets(self, context: str) -> tuple[str, ...]:
        spec = self._context_spec(context)
        return tuple(spec.validation_datasets or ())

    def _active_transfer_datasets(self, discovery_dataset: str) -> tuple[str, ...]:
        active_datasets = self.config.phase_datasets.get(
            self.state.phase,
            EXTENDED_DATASETS if self.state.phase >= 2 else DEFAULT_ACTIVE_DATASETS,
        )
        return tuple(dataset for dataset in active_datasets if dataset != discovery_dataset)

    def _seed_stable_from_history(
        self,
        proposal: StrategySpec,
        context: str,
        prior_rows: list[dict[str, Any]],
    ) -> tuple[bool, int, int]:
        """Check stability by counting how many of the last N unique iterations contained the
        same feature+direction signature for this context. O(n) against the existing ledger —
        no expensive proposal re-generation."""
        runs = max(1, int(getattr(self.config, "seed_stability_runs", SEED_STABILITY_RUNS)))
        min_votes = max(1, int(getattr(self.config, "seed_stability_min_votes", SEED_STABILITY_MIN_VOTES)))
        target_sig = _strategy_stability_signature(proposal)

        context_rows = [r for r in prior_rows if str(r.get("context", "")) == context]
        if not context_rows:
            return True, 0, 0  # no history — pass through

        recent_iters = sorted(
            {int(r["iteration"]) for r in context_rows if r.get("iteration") is not None},
            reverse=True,
        )[:runs]
        if len(recent_iters) < min_votes:
            return True, 0, 0  # not enough rounds of history yet

        def _sig_from_row(row: dict[str, Any]) -> tuple[object, ...] | None:
            raw = str(row.get("components") or "")
            if not raw:
                return None
            try:
                parts = [item.strip() for item in raw.split(";") if item.strip()]
                components = []
                for part in parts:
                    tokens = part.split(":")
                    if len(tokens) >= 3:
                        feat, _weight, direction = tokens[0], tokens[1], tokens[2]
                        components.append((feat.strip(), direction.strip()))
                return (
                    str(row.get("operator_family", "")),
                    str(row.get("discovery_kind", "")),
                    tuple(components),
                )
            except Exception:
                return None

        target_normalized = (
            target_sig[0],
            target_sig[1],
            tuple((feat, direction) for feat, direction in target_sig[2]),
        )

        votes = 0
        for it in recent_iters:
            iter_rows = [r for r in context_rows if int(r.get("iteration", -1)) == it]
            for row in iter_rows:
                row_sig = _sig_from_row(row)
                if row_sig is not None and row_sig == target_normalized:
                    votes += 1
                    break  # count at most once per iteration

        stable = votes >= min_votes
        return stable, votes, len(recent_iters)

    def _paper_audit_strategy(
        self,
        *,
        context: str,
        strategy: StrategySpec,
        discovery_dataset: str | None = None,
        validation_datasets: tuple[str, ...] | None = None,
        transfer_datasets: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        spec = self._context_spec(context)
        home_dataset = discovery_dataset or spec.discovery_datasets[0]
        home_task_type = spec.task_type
        home_df = load_dataset(home_dataset)
        baseline = _select_binding_baseline()
        home_eval = evaluate_strategies_for_dataset(
            home_df,
            [baseline, strategy],
            context=context,
            dataset=home_dataset,
            task_type=home_task_type,
            selection_status="autoresearch_paper_audit",
            validation_status="home_validation",
        )
        home_map = {row.strategy_name: row for row in home_eval}
        validation_rows: list[dict[str, Any]] = []
        validation_datasets = validation_datasets if validation_datasets is not None else tuple(spec.validation_datasets or ())
        for dataset in validation_datasets:
            df = load_dataset(dataset)
            rows = evaluate_strategies_for_dataset(
                df,
                [baseline, strategy],
                context=context,
                dataset=dataset,
                task_type=spec.task_type,
                selection_status="autoresearch_paper_audit",
                validation_status="held_out_validation",
            )
            row_map = {row.strategy_name: row for row in rows}
            candidate = row_map.get(strategy.strategy_name)
            base = row_map.get("binding_only")
            validation_rows.append(
                {
                    "dataset": dataset,
                    "candidate_metric": candidate.primary_metric if candidate is not None else None,
                    "baseline_metric": base.primary_metric if base is not None else None,
                    "candidate_delta_vs_binding": (
                        candidate.primary_metric - base.primary_metric
                        if candidate is not None
                        and candidate.primary_metric is not None
                        and base is not None
                        and base.primary_metric is not None
                        else None
                    ),
                    "candidate_ci": candidate.bootstrap_ci if candidate is not None else None,
                    "baseline_ci": base.bootstrap_ci if base is not None else None,
                }
            )

        transfer_rows: list[dict[str, Any]] = []
        for dataset in transfer_datasets if transfer_datasets is not None else self._active_transfer_datasets(home_dataset):
            if dataset == home_dataset or dataset in validation_datasets:
                continue
            transfer_context = _dataset_context(dataset)
            transfer_task_type = CONTEXT_SPECS[transfer_context].task_type if transfer_context in CONTEXT_SPECS else spec.task_type
            df = load_dataset(dataset)
            rows = evaluate_strategies_for_dataset(
                df,
                [baseline, strategy],
                context=transfer_context,
                dataset=dataset,
                task_type=transfer_task_type,
                selection_status="autoresearch_paper_audit",
                validation_status="transfer",
            )
            row_map = {row.strategy_name: row for row in rows}
            candidate = row_map.get(strategy.strategy_name)
            base = row_map.get("binding_only")
            transfer_rows.append(
                {
                    "dataset": dataset,
                    "context": transfer_context,
                    "candidate_metric": candidate.primary_metric if candidate is not None else None,
                    "baseline_metric": base.primary_metric if base is not None else None,
                    "candidate_delta_vs_binding": (
                        candidate.primary_metric - base.primary_metric
                        if candidate is not None
                        and candidate.primary_metric is not None
                        and base is not None
                        and base.primary_metric is not None
                        else None
                    ),
                    "candidate_ci": candidate.bootstrap_ci if candidate is not None else None,
                    "baseline_ci": base.bootstrap_ci if base is not None else None,
                }
            )

        return {
            "home_dataset": home_dataset,
            "home_evaluations": [row.to_dict() for row in home_eval],
            "home_row": home_map.get(strategy.strategy_name).to_dict() if home_map.get(strategy.strategy_name) is not None else None,
            "home_baseline_row": home_map.get("binding_only").to_dict() if home_map.get("binding_only") is not None else None,
            "validation_rows": validation_rows,
            "transfer_rows": transfer_rows,
        }

    def _specificity_table(
        self,
        *,
        context: str,
        proposal: StrategySpec,
        home_audit: dict[str, Any],
        transfer_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        home_row = home_audit.get("home_row") or {}
        home_delta = _as_float(home_row.get("primary_metric"))
        home_baseline = _as_float((home_audit.get("home_baseline_row") or {}).get("primary_metric"))
        home_delta_vs_binding = home_delta - home_baseline if home_delta is not None and home_baseline is not None else None
        rows: list[dict[str, Any]] = []
        for item in transfer_rows:
            away_delta = _as_float(item.get("candidate_delta_vs_binding"))
            specificity_gap = (
                home_delta_vs_binding - away_delta
                if home_delta_vs_binding is not None and away_delta is not None
                else None
            )
            if home_delta_vs_binding is not None and away_delta is not None:
                if away_delta > SPECIFICITY_FLATTEN_EPS:
                    specificity_label = "non_specific"
                elif away_delta >= -SPECIFICITY_FLATTEN_EPS:
                    specificity_label = "thesis_supporting"
                else:
                    specificity_label = "home_specific"
            else:
                specificity_label = "insufficient_data"
            rows.append(
                {
                    "context": context,
                    "strategy_name": proposal.strategy_name,
                    "display_name": proposal.display_name,
                    "home_dataset": home_audit.get("home_dataset"),
                    "away_dataset": item.get("dataset"),
                    "away_context": item.get("context"),
                    "home_metric": home_delta,
                    "home_baseline_metric": home_baseline,
                    "home_delta_vs_binding": home_delta_vs_binding,
                    "away_metric": item.get("candidate_metric"),
                    "away_baseline_metric": item.get("baseline_metric"),
                    "away_delta_vs_binding": away_delta,
                    "specificity_gap": specificity_gap,
                    "specificity_label": specificity_label,
                }
            )
        return rows

    def _crystallize_record(
        self,
        *,
        context: str,
        proposal: StrategySpec,
        home_audit: dict[str, Any],
        specificity_rows: list[dict[str, Any]],
        source_row: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now_iso()
        return {
            "strategy_name": proposal.strategy_name,
            "display_name": proposal.display_name,
            "strategy_origin": proposal.strategy_origin,
            "origin_dataset": proposal.origin_dataset,
            "family": proposal.family,
            "discovery_kind": proposal.discovery_kind,
            "context": context,
            "registry_state": "crystallized",
            "created_at": now,
            "last_modified": now,
            "components": [
                {"feature": feature, "weight": round(float(weight), 4), "direction": direction}
                for feature, weight, direction in proposal.components
            ],
            "crystallization": {
                "iteration": source_row.get("iteration"),
                "phase": source_row.get("phase"),
                "reason": source_row.get("reason"),
                "seed_stability_votes": source_row.get("seed_stability_votes"),
                "seed_stability_runs": source_row.get("seed_stability_runs"),
                "probable_overfit": source_row.get("probable_overfit"),
                "registry_context": True,
                "frozen": True,
            },
            "paper_audit": home_audit,
            "home_vs_away_specificity_table": specificity_rows,
        }

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_path.write_text(json.dumps(make_jsonable(self.state.to_dict()), indent=2), encoding="utf-8")

    def _write_progress_snapshot(self, *, current_context: str | None = None, completed_contexts: int | None = None, total_contexts: int | None = None, round_rows: int | None = None) -> None:
        payload = {
            "timestamp": _now_iso(),
            "iteration": self.state.iteration,
            "phase": self.state.phase,
            "stall_count": self.state.stall_count,
            "exploration_pressure": self.state.exploration_pressure,
            "current_context": current_context,
            "completed_contexts": completed_contexts,
            "total_contexts": total_contexts,
            "round_rows": round_rows,
            "best_by_context": {
                context: {
                    "strategy_name": row.get("strategy_name"),
                    "delta_vs_binding": row.get("delta_vs_binding"),
                    "operator_family": row.get("operator_family"),
                    "promoted": row.get("promoted"),
                }
                for context, row in self.state.best_by_context.items()
            },
        }
        self.config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.progress_path.write_text(json.dumps(make_jsonable(payload), indent=2), encoding="utf-8")

    def _family_weights(self) -> dict[str, float]:
        weights = {spec.family: float(spec.base_weight) for spec in self.config.operator_families}
        for family, defaults in self.state.family_stats.items():
            weights[family] = float(defaults.get("weight") or 1.0)
        if self.state.exploration_pressure > 0:
            exploratory = {"mutation", "random_blend", "recombination", "toolstack", "ml_importance", "icefire"}
            for family in exploratory:
                if family in weights:
                    weights[family] *= 1.0 + min(0.75, float(self.state.exploration_pressure))
        total = float(sum(max(v, 0.0) for v in weights.values()))
        if total <= 0:
            return {key: 1.0 / len(weights) for key in weights}
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def _feature_space(self, frame: pd.DataFrame) -> list[str]:
        return _feature_space(frame, self.state.phase, self.config.feature_pools)

    def _enabled_operator_specs(self) -> list[OperatorFamilySpec]:
        enabled: list[OperatorFamilySpec] = []
        for spec in self.config.operator_families:
            if self.state.phase < spec.enabled_from_phase:
                continue
            if spec.requires_tool and not self.tool_registry.available(spec.requires_tool):
                continue
            enabled.append(spec)
        return enabled

    def _operator_budget(self, spec: OperatorFamilySpec, weight: float) -> int:
        raw_budget = self.config.max_candidates_per_task * float(spec.base_weight) * max(weight, 0.0)
        budget = int(round(raw_budget))
        if spec.min_candidates > 0:
            budget = max(spec.min_candidates, budget)
        if spec.max_candidates > 0:
            budget = min(spec.max_candidates, budget)
        return max(0, budget)

    def _elite_payloads(self, context: str) -> list[dict[str, Any]]:
        payloads = list(self.state.elite_bank.get(context, []))
        payloads.sort(
            key=lambda row: (
                _sort_val(row, "robust_delta_vs_binding", _sort_val(row, "heldout_delta_vs_binding")),
                _sort_val(row, "transfer_delta_vs_binding"),
                _sort_val(row, "bio_score"),
                _sort_val(row, "stability"),
            ),
            reverse=True,
        )
        return payloads[: max(0, int(self.config.elite_bank_size))]

    def _elite_strategies(self, context: str) -> list[StrategySpec]:
        strategies: list[StrategySpec] = []
        for payload in self._elite_payloads(context):
            try:
                strategies.append(_strategy_from_payload(payload))
            except Exception:
                continue
        return strategies

    def _remember_elite(self, context: str, row: dict[str, Any], proposal: StrategySpec) -> None:
        payload = _strategy_payload(proposal)
        signature = json.dumps(_candidate_signature(proposal), default=str)
        structure = _structure_signature_text(proposal)
        payload.update(
            {
                "context": context,
                "iteration": self.state.iteration,
                "phase": self.state.phase,
                "signature": signature,
                "structure_signature": structure,
                "heldout_delta_vs_binding": row.get("heldout_delta_vs_binding"),
                "transfer_delta_vs_binding": row.get("transfer_delta_vs_binding"),
                "bio_score": row.get("bio_score"),
                "stability": row.get("stability"),
                "kept": row.get("kept"),
                "promoted": row.get("promoted"),
            }
        )
        bank = self.state.elite_bank.setdefault(context, [])
        bank = [item for item in bank if item.get("signature") != signature and item.get("structure_signature") != structure]
        bank.append(payload)
        bank.sort(
            key=lambda item: (
                _sort_val(item, "robust_delta_vs_binding", _sort_val(item, "heldout_delta_vs_binding")),
                _sort_val(item, "transfer_delta_vs_binding"),
                _sort_val(item, "bio_score"),
                _sort_val(item, "stability"),
            ),
            reverse=True,
        )
        self.state.elite_bank[context] = bank[: max(0, int(self.config.elite_bank_size))]

    def _bootstrap_margin(self, candidate_row: StrategyEvaluation | None, baseline_row: StrategyEvaluation | None) -> float | None:
        if candidate_row is None or baseline_row is None:
            return None
        candidate_lo = _as_float(candidate_row.bootstrap_ci.get("ci_lo"))
        baseline_hi = _as_float(baseline_row.bootstrap_ci.get("ci_hi"))
        if candidate_lo is None or baseline_hi is None:
            return None
        return candidate_lo - baseline_hi

    def _promotion_decision(
        self,
        *,
        registry_eligible: bool,
        heldout_delta: float | None,
        transfer_delta: float | None,
        heldout_ci_margin: float | None,
        seed_stable: bool,
        probable_overfit: bool,
        discovery_delta: float | None = None,
    ) -> tuple[bool, bool, str]:
        # Non-registry contexts (no held-out partner): keep if they beat baseline on discovery,
        # but never promote — they can seed future mutations but can't crystallize.
        if not registry_eligible:
            if probable_overfit:
                return False, False, "exploratory_overfit"
            beats_baseline = (
                (heldout_delta is not None and heldout_delta >= self.config.promote_delta)
                or (discovery_delta is not None and discovery_delta >= self.config.promote_delta)
            )
            if beats_baseline:
                return True, False, "exploratory_kept"
            return False, False, "exploratory_discarded"

        # Registry contexts (held-out partner available):
        if probable_overfit and heldout_delta is None:
            return False, False, "probable_overfit"
        if heldout_delta is None:
            return False, False, "no_heldout_partner"
        if heldout_delta < self.config.promote_delta:
            return False, False, "discarded"
        if self.config.require_transfer_support_when_available:
            if transfer_delta is None:
                return False, False, "no_transfer_partner"
            if transfer_delta < max(self.config.transfer_floor, self.config.promote_delta):
                return False, False, "discarded_transfer"
        if not seed_stable:
            # Still keep — good-delta unstable strategies can seed future rounds.
            return True, False, "kept_seed_unstable"
        ci_ready = heldout_ci_margin is None or heldout_ci_margin >= self.config.ci_margin
        if heldout_delta >= self.config.substantial_delta and ci_ready:
            return True, True, "promoted_home_only"
        if ci_ready:
            return True, False, "kept_home_only"
        # Good delta but CI overlaps too much — keep for seeding, don't promote.
        return True, False, "kept_ci_marginal"

    def _load_frame(self, dataset: str) -> pd.DataFrame:
        df = self.dataset_loader(dataset)
        if self.state.phase >= 3:
            df = _engineer_phase3_features(df)
        return df

    def _baseline_for_dataset(self, dataset: str, context: str, task_type: str, frame: pd.DataFrame) -> StrategySpec:
        return _select_binding_baseline()

    def _evaluate_dataset(self, dataset: str, context: str, task_type: str, strategies: list[StrategySpec], selection_status: str, validation_status: str) -> list[StrategyEvaluation]:
        frame = self._load_frame(dataset)
        return evaluate_strategies_for_dataset(
            frame,
            strategies,
            context=context,
            dataset=dataset,
            task_type=task_type,
            selection_status=selection_status,
            validation_status=validation_status,
        )

    def _select_seeds(self, ranked: list[StrategyEvaluation], strategy_library: list[StrategySpec], max_seeds: int) -> list[StrategySpec]:
        lookup = {strategy.strategy_name: strategy for strategy in strategy_library}
        seeds: list[StrategySpec] = []
        seen_families: set[str] = set()
        seen_structures: set[str] = set(self.state.seen_structures)
        seen_domains: set[str] = set()
        pool: list[tuple[float, float, float, StrategySpec]] = []
        for evaluation in ranked:
            candidate = lookup.get(evaluation.strategy_name)
            if candidate is None or candidate.strategy_name == "binding_only":
                continue
            pool.append(
                (
                    _as_float(evaluation.primary_metric) or float("-inf"),
                    _strategy_bio_score(candidate),
                    -float(_strategy_complexity(candidate)),
                    candidate.strategy_name,
                    candidate,
                )
            )
        pool.sort(reverse=True)
        for _metric, _bio, _complexity, _name, candidate in pool:
            family = str(candidate.family or "")
            structure = _structure_signature_text(candidate)
            domains = set(_strategy_bio_domains(candidate))
            if structure in seen_structures:
                continue
            if family in seen_families:
                continue
            if domains and domains.issubset(seen_domains) and len(seeds) < max_seeds - 1:
                continue
            seeds.append(
                replace(
                    candidate,
                    discovery_kind="seed_from_previous_round",
                    notes="Seeded from a previous surviving candidate.",
                )
            )
            seen_families.add(family)
            seen_structures.add(structure)
            seen_domains.update(domains)
            if len(seeds) >= max_seeds:
                break
        if len(seeds) < max_seeds:
            for _metric, _bio, _complexity, _name, candidate in pool:
                if len(seeds) >= max_seeds:
                    break
                structure = _structure_signature_text(candidate)
                domains = set(_strategy_bio_domains(candidate))
                if structure in seen_structures:
                    continue
                seeds.append(
                    replace(
                        candidate,
                        discovery_kind="seed_from_previous_round",
                        notes="Seeded from a previous surviving candidate.",
                    )
                )
                seen_structures.add(structure)
                seen_domains.update(domains)
        return seeds

    def _generate_proposals_for_task(self, task: dict[str, Any], rng: np.random.Generator) -> list[StrategySpec]:
        discovery_dataset = str(task["discovery_dataset"])
        context = str(task["context"])
        task_type = str(task["task_type"])
        phase = int(self.state.phase)
        discovery_df = self._load_frame(discovery_dataset)
        tool_plans = [
            {
                "template_id": plan.template_id,
                "family": plan.family,
                "display_name": plan.display_name,
                "required_tools": plan.required_tools,
                "required_modalities": plan.required_modalities,
                "base_weight": plan.base_weight,
                "min_candidates": plan.min_candidates,
                "max_candidates": plan.max_candidates,
                "feature_sets": [tuple(feature_set) for feature_set in plan.feature_sets],
                "notes": plan.notes,
            }
            for plan in self.research_registry.plans_for_dataset(discovery_dataset, phase)
        ]
        allowed_features = set(self._feature_space(discovery_df))
        seeds = _top_existing_seeds(discovery_df, context=context, origin_dataset=discovery_dataset, task_type=task_type)
        elites = self._elite_strategies(context)
        if elites:
            seeds.extend(elites)
        seeds = [
            seed
            for seed in seeds
            if _strategy_uses_safe_features(seed, allowed_features)
            and _structure_signature_text(seed) not in set(self.state.seen_structures)
        ]
        if not seeds:
            seeds = _top_existing_seeds(discovery_df, context=context, origin_dataset=discovery_dataset, task_type=task_type)
            seeds = [seed for seed in seeds if _strategy_uses_safe_features(seed, allowed_features)]
        ranked_base = _sort_evaluations(
            evaluate_strategies_for_dataset(
                discovery_df,
                seeds,
                context=context,
                dataset=discovery_dataset,
                task_type=task_type,
                selection_status="autoresearch_seed",
                validation_status="discovery",
            )
        )
        seed_pool = self._select_seeds(ranked_base, seeds, max_seeds=4)
        candidates: list[StrategySpec] = []
        family_weights = self._family_weights()
        enabled_specs = self._enabled_operator_specs()
        family_budget = {spec.family: self._operator_budget(spec, family_weights.get(spec.family, spec.base_weight)) for spec in enabled_specs}

        for spec in enabled_specs:
            budget = family_budget.get(spec.family, 0)
            if budget <= 0:
                continue
            generator = OPERATOR_GENERATORS.get(spec.generator)
            if generator is None:
                continue
            mutation_pool = seed_pool or seeds[:2]
            generated = generator(
                discovery_df=discovery_df,
                context=context,
                task_type=task_type,
                rng=rng,
                phase=phase,
                feature_pools=self.config.feature_pools,
                seed_pool=mutation_pool,
                seeds=seeds,
                tool_plans=tool_plans,
                budget=budget,
            )
            candidates.extend(generated[:budget])

        unique: list[StrategySpec] = []
        seen: set[tuple[object, ...]] = set()
        seen_structures = set(self.state.seen_structures)
        for candidate in candidates:
            if not _strategy_uses_safe_features(candidate, allowed_features):
                continue
            signature = _candidate_signature(candidate)
            structure = _structure_signature_text(candidate)
            if signature in seen:
                continue
            if structure in seen_structures:
                continue
            seen.add(signature)
            seen_structures.add(structure)
            unique.append(candidate)

        if not unique and candidates:
            fallback_seen: set[tuple[object, ...]] = set()
            for candidate in candidates:
                if not _strategy_uses_safe_features(candidate, allowed_features):
                    continue
                signature = _candidate_signature(candidate)
                if signature in fallback_seen:
                    continue
                fallback_seen.add(signature)
                unique.append(candidate)

        by_family: dict[str, list[StrategySpec]] = defaultdict(list)
        for candidate in unique:
            # Map to operator key so ml_logreg/ml_random_forest etc. are found under "ml_importance"
            op_key = DISCOVERY_KIND_TO_OPERATOR.get(candidate.discovery_kind, candidate.family)
            by_family[op_key].append(candidate)
        for family, family_candidates in by_family.items():
            family_candidates.sort(
                key=lambda candidate: (
                    _strategy_bio_score(candidate),
                    -float(_strategy_complexity(candidate)),
                    _structure_signature_text(candidate),
                ),
                reverse=True,
            )

        ordered: list[StrategySpec] = []
        ordered_signatures: set[tuple[object, ...]] = set()
        for spec in self.config.operator_families:
            family = spec.family
            if not by_family.get(family):
                continue
            candidate = by_family[family][0]
            signature = _candidate_signature(candidate)
            if signature in ordered_signatures:
                continue
            ordered.append(candidate)
            ordered_signatures.add(signature)

        for candidate in unique:
            signature = _candidate_signature(candidate)
            if signature in ordered_signatures:
                continue
            ordered.append(candidate)
            ordered_signatures.add(signature)
            if len(ordered) >= self.config.max_candidates_per_task:
                break

        ordered.sort(
            key=lambda candidate: (
                _strategy_bio_score(candidate),
                -float(_strategy_complexity(candidate)),
                _structure_signature_text(candidate),
            ),
            reverse=True,
        )

        return ordered[: self.config.max_candidates_per_task]

    def _build_evaluation_bundle(
        self,
        task: dict[str, Any],
        proposals: list[StrategySpec],
        prior_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        context = str(task["context"])
        task_type = str(task["task_type"])
        discovery_dataset = str(task["discovery_dataset"])
        validation_datasets = tuple(task.get("validation_datasets") or ())
        transfer_datasets = self._active_transfer_datasets(discovery_dataset)
        baseline = _select_binding_baseline()

        discovery_rows = self._evaluate_dataset(
            discovery_dataset,
            context,
            task_type,
            [baseline, *proposals],
            selection_status="autoresearch_discovery",
            validation_status="discovery",
        )
        discovery_map = {row.strategy_name: row for row in discovery_rows}

        validation_map: dict[str, dict[str, StrategyEvaluation]] = {}
        for dataset in validation_datasets:
            validation_rows = self._evaluate_dataset(
                dataset,
                context,
                task_type,
                [baseline, *proposals],
                selection_status="autoresearch_validation",
                validation_status="held_out_validation",
            )
            validation_map[dataset] = {row.strategy_name: row for row in validation_rows}

        # Only run transfer evaluation for registry contexts (those with a held-out partner).
        # Non-registry contexts discard all transfer results anyway — skip to save ~80% of compute.
        registry_eligible_context = self._is_registry_context(context)
        transfer_map: dict[str, dict[str, StrategyEvaluation]] = {}
        if registry_eligible_context:
            for dataset in transfer_datasets:
                transfer_context = _dataset_context(dataset)
                transfer_task_type = CONTEXT_SPECS[transfer_context].task_type if transfer_context in CONTEXT_SPECS else task_type
                transfer_rows = self._evaluate_dataset(
                    dataset,
                    transfer_context,
                    transfer_task_type,
                    [baseline, *proposals],
                    selection_status="autoresearch_transfer",
                    validation_status="transfer",
                )
                transfer_map[dataset] = {row.strategy_name: row for row in transfer_rows}

        discovery_frame = self._load_frame(discovery_dataset)

        def _proposal_score_col(proposal: StrategySpec, df: pd.DataFrame) -> pd.Series | None:
            """Build a composite score series for recall computation."""
            score = pd.Series(0.0, index=df.index)
            valid = False
            for feat, weight, direction in proposal.components:
                if feat not in df.columns:
                    continue
                col = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
                score += float(weight) * (col if direction == "desc" else -col)
                valid = True
            return score if valid else None

        ranking_rows: list[dict[str, Any]] = []
        specificity_rows: list[dict[str, Any]] = []
        for proposal in proposals:
            structure_str = _structure_signature_text(proposal)
            structure_seen_before = structure_str in set(self.state.seen_structures)
            discovery_row = discovery_map.get(proposal.strategy_name)
            baseline_discovery = discovery_map.get("binding_only")
            validation_rows = [mapping.get(proposal.strategy_name) for mapping in validation_map.values() if mapping.get(proposal.strategy_name) is not None]
            baseline_validation_rows = [mapping.get("binding_only") for mapping in validation_map.values() if mapping.get("binding_only") is not None]
            transfer_rows = [mapping.get(proposal.strategy_name) for mapping in transfer_map.values() if mapping.get(proposal.strategy_name) is not None]
            baseline_transfer_rows = [mapping.get("binding_only") for mapping in transfer_map.values() if mapping.get("binding_only") is not None]

            heldout_metric = _safe_mean(row.primary_metric for row in validation_rows)
            baseline_heldout = _safe_mean(row.primary_metric for row in baseline_validation_rows)
            transfer_metric = _safe_mean(row.primary_metric for row in transfer_rows)
            baseline_transfer = _safe_mean(row.primary_metric for row in baseline_transfer_rows)
            discovery_metric = _as_float(discovery_row.primary_metric) if discovery_row is not None else None
            baseline_discovery_metric = _as_float(baseline_discovery.primary_metric) if baseline_discovery is not None else None
            heldout_delta = None
            if heldout_metric is not None and baseline_heldout is not None:
                heldout_delta = heldout_metric - baseline_heldout
            transfer_delta = None
            if transfer_metric is not None and baseline_transfer is not None:
                transfer_delta = transfer_metric - baseline_transfer
            transfer_ci_margin = _safe_mean(
                [
                    self._bootstrap_margin(candidate_row, baseline_row)
                    for candidate_row, baseline_row in zip(transfer_rows, baseline_transfer_rows)
                    if candidate_row is not None and baseline_row is not None
                ]
            )
            heldout_ci_margin = _safe_mean(
                [
                    self._bootstrap_margin(candidate_row, baseline_row)
                    for candidate_row, baseline_row in zip(validation_rows, baseline_validation_rows)
                    if candidate_row is not None and baseline_row is not None
                ]
            )
            registry_eligible = self._is_registry_context(context)
            probable_overfit = bool(discovery_metric is not None and discovery_metric >= float(self.config.overfit_auc_ceiling))
            seed_votes = 0
            seed_runs = 0
            strong_bypass = float(getattr(self.config, "seed_stability_strong_delta_bypass", 0.08))
            stability_gate = registry_eligible and heldout_delta is not None and heldout_delta >= self.config.promote_delta
            if stability_gate and (heldout_delta is None or heldout_delta < strong_bypass):
                seed_stable, seed_votes, seed_runs = self._seed_stable_from_history(
                    proposal, context, prior_rows or []
                )
            else:
                seed_stable = True  # bypass: either not gated or delta large enough to trust

            discovery_delta = (
                (discovery_metric - baseline_discovery_metric)
                if discovery_metric is not None and baseline_discovery_metric is not None
                else None
            )
            kept, promoted, reason = self._promotion_decision(
                registry_eligible=registry_eligible,
                heldout_delta=heldout_delta,
                transfer_delta=transfer_delta,
                heldout_ci_margin=heldout_ci_margin,
                seed_stable=seed_stable,
                probable_overfit=probable_overfit,
                discovery_delta=discovery_delta,
            )
            best_metric_kind, best_metric_value = _robust_validation_value(
                heldout_delta,
                transfer_delta,
                discovery_delta,
            )

            ranking_rows.append(
                {
                    "iteration": self.state.iteration,
                    "phase": self.state.phase,
                    "context": context,
                    "discovery_dataset": discovery_dataset,
                    "validation_dataset": ";".join(validation_datasets) if validation_datasets else None,
                    "operator_family": proposal.family,
                    "discovery_kind": proposal.discovery_kind,
                    "strategy_name": proposal.strategy_name,
                    "display_name": proposal.display_name,
                    "strategy_origin": proposal.strategy_origin,
                    "origin_dataset": proposal.origin_dataset,
                    "structure_signature": structure_str,
                    "structure_seen_before": structure_seen_before,
                    "components": "; ".join(f"{feature}:{weight:.3f}:{direction}" for feature, weight, direction in proposal.components),
                    "complexity": _strategy_complexity(proposal),
                    "bio_score": _strategy_bio_score(proposal),
                    "discovery_metric": discovery_metric,
                    "baseline_discovery_metric": baseline_discovery_metric,
                    "discovery_delta_vs_binding": discovery_delta,
                    "heldout_metric": heldout_metric,
                    "baseline_heldout_metric": baseline_heldout,
                    "transfer_metric": transfer_metric,
                    "baseline_transfer_metric": baseline_transfer,
                    "heldout_delta_vs_binding": heldout_delta,
                    "transfer_delta_vs_binding": transfer_delta,
                    "robust_delta_vs_binding": best_metric_value,
                    "best_metric_kind": best_metric_kind,
                    "best_metric_value": best_metric_value,
                    "heldout_ci_margin": heldout_ci_margin,
                    "transfer_ci_margin": transfer_ci_margin,
                    "probable_overfit": probable_overfit,
                    "seed_stability_votes": seed_votes,
                    "seed_stability_runs": seed_runs,
                    "seed_stable": seed_stable,
                    "kept": kept,
                    "promoted": promoted,
                    "reason": reason,
                    "discovery_ci_lo": discovery_row.bootstrap_ci.get("ci_lo") if discovery_row is not None else None,
                    "discovery_ci_hi": discovery_row.bootstrap_ci.get("ci_hi") if discovery_row is not None else None,
                    "stability": _safe_mean(
                        [discovery_metric, heldout_metric, transfer_metric]
                    ),
                    "stability_n_splits": sum(1 for v in [discovery_metric, heldout_metric, transfer_metric] if v is not None),
                    "recall_at_5": None,
                    "recall_at_10": None,
                    "recall_at_20": None,
                }
            )
            # Compute per-patient recall@N and standardised contributions; patch into last row.
            score_series = _proposal_score_col(proposal, discovery_frame)
            if score_series is not None:
                tmp_col = f"__score_tmp_{id(proposal)}"
                discovery_frame[tmp_col] = score_series
                recall = _top_n_recall_per_patient(discovery_frame, tmp_col)
                for k, v in recall.items():
                    ranking_rows[-1][k] = v
                del discovery_frame[tmp_col]
            contribs = _standardised_contributions(proposal, discovery_frame)
            ranking_rows[-1].update(contribs)

            operator_key = DISCOVERY_KIND_TO_OPERATOR.get(proposal.discovery_kind, proposal.family)
            family_stats = self.state.family_stats[operator_key]
            family_stats["tried"] = float(family_stats.get("tried") or 0.0) + 1.0
            if kept:
                family_stats["kept"] = float(family_stats.get("kept") or 0.0) + 1.0
                if promoted:
                    family_stats["promoted"] = float(family_stats.get("promoted") or 0.0) + 1.0

            if kept or promoted:
                sig_str = json.dumps(_candidate_signature(proposal), default=str)
                structure_str = _structure_signature_text(proposal)
                if sig_str not in self.state.seen_signatures:
                    self.state.seen_signatures.append(sig_str)
                if structure_str not in self.state.seen_structures:
                    self.state.seen_structures.append(structure_str)
                best = self.state.best_by_context.get(context)
                if best is None or (best_metric_value is not None and best_metric_value > float(best.get("best_metric_value") or best.get("delta_vs_binding") or float("-inf"))):
                    self.state.best_by_context[context] = {
                        "strategy_name": proposal.strategy_name,
                        "display_name": proposal.display_name,
                        "operator_family": proposal.family,
                        "origin_dataset": proposal.origin_dataset,
                        "heldout_metric": heldout_metric,
                        "transfer_metric": transfer_metric,
                        "discovery_metric": discovery_metric,
                        "heldout_delta_vs_binding": heldout_delta,
                        "transfer_delta_vs_binding": transfer_delta,
                        "discovery_delta_vs_binding": discovery_delta,
                        "best_metric_kind": best_metric_kind,
                        "best_metric_value": best_metric_value,
                        "robust_delta_vs_binding": best_metric_value,
                        "delta_vs_binding": best_metric_value,
                        "promoted": promoted,
                        "phase": self.state.phase,
                        "registry_eligible": registry_eligible,
                        "probable_overfit": probable_overfit,
                        "seed_stable": seed_stable,
                    }
                if promoted:
                    self.state.substantial_promotions.append(
                        {
                            "iteration": self.state.iteration,
                            "phase": self.state.phase,
                            "context": context,
                            "strategy_name": proposal.strategy_name,
                            "operator_family": proposal.family,
                            "delta_vs_binding": best_metric_value,
                        }
                    )
                self._remember_elite(context, ranking_rows[-1], proposal)
                if promoted and validation_datasets:
                    home_audit = self._paper_audit_strategy(
                        context=context,
                        strategy=proposal,
                        discovery_dataset=discovery_dataset,
                        validation_datasets=validation_datasets,
                        transfer_datasets=transfer_datasets,
                    )
                    specificity_rows.extend(
                        self._specificity_table(
                            context=context,
                            proposal=proposal,
                            home_audit=home_audit,
                            transfer_rows=home_audit["transfer_rows"],
                        )
                    )
                    ranking_rows[-1]["paper_audit"] = home_audit
                    ranking_rows[-1]["home_vs_away_specificity"] = home_audit["transfer_rows"]

        all_validation_evals = [item for rows in validation_map.values() for item in rows.values()]
        all_transfer_evals = [item for rows in transfer_map.values() for item in rows.values()]
        return ranking_rows, list(discovery_rows), all_validation_evals + all_transfer_evals, specificity_rows

    def _update_family_weights(self) -> None:
        for family, stats in self.state.family_stats.items():
            tried = float(stats.get("tried") or 0.0)
            kept = float(stats.get("kept") or 0.0)
            promoted = float(stats.get("promoted") or 0.0)
            weight = float(stats.get("weight") or 1.0)
            if tried <= 0:
                continue
            keep_rate = kept / tried
            promote_rate = promoted / tried
            updated = weight * (0.90 + 0.15 * keep_rate + 0.25 * promote_rate)
            stats["weight"] = float(max(updated, 0.05))
        total = sum(float(stats.get("weight") or 0.0) for stats in self.state.family_stats.values())
        if total > 0:
            for stats in self.state.family_stats.values():
                stats["weight"] = float(stats.get("weight") or 0.0) / total

    def _normalize_family_weights(self) -> None:
        total = sum(float(stats.get("weight") or 0.0) for stats in self.state.family_stats.values())
        if total <= 0:
            return
        for stats in self.state.family_stats.values():
            stats["weight"] = float(stats.get("weight") or 0.0) / total

    def _apply_context_pivot(
        self,
        context: str,
        *,
        current: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current_metric = _as_float(current.get("best_metric_value"))
        previous_metric = _as_float(previous.get("best_metric_value")) if previous else None
        current_strategy = str(current.get("strategy_name") or "")
        current_family = str(current.get("operator_family") or "")
        previous_strategy = str(previous.get("strategy_name") or "") if previous else ""
        previous_family = str(previous.get("operator_family") or "") if previous else ""

        state = self.state.context_pivot_state.setdefault(
            context,
            {
                "streak": 0,
                "pivot_count": 0,
                "last_strategy_name": None,
                "last_operator_family": None,
                "last_metric_value": None,
                "last_pivot_iteration": None,
            },
        )
        state["last_strategy_name"] = current_strategy
        state["last_operator_family"] = current_family
        state["last_metric_value"] = current_metric

        improvement = None
        if current_metric is not None and previous_metric is not None:
            improvement = current_metric - previous_metric

        same_family = bool(current_family and current_family == previous_family)
        same_strategy = bool(current_strategy and current_strategy == previous_strategy)
        stagnating = False
        if previous is not None:
            if same_strategy and (improvement is None or improvement <= float(self.config.pivot_metric_epsilon)):
                stagnating = True
            elif same_family and (improvement is None or improvement <= float(self.config.pivot_metric_epsilon) * 1.5):
                stagnating = True

        if stagnating:
            state["streak"] = int(state.get("streak") or 0) + 1
        else:
            state["streak"] = 0

        if int(state.get("streak") or 0) < int(self.config.pivot_patience):
            return None

        dominant_family = current_family or previous_family
        retained_payloads: list[dict[str, Any]] = []
        for payload in self.state.elite_bank.get(context, []):
            if dominant_family and payload.get("family") == dominant_family:
                continue
            retained_payloads.append(payload)
        if not retained_payloads:
            retained_payloads = list(self.state.elite_bank.get(context, []))
        retained_payloads.sort(
            key=lambda row: (
                _sort_val(row, "bio_score"),
                _sort_val(row, "heldout_delta_vs_binding"),
                _sort_val(row, "transfer_delta_vs_binding"),
                _sort_val(row, "stability"),
            ),
            reverse=True,
        )
        self.state.elite_bank[context] = retained_payloads[: max(0, int(self.config.pivot_elite_retention))]

        if dominant_family and dominant_family in self.state.family_stats:
            stats = self.state.family_stats[dominant_family]
            current_weight = float(stats.get("weight") or 0.0)
            dampened = max(0.05, current_weight * (1.0 - float(self.config.pivot_family_penalty)))
            stats["weight"] = dampened
            self._normalize_family_weights()

        self.state.stall_count = 0
        self.state.exploration_pressure = min(1.0, float(self.state.exploration_pressure) + float(self.config.pivot_exploration_boost))

        state["streak"] = 0
        state["pivot_count"] = int(state.get("pivot_count") or 0) + 1
        state["last_pivot_iteration"] = self.state.iteration
        state["last_pivot_family"] = dominant_family
        state["last_pivot_metric"] = current_metric
        return {
            "context": context,
            "strategy_name": current_strategy,
            "operator_family": dominant_family,
            "improvement": improvement,
            "pivot_count": state["pivot_count"],
            "metric": current_metric,
        }

    def _advance_phase_if_needed(self, round_rows: list[dict[str, Any]], context_promotions: dict[str, bool] | None = None) -> None:
        if context_promotions is None:
            context_promotions = {}
        promoted_contexts = sum(1 for v in context_promotions.values() if v)
        min_required = getattr(self.config, "min_contexts_for_phase_advance", 2)
        enough_promotions = promoted_contexts >= min_required
        if enough_promotions:
            self.state.stall_count = 0
            self.state.exploration_pressure = max(0.0, self.state.exploration_pressure * 0.5)
            if self.state.phase < 3:
                self.state.phase += 1
            return
        # Single-context promotion counts as partial progress — reduce stall but don't reset.
        if promoted_contexts >= 1:
            self.state.stall_count = max(0, self.state.stall_count - 1)
            self.state.exploration_pressure = max(0.0, self.state.exploration_pressure * 0.75)
            return
        self.state.stall_count += 1
        self.state.exploration_pressure = min(1.0, self.state.exploration_pressure + 0.10)
        if self.state.stall_count >= self.config.stall_patience and self.state.phase < 3:
            self.state.phase += 1
            self.state.stall_count = 0

    def _write_outputs(
        self,
        ledger_rows: list[dict[str, Any]],
        current_round_rows: list[dict[str, Any]] | None = None,
        pivot_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, int]:
            robust = _sort_val(row, "robust_delta_vs_binding", _sort_val(row, "heldout_delta_vs_binding"))
            return (
                -float(_support_tier(row)),
                _novelty_bonus(row, float(self.config.novelty_bonus)),
                robust,
                _sort_val(row, "heldout_delta_vs_binding"),
                _sort_val(row, "transfer_delta_vs_binding"),
                _sort_val(row, "heldout_metric"),
                _sort_val(row, "transfer_metric"),
                _sort_val(row, "discovery_metric"),
                _sort_val(row, "bio_score"),
                -int(row.get("complexity") or 0),
            )

        deduped_rows: dict[str, dict[str, Any]] = {}
        deduped_scores: dict[str, tuple[float, float, float, float, float, float, float, int]] = {}
        for row in ledger_rows:
            strategy_name = str(row.get("strategy_name") or "").strip()
            if not strategy_name:
                continue
            score = _rank_key(row)
            previous = deduped_scores.get(strategy_name)
            if previous is None or score > previous:
                deduped_scores[strategy_name] = score
                deduped_rows[strategy_name] = row

        ranked_rows = sorted(deduped_rows.values(), key=_rank_key, reverse=True)
        recomputed_best_by_context: dict[str, dict[str, Any]] = {}
        for row in ranked_rows:
            context = str(row.get("context") or "").strip()
            strategy_name = str(row.get("strategy_name") or "").strip()
            if not context or not strategy_name or context in recomputed_best_by_context:
                continue
            # Backfill discovery_delta_vs_binding for old rows that predate the field.
            if row.get("discovery_delta_vs_binding") is None:
                disc = _as_float(row.get("discovery_metric"))
                base = _as_float(row.get("baseline_discovery_metric"))
                if disc is not None and base is not None:
                    row = dict(row)
                    row["discovery_delta_vs_binding"] = disc - base

            # For non-registry contexts (no held-out partner), only trust discovery_delta.
            # Old ledger rows may have stale transfer_delta from before transfer was skipped
            # for non-registry contexts — ignore those values here.
            row_registry = row.get("registry_eligible")
            if row_registry is False or (row_registry is None and not self._is_registry_context(context)):
                best_metric_kind, best_metric_value = _robust_validation_value(
                    None, None, _as_float(row.get("discovery_delta_vs_binding")),
                )
            else:
                best_metric_kind, best_metric_value = _robust_validation_value(
                    _as_float(row.get("heldout_delta_vs_binding")),
                    _as_float(row.get("transfer_delta_vs_binding")),
                    _as_float(row.get("discovery_delta_vs_binding")),
                )
            if best_metric_value is None:
                best_metric_kind = "delta_vs_binding"
                best_metric_value = _as_float(row.get("delta_vs_binding"))
            if best_metric_value is None:
                best_metric_value = _as_float(row.get("heldout_delta_vs_binding"))
            recomputed_best_by_context[context] = {
                "strategy_name": strategy_name,
                "display_name": row.get("display_name"),
                "operator_family": row.get("operator_family"),
                "origin_dataset": row.get("origin_dataset"),
                "heldout_metric": row.get("heldout_metric"),
                "transfer_metric": row.get("transfer_metric"),
                "discovery_metric": row.get("discovery_metric"),
                "heldout_delta_vs_binding": row.get("heldout_delta_vs_binding"),
                "transfer_delta_vs_binding": row.get("transfer_delta_vs_binding"),
                "discovery_delta_vs_binding": row.get("discovery_delta_vs_binding"),
                "robust_delta_vs_binding": row.get("robust_delta_vs_binding"),
                "best_metric_kind": row.get("best_metric_kind") or best_metric_kind,
                "best_metric_value": row.get("best_metric_value") if row.get("best_metric_value") is not None else best_metric_value,
                "delta_vs_binding": row.get("best_metric_value") if row.get("best_metric_value") is not None else best_metric_value,
                "promoted": row.get("promoted"),
                "phase": row.get("phase"),
                "registry_eligible": row.get("registry_eligible"),
                "probable_overfit": row.get("probable_overfit"),
                "seed_stable": row.get("seed_stable"),
            }
        if recomputed_best_by_context:
            self.state.best_by_context = recomputed_best_by_context
        kept_rows = [row for row in ranked_rows if bool(row.get("kept"))]
        pareto_rows = _pareto_front(kept_rows or ranked_rows)

        self.config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ledger_rows).to_csv(self.config.ledger_path, index=False)
        pd.DataFrame(ranked_rows[: max(1, min(50, len(ranked_rows)))]).to_csv(self.config.ranked_path, index=False)
        pd.DataFrame(pareto_rows).to_csv(self.config.pareto_path, index=False)
        specificity_rows = [
            row
            for row in ledger_rows
            if row.get("home_vs_away_specificity")
        ]
        specificity_table = [item for row in ledger_rows for item in (row.get("home_vs_away_specificity") or [])]
        specificity_path = self.config.ledger_path.parent / "autoresearch_loop_specificity.csv"
        pd.DataFrame(specificity_table).to_csv(specificity_path, index=False)
        (self.config.ledger_path.parent / "autoresearch_loop_specificity.json").write_text(
            json.dumps(make_jsonable(specificity_table), indent=2),
            encoding="utf-8",
        )

        summary = {
            "timestamp": _now_iso(),
            "state": self.state.to_dict(),
            "best_by_context": self.state.best_by_context,
            "crystallized_contexts": self.state.crystallized_contexts,
            "pivot_events": pivot_events or [],
            "family_stats": {key: dict(value) for key, value in self.state.family_stats.items()},
            "plan": {
                "datasets_manifest": str(self.config.datasets_manifest) if self.config.datasets_manifest else None,
                "operators_manifest": str(self.config.operators_manifest) if self.config.operators_manifest else None,
                "phase_datasets": {str(key): list(value) for key, value in self.config.phase_datasets.items()},
                "feature_pools": {str(key): list(value) for key, value in self.config.feature_pools.items()},
                "operator_families": [
                    {
                        "family": spec.family,
                        "generator": spec.generator,
                        "enabled_from_phase": spec.enabled_from_phase,
                        "base_weight": spec.base_weight,
                        "min_candidates": spec.min_candidates,
                        "max_candidates": spec.max_candidates,
                        "requires_tool": spec.requires_tool,
                    }
                    for spec in self.config.operator_families
                ],
                "promotion": {
                    "promote_delta": self.config.promote_delta,
                    "substantial_delta": self.config.substantial_delta,
                    "ci_margin": self.config.ci_margin,
                    "require_transfer_support_when_available": self.config.require_transfer_support_when_available,
                    "require_heldout_support_when_available": self.config.require_heldout_support_when_available,
                },
                "elite_bank_size": self.config.elite_bank_size,
                "exploration_pressure": self.state.exploration_pressure,
            },
            "safety": {
                "blocked_feature_names": sorted(LEAKAGE_FEATURE_NAMES),
                "blocked_feature_token_count": len(LEAKAGE_NAME_TOKENS),
                "safe_feature_policy": "manifest_only_plus_leakage_filter",
                "cross_fit_ml_importance": True,
                "require_transfer_support_when_available": self.config.require_transfer_support_when_available,
                "require_heldout_support_when_available": self.config.require_heldout_support_when_available,
                "overfit_auc_ceiling": float(self.config.overfit_auc_ceiling),
                "promotion_gates_home_only": True,
                "bootstrap_unit": "patient_level_mean_auc",
                "multiple_comparison_caveat": (
                    f"The loop evaluated {len(ledger_rows)} strategy-dataset pairs across "
                    f"{self.state.iteration} iterations with no formal multiple-comparison "
                    "correction (no Bonferroni, no FDR). All loop-derived results should be "
                    "treated as exploratory. Only strategies re-evaluated through the paper "
                    "audit pipeline (crystallize()) with a pre-specified evaluation protocol "
                    "are suitable for main-text reporting."
                ),
            },
            "ranked_winners": ranked_rows[:25],
            "pareto_front": pareto_rows[:25],
            "home_vs_away_specificity_table": specificity_table[:250],
            "tool_registry": self.tool_registry.status("icefire"),
            "registry": self.research_registry.summary(),
        }
        self.config.summary_path.write_text(json.dumps(make_jsonable(summary), indent=2), encoding="utf-8")
        lines = [
            "# Autoresearch Loop",
            "",
            f"- Timestamp: `{summary['timestamp']}`",
            f"- Iteration: `{self.state.iteration}`",
            f"- Phase: `{self.state.phase}`",
            f"- Stall count: `{self.state.stall_count}`",
            f"- Exploration pressure: `{self.state.exploration_pressure:.2f}`",
            f"- Icefire status: `{summary['tool_registry']['status']}`",
            "",
            "## Active Plan",
        ]
        lines.append(
            f"- Datasets: {', '.join(self.config.phase_datasets.get(self.state.phase, ())) if self.config.phase_datasets else 'n/a'}"
        )
        lines.append(
            f"- Operators: {', '.join(spec.family for spec in self._enabled_operator_specs()) or 'none'}"
        )
        lines.append(f"- Dataset manifest: `{self.config.datasets_manifest or 'embedded defaults'}`")
        lines.append(f"- Operator manifest: `{self.config.operators_manifest or 'embedded defaults'}`")
        lines.append(f"- Tool registry manifest: `{self.config.tool_registry_manifest or 'embedded defaults'}`")
        lines.append(f"- Dataset capabilities manifest: `{self.config.dataset_capabilities_manifest or 'embedded defaults'}`")
        if self.research_registry.tool_templates:
            lines.append(
                f"- Tool stacks: {', '.join(sorted(self.research_registry.tool_templates))}"
            )
        lines.append(f"- Safety policy: manifest-only + leakage filter + cross-fit ML importance")
        lines.extend(
            [
                "## Best By Context",
            ]
        )
        for context, row in self.state.best_by_context.items():
            r20 = next(
                (r.get("recall_at_20") for r in ranked_rows if r.get("strategy_name") == row.get("strategy_name") and r.get("context") == context),
                None,
            )
            r20_str = f"recall@20={r20:.3f}" if isinstance(r20, float) else ""
            best_metric_kind = str(row.get("best_metric_kind") or "delta_vs_binding")
            best_metric_value = _as_float(row.get("best_metric_value"))
            metric_label = {
                "heldout_delta_vs_binding": "heldout_delta",
                "transfer_delta_vs_binding": "transfer_delta",
                "robust_transfer_delta_vs_binding": "robust_delta",
                "discovery_delta_vs_binding": "discovery_delta",
            }.get(best_metric_kind, "delta")
            metric_str = f"{metric_label}={best_metric_value:+.4f}" if best_metric_value is not None else f"{metric_label}=n/a"
            lines.append(
                f"- {context}: {row.get('strategy_name')} "
                f"{metric_str} {r20_str} "
                f"family={row.get('operator_family')} promoted={row.get('promoted')}"
            )
        if self.state.elite_bank:
            lines.extend(["", "## Elite Bank"])
            for context, payloads in self.state.elite_bank.items():
                lines.append(f"- {context}: {len(payloads)} stored survivor(s)")
        lines.extend(["", "## Top Ranked (all history) — transfer-supported first, then novelty, then held-out, then discovery"])
        for row in ranked_rows[:10]:
            r20 = row.get("recall_at_20")
            r20_str = f"{r20:.3f}" if isinstance(r20, float) else "n/a"
            hd = row.get("heldout_delta_vs_binding")
            hd_str = f"{hd:+.4f}" if isinstance(hd, float) else "n/a"
            novelty = "new" if not bool(row.get("structure_seen_before")) else "seen"
            lines.append(
                f"- {row.get('context','?')} | {row['strategy_name']}: "
                f"novelty={novelty} recall@20={r20_str} heldout_delta={hd_str} "
                f"reason={row.get('reason')} complexity={row.get('complexity')}"
            )
        if specificity_table:
            lines.extend(["", "## Home vs Away Specificity"])
            for row in specificity_table[:10]:
                lines.append(
                    f"- {row.get('strategy_name')} @ {row.get('away_dataset')}: "
                    f"home_delta={row.get('home_delta_vs_binding')} away_delta={row.get('away_delta_vs_binding')} "
                    f"label={row.get('specificity_label')}"
                )
        if current_round_rows:
            this_round_sorted = sorted(
                current_round_rows,
                key=lambda r: (
                    -float(_support_tier(r)),
                    _novelty_bonus(r, float(self.config.novelty_bonus)),
                    _sort_val(r, "robust_delta_vs_binding", _sort_val(r, "heldout_delta_vs_binding")),
                    _sort_val(r, "transfer_delta_vs_binding"),
                ),
                reverse=True,
            )
            lines.extend(["", "## This Round Top-5"])
            for row in this_round_sorted[:5]:
                novelty = "new" if not bool(row.get("structure_seen_before")) else "seen"
                lines.append(
                    f"- {row['strategy_name']}: novelty={novelty} robust_delta={row.get('robust_delta_vs_binding')} "
                    f"heldout_delta={row.get('heldout_delta_vs_binding')} "
                    f"transfer_delta={row.get('transfer_delta_vs_binding')} reason={row.get('reason')}"
                )
        if pivot_events:
            lines.extend(["", "## Pivot Events"])
            for event in pivot_events:
                lines.append(
                    f"- {event['context']}: pivoted family={event.get('operator_family')} "
                    f"after streak reset; improvement={event.get('improvement')} "
                    f"metric={event.get('metric')} strategy={event.get('strategy_name')}"
                )
        self.config.runlog_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._save_state()
        return summary

    def _load_ledger_history(self) -> list[dict[str, Any]]:
        if not self.config.ledger_path.exists():
            return []
        try:
            return pd.read_csv(self.config.ledger_path).to_dict("records")
        except Exception:
            return []

    def run_round(self) -> dict[str, Any]:
        rng = np.random.default_rng(self.config.seed + self.state.iteration)
        prior_rows = self._load_ledger_history()
        prior_best_by_context = {context: dict(row or {}) for context, row in self.state.best_by_context.items()}
        round_rows: list[dict[str, Any]] = []
        context_promotions: dict[str, bool] = {}
        pivot_events: list[dict[str, Any]] = []
        all_eval_detail: list[dict[str, Any]] = []
        contexts = list(CONTEXT_SPECS.items())
        self._write_progress_snapshot(current_context="starting", completed_contexts=0, total_contexts=len(contexts), round_rows=0)
        for index, (context, spec) in enumerate(contexts, start=1):
            if context in self.state.crystallized_contexts:
                self._write_progress_snapshot(current_context=f"skip:{context}", completed_contexts=index, total_contexts=len(contexts), round_rows=len(round_rows))
                continue
            self._write_progress_snapshot(current_context=context, completed_contexts=index - 1, total_contexts=len(contexts), round_rows=len(round_rows))
            task = _task_spec(context)
            proposals = self._generate_proposals_for_task(task, rng)
            rows, discovery_evals, holdout_transfer_evals, specificity_rows = self._build_evaluation_bundle(task, proposals, prior_rows=prior_rows)
            round_rows.extend(rows)
            context_promotions[context] = any(bool(r.get("promoted")) for r in rows)
            all_eval_detail.extend(make_jsonable(e.to_dict()) for e in discovery_evals)
            all_eval_detail.extend(make_jsonable(e.to_dict()) for e in holdout_transfer_evals)
            all_eval_detail.extend(make_jsonable(row) for row in specificity_rows)
            self._write_progress_snapshot(current_context=f"done:{context}", completed_contexts=index, total_contexts=len(contexts), round_rows=len(round_rows))
        self.state.iteration += 1
        self._update_family_weights()
        self._advance_phase_if_needed(round_rows, context_promotions)
        for context, current in list(self.state.best_by_context.items()):
            previous = prior_best_by_context.get(context)
            event = self._apply_context_pivot(context, current=current, previous=previous)
            if event is not None:
                pivot_events.append(event)
        all_ledger_rows = prior_rows + round_rows
        eval_detail_path = self.config.ledger_path.parent / "autoresearch_loop_eval_detail.jsonl"
        try:
            with open(eval_detail_path, "a", encoding="utf-8") as fh:
                for record in all_eval_detail:
                    fh.write(json.dumps(make_jsonable(record)) + "\n")
        except Exception:
            pass
        self._write_progress_snapshot(current_context="round_complete", completed_contexts=len(contexts), total_contexts=len(contexts), round_rows=len(round_rows))
        return self._write_outputs(all_ledger_rows, current_round_rows=round_rows, pivot_events=pivot_events)

    def run(self, *, max_rounds: int | None = None, daemon: bool = False) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        if daemon:
            # In daemon mode, max_rounds=None means run forever.
            # config.max_rounds is a one-shot batch limit and does not apply here.
            completed = 0
            while max_rounds is None or completed < int(max_rounds):
                latest = self.run_round()
                completed += 1
                if self.config.daemon_sleep_seconds > 0:
                    time.sleep(self.config.daemon_sleep_seconds)
        else:
            rounds = max_rounds if max_rounds is not None else self.config.max_rounds
            for _ in range(max(1, int(rounds or 1))):
                latest = self.run_round()
        return latest

    def crystallize(self, context: str, output_dir: Path | None = None) -> list[Path]:
        if context not in CONTEXT_SPECS:
            raise KeyError(context)
        if not self._is_registry_context(context):
            raise ValueError(f"{context} does not have a held-out registry partner and cannot crystallize.")
        if context in self.state.crystallized_contexts:
            stored = self.state.crystallized_contexts[context]
            artifact = stored.get("artifact_path")
            return [Path(str(artifact))] if artifact else []

        payloads = self._elite_payloads(context)
        if not payloads:
            raise ValueError(f"No elite strategy available to crystallize for context {context}.")

        source_row = payloads[0]
        proposal = _strategy_from_payload(source_row)
        home_audit = self._paper_audit_strategy(context=context, strategy=proposal)
        specificity_rows = self._specificity_table(
            context=context,
            proposal=proposal,
            home_audit=home_audit,
            transfer_rows=home_audit["transfer_rows"],
        )
        record = self._crystallize_record(
            context=context,
            proposal=proposal,
            home_audit=home_audit,
            specificity_rows=specificity_rows,
            source_row=source_row,
        )

        try:
            import yaml
        except ImportError:
            print("[autoresearch] WARNING: pyyaml not installed — cannot crystallize strategy.", file=sys.stderr)
            return []

        out_dir = output_dir or (ROOT / "configs" / "strategies_extra")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in proposal.strategy_name)
        out_path = out_dir / f"crystallized_{context}_{safe_name}_v{self.state.iteration:04d}.yaml"
        out_path.write_text(yaml.dump(record, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")

        self.state.crystallized_contexts[context] = {
            "strategy_name": proposal.strategy_name,
            "display_name": proposal.display_name,
            "strategy_origin": proposal.strategy_origin,
            "origin_dataset": proposal.origin_dataset,
            "family": proposal.family,
            "discovery_kind": proposal.discovery_kind,
            "components": record["components"],
            "artifact_path": str(out_path),
            "paper_audit": record["paper_audit"],
            "home_vs_away_specificity_table": specificity_rows,
            "crystallized_at": record["created_at"],
        }
        best = self.state.best_by_context.get(context, {})
        best.update(
            {
                "strategy_name": proposal.strategy_name,
                "display_name": proposal.display_name,
                "operator_family": proposal.family,
                "origin_dataset": proposal.origin_dataset,
                "paper_audit": record["paper_audit"],
                "crystallized": True,
            }
        )
        self.state.best_by_context[context] = best
        self._save_state()
        return [out_path]

    def export_to_registry(self, output_dir: Path | None = None, min_status: str = "kept") -> list[Path]:
        """Write promoted/kept strategies as YAML files loadable by neoresist/strategy_registry.py.

        Args:
            output_dir: Target directory. Defaults to configs/strategies_extra/.
            min_status: Minimum reason to export — "kept" (default) or "promoted".
        Returns:
            List of written file paths.
        """
        try:
            import yaml
        except ImportError:
            print("[autoresearch] WARNING: pyyaml not installed — cannot export to registry.", file=sys.stderr)
            return []

        out_dir = output_dir or (ROOT / "configs" / "strategies_extra")
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        promoted_only = min_status == "promoted"
        if self.state.crystallized_contexts:
            for context, payload in self.state.crystallized_contexts.items():
                strategy_name = str(payload.get("strategy_name") or "")
                if not strategy_name:
                    continue
                record = {
                    "strategy_name": strategy_name,
                    "display_name": str(payload.get("display_name") or strategy_name),
                    "strategy_origin": str(payload.get("strategy_origin") or "autoresearch"),
                    "origin_dataset": payload.get("origin_dataset"),
                    "family": str(payload.get("family") or "autoresearch"),
                    "discovery_kind": str(payload.get("discovery_kind") or "autoresearch"),
                    "context": context,
                    "components": payload.get("components") or [],
                    "registry_state": "crystallized",
                    "paper_audit": payload.get("paper_audit"),
                    "home_vs_away_specificity_table": payload.get("home_vs_away_specificity_table"),
                    "autoresearch_metadata": {
                        "artifact_path": payload.get("artifact_path"),
                        "crystallized_at": payload.get("crystallized_at"),
                    },
                }
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy_name)
                out_path = out_dir / f"crystallized_{context}_{safe_name}.yaml"
                out_path.write_text(yaml.dump(record, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")
                written.append(out_path)
        else:
            for context, payloads in self.state.elite_bank.items():
                best_payloads = payloads[:1] if payloads else []
                for payload in best_payloads:
                    reason = str(payload.get("reason") or "")
                    if promoted_only and "promoted" not in reason:
                        continue
                    strategy_name = str(payload.get("strategy_name") or "")
                    if not strategy_name:
                        continue
                    components_raw = payload.get("components") or []
                    try:
                        components = [
                            {"feature": f, "weight": round(float(w), 4), "direction": d}
                            for f, w, d in components_raw
                        ]
                    except Exception:
                        continue
                    audit = self._paper_audit_strategy(
                        context=context,
                        strategy=_strategy_from_payload(payload),
                        discovery_dataset=payload.get("origin_dataset") or _task_spec(context)["discovery_dataset"],
                    ) if self._is_registry_context(context) else None
                    record = {
                        "strategy_name": strategy_name,
                        "display_name": str(payload.get("display_name") or strategy_name),
                        "strategy_origin": str(payload.get("strategy_origin") or "autoresearch"),
                        "origin_dataset": payload.get("origin_dataset"),
                        "family": str(payload.get("family") or "autoresearch"),
                        "discovery_kind": str(payload.get("discovery_kind") or "autoresearch"),
                        "context": context,
                        "components": components,
                        "autoresearch_metadata": {
                            "iteration": payload.get("iteration"),
                            "phase": payload.get("phase"),
                            "heldout_delta_vs_binding": payload.get("heldout_delta_vs_binding"),
                            "transfer_delta_vs_binding": payload.get("transfer_delta_vs_binding"),
                            "stability": payload.get("stability"),
                            "reason": reason,
                        },
                        "paper_audit": audit,
                    }
                    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy_name)
                    out_path = out_dir / f"autoresearch_{safe_name}.yaml"
                    out_path.write_text(yaml.dump(record, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    written.append(out_path)

        print(f"[autoresearch] Exported {len(written)} strategies to {out_dir}", file=sys.stderr)
        return written

    def launch_background(self) -> subprocess.Popen[bytes] | None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out_log = open(ARTIFACTS / "autoresearch_loop_daemon.out.log", "a", encoding="utf-8")
        err_log = open(ARTIFACTS / "autoresearch_loop_daemon.err.log", "a", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(
            [sys.executable, "-m", "backend.strategy_engine.autoresearch_loop", "--daemon"],
            cwd=str(ROOT),
            stdout=out_log,
            stderr=err_log,
            creationflags=creationflags,
        )


def load_config(path: Path | None = None) -> AutoResearchConfig:
    if path is None:
        path = CONFIGS / "autoresearch_loop.yaml"
    if not path.exists():
        return AutoResearchConfig()
    payload = _load_yaml_like(path)
    kwargs = {}
    for key in ("state_path", "ledger_path", "ranked_path", "pareto_path", "progress_path", "summary_path", "runlog_path"):
        if key in payload and payload[key]:
            kwargs[key] = Path(str(payload[key]))
    for key in ("max_rounds", "stall_patience", "promote_delta", "substantial_delta", "max_candidates_per_task", "seed", "daemon_sleep_seconds"):
        if key in payload and payload[key] is not None:
            kwargs[key] = payload[key]
    for key in (
        "ci_margin",
        "elite_bank_size",
        "min_contexts_for_phase_advance",
        "soft_promote_delta",
        "overfit_auc_ceiling",
        "seed_stability_strong_delta_bypass",
        "pivot_patience",
        "pivot_metric_epsilon",
        "pivot_exploration_boost",
        "pivot_family_penalty",
        "pivot_elite_retention",
        "transfer_floor",
        "novelty_bonus",
    ):
        if key in payload and payload[key] is not None:
            kwargs[key] = payload[key]
    for key in ("seed_stability_runs", "seed_stability_min_votes"):
        if key in payload and payload[key] is not None:
            kwargs[key] = int(payload[key])
    for key in ("require_transfer_support_when_available", "require_heldout_support_when_available"):
        if key in payload and payload[key] is not None:
            kwargs[key] = bool(payload[key])
    if payload.get("operator_priority"):
        kwargs["operator_priority"] = tuple(str(item) for item in payload["operator_priority"])
    if payload.get("feature_pools"):
        feature_payload: dict[int, tuple[str, ...]] = {}
        for key, value in dict(payload["feature_pools"]).items():
            try:
                phase = int(key)
            except (TypeError, ValueError):
                continue
            feature_payload[phase] = tuple(str(item) for item in (value or []))
        if feature_payload:
            kwargs["feature_pools"] = feature_payload
    if payload.get("operator_families"):
        families: list[OperatorFamilySpec] = []
        for item in payload["operator_families"]:
            entry = dict(item or {})
            family = str(entry.get("family") or entry.get("generator") or "custom")
            generator = str(entry.get("generator") or family)
            families.append(
                OperatorFamilySpec(
                    family=family,
                    generator=generator,
                    enabled_from_phase=int(entry.get("enabled_from_phase") or 1),
                    base_weight=float(entry.get("base_weight") or 1.0),
                    min_candidates=int(entry.get("min_candidates") or 0),
                    max_candidates=int(entry.get("max_candidates") or 4),
                    requires_tool=str(entry.get("requires_tool")) if entry.get("requires_tool") else None,
                )
            )
        if families:
            kwargs["operator_families"] = tuple(families)
    if "operator_families" not in kwargs and kwargs.get("operator_priority"):
        generator_map = {
            "existing": "existing",
            "mutation": "mutation",
            "recombination": "recombination",
            "toolstack": "toolstack",
            "random_blend": "random_blend",
            "ml_importance": "ml_importance",
            "icefire": "icefire",
        }
        kwargs["operator_families"] = tuple(
            OperatorFamilySpec(
                family=str(family),
                generator=generator_map.get(str(family), str(family)),
                enabled_from_phase=1 if str(family) != "ml_importance" else 2,
                base_weight=1.0 if str(family) != "icefire" else 0.7,
                min_candidates=0 if str(family) == "icefire" else 1,
                max_candidates=2 if str(family) == "icefire" else 4,
                requires_tool="icefire" if str(family) == "icefire" else None,
            )
            for family in kwargs["operator_priority"]
        )
    if payload.get("phase_datasets"):
        phase_payload = {}
        for key, value in dict(payload["phase_datasets"]).items():
            try:
                phase = int(key)
            except (TypeError, ValueError):
                continue
            phase_payload[phase] = tuple(str(item) for item in (value or []))
        if phase_payload:
            kwargs["phase_datasets"] = phase_payload
    if payload.get("datasets_manifest"):
        datasets_manifest = _resolve_relative_path(path, payload.get("datasets_manifest"))
        if datasets_manifest and datasets_manifest.exists():
            datasets_payload = _load_yaml_like(datasets_manifest)
            if datasets_payload.get("feature_pools"):
                feature_payload: dict[int, tuple[str, ...]] = {}
                for key, value in dict(datasets_payload["feature_pools"]).items():
                    try:
                        phase = int(key)
                    except (TypeError, ValueError):
                        continue
                    feature_payload[phase] = tuple(str(item) for item in (value or []))
                if feature_payload:
                    kwargs["feature_pools"] = feature_payload
            if datasets_payload.get("phase_datasets"):
                phase_payload = {}
                for key, value in dict(datasets_payload["phase_datasets"]).items():
                    try:
                        phase = int(key)
                    except (TypeError, ValueError):
                        continue
                    phase_payload[phase] = tuple(str(item) for item in (value or []))
                if phase_payload:
                    kwargs["phase_datasets"] = phase_payload
            kwargs["datasets_manifest"] = datasets_manifest
    if payload.get("operators_manifest"):
        operators_manifest = _resolve_relative_path(path, payload.get("operators_manifest"))
        if operators_manifest and operators_manifest.exists():
            operators_payload = _load_yaml_like(operators_manifest)
            if operators_payload.get("operator_families"):
                families: list[OperatorFamilySpec] = []
                for item in operators_payload["operator_families"]:
                    entry = dict(item or {})
                    family = str(entry.get("family") or entry.get("generator") or "custom")
                    generator = str(entry.get("generator") or family)
                    families.append(
                        OperatorFamilySpec(
                            family=family,
                            generator=generator,
                            enabled_from_phase=int(entry.get("enabled_from_phase") or 1),
                            base_weight=float(entry.get("base_weight") or 1.0),
                            min_candidates=int(entry.get("min_candidates") or 0),
                            max_candidates=int(entry.get("max_candidates") or 4),
                            requires_tool=str(entry.get("requires_tool")) if entry.get("requires_tool") else None,
                        )
                    )
                if families:
                    kwargs["operator_families"] = tuple(families)
            kwargs["operators_manifest"] = operators_manifest
    if payload.get("tool_registry_manifest"):
        tool_registry_manifest = _resolve_relative_path(path, payload.get("tool_registry_manifest"))
        if tool_registry_manifest and tool_registry_manifest.exists():
            kwargs["tool_registry_manifest"] = tool_registry_manifest
    if payload.get("dataset_capabilities_manifest"):
        dataset_capabilities_manifest = _resolve_relative_path(path, payload.get("dataset_capabilities_manifest"))
        if dataset_capabilities_manifest and dataset_capabilities_manifest.exists():
            kwargs["dataset_capabilities_manifest"] = dataset_capabilities_manifest
    return AutoResearchConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the background autoresearch loop for strategy discovery.")
    parser.add_argument("--daemon", action="store_true", help="Run continuously instead of a single batch round.")
    parser.add_argument("--rounds", type=int, default=None, help="Number of batch rounds to execute.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON or YAML config file.")
    parser.add_argument("--export", action="store_true", help="Export kept/promoted strategies to configs/strategies_extra/ after running.")
    parser.add_argument("--export-promoted-only", action="store_true", help="Export only formally promoted strategies.")
    parser.add_argument("--crystallize-context", type=str, default=None, help="Freeze the best elite strategy for the given context and write a versioned registry YAML.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    loop = AutoResearchLoop(config=config)
    loop.run(max_rounds=args.rounds, daemon=bool(args.daemon))
    crystallized_paths: list[Path] = []
    if args.crystallize_context:
        crystallized_paths = loop.crystallize(args.crystallize_context)
    if getattr(args, "export", False) or getattr(args, "export_promoted_only", False):
        min_status = "promoted" if getattr(args, "export_promoted_only", False) else "kept"
        loop.export_to_registry(min_status=min_status)
    payload = {"state": loop.state.to_dict(), "summary": str(config.summary_path)}
    if crystallized_paths:
        payload["crystallized_paths"] = [str(path) for path in crystallized_paths]
    print(json.dumps(make_jsonable(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
