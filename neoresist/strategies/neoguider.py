"""kind=neoguider handler.

Scores candidates using the NeoGuider pipeline (Zhao et al. Genome Medicine 2026):
applies a per-feature aKDE -> IR -> CIR transformation to F = {ScoreEL, ICfiftyBA,
BindStab, NeoAbundance, Agretop}, then passes the calibrated feature matrix
through a packaged logistic regression head to produce a probability in [0, 1].

Feature name bridge (NeoGuider F -> candidate DataFrame columns) is declared in
the strategy YAML's `params.feature_map`. Default mapping:

    ScoreEL      -> presentation_score_el
    ICfiftyBA    -> binding_nm
    BindStab     -> binding_stability
    NeoAbundance -> expression_log2
    Agretop      -> dai_agretopicity

Training lives in a separate CLI step (neoresist/strategies/neoguider_train.py,
written in a later commit) that fits the transforms + LR on Ott+Sahin+NCI and
serializes the whole pipeline to a joblib .pkl at params.model_path.

Cold-start behavior: if the model artifact is missing, we return rl_priority=0.0
for every candidate. The strategy still appears in the dropdown (so the UI
shows the placeholder), but users get a clear zero until the pipeline is trained.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from neoresist.paths import repo_root
from neoresist.strategies.neoguider_transform import build_feature_matrix

if TYPE_CHECKING:
    from neoresist.strategy_registry import ResolvedStrategy

DEFAULT_FEATURE_ORDER = ["ScoreEL", "ICfiftyBA", "BindStab", "NeoAbundance", "Agretop"]
DEFAULT_FEATURE_MAP = {
    "ScoreEL": "presentation_score_el",
    "ICfiftyBA": "binding_nm",
    "BindStab": "binding_stability",
    "NeoAbundance": "expression_log2",
    "Agretop": "dai_agretopicity",
}
LOGGER = logging.getLogger(__name__)
_MODEL_CACHE: dict[str, dict[str, Any] | None] = {}

def score_neoguider(df: pd.DataFrame, strategy: "ResolvedStrategy") -> pd.DataFrame:
    params: dict[str, Any] = dict(strategy.params or {})
    feature_order = list(params.get("features") or DEFAULT_FEATURE_ORDER)
    feature_map = dict(params.get("feature_map") or DEFAULT_FEATURE_MAP)
    model_path_raw = str(params.get("model_pkl") or params.get("model_path") or "")

    out = df.copy()
    bundle = _load_bundle(model_path_raw)
    if not bundle:
        out["rl_priority"] = _binding_fallback_scores(out)
        out = _apply_frameshift_overrides(out)
        return out

    X_raw = build_feature_matrix(df, feature_order, feature_map)
    if X_raw.size == 0:
        out["rl_priority"] = _binding_fallback_scores(out)
        out = _apply_frameshift_overrides(out)
        return out
    medians = dict(bundle.get("feature_medians") or {})
    for idx, feat in enumerate(bundle.get("features") or feature_order):
        median = float(medians.get(feat, 0.0))
        X_raw[:, idx] = np.where(np.isnan(X_raw[:, idx]), median, X_raw[:, idx])

    transformer = bundle.get("transformer")
    classifier = bundle.get("lr")
    if transformer is None or classifier is None:
        out["rl_priority"] = _binding_fallback_scores(out)
        out = _apply_frameshift_overrides(out)
        return out
    X_calibrated = transformer.transform(X_raw)

    try:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(X_calibrated)
            scores = np.asarray(proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba).ravel()
        else:
            scores = np.asarray(classifier.predict(X_calibrated), dtype=float).ravel()
    except Exception:
        scores = _binding_fallback_scores(out)

    out["rl_priority"] = np.clip(scores, 0.0, 1.0)
    out = _apply_frameshift_overrides(out)
    return out


def _load_bundle(model_path_raw: str) -> dict[str, Any] | None:
    if not model_path_raw:
        LOGGER.warning("NeoGuider model not trained. Run neoguider_train.py. Using binding fallback.")
        return None
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = repo_root() / model_path
    cache_key = str(model_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        import joblib  # type: ignore
    except Exception:
        LOGGER.warning("NeoGuider model not trained. Run neoguider_train.py. Using binding fallback.")
        _MODEL_CACHE[cache_key] = None
        return None
    if not model_path.is_file():
        LOGGER.warning("NeoGuider model not trained. Run neoguider_train.py. Using binding fallback.")
        _MODEL_CACHE[cache_key] = None
        return None
    try:
        bundle = joblib.load(model_path)
    except Exception:
        LOGGER.warning("NeoGuider model not trained. Run neoguider_train.py. Using binding fallback.")
        bundle = None
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _binding_fallback_scores(df: pd.DataFrame) -> np.ndarray:
    if "binding_nm" not in df.columns:
        return np.zeros(len(df), dtype=float)
    binding = pd.to_numeric(df["binding_nm"], errors="coerce")
    if binding.notna().sum() == 0:
        return np.zeros(len(df), dtype=float)
    ranked = binding.rank(method="average", ascending=True, na_option="bottom")
    max_rank = float(ranked.max()) if len(ranked) else 1.0
    scores = 1.0 - ((ranked - 1.0) / max(max_rank - 1.0, 1.0))
    scores = scores.fillna(0.0)
    return scores.to_numpy(dtype=float)


def _apply_frameshift_overrides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_frameshift" not in out.columns:
        return out
    mask = out["is_frameshift"].fillna(False).astype(bool)
    if not mask.any():
        return out
    out.loc[mask, "presentation_score_el"] = np.nan
    out.loc[mask, "binding_nm"] = np.nan
    out.loc[mask, "forced_tcr_scorer"] = True
    out.loc[mask, "rl_priority"] = _frameshift_tcr_scores(out.loc[mask])
    return out


def _frameshift_tcr_scores(df: pd.DataFrame) -> np.ndarray:
    features = ["tcr_hydro_mean", "tcr_hydro_max", "blosum62_score", "self_dissimilarity"]
    cols: list[np.ndarray] = []
    for feature in features:
        if feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        else:
            values = np.full(len(df), np.nan, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            cols.append(np.zeros(len(df), dtype=float))
            continue
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi - lo <= 1e-12:
            norm = np.ones(len(df), dtype=float) * 0.5
        else:
            norm = (values - lo) / (hi - lo)
        norm = np.where(np.isfinite(norm), np.clip(norm, 0.0, 1.0), 0.0)
        cols.append(norm)
    if not cols:
        return np.zeros(len(df), dtype=float)
    return np.mean(np.column_stack(cols), axis=1)
