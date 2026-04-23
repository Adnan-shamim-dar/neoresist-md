"""kind=ml_model handler.

Runs a fitted sklearn-compatible classifier (XGBoost, LogisticRegression, ...)
stored as a joblib .pkl. Resolves feature columns from `params.features` with
an optional `params.feature_map` renaming bridge for strategies whose feature
names differ from the candidate DataFrame's column names (e.g. NeoGuider's
'ScoreEL' -> our 'presentation_score_el').

Strategy YAML shape (under configs/strategies_extra/):

    strategy_id: presentation_model_v1_ui
    display_name: "Stage 1 Presentation (NCI, XGBoost)"
    kind: ml_model
    params:
      model_path: backend/strategy_engine/artifacts/stage1_presentation_model.pkl
      features: [binding_nm, binding_log, binding_sigmoid, binding_stability,
                 presentation_score_el, expression_log2, pep_length]
      # Optional rename map: the keys on the left are the feature names the
      # model was trained on, values are candidate-DataFrame columns to pull.
      feature_map: {}
      # Optional: clip model output into [0, 1] if it's not already a probability.
      normalize: clip01
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from neoresist.paths import repo_root

if TYPE_CHECKING:
    from neoresist.strategy_registry import ResolvedStrategy


def score_ml_model(df: pd.DataFrame, strategy: "ResolvedStrategy") -> pd.DataFrame:
    params: dict[str, Any] = dict(strategy.params or {})
    features: list[str] = [str(f) for f in (params.get("features") or [])]
    if not features:
        return _empty_with_zero_priority(df)

    feature_map: dict[str, str] = dict(params.get("feature_map") or {})
    model_path_raw = str(params.get("model_path") or "")
    if not model_path_raw:
        return _empty_with_zero_priority(df)
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = repo_root() / model_path
    if not model_path.is_file():
        return _empty_with_zero_priority(df)

    try:
        import joblib  # type: ignore
    except Exception:
        return _empty_with_zero_priority(df)
    try:
        model = joblib.load(model_path)
    except Exception:
        return _empty_with_zero_priority(df)

    X = _build_feature_matrix(df, features, feature_map)
    if X is None or len(X) == 0:
        return _empty_with_zero_priority(df)

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Binary classifier: take probability of positive class.
            if proba.ndim == 2 and proba.shape[1] >= 2:
                scores = np.asarray(proba[:, 1], dtype=float)
            else:
                scores = np.asarray(proba, dtype=float).ravel()
        else:
            scores = np.asarray(model.predict(X), dtype=float)
    except Exception:
        return _empty_with_zero_priority(df)

    if str(params.get("normalize") or "clip01") == "clip01":
        scores = np.clip(scores, 0.0, 1.0)

    out = df.copy()
    out["rl_priority"] = scores
    return out


def _build_feature_matrix(
    df: pd.DataFrame, features: list[str], feature_map: dict[str, str]
) -> np.ndarray | None:
    cols: list[np.ndarray] = []
    for feat in features:
        source = feature_map.get(feat, feat)
        if source in df.columns:
            series = pd.to_numeric(df[source], errors="coerce").fillna(0.0)
        else:
            series = pd.Series([0.0] * len(df), index=df.index)
        cols.append(series.to_numpy(dtype=float))
    if not cols:
        return None
    return np.column_stack(cols)


def _empty_with_zero_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rl_priority"] = 0.0
    return out
