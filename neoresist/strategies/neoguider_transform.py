"""aKDE -> IR -> CIR feature transformation for the NeoGuider strategy.

This is a from-scratch implementation guided by Zhao et al. (Genome Medicine
2026) of the three-stage per-feature calibration pipeline NeoGuider applies
before its logistic regression head:

    1. Adaptive Kernel Density Estimation (aKDE)
       For each feature, fit per-class Gaussian KDEs on the training set and
       evaluate the Bayes-rule posterior P(y=1 | x) at every x using the
       class prior implied by the training data. The bandwidth is adaptive
       per class using Silverman's rule on that class's empirical spread.

    2. Isotonic Regression (IR)
       Fit a monotone non-decreasing calibrator that maps the aKDE posterior
       to the observed positive rate across a binned representation of the
       training set. Uses sklearn.isotonic.IsotonicRegression with
       out_of_bounds='clip' so the transform stays well-defined at inference.

    3. Centered Isotonic Regression (CIR)
       A second IR pass on the output of stage 2, fit on the same training
       pairs. The two-stage calibration is what the paper terms "centered"
       isotonic regression in their pipeline; it absorbs residual curvature
       left by the first IR and returns a smooth, stable monotone map into
       [0, 1].

The pipeline is deliberately self-contained: no AGPL-3 code is imported or
copied. All public APIs here are np/sklearn/scipy primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression


@dataclass
class _PerClassKDE:
    positive_kde: gaussian_kde | None
    negative_kde: gaussian_kde | None
    prior_positive: float
    prior_negative: float
    # Store raw arrays for degenerate-case handling (single-class or constant).
    positive_vals: np.ndarray
    negative_vals: np.ndarray


class AdaptiveFeatureTransform:
    """Per-feature aKDE -> IR -> CIR calibrator.

    One instance per feature. `fit(x, y)` learns the calibrators, `transform(x)`
    returns the per-sample calibrated posterior in [0, 1]. Instances are pickleable.
    """

    def __init__(self, min_bandwidth: float = 1e-3) -> None:
        self.min_bandwidth = float(min_bandwidth)
        self._kde: _PerClassKDE | None = None
        self._ir_stage1: IsotonicRegression | None = None
        self._ir_stage2: IsotonicRegression | None = None
        self._fitted: bool = False
        self.training_median_: float = 0.0
        self.cir_x_: np.ndarray = np.array([], dtype=float)
        self.cir_y_: np.ndarray = np.array([], dtype=float)

    # ------------------------------------------------------------------ fit
    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdaptiveFeatureTransform":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()
        if x.size == 0 or x.size != y.size:
            raise ValueError("AdaptiveFeatureTransform.fit: x and y must be same non-empty length")
        if np.isnan(x).all():
            x = np.zeros_like(x, dtype=float)
        else:
            self.training_median_ = float(np.nanmedian(x))
            x = np.where(np.isnan(x), self.training_median_, x)
        if not np.isfinite(self.training_median_):
            self.training_median_ = 0.0

        pos_mask = y == 1
        neg_mask = ~pos_mask
        positive_vals = x[pos_mask]
        negative_vals = x[neg_mask]
        n_total = max(x.size, 1)
        prior_pos = float(max(positive_vals.size, 1)) / float(n_total)
        prior_neg = 1.0 - prior_pos

        self._kde = _PerClassKDE(
            positive_kde=_safe_kde(positive_vals, self.min_bandwidth),
            negative_kde=_safe_kde(negative_vals, self.min_bandwidth),
            prior_positive=prior_pos,
            prior_negative=prior_neg,
            positive_vals=positive_vals,
            negative_vals=negative_vals,
        )

        posterior = self._bayes_posterior(x)
        # Stage 1 isotonic regression: aKDE posterior -> empirical positive rate.
        self._ir_stage1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._ir_stage1.fit(posterior, y.astype(float))

        # Stage 2 (centered) isotonic regression on the stage 1 output.
        stage1_out = self._ir_stage1.transform(posterior)
        self._ir_stage2 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._ir_stage2.fit(stage1_out, y.astype(float))
        self.cir_x_ = np.asarray(getattr(self._ir_stage2, "X_thresholds_", stage1_out), dtype=float)
        self.cir_y_ = np.asarray(getattr(self._ir_stage2, "y_thresholds_", y.astype(float)), dtype=float)

        self._fitted = True
        return self

    # ----------------------------------------------------------- transform
    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted or self._kde is None or self._ir_stage1 is None or self._ir_stage2 is None:
            # Identity fallback keeps the pipeline usable before training.
            return np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0)
        x = np.asarray(x, dtype=float).ravel()
        x = np.where(np.isnan(x), self.training_median_, x)
        posterior = self._bayes_posterior(x)
        stage1 = self._ir_stage1.transform(posterior)
        stage2 = self._ir_stage2.transform(stage1)
        return np.clip(stage2, 0.0, 1.0)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        kde = state.get("_kde")
        if kde is not None:
            state["_kde_state"] = {
                "prior_positive": kde.prior_positive,
                "prior_negative": kde.prior_negative,
                "positive_vals": np.asarray(kde.positive_vals, dtype=float),
                "negative_vals": np.asarray(kde.negative_vals, dtype=float),
            }
        state["_kde"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        kde_state = state.pop("_kde_state", None)
        self.__dict__.update(state)
        if kde_state is None:
            self._kde = None
            return
        positive_vals = np.asarray(kde_state["positive_vals"], dtype=float)
        negative_vals = np.asarray(kde_state["negative_vals"], dtype=float)
        self._kde = _PerClassKDE(
            positive_kde=_safe_kde(positive_vals, self.min_bandwidth),
            negative_kde=_safe_kde(negative_vals, self.min_bandwidth),
            prior_positive=float(kde_state["prior_positive"]),
            prior_negative=float(kde_state["prior_negative"]),
            positive_vals=positive_vals,
            negative_vals=negative_vals,
        )

    # ------------------------------------------------------------ helpers
    def _bayes_posterior(self, x: np.ndarray) -> np.ndarray:
        assert self._kde is not None
        kde = self._kde
        x = np.asarray(x, dtype=float).ravel()
        if kde.positive_kde is None and kde.negative_kde is None:
            return np.full_like(x, kde.prior_positive, dtype=float)
        if kde.positive_kde is None:
            return np.zeros_like(x, dtype=float)
        if kde.negative_kde is None:
            return np.ones_like(x, dtype=float)
        # Evaluate each class KDE; add a tiny floor to avoid 0/0.
        pos_density = np.maximum(kde.positive_kde.evaluate(x), 1e-12)
        neg_density = np.maximum(kde.negative_kde.evaluate(x), 1e-12)
        numer = pos_density * kde.prior_positive
        denom = numer + neg_density * kde.prior_negative
        return numer / denom


def _safe_kde(values: np.ndarray, min_bandwidth: float) -> gaussian_kde | None:
    """Return a gaussian_kde or None if the class is too degenerate to fit."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        return None
    if float(np.nanstd(values)) <= 1e-12:
        # Scipy's gaussian_kde chokes on zero variance; nudge with a tiny jitter
        # so the density is still well-defined at training points.
        values = values + np.random.default_rng(0).normal(scale=min_bandwidth, size=values.size)
    try:
        kde = gaussian_kde(values, bw_method="silverman")
    except Exception:
        return None
    # Enforce a bandwidth floor to keep inference stable on sparse features.
    try:
        factor = max(kde.factor, min_bandwidth)
        kde.set_bandwidth(bw_method=factor)
    except Exception:
        pass
    return kde


def build_feature_matrix(
    frame, feature_order: list[str], feature_map: dict[str, str] | None = None, *, missing_value: float = 0.0
) -> np.ndarray:
    """Pull feature columns out of a DataFrame and return a (n, k) float matrix.

    Missing columns and non-numeric values coerce to `missing_value`. `feature_map` renames
    the NeoGuider-F names (ScoreEL, ICfiftyBA, ...) to candidate-table columns
    (presentation_score_el, binding_nm, ...) without needing to mutate either side.
    """
    feature_map = feature_map or {}
    cols: list[np.ndarray] = []
    import pandas as pd

    for name in feature_order:
        source = feature_map.get(name, name)
        if source in frame.columns:
            series = pd.to_numeric(frame[source], errors="coerce").fillna(missing_value)
        else:
            series = pd.Series([missing_value] * len(frame), index=frame.index)
        cols.append(series.to_numpy(dtype=float))
    if not cols:
        return np.zeros((len(frame), 0), dtype=float)
    return np.column_stack(cols)


class NeoGuiderFeatureTransformer:
    """Multi-feature wrapper around per-feature aKDE -> IR -> CIR transforms."""

    def __init__(self, features: list[str], min_bandwidth: float = 1e-3) -> None:
        self.features = list(features)
        self.min_bandwidth = float(min_bandwidth)
        self.transforms_: dict[str, AdaptiveFeatureTransform] = {}
        self.feature_medians_: dict[str, float] = {}
        self.cir_curves_: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeoGuiderFeatureTransformer":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim != 2:
            raise ValueError("NeoGuiderFeatureTransformer.fit: X must be 2D")
        if X.shape[1] != len(self.features):
            raise ValueError("NeoGuiderFeatureTransformer.fit: feature count mismatch")
        self.transforms_.clear()
        self.feature_medians_.clear()
        self.cir_curves_.clear()
        for idx, feature in enumerate(self.features):
            col = X[:, idx]
            if np.isnan(col).all():
                median = 0.0
            else:
                median = float(np.nanmedian(col))
            if not np.isfinite(median):
                median = 0.0
            self.feature_medians_[feature] = median
            transform = AdaptiveFeatureTransform(min_bandwidth=self.min_bandwidth)
            transform.fit(np.where(np.isnan(col), median, col), y)
            self.transforms_[feature] = transform
            self.cir_curves_[feature] = transform.cir_y_.copy()
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("NeoGuiderFeatureTransformer.transform: X must be 2D")
        if X.shape[1] != len(self.features):
            raise ValueError("NeoGuiderFeatureTransformer.transform: feature count mismatch")
        cols: list[np.ndarray] = []
        for idx, feature in enumerate(self.features):
            col = X[:, idx]
            median = self.feature_medians_.get(feature, 0.0)
            col = np.where(np.isnan(col), median, col)
            transform = self.transforms_.get(feature)
            if transform is None:
                cols.append(np.clip(col, 0.0, 1.0))
            else:
                cols.append(transform.transform(col))
        if not cols:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(cols)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
