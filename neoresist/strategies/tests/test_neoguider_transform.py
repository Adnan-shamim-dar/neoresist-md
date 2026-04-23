from __future__ import annotations

import numpy as np

from neoresist.strategies.neoguider_transform import NeoGuiderFeatureTransformer


def test_fit_transform_basic():
    X = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    y = np.array([0, 0, 1, 0, 1])
    transformer = NeoGuiderFeatureTransformer(features=["f1", "f2"])
    Xt = transformer.fit_transform(X, y)
    assert Xt.shape == X.shape, "Output shape must match input"
    assert np.isfinite(Xt).all(), "No NaN or Inf in output"


def test_monotonicity():
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.6, 0.1], [0.8, 0.3], [0.9, 0.5]])
    y = np.array([0, 0, 0, 1, 1])
    transformer = NeoGuiderFeatureTransformer(features=["f1", "f2"])
    transformer.fit(X, y)
    curve = transformer.cir_curves_["f1"]
    assert np.all(np.diff(curve) >= -1e-9), "CIR curve must be monotone non-decreasing"


def test_single_positive():
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = np.zeros(20)
    y[5] = 1
    transformer = NeoGuiderFeatureTransformer(features=["a", "b"])
    Xt = transformer.fit_transform(X, y)
    assert Xt.shape == X.shape


def test_nan_handling():
    X = np.array([[np.nan, 0.5], [0.4, 0.2], [0.8, 0.1], [np.nan, 0.9]])
    y = np.array([0, 0, 1, 1])
    transformer = NeoGuiderFeatureTransformer(features=["a", "b"])
    Xt = transformer.fit_transform(X, y)
    assert np.isfinite(Xt).all()

