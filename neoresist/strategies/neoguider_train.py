from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from neoresist.paths import repo_root
from neoresist.strategies.neoguider_transform import NeoGuiderFeatureTransformer

F_FEATURES = [
    "presentation_score_el",
    "binding_nm",
    "binding_stability",
    "expression_log2",
    "dai_agretopicity",
]

DATASET_PATHS = {
    "ott": Path("backend/strategy_engine/artifacts/ott_2017_features.csv"),
    "nci": Path("validation_papers/muller2023/muller_nci_features_v2.csv"),
    "sahin": Path("backend/strategy_engine/artifacts/sahin_2017_features.csv"),
}
MAX_NEGATIVE_MULTIPLIER = 10
MAX_ROWS_PER_DATASET = 5000


def _normalize_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    if "immunogenic" not in out.columns:
        raise ValueError(f"{dataset_name}: missing immunogenic column")
    out = out[out["immunogenic"].isin([0, 1])].copy()

    if "presentation_score_el" not in out.columns and "presentation_score" in out.columns:
        out["presentation_score_el"] = pd.to_numeric(out["presentation_score"], errors="coerce")
    if "binding_nm" not in out.columns:
        for alias in ("final_binding_affinity_nm", "binding_affinity_nm_numeric", "binding_affinity_nm"):
            if alias in out.columns:
                out["binding_nm"] = pd.to_numeric(out[alias], errors="coerce")
                break
    if "binding_stability" not in out.columns:
        out["binding_stability"] = np.nan
    if "expression_log2" not in out.columns:
        if "expression_tpm" in out.columns:
            expr = pd.to_numeric(out["expression_tpm"], errors="coerce").clip(lower=0.0)
            out["expression_log2"] = np.log2(1.0 + expr)
        else:
            out["expression_log2"] = np.nan
    if "dai_agretopicity" not in out.columns:
        if "dai" in out.columns:
            out["dai_agretopicity"] = pd.to_numeric(out["dai"], errors="coerce")
        else:
            out["dai_agretopicity"] = np.nan

    for feature in F_FEATURES:
        if feature not in out.columns:
            out[feature] = np.nan
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    out["dataset_name"] = dataset_name
    return out


def _downsample_for_training(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if len(df) <= MAX_ROWS_PER_DATASET:
        return df
    positives = df[df["immunogenic"] == 1].copy()
    negatives = df[df["immunogenic"] == 0].copy()
    keep_neg = min(len(negatives), max(MAX_ROWS_PER_DATASET - len(positives), len(positives) * MAX_NEGATIVE_MULTIPLIER))
    if keep_neg <= 0:
        sampled = positives
    else:
        sampled = pd.concat(
            [positives, negatives.sample(n=keep_neg, random_state=0)],
            ignore_index=True,
        )
    sampled = sampled.sample(frac=1.0, random_state=0).reset_index(drop=True)
    print(f"{dataset_name}: downsampled from {len(df)} to {len(sampled)} rows for KDE stability")
    return sampled


def load_training_rows(dataset_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    for dataset_name in dataset_names:
        path = DATASET_PATHS[dataset_name]
        full_path = repo_root() / path
        if not full_path.is_file():
            continue
        df = pd.read_csv(full_path, low_memory=False)
        normalized = _normalize_dataset(df, dataset_name)
        if normalized.empty:
            continue
        normalized = _downsample_for_training(normalized, dataset_name)
        frames.append(normalized)
        used.append(dataset_name)
    if not frames:
        return pd.DataFrame(columns=F_FEATURES + ["immunogenic"]), used
    return pd.concat(frames, ignore_index=True), used


def train_model(dataset_names: list[str], output: Path) -> dict:
    train_df, used = load_training_rows(dataset_names)
    if train_df.empty:
        raise SystemExit(
            "No labeled NeoGuider training datasets found. Expected one or more of: "
            + ", ".join(str(DATASET_PATHS[name]) for name in dataset_names)
        )
    X = train_df[F_FEATURES].to_numpy(dtype=float)
    y = train_df["immunogenic"].to_numpy(dtype=int)
    medians = {
        feature: float(np.nanmedian(X[:, idx])) if not np.isnan(X[:, idx]).all() else 0.0
        for idx, feature in enumerate(F_FEATURES)
    }
    for idx, feature in enumerate(F_FEATURES):
        X[:, idx] = np.where(np.isnan(X[:, idx]), medians[feature], X[:, idx])

    transformer = NeoGuiderFeatureTransformer(features=F_FEATURES)
    X_transformed = transformer.fit_transform(X, y)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    lr.fit(X_transformed, y)

    bundle = {
        "transformer": transformer,
        "lr": lr,
        "features": F_FEATURES,
        "feature_medians": medians,
        "trained_on": used,
        "n_positive": int(y.sum()),
        "n_total": int(len(y)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return bundle


def _print_summary(bundle: dict) -> None:
    n_total = int(bundle["n_total"])
    n_positive = int(bundle["n_positive"])
    rate = (n_positive / n_total) if n_total else 0.0
    print("Datasets used:", ", ".join(bundle["trained_on"]))
    print(f"n_total={n_total}")
    print(f"n_positive={n_positive}")
    print(f"positive_rate={rate:.4f}")
    transformer = bundle["transformer"]
    print("Feature transform ranges:")
    for feature in bundle["features"]:
        curve = transformer.cir_curves_.get(feature)
        if curve is None or len(curve) == 0:
            lo = hi = 0.0
        else:
            lo = float(np.min(curve))
            hi = float(np.max(curve))
        print(f"  {feature}: min={lo:.4f} max={hi:.4f}")
    print("LR coefficients:")
    coef = np.asarray(bundle["lr"].coef_).ravel()
    for feature, weight in zip(bundle["features"], coef, strict=False):
        print(f"  {feature}: {float(weight):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train NeoGuider transformer + logistic regression bundle.")
    parser.add_argument("--datasets", default="ott,nci,sahin", help="Comma-separated dataset ids to consider.")
    parser.add_argument(
        "--output",
        default="configs/strategies_extra/neoguider_v1_model.pkl",
        help="Output pickle path.",
    )
    args = parser.parse_args()
    dataset_names = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root() / output
    bundle = train_model(dataset_names, output)
    _print_summary(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
