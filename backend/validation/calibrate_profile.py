from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.validation.validation_harness import _build_labels, _load_dataset, _normalise_maf
from neoresist_md.backend.core.module_runner import ModuleRunner

FEATURE_COLUMNS = [
    "expression_score",
    "presentation_score",
    "ccf_score",
    "self_dissimilarity_score",
    "escape_penalty",
]


def _expression_score(values: pd.Series, cap: float = 1000.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    return (np.log1p(numeric) / math.log1p(max(cap, 1.0))).clip(0.0, 1.0)


def _presentation_score(df: pd.DataFrame) -> pd.Series:
    if "presentation_score" in df.columns:
        return pd.to_numeric(df["presentation_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    rank = pd.to_numeric(df.get("binding_rank"), errors="coerce").fillna(100.0)
    return (1.0 - (rank / 100.0)).clip(0.0, 1.0)


def build_training_matrix(dataset_name: str) -> tuple[pd.DataFrame, str]:
    dataset = _load_dataset(dataset_name)
    runner = ModuleRunner(config_path=str(ROOT / "neoresist_md" / "config"))
    positives = {x.strip().upper() for x in dataset.immunogenic_neoantigens}
    negatives = {x.strip().upper() for x in dataset.non_immunogenic_neoantigens}

    frames: list[pd.DataFrame] = []
    for patient_id, maf_path in zip(dataset.patient_ids, dataset.maf_paths):
        maf_df = _normalise_maf(maf_path, patient_id)
        out = runner.run_pipeline(
            maf_df,
            enabled_modules=["generation", "expression", "clonality", "recognition", "resistance", "tiering"],
        ).copy()
        out["patient_id"] = str(patient_id)
        frames.append(out)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        raise RuntimeError("Validation produced no candidate rows for calibration.")

    peptides = merged.get("mutant_peptide", pd.Series([""] * len(merged))).astype(str).str.upper()
    labels = _build_labels(peptides, positives, negatives)
    matrix = pd.DataFrame(
        {
            "dataset": dataset.dataset_name,
            "patient_id": merged.get("patient_id", ""),
            "gene": merged.get("gene", ""),
            "mutant_peptide": peptides,
            "expression_score": _expression_score(merged.get("expression_tpm", pd.Series([0.0] * len(merged)))),
            "presentation_score": _presentation_score(merged),
            "ccf_score": pd.to_numeric(merged.get("ccf"), errors="coerce").fillna(0.0).clip(0.0, 1.0),
            "self_dissimilarity_score": pd.to_numeric(merged.get("self_dissimilarity"), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0),
            "escape_penalty": pd.to_numeric(merged.get("resistance_composite"), errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0),
            "immunogenicity_label": labels,
        }
    )
    return matrix, dataset.dataset_name


def _fit_weights(matrix: pd.DataFrame) -> tuple[dict[str, float], float, float]:
    valid = matrix["immunogenicity_label"].notna()
    X = matrix.loc[valid, FEATURE_COLUMNS]
    y = matrix.loc[valid, "immunogenicity_label"].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Calibration labels collapsed to one class.")

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    coefs = dict(zip(FEATURE_COLUMNS, model.coef_[0]))
    positive = {
        "expression_norm": max(0.0, float(coefs["expression_score"])),
        "presentation": max(0.0, float(coefs["presentation_score"])),
        "ccf": max(0.0, float(coefs["ccf_score"])),
        "self_dissimilarity": max(0.0, float(coefs["self_dissimilarity_score"])),
    }
    pos_sum = sum(positive.values())
    if pos_sum <= 1e-12:
        positive = {k: 0.25 for k in positive}
    else:
        positive = {k: v / pos_sum for k, v in positive.items()}

    escape_coef = float(coefs["escape_penalty"])
    resistance_weight = abs(escape_coef) if escape_coef < 0.0 else 0.0
    return {**positive, "escape_penalty_weight": -resistance_weight}, auc, escape_coef


def _profile_yaml(weights: dict[str, float], dataset_name: str, auc: float, escape_coef: float) -> str:
    escape_note = (
        "Escape coefficient was non-negative in this training matrix; calibrated penalty is set to 0.0."
        if escape_coef >= 0.0
        else "Escape penalty weight uses the absolute negative logistic coefficient."
    )
    return f"""# rl_calibrated — weights derived from logistic regression on published trial validation data
# Training data: {dataset_name}
# Date: {date.today().isoformat()}
# AUC-ROC on training data: {auc:.4f}
profile_id: rl_calibrated
display_name: RL calibrated
version: "1.0"
description: >
  Logistic-regression calibrated ResistanceLoop profile derived from the validation
  training matrix in backend/validation/artifacts/training_matrix.csv. {escape_note}
higher_is_better: true
primary_score_column: rl_priority

expression_tpm_cap: 1000.0
blend:
  real_expression_weight: 0.85
  real_ccf_weight: 0.85

weights:
  expression_norm: {weights["expression_norm"]:.6f}
  presentation: {weights["presentation"]:.6f}
  ccf: {weights["ccf"]:.6f}
  self_dissimilarity: {weights["self_dissimilarity"]:.6f}
escape_penalty_weight: {weights["escape_penalty_weight"]:.6f}
"""


def calibrate(dataset_name: str, artifacts_dir: Path, profile_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    matrix, source_name = build_training_matrix(dataset_name)
    matrix_path = artifacts_dir / "training_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    weights, auc, escape_coef = _fit_weights(matrix)
    profile_path = profile_dir / "rl_calibrated.yaml"
    profile_path.write_text(_profile_yaml(weights, source_name, auc, escape_coef), encoding="utf-8")
    print(f"Training matrix: {matrix_path}")
    print(f"Calibrated profile: {profile_path}")
    print(f"AUC-ROC: {auc:.4f}")
    return profile_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build rl_calibrated from validation artifacts.")
    parser.add_argument("--dataset", default="ott2017")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "backend" / "validation" / "artifacts")
    parser.add_argument("--profiles", type=Path, default=ROOT / "configs" / "profiles")
    args = parser.parse_args(argv)
    calibrate(args.dataset, args.artifacts, args.profiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
