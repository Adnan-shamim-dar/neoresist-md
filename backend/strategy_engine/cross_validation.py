"""
PHASE 4b — Specific cross-dataset pairs with bootstrap CI.

Pairs:
  (a) Train XGBoost on ott_2017  → predict sahin_2017
  (b) Train XGBoost on sahin_2017 → predict ott_2017
  (c) Train XGBoost on hilf_2019  → predict keskin_2019

Run: py -3.11 backend/strategy_engine/cross_validation.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")

try:
    from xgboost import XGBClassifier
    def _model(spw):
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=42, verbosity=0)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def _model(spw):
        return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.1, random_state=42)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / f"{name}_features.csv", low_memory=False)
    return df[df["immunogenic"].isin([0, 1])].copy()


def prep(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = df[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
    for i in range(X.shape[1]):
        med = np.nanmedian(X[:, i])
        X[np.isnan(X[:, i]), i] = med if np.isfinite(med) else 0.0
    return X


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    aucs = [roc_auc_score(y_true[idx := rng.choice(len(y_true), len(y_true), replace=True)],
                          y_pred[idx])
            for _ in range(n)
            if len(np.unique(y_true[rng.choice(len(y_true), len(y_true), replace=True)])) > 1]
    # recompute cleanly
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, yp))
    if len(aucs) < 10:
        return [None, None]
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


def run_pair(train_name: str, test_name: str) -> dict:
    train_df, test_df = load(train_name), load(test_name)
    y_tr = train_df["immunogenic"].values.astype(int)
    y_te = test_df["immunogenic"].values.astype(int)

    # Features: intersection of non-sparse columns in both datasets
    # (exclude presentation_probability — added at runtime, not in saved CSVs)
    candidates = [c for c in train_df.columns
                  if c not in ("immunogenic", "patient_id", "mutant_peptide",
                               "hla_allele", "presentation_probability")
                  and pd.api.types.is_numeric_dtype(train_df[c])
                  and pd.to_numeric(train_df[c], errors="coerce").notna().mean() > 0.3
                  and c in test_df.columns
                  and pd.to_numeric(test_df[c], errors="coerce").notna().mean() > 0.3]

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    X_tr, X_te = prep(train_df, candidates), prep(test_df, candidates)

    m = _model(spw)
    m.fit(X_tr, y_tr)
    y_pred = m.predict_proba(X_te)[:, 1]

    auc = float(roc_auc_score(y_te, y_pred)) if len(np.unique(y_te)) > 1 else None
    ci = bootstrap_ci(y_te, y_pred)

    print(f"  ({train_name} → {test_name}): "
          f"n_train={len(train_df)} n_test={len(test_df)} pos_test={y_te.sum()} "
          f"AUC={auc:.4f}  CI={ci}")
    return {
        "train": train_name, "test": test_name,
        "n_train": len(train_df), "n_test": len(test_df),
        "n_test_positive": int(y_te.sum()),
        "auc": round(auc, 4) if auc else None,
        "ci_95": ci,
        "n_features": len(candidates),
    }


PAIRS = [
    ("ott_2017",   "sahin_2017"),   # (a)
    ("sahin_2017", "ott_2017"),     # (b)
    ("hilf_2019",  "keskin_2019"),  # (c)
]

# rl_tcr_v1 on sahin_2017 WITHOUT any sahin training — from AGENTS.md validated results
# Discovered on Ott (zero-shot transfer to Sahin)
RL_TCR_V1_SAHIN_ZERO_SHOT = 0.6599


def main() -> int:
    print("=" * 60)
    print("PHASE 4b — CROSS-DATASET PAIRS WITH BOOTSTRAP CI")
    print("=" * 60)

    results = [run_pair(tr, te) for tr, te in PAIRS]

    print(f"\nrl_tcr_v1 on sahin_2017 (zero-shot, trained on Ott only): "
          f"AUC={RL_TCR_V1_SAHIN_ZERO_SHOT}")
    print("  Source: AGENTS.md validated results (rerun_sahin_with_binding.py)")

    out = {
        "pairs": results,
        "rl_tcr_v1_sahin_zero_shot": {
            "auc": RL_TCR_V1_SAHIN_ZERO_SHOT,
            "note": "rl_tcr_v1 discovered on Ott, applied to Sahin without any Sahin training",
            "source": "artifacts/sahin_cross_validation_with_binding.json",
        },
    }
    path = ARTIFACTS / "cross_validation_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")

    print("\n--- SUMMARY ---")
    for r in results:
        print(f"  ({r['train']:12s} → {r['test']:12s})  "
              f"AUC={r['auc']}  CI={r['ci_95']}")
    print(f"  rl_tcr_v1 Sahin zero-shot:              AUC={RL_TCR_V1_SAHIN_ZERO_SHOT}  "
          f"(no CI — from prior LOPO run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
