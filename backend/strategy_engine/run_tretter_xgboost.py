"""
XGBoost on Tretter 2023 features — LOPO CV, RFECV, Phase comparison.
Usage: py -3.11 backend/strategy_engine/run_tretter_xgboost.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

try:
    import xgboost as xgb
except ImportError:
    print("pip install xgboost"); sys.exit(1)

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

MODEL_OUT   = ARTIFACTS / "tretter_combination_model.pkl"
RESULTS_OUT = ARTIFACTS / "tretter_combination_results.json"

XGB_PARAMS = dict(
    max_depth=2,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    scale_pos_weight=2.9,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("backend/strategy_engine/artifacts/tretter_2023_features.csv")
print(f"Tretter 2023: {len(df)} rows, {int(df['immunogenic'].sum())} positives, "
      f"{df['patient_id'].nunique()} patients")

EXCL = {"immunogenic", "paper_source", "patient_id", "cancer_type",
        "mutant_peptide", "wt_peptide", "hla_allele", "clonality_class"}

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in num_cols
            if c not in EXCL and df[c].notna().mean() > 0.50]
print(f"Usable features ({len(features)}): {features}")

# hydro_full_mean coverage
hfm_cov = df["hydro_full_mean"].notna().mean() if "hydro_full_mean" in df.columns else None
print(f"\nhydro_full_mean coverage: {hfm_cov:.1%}" if hfm_cov is not None else "\nhydro_full_mean: ABSENT")

y        = df["immunogenic"].values.astype(int)
X        = df[features].copy()
patients = df["patient_id"].values

# ── LOPO CV (folds with >=1 positive only) ────────────────────────────────────
print("\n[1] LOPO cross-validation ...")
unique_patients = sorted(df["patient_id"].unique())
fold_aucs: list[float] = []
oof_scores = np.full(len(df), np.nan)
valid_folds: list[str] = []

for pat in unique_patients:
    test_idx  = np.where(patients == pat)[0]
    train_idx = np.where(patients != pat)[0]
    y_test = y[test_idx]
    if y_test.sum() == 0:
        continue  # no positives — skip

    y_train = y[train_idx]
    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()

    fold_med = X_train.median()
    X_train  = X_train.fillna(fold_med)
    X_test   = X_test.fillna(fold_med)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    probs = model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) < 2:
        print(f"  {pat}: skipped (all-positive test fold)")
        continue

    auc = roc_auc_score(y_test, probs)
    fold_aucs.append(auc)
    oof_scores[test_idx] = probs
    valid_folds.append(pat)
    print(f"  {pat}: n={len(test_idx)}, pos={int(y_test.sum())}, AUC={auc:.3f}")

mean_auc = float(np.mean(fold_aucs))
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(fold_aucs, size=len(fold_aucs), replace=True))
              for _ in range(1000)]
ci_lo = float(np.percentile(boot_means, 2.5))
ci_hi = float(np.percentile(boot_means, 97.5))
print(f"\nMean LOPO AUC: {mean_auc:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")


# ── Baseline single-feature LOPO AUCs ────────────────────────────────────────
print("\n[2] Baseline LOPO AUCs ...")
baselines = ["binding_nm", "self_dissimilarity", "calis_simplified"]
baseline_results: dict[str, dict] = {}

for feat in baselines:
    if feat not in df.columns:
        print(f"  {feat}: ABSENT"); continue
    b_aucs = []
    for pat in valid_folds:
        test_idx  = np.where(patients == pat)[0]
        train_idx = np.where(patients != pat)[0]
        y_test    = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        col = pd.to_numeric(df[feat], errors="coerce")
        # impute with training median
        train_med = float(col.iloc[train_idx].median())
        scores = col.iloc[test_idx].fillna(train_med).values
        # binding_nm: lower = better binder, invert
        if feat == "binding_nm":
            scores = -scores
        b_aucs.append(roc_auc_score(y_test, scores))
    b_mean = float(np.mean(b_aucs))
    b_boots = [np.mean(rng.choice(b_aucs, size=len(b_aucs), replace=True)) for _ in range(1000)]
    b_lo = float(np.percentile(b_boots, 2.5))
    b_hi = float(np.percentile(b_boots, 97.5))
    delta = mean_auc - b_mean
    overlaps = not (ci_lo > b_hi or b_lo > ci_hi)
    baseline_results[feat] = {
        "mean_auc": round(b_mean, 3),
        "ci_95": [round(b_lo, 3), round(b_hi, 3)],
        "delta_vs_xgb": round(delta, 3),
        "ci_overlap": overlaps,
    }
    print(f"  {feat:<25} AUC={b_mean:.3f} CI=[{b_lo:.3f},{b_hi:.3f}]  "
          f"delta={delta:+.3f}  CI overlap: {overlaps}")


# ── Recall@k (OOF) ───────────────────────────────────────────────────────────
print("\n[3] Recall@k on Tretter cohort (OOF) ...")
valid_mask = ~np.isnan(oof_scores)
recall_at: dict[int, float] = {}
for k in (5, 10):
    hits, total_pos = 0, 0
    for pat in valid_folds:
        idx = np.where((patients == pat) & valid_mask)[0]
        if len(idx) == 0: continue
        order = np.argsort(oof_scores[idx])[::-1][:k]
        hits      += int(y[idx][order].sum())
        total_pos += int(y[idx].sum())
    recall_at[k] = hits / total_pos if total_pos else 0.0
    print(f"  Recall@{k} = {hits}/{total_pos} = {recall_at[k]:.3f}")


# ── Final model + feature importances ─────────────────────────────────────────
print("\n[4] Final model (all 77 rows) ...")
train_med = X.median()
X_all     = X.fillna(train_med)
final_model = xgb.XGBClassifier(**XGB_PARAMS)
final_model.fit(X_all, y, verbose=False)

gain  = final_model.get_booster().get_score(importance_type="gain")
top10 = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 by gain:")
for feat, g in top10:
    hydro_marker = " <-- hydro_full_mean" if feat == "hydro_full_mean" else ""
    print(f"  {feat:<35} {g:.1f}{hydro_marker}")

# hydro_full_mean rank
all_ranked = sorted(gain.items(), key=lambda x: x[1], reverse=True)
hfm_rank = next((i+1 for i, (f, _) in enumerate(all_ranked) if f == "hydro_full_mean"), None)
print(f"\nhydro_full_mean importance rank: {hfm_rank} / {len(all_ranked)}")
hfm_gain = gain.get("hydro_full_mean", 0.0)
print(f"hydro_full_mean gain: {hfm_gain:.2f}")


# ── RFECV ────────────────────────────────────────────────────────────────────
print("\n[5] RFECV feature selection ...")
rfecv_model = xgb.XGBClassifier(**XGB_PARAMS)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(
    estimator=rfecv_model,
    step=1,
    cv=cv,
    scoring="roc_auc",
    min_features_to_select=3,
    n_jobs=-1,
)
rfecv.fit(X_all, y)
selected = [f for f, s in zip(features, rfecv.support_) if s]
print(f"RFECV selected {len(selected)}/{len(features)} features:")
for f in selected:
    print(f"  {f}")
print(f"  hydro_full_mean selected: {'hydro_full_mean' in selected}")


# ── Save ──────────────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as f:
    pickle.dump({"model": final_model, "features": features,
                 "medians": train_med.to_dict(), "rfecv_selected": selected}, f)
print(f"\nModel saved: {MODEL_OUT}")

results = {
    "model": "tretter_combination_xgb",
    "n_samples": int(len(df)),
    "n_positives": int(y.sum()),
    "n_features": len(features),
    "features": features,
    "xgb_params": XGB_PARAMS,
    "hydro_full_mean": {
        "coverage": float(hfm_cov) if hfm_cov is not None else None,
        "importance_rank": hfm_rank,
        "gain": round(float(hfm_gain), 2),
        "rfecv_selected": "hydro_full_mean" in selected,
    },
    "lopo": {
        "valid_folds": valid_folds,
        "n_folds": len(fold_aucs),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "mean_auc": round(mean_auc, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    },
    "baselines": baseline_results,
    "prescreened_cohort": {
        "recall_at_5": round(recall_at[5], 4),
        "recall_at_10": round(recall_at[10], 4),
    },
    "top10_features_gain": {k: round(v, 2) for k, v in top10},
    "rfecv": {
        "n_selected": len(selected),
        "selected_features": selected,
        "optimal_n_features": int(rfecv.n_features_),
    },
}

with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {RESULTS_OUT}")
