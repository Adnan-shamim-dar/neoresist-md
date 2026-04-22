"""
XGBoost on Ott 2017 (ott_2017_features.csv) — LOPO CV + Phase 8 haystack evaluation.

Usage: py -3.11 backend/strategy_engine/run_ott_xgboost.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

try:
    import xgboost as xgb
except ImportError:
    print("xgboost not installed — pip install xgboost")
    sys.exit(1)

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

OTT_FEAT    = ARTIFACTS / "ott_2017_features.csv"
HAYSTACK    = Path("backend/validation/artifacts/ott_haystack_scored.csv")
MODEL_OUT   = ARTIFACTS / "ott_full_combination_model.pkl"
RESULTS_OUT = ARTIFACTS / "ott_full_combination_results.json"

XGB_PARAMS = dict(
    max_depth=2,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=3,
    reg_alpha=0.5,
    scale_pos_weight=5.47,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(OTT_FEAT)
print(f"Ott 2017: {len(df)} rows, {int(df['immunogenic'].sum())} positives, "
      f"{df['patient_id'].nunique()} patients")

EXCL = {"immunogenic", "paper_source", "patient_id", "cancer_type",
        "mutant_peptide", "wt_peptide", "hla_allele", "clonality_class",
        "gene", "protein_change"}

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in num_cols
            if c not in EXCL and df[c].notna().mean() > 0.50]
print(f"Usable features (>50% non-null): {len(features)}")

y  = df["immunogenic"].values.astype(int)
X  = df[features].copy()
patients = df["patient_id"].values


# ── LOPO CV ───────────────────────────────────────────────────────────────────
print("\n[1] LOPO cross-validation ...")
unique_patients = sorted(df["patient_id"].unique())
fold_aucs: list[float] = []
oof_scores = np.full(len(df), np.nan)

for pat in unique_patients:
    test_idx  = np.where(patients == pat)[0]
    train_idx = np.where(patients != pat)[0]

    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()

    # Impute with training fold medians
    fold_medians = X_train.median()
    X_train = X_train.fillna(fold_medians)
    X_test  = X_test.fillna(fold_medians)

    # Need both classes to compute AUC
    if len(np.unique(y_test)) < 2:
        print(f"  {pat}: skipped (single class in test fold)")
        continue

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    fold_aucs.append(auc)
    oof_scores[test_idx] = probs
    print(f"  {pat}: n_test={len(test_idx)}, pos={int(y_test.sum())}, AUC={auc:.3f}")

print(f"\nLOPO AUCs: {[f'{a:.3f}' for a in fold_aucs]}")
mean_auc = float(np.mean(fold_aucs))
print(f"Mean LOPO AUC: {mean_auc:.3f}")

# 95% bootstrap CI over fold AUCs
rng = np.random.default_rng(42)
boot_means = [np.mean(rng.choice(fold_aucs, size=len(fold_aucs), replace=True))
              for _ in range(1000)]
ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
print(f"95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]")


# ── Recall@k on pre-screened cohort (OOF) ────────────────────────────────────
print("\n[2] Recall@k on pre-screened Ott cohort (OOF scores) ...")
valid_mask = ~np.isnan(oof_scores)
recall_at = {}
for k in (10, 20):
    hits = 0
    total_pos = 0
    for pat in unique_patients:
        idx = np.where((patients == pat) & valid_mask)[0]
        if len(idx) == 0:
            continue
        order = np.argsort(oof_scores[idx])[::-1][:k]
        hits      += int(y[idx][order].sum())
        total_pos += int(y[idx].sum())
    recall_at[k] = hits / total_pos if total_pos else 0.0
    print(f"  Recall@{k} = {hits}/{total_pos} = {recall_at[k]:.3f}")


# ── Feature importances ───────────────────────────────────────────────────────
print("\n[3] Training final model on all 97 rows ...")
X_all = X.copy()
train_medians = X_all.median()
X_all = X_all.fillna(train_medians)

final_model = xgb.XGBClassifier(**XGB_PARAMS)
final_model.fit(X_all, y, verbose=False)

gain = final_model.get_booster().get_score(importance_type="gain")
top10 = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 features by gain:")
for feat, g in top10:
    print(f"  {feat:<35} {g:.1f}")


# ── Phase 8 haystack evaluation ───────────────────────────────────────────────
print("\n[4] Phase 8 haystack evaluation ...")
hay = pd.read_csv(HAYSTACK)
print(f"Haystack: {len(hay)} rows, patients: {hay['patient_id'].nunique()}")

# Map haystack feature columns to model feature names
# Only 3 overlap: presentation_score, expression_score, self_dissimilarity
hay_feat_map = {f: f for f in features if f in hay.columns}
print(f"  Features available in haystack: {len(hay_feat_map)} / {len(features)}")
print(f"  Available: {list(hay_feat_map.keys())}")

# Build feature matrix for haystack: median-impute everything
X_hay = pd.DataFrame(index=hay.index, columns=features, dtype=float)
for feat in features:
    if feat in hay.columns:
        X_hay[feat] = pd.to_numeric(hay[feat], errors="coerce")
    else:
        X_hay[feat] = np.nan
X_hay = X_hay.fillna(train_medians)

hay_scores = final_model.predict_proba(X_hay)[:, 1]
hay = hay.copy()
hay["model_score"] = hay_scores

# Recall@20 per-patient on haystack (patients ott_1..ott_6 only; others have no labeled data)
labeled_patients = [p for p in sorted(hay["patient_id"].unique())
                    if hay.loc[hay["patient_id"] == p, "is_labeled_mutation"].any()]

print(f"\n  Evaluating on {len(labeled_patients)} patients with labeled data")
hay_hits, hay_total = 0, 0
per_patient_r20 = {}

for pat in labeled_patients:
    pmask  = hay["patient_id"] == pat
    pdata  = hay[pmask].copy().sort_values("model_score", ascending=False).reset_index(drop=True)
    n_pos  = int((pdata["immunogenic_labeled"] == 1).sum())
    if n_pos == 0:
        continue
    top20  = pdata.head(20)
    hits   = int((top20["immunogenic_labeled"] == 1).sum())
    hay_hits  += hits
    hay_total += n_pos
    pos_ranks = pdata.index[pdata["immunogenic_labeled"] == 1].tolist()
    per_patient_r20[pat] = {
        "n_total": int(pmask.sum()),
        "n_positive": n_pos,
        "hits_in_top20": hits,
        "positive_ranks": [r + 1 for r in pos_ranks],
    }
    print(f"  {pat}: n={int(pmask.sum())}, pos={n_pos}, hits@20={hits}, "
          f"ranks={[r+1 for r in pos_ranks[:3]]}")

hay_r20 = hay_hits / hay_total if hay_total else 0.0
print(f"\n  Haystack Recall@20: {hay_hits}/{hay_total} = {hay_r20:.3f} ({hay_r20:.1%})")
print(f"  Baseline binding_only R@20 = 36.4%")
print(f"  melanoma_ml_v1     R@20 = 63.6%")
print(f"  NOTE: haystack features limited to {len(hay_feat_map)}/{{len(features)}} "
      f"({', '.join(hay_feat_map.keys())}). Remaining features median-imputed.")


# ── Save artifacts ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as f:
    pickle.dump({"model": final_model, "features": features, "medians": train_medians.to_dict()}, f)
print(f"\nModel saved: {MODEL_OUT}")

results = {
    "model": "ott_full_combination_xgb",
    "training_data": "ott_2017_features.csv",
    "n_samples": int(len(df)),
    "n_positives": int(y.sum()),
    "n_features": len(features),
    "features": features,
    "xgb_params": XGB_PARAMS,
    "lopo": {
        "n_folds": len(fold_aucs),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "mean_auc": round(mean_auc, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    },
    "prescreened_cohort": {
        "recall_at_10": round(recall_at[10], 4),
        "recall_at_20": round(recall_at[20], 4),
    },
    "top10_features_gain": {k: round(v, 2) for k, v in top10},
    "phase8_haystack": {
        "n_haystack": int(len(hay)),
        "n_labeled_patients": len(labeled_patients),
        "features_available": list(hay_feat_map.keys()),
        "features_median_imputed": len(features) - len(hay_feat_map),
        "recall_at_20": round(hay_r20, 4),
        "hits": hay_hits,
        "total_positives": hay_total,
        "per_patient": per_patient_r20,
        "note": "Most features median-imputed; only presentation_score, expression_score, self_dissimilarity available for full haystack.",
        "comparison": {
            "binding_only_r20": 0.364,
            "melanoma_ml_v1_r20": 0.636,
        },
    },
}

with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {RESULTS_OUT}")
