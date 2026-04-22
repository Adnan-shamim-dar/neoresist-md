"""
Cross-dataset transfer: Ott 2017 <-> Tretter 2023.
Test A: Train Ott  → score Tretter (no retraining)
Test B: Train Tretter → score Ott LOPO
Test C: Combined Ott+Tretter LOPO on all patients
Usage: py -3.11 backend/strategy_engine/run_cross_dataset_transfer.py
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

try:
    import xgboost as xgb
except ImportError:
    print("pip install xgboost"); sys.exit(1)

ARTIFACTS = Path("backend/strategy_engine/artifacts")
RESULTS_OUT = ARTIFACTS / "cross_dataset_transfer_results.json"

XGB_PARAMS = dict(
    max_depth=2, n_estimators=200, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.6, min_child_weight=3,
    reg_alpha=0.5, objective="binary:logistic",
    eval_metric="auc", random_state=42, n_jobs=-1,
)

EXCL = {"immunogenic", "paper_source", "patient_id", "cancer_type",
        "mutant_peptide", "wt_peptide", "hla_allele", "clonality_class",
        "gene", "protein_change"}


def usable_features(df: pd.DataFrame) -> set[str]:
    return {c for c in df.select_dtypes(include=[np.number]).columns
            if c not in EXCL and df[c].notna().mean() > 0.5}


def bootstrap_ci(aucs: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(aucs, len(aucs), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray,
              spw: float) -> xgb.XGBClassifier:
    p = {**XGB_PARAMS, "scale_pos_weight": spw}
    m = xgb.XGBClassifier(**p)
    m.fit(X_train, y_train, verbose=False)
    return m


def impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians)


# ── Load ──────────────────────────────────────────────────────────────────────
ott  = pd.read_csv(ARTIFACTS / "ott_2017_features.csv")
tret = pd.read_csv("backend/strategy_engine/artifacts/tretter_2023_features.csv")

ott_f  = usable_features(ott)
tret_f = usable_features(tret)
COMMON = sorted(ott_f & tret_f)

print(f"Ott:     {len(ott)} rows, {int(ott['immunogenic'].sum())} pos, "
      f"{ott['patient_id'].nunique()} patients")
print(f"Tretter: {len(tret)} rows, {int(tret['immunogenic'].sum())} pos, "
      f"{tret['patient_id'].nunique()} patients")
print(f"Common features: {len(COMMON)}")
print(f"  {COMMON}\n")

SPW_OTT  = round((len(ott)  - ott['immunogenic'].sum())  / ott['immunogenic'].sum(),  2)
SPW_TRET = round((len(tret) - tret['immunogenic'].sum()) / tret['immunogenic'].sum(), 2)
print(f"scale_pos_weight  Ott={SPW_OTT}  Tretter={SPW_TRET}\n")


# ── Test A: Train Ott → Score Tretter ─────────────────────────────────────────
print("=" * 60)
print("TEST A: Train on Ott (all), score Tretter (all)")
print("=" * 60)

X_ott  = ott[COMMON].copy()
y_ott  = ott["immunogenic"].values.astype(int)
ott_med = X_ott.median()

X_tret = tret[COMMON].copy()
y_tret = tret["immunogenic"].values.astype(int)

model_a = train_xgb(impute(X_ott, ott_med), y_ott, SPW_OTT)
probs_a = model_a.predict_proba(impute(X_tret, ott_med))[:, 1]
auc_a   = roc_auc_score(y_tret, probs_a)

# per-patient AUCs for Tretter
pat_aucs_a = []
for pat in sorted(tret["patient_id"].unique()):
    idx = tret["patient_id"].values == pat
    yt  = y_tret[idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    pat_aucs_a.append(roc_auc_score(yt, probs_a[idx]))

ci_a_lo, ci_a_hi = bootstrap_ci(pat_aucs_a)
print(f"  Pooled AUC (all Tretter rows):  {auc_a:.4f}")
print(f"  Per-patient mean AUC:           {np.mean(pat_aucs_a):.4f}  "
      f"95% CI [{ci_a_lo:.4f}, {ci_a_hi:.4f}]  (n={len(pat_aucs_a)} folds)")


# ── Test B: Train Tretter → Score Ott LOPO ────────────────────────────────────
print()
print("=" * 60)
print("TEST B: Train on Tretter (all), score Ott LOPO")
print("=" * 60)

X_tret_full = impute(tret[COMMON].copy(), tret[COMMON].median())
model_b = train_xgb(X_tret_full, y_tret, SPW_TRET)

ott_pats = sorted(ott["patient_id"].unique())
fold_aucs_b: list[float] = []
for pat in ott_pats:
    idx = ott["patient_id"].values == pat
    yt  = y_ott[idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        print(f"  {pat}: skipped (single class)")
        continue
    # impute using Tretter medians (model was trained on Tretter)
    Xt = impute(ott.loc[idx, COMMON].copy(), tret[COMMON].median())
    probs = model_b.predict_proba(Xt)[:, 1]
    auc = roc_auc_score(yt, probs)
    fold_aucs_b.append(auc)
    print(f"  {pat}: n={idx.sum()}, pos={yt.sum()}, AUC={auc:.3f}")

mean_b = float(np.mean(fold_aucs_b))
ci_b_lo, ci_b_hi = bootstrap_ci(fold_aucs_b)
pool_probs_b = model_b.predict_proba(impute(ott[COMMON].copy(), tret[COMMON].median()))[:, 1]
pooled_b = roc_auc_score(y_ott, pool_probs_b)
print(f"\n  Per-patient mean AUC: {mean_b:.4f}  95% CI [{ci_b_lo:.4f}, {ci_b_hi:.4f}]")
print(f"  Pooled AUC (all Ott rows):      {pooled_b:.4f}")


# ── Test C: Combined LOPO ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST C: Combined Ott+Tretter LOPO (all patients)")
print("=" * 60)

combined = pd.concat([
    ott[COMMON  + ["immunogenic", "patient_id", "paper_source"]],
    tret[COMMON + ["immunogenic", "patient_id", "paper_source"]],
], ignore_index=True)
y_comb = combined["immunogenic"].values.astype(int)
pats_comb = combined["patient_id"].values
X_comb = combined[COMMON].copy()

all_pats = sorted(combined["patient_id"].unique())
fold_aucs_c: list[float] = []
oof_c = np.full(len(combined), np.nan)

for pat in all_pats:
    test_idx  = np.where(pats_comb == pat)[0]
    train_idx = np.where(pats_comb != pat)[0]
    yt = y_comb[test_idx]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue  # skip all-neg or all-pos

    X_tr = X_comb.iloc[train_idx].copy()
    X_te = X_comb.iloc[test_idx].copy()
    fold_med = X_tr.median()
    X_tr = impute(X_tr, fold_med)
    X_te = impute(X_te, fold_med)

    y_tr = y_comb[train_idx]
    spw = round((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 2)
    m = train_xgb(X_tr, y_tr, spw)
    probs = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(yt, probs)
    fold_aucs_c.append(auc)
    oof_c[test_idx] = probs
    src = combined.loc[test_idx[0], "paper_source"]
    print(f"  {pat} [{src}]: n={len(test_idx)}, pos={yt.sum()}, AUC={auc:.3f}")

mean_c = float(np.mean(fold_aucs_c))
ci_c_lo, ci_c_hi = bootstrap_ci(fold_aucs_c)

valid = ~np.isnan(oof_c)
pooled_c = roc_auc_score(y_comb[valid], oof_c[valid])
print(f"\n  Per-patient mean AUC: {mean_c:.4f}  95% CI [{ci_c_lo:.4f}, {ci_c_hi:.4f}]")
print(f"  Pooled AUC (concat OOF): {pooled_c:.4f}")

# Break down pooled by source
for src, df_src in [("ott_2017", ott), ("tretter_2023", tret)]:
    src_mask = combined["paper_source"].str.contains(src.split("_")[0]) & valid
    if src_mask.sum() > 0:
        src_auc = roc_auc_score(y_comb[src_mask], oof_c[src_mask])
        print(f"  Pooled AUC [{src}]: {src_auc:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
threshold = 0.60
results = {
    "A_ott_to_tretter": {"pooled_auc": round(auc_a, 4),
                         "per_patient_mean": round(float(np.mean(pat_aucs_a)), 4),
                         "ci_95": [round(ci_a_lo, 4), round(ci_a_hi, 4)]},
    "B_tretter_to_ott": {"pooled_auc": round(pooled_b, 4),
                         "per_patient_mean": round(mean_b, 4),
                         "ci_95": [round(ci_b_lo, 4), round(ci_b_hi, 4)]},
    "C_combined_lopo":  {"pooled_auc": round(pooled_c, 4),
                         "per_patient_mean": round(mean_c, 4),
                         "ci_95": [round(ci_c_lo, 4), round(ci_c_hi, 4)]},
    "n_common_features": len(COMMON),
    "common_features": COMMON,
}

for label, r in results.items():
    if label == "n_common_features" or label == "common_features":
        continue
    pa = r["per_patient_mean"]
    verdict = "TRANSFERS (>0.60)" if pa > threshold else "COLLAPSES (<=0.60)"
    print(f"  {label:<25} pooled={r['pooled_auc']:.4f}  "
          f"per-pat={pa:.4f} CI[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]  {verdict}")

generalizes = (results["A_ott_to_tretter"]["per_patient_mean"] > threshold and
               results["B_tretter_to_ott"]["per_patient_mean"] > threshold)
print(f"\n  Stage 2 signal generalizes: {generalizes}")
print(f"  Architecture implication: "
      f"{'shared weights feasible' if generalizes else 'registry (per-dataset weights) supported'}")

results["interpretation"] = {
    "threshold": threshold,
    "generalizes": generalizes,
    "implication": "shared weights feasible" if generalizes else "registry architecture supported",
}

with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {RESULTS_OUT}")
