"""
PHASE 3 — Two-Stage Model.

Stage 1: Presentation model trained on NCI full mutanome (292K rows).
         Predicts whether a peptide is processed and displayed on MHC.
         Expected AUC > 0.90 — binding dominates at this stage.

Stage 2: Recognition model trained on pre-screened cohorts (per cancer type).
         Given presentation, does a T-cell recognize the peptide?
         TCR contact, foreignness, expression take over from binding.

Run: py -3.11 backend/strategy_engine/two_stage_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Model backend ─────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
    print("Using XGBoost")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False
    print("WARNING: XGBoost not found. Using sklearn GradientBoostingClassifier.")


def make_stage1_model(scale_pos_weight: float):
    if USE_XGBOOST:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )


def make_stage2_model(scale_pos_weight: float):
    if USE_XGBOOST:
        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)


def safe_auc(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    if m.sum() < 2 or len(np.unique(y_true[m])) < 2:
        return None
    return float(roc_auc_score(y_true[m], y_score[m]))


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if len(aucs) < 10:
        return [None, None]
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


def prepare_X(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix, filling missing with NaN (XGBoost handles native NaN)."""
    cols = []
    for f in features:
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce")
            if s.notna().mean() > 0.10:
                cols.append(f)
    X = df[cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    return X, cols


# ── Stage 1 features (from NCI full mutanome) ─────────────────────────────────
STAGE1_FEATURES = [
    "binding_nm", "binding_log", "binding_sigmoid",
    "binding_stability", "presentation_score_el",
    "expression_log2", "pep_length",
]

# ── Stage 2 features (pre-screened cohorts) ───────────────────────────────────
STAGE2_FEATURES = [
    "tcr_hydro_mean", "tcr_hydro_max", "tcr_volume_mean",
    "tcr_charge_sum", "tcr_aromatic_count",
    "self_dissimilarity", "hamming_distance",
    "blosum62_score", "blosum62_at_mutation",
    "dai_agretopicity",
    "calis_simplified",
    "pep_length", "mutation_position",
    "hydro_full_mean", "aliphatic_index",
    "expression_log2", "expression_zscore",
    "vaf",
    "binding_log", "binding_sigmoid",
    "presentation_probability",  # added after Stage 1
]

# Cancer type groupings
CANCER_GROUPS: dict[str, list[str]] = {
    "melanoma": ["ott_2017", "sahin_2017"],
    "gbm": ["hilf_2019", "keskin_2019"],
    "pancreatic": ["rojas_2023"],
    "mixed_tesla": ["tesla_2020"],
}


def load_features(name: str) -> pd.DataFrame:
    path = ARTIFACTS / f"{name}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    return pd.read_csv(path, low_memory=False)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage1() -> dict:
    print("\n" + "=" * 60)
    print("STAGE 1 — PRESENTATION MODEL (NCI full mutanome)")
    print("=" * 60)

    nci = load_features("muller_nci")
    y = nci["immunogenic"].values.astype(int)
    X, used_feats = prepare_X(nci, STAGE1_FEATURES)
    patient_ids = nci["patient_id"].values

    missing_feats = [f for f in STAGE1_FEATURES if f not in used_feats]
    print(f"Features used: {used_feats}")
    print(f"Features dropped (low coverage): {missing_feats}")
    print(f"  Rows: {len(nci):,} | Positives: {y.sum()} | Patients: {len(np.unique(patient_ids))}")

    # Class balance
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  Class ratio: 1:{spw:.0f} → scale_pos_weight={spw:.1f}")

    # LOPO validation
    print("\n  Running LOPO validation...")
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    per_patient: dict[str, dict] = {}

    for train_idx, test_idx in logo.split(X, y, groups=patient_ids):
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        m = make_stage1_model(spw)
        fit_model(m, X[train_idx], y[train_idx])
        y_pred = m.predict_proba(X[test_idx])[:, 1]
        pt = patient_ids[test_idx[0]]
        auc = safe_auc(y_test, y_pred)
        per_patient[str(pt)] = {"auc": round(auc, 4) if auc else None}
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    stage1_auc = safe_auc(y_true_all, y_pred_all)
    stage1_aupr = float(average_precision_score(y_true_all, y_pred_all))
    ci = bootstrap_ci(y_true_all, y_pred_all)
    print(f"  Stage 1 AUC:  {stage1_auc:.4f}  CI={ci}")
    print(f"  Stage 1 AUPR: {stage1_aupr:.4f}")
    print(f"  LOPO folds: {len(per_patient)}")

    if stage1_auc < 0.80:
        print(f"  ERROR: Stage 1 AUC={stage1_auc:.4f} < 0.80. Check data!")
        print(f"  Features used: {used_feats}")

    # Train final model on ALL NCI data
    print("\n  Training final Stage 1 model on all NCI data...")
    final_model = make_stage1_model(spw)
    fit_model(final_model, X, y)

    model_path = ARTIFACTS / "stage1_presentation_model.pkl"
    joblib.dump({"model": final_model, "features": used_feats}, model_path)
    print(f"  Saved: {model_path}")

    result = {
        "stage": 1,
        "training_data": "nci_full_mutanome",
        "n_rows": len(nci),
        "n_immunogenic": int(y.sum()),
        "n_patients": int(len(np.unique(patient_ids))),
        "features_used": used_feats,
        "features_dropped": missing_feats,
        "lopo_auc": round(stage1_auc, 4) if stage1_auc else None,
        "lopo_auc_ci": ci,
        "lopo_aupr": round(stage1_aupr, 4),
        "n_lopo_folds": len(per_patient),
        "per_patient_auc": per_patient,
        "model_backend": "xgboost" if USE_XGBOOST else "sklearn_gbm",
    }
    return result, final_model, used_feats


def add_presentation_probabilities(
    stage1_model,
    stage1_features: list[str],
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply Stage 1 model to all pre-screened datasets."""
    print("\n  Adding presentation_probability to all datasets...")
    updated = {}
    for name, df in datasets.items():
        X_pred = np.column_stack([
            pd.to_numeric(df[f], errors="coerce").fillna(np.nanmedian(
                pd.to_numeric(df[f], errors="coerce").dropna()
            ) if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().any()
              else 0.0).values
            if f in df.columns
            else np.full(len(df), np.nan)
            for f in stage1_features
        ])
        # For sklearn/XGBoost NaN handling: replace NaN with column median
        for col_idx in range(X_pred.shape[1]):
            col = X_pred[:, col_idx]
            valid = col[np.isfinite(col)]
            if len(valid) > 0:
                X_pred[np.isnan(col), col_idx] = np.median(valid)
            else:
                X_pred[:, col_idx] = 0.0
        df = df.copy()
        df["presentation_probability"] = stage1_model.predict_proba(X_pred)[:, 1]
        updated[name] = df
        print(f"    {name}: mean P={df['presentation_probability'].mean():.3f}")
    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage2_cancer_type(
    cancer_type: str,
    dataset_names: list[str],
    datasets: dict[str, pd.DataFrame],
) -> dict:
    print(f"\n  --- {cancer_type} ({'+'.join(dataset_names)}) ---")

    dfs = []
    for dn in dataset_names:
        if dn in datasets:
            df = datasets[dn].copy()
            df["_source"] = dn
            dfs.append(df)

    if not dfs:
        return {"cancer_type": cancer_type, "error": "no data"}

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["immunogenic"].isin([0, 1])].copy()
    y = combined["immunogenic"].values.astype(int)
    patient_ids = combined["patient_id"].values
    n_patients = len(np.unique(patient_ids))
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    print(f"    Combined: {len(combined)} rows | {n_pos} pos | {n_neg} neg | {n_patients} patients")

    if n_pos < 3 or n_patients < 2:
        print(f"    Insufficient data for LOPO (need ≥3 pos and ≥2 patients)")
        return {
            "cancer_type": cancer_type,
            "datasets_used": dataset_names,
            "n_rows": len(combined),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_patients": n_patients,
            "lopo_auc": None,
            "error": "insufficient_data",
        }

    # Feature selection: only use if >30% coverage
    X_full, used_feats = prepare_X(combined, STAGE2_FEATURES)
    dropped = [f for f in STAGE2_FEATURES if f not in used_feats]
    print(f"    Features used ({len(used_feats)}): {used_feats[:8]}...")
    if dropped:
        print(f"    Features dropped (low coverage): {dropped[:5]}...")

    if len(used_feats) == 0:
        return {"cancer_type": cancer_type, "error": "no usable features",
                "datasets_used": dataset_names}

    spw = n_neg / n_pos if n_pos > 0 else 1.0

    # LOPO validation
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    per_patient: dict[str, float | None] = {}
    lopo_eligible = 0

    for train_idx, test_idx in logo.split(X_full, y, groups=patient_ids):
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            per_patient[str(patient_ids[test_idx[0]])] = None
            continue
        lopo_eligible += 1
        X_train, X_test = X_full[train_idx], X_full[test_idx]

        if not USE_XGBOOST:
            # Impute with training median for sklearn
            for col_i in range(X_train.shape[1]):
                col = X_train[:, col_i]
                med = np.nanmedian(col)
                X_train[np.isnan(col), col_i] = med
                X_test[np.isnan(X_test[:, col_i]), col_i] = med

        m = make_stage2_model(spw)
        fit_model(m, X_train, y[train_idx])
        y_pred = m.predict_proba(X_test)[:, 1]
        pt = str(patient_ids[test_idx[0]])
        per_patient[pt] = round(float(safe_auc(y_test, y_pred) or 0), 4)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Handle Keskin (2 patients, no LOPO)
    is_keskin_only = dataset_names == ["keskin_2019"]
    if is_keskin_only or lopo_eligible == 0:
        print(f"    Keskin / <2 LOPO folds — in-sample AUC only")
        final_m = make_stage2_model(spw)
        fit_model(final_m, X_full, y)
        y_is = final_m.predict_proba(X_full)[:, 1]
        is_auc = safe_auc(y, y_is)
        return {
            "cancer_type": cancer_type,
            "datasets_used": dataset_names,
            "n_rows": len(combined),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_patients": n_patients,
            "lopo_auc": None,
            "insample_auc": round(is_auc, 4) if is_auc else None,
            "in_sample_only": True,
            "features_used": used_feats,
            "features_dropped_low_coverage": dropped,
            "per_patient_auc": per_patient,
        }

    lopo_auc = safe_auc(y_true_all, y_pred_all)
    lopo_aupr = float(average_precision_score(y_true_all, y_pred_all))
    ci = bootstrap_ci(y_true_all, y_pred_all)
    print(f"    LOPO AUC={lopo_auc:.4f}  CI={ci}  AUPR={lopo_aupr:.4f}  "
          f"folds={lopo_eligible}")

    # Binding-only baseline for comparison
    bind_feat = ["binding_log"]
    X_bind, bind_used = prepare_X(combined, bind_feat)
    y_true_bind, y_pred_bind = [], []
    for train_idx, test_idx in logo.split(X_bind, y, groups=patient_ids):
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        if not USE_XGBOOST:
            X_b_train = X_bind[train_idx].copy()
            X_b_test = X_bind[test_idx].copy()
            med = np.nanmedian(X_b_train)
            X_b_train[np.isnan(X_b_train)] = med
            X_b_test[np.isnan(X_b_test)] = med
        else:
            X_b_train, X_b_test = X_bind[train_idx], X_bind[test_idx]
        if len(bind_used) > 0:
            mb = make_stage2_model(spw)
            fit_model(mb, X_b_train, y[train_idx])
            y_true_bind.extend(y_test)
            y_pred_bind.extend(mb.predict_proba(X_b_test)[:, 1])

    baseline_auc = safe_auc(y_true_bind, y_pred_bind)
    baseline_ci = bootstrap_ci(y_true_bind, y_pred_bind) if y_true_bind else [None, None]
    improvement = round(lopo_auc - (baseline_auc or 0), 4) if lopo_auc and baseline_auc else None
    print(f"    Binding baseline: {baseline_auc:.4f}  CI={baseline_ci}  "
          f"Improvement: {'+' if improvement and improvement>0 else ''}{improvement}")

    # Train final model on all data
    final_m = make_stage2_model(spw)
    fit_model(final_m, X_full, y)

    # Feature importances
    if hasattr(final_m, "feature_importances_"):
        imps = dict(zip(used_feats,
                        [round(float(v), 4) for v in final_m.feature_importances_]))
        imps = dict(sorted(imps.items(), key=lambda x: x[1], reverse=True))
    else:
        imps = {}

    # Save model
    model_path = ARTIFACTS / f"stage2_{cancer_type}_model.pkl"
    joblib.dump({"model": final_m, "features": used_feats}, model_path)

    return {
        "cancer_type": cancer_type,
        "datasets_used": dataset_names,
        "n_rows": len(combined),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_patients": n_patients,
        "n_lopo_folds": lopo_eligible,
        "features_used": used_feats,
        "features_dropped_low_coverage": dropped,
        "lopo_auc": round(lopo_auc, 4) if lopo_auc else None,
        "lopo_auc_ci": ci,
        "lopo_aupr": round(lopo_aupr, 4),
        "baseline_binding_only_auc": round(baseline_auc, 4) if baseline_auc else None,
        "baseline_binding_only_ci": baseline_ci,
        "improvement_over_binding": improvement,
        "improvement_significant": bool(ci[0] is not None and baseline_auc is not None
                                        and ci[0] > baseline_auc),
        "per_patient_auc": per_patient,
        "feature_importances": imps,
        "no_transfer_partner": cancer_type == "pancreatic",
        "in_sample_only": False,
    }


def run_stage2_universal(datasets: dict[str, pd.DataFrame]) -> dict:
    """Train a universal model on ALL pre-screened cohorts pooled."""
    print("\n  --- universal (all cohorts pooled) ---")
    all_names = [k for k in datasets if k != "muller_nci"]
    return run_stage2_cancer_type("universal", all_names, datasets)


def main() -> int:
    print("=" * 60)
    print("PHASE 3 — TWO-STAGE MODEL")
    print("=" * 60)

    # Load all pre-screened feature matrices
    dataset_names = ["ott_2017", "tesla_2020", "sahin_2017",
                     "hilf_2019", "rojas_2023", "keskin_2019"]
    datasets: dict[str, pd.DataFrame] = {}
    for dn in dataset_names:
        try:
            datasets[dn] = load_features(dn)
        except FileNotFoundError:
            print(f"  WARNING: {dn} features not found — skipping")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    stage1_result, stage1_model, stage1_feats = run_stage1()

    out_path = ARTIFACTS / "stage1_results.json"
    out_path.write_text(json.dumps(stage1_result, indent=2, default=str), encoding="utf-8")
    print(f"  Saved Stage 1 results: {out_path}")

    if stage1_result.get("lopo_auc") and stage1_result["lopo_auc"] < 0.80:
        print("STOPPING: Stage 1 AUC < 0.80. Check data pipeline.")
        return 1

    # Add presentation_probability to all datasets
    datasets = add_presentation_probabilities(stage1_model, stage1_feats, datasets)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — RECOGNITION MODELS (per cancer type)")
    print("=" * 60)

    all_stage2_results: dict[str, dict] = {}

    for cancer_type, ds_names in CANCER_GROUPS.items():
        result = run_stage2_cancer_type(cancer_type, ds_names, datasets)
        all_stage2_results[cancer_type] = result
        out = ARTIFACTS / f"stage2_{cancer_type}_results.json"
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"    Saved: {out}")

    # Universal model
    universal_result = run_stage2_universal(datasets)
    all_stage2_results["universal"] = universal_result
    out = ARTIFACTS / "stage2_universal_results.json"
    out.write_text(json.dumps(universal_result, indent=2, default=str), encoding="utf-8")
    print(f"    Saved: {out}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    print(f"Stage 1 AUC: {stage1_result.get('lopo_auc')}  CI={stage1_result.get('lopo_auc_ci')}")
    print(f"\nStage 2 per cancer type:")
    print(f"  {'Cancer type':<20} {'AUC':>8}  {'CI':>18}  {'Baseline':>10}  {'Improvement':>12}  {'Sig?':>6}")
    for ct, r in all_stage2_results.items():
        auc = r.get("lopo_auc")
        ci = r.get("lopo_auc_ci", ["?", "?"])
        base = r.get("baseline_binding_only_auc")
        imp = r.get("improvement_over_binding")
        sig = r.get("improvement_significant", False)
        is_only = r.get("in_sample_only", False)
        auc_s = f"{auc:.4f}" if auc else "in-sample" if is_only else "N/A"
        ci_s = f"[{ci[0]:.4f},{ci[1]:.4f}]" if ci and ci[0] else "  N/A"
        base_s = f"{base:.4f}" if base else "  N/A"
        imp_s = f"{'+' if imp and imp>0 else ''}{imp:.4f}" if imp else "  N/A"
        print(f"  {ct:<20} {auc_s:>8}  {ci_s:>18}  {base_s:>10}  {imp_s:>12}  {'YES' if sig else 'no':>6}")

    # Save combined summary
    summary_path = ARTIFACTS / "stage2_all_results.json"
    summary_path.write_text(json.dumps(all_stage2_results, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nAll Stage 2 results: {summary_path}")
    print("\nPhase 3 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
