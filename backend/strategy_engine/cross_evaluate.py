"""
PHASE 4 — Cross-Cancer Transfer Evaluation.

1. Apply each trained Stage 2 cancer-type model to held-out cancer types.
2. Leave-one-dataset-out within melanoma (Ott↔Sahin).
3. Generate final paper comparison table vs RL hand-crafted strategies.

Run: py -3.11 backend/strategy_engine/cross_evaluate.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")

# ── RL hand-crafted strategy scores (from AGENTS.md validated results) ─────────
RL_RESULTS = {
    "ott_2017":   {"binding_only": 0.649,  "rl_tcr_v1": 0.758, "rl_expression_v1": 0.716, "binding_plus_calis": 0.667},
    "tesla_2020": {"binding_only": 0.771,  "rl_tcr_v1": 0.510, "rl_expression_v1": 0.813, "binding_plus_calis": 0.762},
    "hilf_2019":  {"binding_only": 0.624,  "rl_tcr_v1": 0.493, "rl_expression_v1": 0.497, "binding_plus_calis": 0.627},
    "rojas_2023": {"binding_only": 0.547,  "rl_tcr_v1": 0.675, "rl_expression_v1": 0.547, "binding_plus_calis": 0.523},
    "sahin_2017": {"binding_only": 0.477,  "rl_tcr_v1": 0.660, "rl_expression_v1": 0.437, "binding_plus_calis": 0.401},
}

CANCER_GROUPS = {
    "melanoma":   ["ott_2017", "sahin_2017"],
    "gbm":        ["hilf_2019", "keskin_2019"],
    "pancreatic": ["rojas_2023"],
    "mixed_tesla": ["tesla_2020"],
}

# Map each dataset to its cancer group
DS_CANCER = {ds: ct for ct, dss in CANCER_GROUPS.items() for ds in dss}


def safe_auc(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    if m.sum() < 2 or len(np.unique(y_true[m])) < 2:
        return None
    return float(roc_auc_score(y_true[m], y_score[m]))


def load_features(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / f"{name}_features.csv", low_memory=False)


def load_model(cancer_type: str):
    path = ARTIFACTS / f"stage2_{cancer_type}_model.pkl"
    if not path.exists():
        return None, None
    obj = joblib.load(path)
    return obj["model"], obj["features"]


def predict(model, features: list[str], df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").values if f in df.columns
        else np.full(len(df), np.nan)
        for f in features
    ])
    for i in range(X.shape[1]):
        col = X[:, i]
        valid = col[np.isfinite(col)]
        X[~np.isfinite(col), i] = np.median(valid) if len(valid) else 0.0
    return model.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Cross-cancer transfer (train on cancer A, predict cancer B)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_cancer_transfer() -> list[dict]:
    print("\n[Task 1] Cross-cancer transfer evaluation")
    results = []
    all_datasets = ["ott_2017", "tesla_2020", "sahin_2017",
                    "hilf_2019", "rojas_2023", "keskin_2019"]

    for src_cancer in ["melanoma", "gbm", "pancreatic", "mixed_tesla"]:
        model, feats = load_model(src_cancer)
        if model is None:
            continue
        src_datasets = CANCER_GROUPS[src_cancer]

        for tgt_ds in all_datasets:
            if tgt_ds in src_datasets:
                continue  # skip in-domain (already in Phase 3 LOPO)
            try:
                df = load_features(tgt_ds)
            except FileNotFoundError:
                continue
            df = df[df["immunogenic"].isin([0, 1])]
            if len(df) < 5 or df["immunogenic"].sum() < 2:
                continue

            y_pred = predict(model, feats, df)
            auc = safe_auc(df["immunogenic"].values, y_pred)
            tgt_cancer = DS_CANCER.get(tgt_ds, "unknown")
            print(f"  {src_cancer:12s} → {tgt_ds:20s} (tgt={tgt_cancer:12s})  AUC={auc:.4f}" if auc else
                  f"  {src_cancer} → {tgt_ds}: insufficient data")
            results.append({
                "train_cancer": src_cancer,
                "test_dataset": tgt_ds,
                "test_cancer": tgt_cancer,
                "n_rows": len(df),
                "n_positive": int(df["immunogenic"].sum()),
                "transfer_auc": round(auc, 4) if auc else None,
                "rl_binding_only": RL_RESULTS.get(tgt_ds, {}).get("binding_only"),
                "rl_tcr_v1": RL_RESULTS.get(tgt_ds, {}).get("rl_tcr_v1"),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Within-melanoma leave-one-dataset-out (Ott ↔ Sahin)
# ═══════════════════════════════════════════════════════════════════════════════

def within_melanoma_lodo() -> list[dict]:
    """Train on one melanoma dataset, evaluate on the other."""
    print("\n[Task 2] Within-melanoma LODO (Ott ↔ Sahin)")
    from sklearn.ensemble import GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
        def make_model(spw):
            return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.7,
                                 eval_metric="logloss", random_state=42, verbosity=0)
    except ImportError:
        def make_model(spw):
            return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=42)

    # Load the feature list from the melanoma model
    _, melanoma_feats = load_model("melanoma")
    if melanoma_feats is None:
        return []

    results = []
    pairs = [("ott_2017", "sahin_2017"), ("sahin_2017", "ott_2017")]
    for train_ds, test_ds in pairs:
        try:
            train_df = load_features(train_ds)
            test_df = load_features(test_ds)
        except FileNotFoundError:
            continue
        train_df = train_df[train_df["immunogenic"].isin([0, 1])]
        test_df = test_df[test_df["immunogenic"].isin([0, 1])]

        y_train = train_df["immunogenic"].values.astype(int)
        y_test = test_df["immunogenic"].values.astype(int)
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # Build feature matrices using only available features
        avail = [f for f in melanoma_feats
                 if f in train_df.columns and f in test_df.columns
                 and pd.to_numeric(train_df[f], errors="coerce").notna().mean() > 0.1]

        X_train = train_df[avail].apply(pd.to_numeric, errors="coerce").values.astype(float)
        X_test = test_df[avail].apply(pd.to_numeric, errors="coerce").values.astype(float)
        for i in range(X_train.shape[1]):
            med = np.nanmedian(X_train[:, i])
            X_train[np.isnan(X_train[:, i]), i] = med
            X_test[np.isnan(X_test[:, i]), i] = med if np.isfinite(med) else 0.0

        m = make_model(spw)
        m.fit(X_train, y_train)
        y_pred = m.predict_proba(X_test)[:, 1]
        auc = safe_auc(y_test, y_pred)
        rl_bind = RL_RESULTS.get(test_ds, {}).get("binding_only")
        rl_tcr = RL_RESULTS.get(test_ds, {}).get("rl_tcr_v1")
        print(f"  train={train_ds} → test={test_ds}  AUC={auc:.4f}  "
              f"(binding={rl_bind}  rl_tcr_v1={rl_tcr})")
        results.append({
            "train_dataset": train_ds,
            "test_dataset": test_ds,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_test_positive": int(y_test.sum()),
            "lodo_auc": round(auc, 4) if auc else None,
            "rl_binding_only": rl_bind,
            "rl_tcr_v1": rl_tcr,
            "features_used": avail,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Paper comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def build_paper_table(cross_results: list[dict], lodo_results: list[dict]) -> pd.DataFrame:
    """Assemble final comparison table for paper."""
    print("\n[Task 3] Building paper comparison table")

    stage2_lopo = {
        "ott_2017":   None,    # in melanoma LOPO pool
        "sahin_2017": None,    # in melanoma LOPO pool
        "hilf_2019":  None,    # in GBM LOPO pool
        "tesla_2020": 0.7444,  # mixed_tesla LOPO
        "rojas_2023": 0.5044,  # pancreatic LOPO
    }

    # melanoma LOPO (pooled Ott+Sahin) maps to both datasets
    melanoma_lopo = 0.7597
    stage2_lopo["ott_2017"] = melanoma_lopo
    stage2_lopo["sahin_2017"] = melanoma_lopo
    stage2_lopo["hilf_2019"] = 0.5946  # GBM LOPO

    # Transfer AUC: universal model → each dataset (cross from cross_results)
    universal_transfer = {r["test_dataset"]: r["transfer_auc"]
                          for r in cross_results if r["train_cancer"] == "melanoma"
                          # will be overridden below with universal model results
                          }
    # Better: load the universal model and predict on each dataset
    univ_model, univ_feats = load_model("universal")

    # Universal model per-dataset breakdown from LOPO results (valid, no leakage)
    # Phase 3 stored per_patient_auc for universal LOPO — aggregate by dataset
    univ_per_ds: dict[str, float | None] = {}
    try:
        univ_res = json.loads((ARTIFACTS / "stage2_universal_results.json").read_text(encoding="utf-8"))
        per_pt = univ_res.get("per_patient_auc", {})
        # patient IDs include dataset prefix (e.g. "ott_2017_P01") — map back
        ds_to_pts: dict[str, list] = {}
        for pt_id, auc in per_pt.items():
            for ds in ["ott_2017", "tesla_2020", "sahin_2017", "hilf_2019",
                       "rojas_2023", "keskin_2019"]:
                if pt_id.startswith(ds.split("_")[0]) or True:
                    # Fall through — patient IDs are numeric within each dataset
                    break
        # Simpler: re-run LOPO per-dataset using the universal model but with
        # leave-one-DATASET-out to get valid transfer AUCs
        # Actually cleanest: use Phase3 per-patient to aggregate per dataset
        # Load the full feature datasets and track which patient belongs to which
        all_dfs = {}
        for ds in ["ott_2017", "tesla_2020", "sahin_2017", "hilf_2019", "rojas_2023"]:
            try:
                all_dfs[ds] = load_features(ds)
            except FileNotFoundError:
                pass

        # Build combined df with dataset label
        combined_parts = []
        for ds, df in all_dfs.items():
            tmp = df.copy()
            tmp["_dataset"] = ds
            combined_parts.append(tmp)
        combined = pd.concat(combined_parts, ignore_index=True)
        combined = combined[combined["immunogenic"].isin([0, 1])]

        # Leave-one-dataset-out with the universal model (train on all except target)
        if univ_model is not None:
            from sklearn.ensemble import GradientBoostingClassifier
            try:
                from xgboost import XGBClassifier
                def _make(spw):
                    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                        scale_pos_weight=spw, eval_metric="logloss",
                                        random_state=42, verbosity=0)
            except ImportError:
                def _make(spw):
                    return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                      learning_rate=0.1, random_state=42)

            # presentation_probability is not in saved CSVs (added at Phase 3 runtime)
            lodo_feats = [f for f in univ_feats
                          if f != "presentation_probability" and f in combined.columns]

            for tgt_ds in all_dfs:
                train_mask = combined["_dataset"] != tgt_ds
                test_mask = combined["_dataset"] == tgt_ds
                train_df = combined[train_mask]
                test_df = combined[test_mask]
                y_tr = train_df["immunogenic"].values.astype(int)
                y_te = test_df["immunogenic"].values.astype(int)
                if len(np.unique(y_te)) < 2 or y_tr.sum() < 2:
                    univ_per_ds[tgt_ds] = None
                    continue
                spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                m = _make(spw)
                X_tr = train_df[lodo_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
                X_te = test_df[lodo_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
                for i in range(X_tr.shape[1]):
                    med = np.nanmedian(X_tr[:, i])
                    X_tr[np.isnan(X_tr[:, i]), i] = med if np.isfinite(med) else 0.0
                    X_te[np.isnan(X_te[:, i]), i] = med if np.isfinite(med) else 0.0
                m.fit(X_tr, y_tr)
                auc = safe_auc(y_te, m.predict_proba(X_te)[:, 1])
                univ_per_ds[tgt_ds] = round(auc, 4) if auc else None
    except Exception as e:
        print(f"  Warning: universal LODO failed ({e})")
        univ_per_ds = {ds: None for ds in all_dfs}

    rows = []
    for ds in ["ott_2017", "tesla_2020", "sahin_2017", "hilf_2019", "rojas_2023"]:
        rl = RL_RESULTS.get(ds, {})
        cancer = DS_CANCER.get(ds, "?")
        rows.append({
            "dataset": ds,
            "cancer_type": cancer,
            "binding_only": rl.get("binding_only"),
            "rl_tcr_v1": rl.get("rl_tcr_v1"),
            "rl_expression_v1": rl.get("rl_expression_v1"),
            "binding_plus_calis": rl.get("binding_plus_calis"),
            "stage2_cancer_lopo": stage2_lopo.get(ds),
            "universal_lodo_AUC": univ_per_ds.get(ds),   # train on all others
        })

    df_table = pd.DataFrame(rows)
    print(df_table.to_string(index=False))
    return df_table, univ_per_ds


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("PHASE 4 — CROSS-CANCER TRANSFER EVALUATION")
    print("=" * 60)

    cross_results = cross_cancer_transfer()
    lodo_results = within_melanoma_lodo()
    paper_table, univ_per_ds = build_paper_table(cross_results, lodo_results)

    # ── Save ────────────────────────────────────────────────────────────────
    (ARTIFACTS / "cross_cancer_transfer.json").write_text(
        json.dumps(cross_results, indent=2), encoding="utf-8")
    (ARTIFACTS / "within_melanoma_lodo.json").write_text(
        json.dumps(lodo_results, indent=2), encoding="utf-8")
    paper_table.to_csv(ARTIFACTS / "paper_comparison_table.csv", index=False)
    (ARTIFACTS / "universal_lodo_per_dataset.json").write_text(
        json.dumps(univ_per_ds, indent=2), encoding="utf-8")

    print(f"\nArtifacts saved to {ARTIFACTS}/")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    print("\nUniversal model LODO (train on all others → test on target):")
    for ds, auc in univ_per_ds.items():
        rl_bind = RL_RESULTS.get(ds, {}).get("binding_only", "N/A")
        print(f"  {ds:<22}: {auc}  vs binding_only={rl_bind}")

    print("\nWithin-melanoma LODO:")
    for r in lodo_results:
        print(f"  train={r['train_dataset']} → test={r['test_dataset']}: "
              f"AUC={r['lodo_auc']}  (rl_tcr_v1={r['rl_tcr_v1']})")

    print("\nPhase 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
