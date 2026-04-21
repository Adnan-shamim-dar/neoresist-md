"""
PHASE 6 — Haystack Evaluation.

Within each melanoma patient, rank ALL their candidates by three scorers:
  1. binding_only   (binding_log, lower nm = higher rank)
  2. rl_tcr_v1      (hand-crafted, from strategy registry)
  3. melanoma_ml_v1 (XGBoost Stage 2 model, LOPO AUC 0.760)

Metrics: per-patient AUC, recall@10, recall@20, median rank of immunogenic.
Datasets: ott_2017 + sahin_2017 (melanoma).

Run: py -3.11 backend/strategy_engine/haystack.py
"""
from __future__ import annotations
import json, sys, warnings
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


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / f"{name}_features.csv", low_memory=False)


def safe_auc(y, s):
    y, s = np.array(y), np.array(s)
    m = np.isfinite(y) & np.isfinite(s) & np.isin(y, [0, 1])
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), s[m]))


def recall_at_k(y, scores, k):
    idx = np.argsort(-np.array(scores))[:k]
    return float(np.array(y)[idx].sum() / max(np.array(y).sum(), 1))


def median_rank(y, scores):
    ranks = pd.Series(scores).rank(ascending=False, method="min").values
    pos_ranks = ranks[np.array(y) == 1]
    return float(np.median(pos_ranks)) if len(pos_ranks) else None


def bootstrap_ci(y, s, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    y, s = np.array(y), np.array(s)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        yt, sp = y[idx], s[idx]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt.astype(int), sp))
    return ([round(float(np.percentile(aucs, 2.5)), 4),
             round(float(np.percentile(aucs, 97.5)), 4)]
            if len(aucs) >= 10 else [None, None])


def ml_score(model, feats, df):
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").values if f in df.columns
        else np.zeros(len(df))   # presentation_probability: ~0 for pre-screened cohorts
        for f in feats
    ])
    for i in range(X.shape[1]):
        col = X[:, i]
        valid = col[np.isfinite(col)]
        X[~np.isfinite(col), i] = np.median(valid) if len(valid) else 0.0
    return model.predict_proba(X)[:, 1]


def rl_tcr_v1_score(df: pd.DataFrame) -> pd.Series:
    """
    rl_tcr_v1 weights from configs/scoring_profiles/rl_tcr_v1.yaml:
      presentation 0.40, tcr_volume 0.20, tcr_charge_diff 0.20, tcr_hydrophobicity 0.20
    Falls back gracefully when WT-derived features are absent (Sahin: no WT peptide).
    """
    def _n(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=s.index)

    # presentation: use binding_nm inverted (higher score = lower nM)
    present = _norm(-_n("binding_log"))  # binding_log is log(nm), invert for presentation
    # try actual presentation_score first
    if "presentation_score" in df.columns:
        ps = pd.to_numeric(df["presentation_score"], errors="coerce")
        if ps.notna().mean() > 0.5:
            present = _norm(ps)

    # tcr_volume: diff preferred (WT present), mean as fallback
    tcr_vol = _n("tcr_volume_diff")
    if tcr_vol.isna().mean() > 0.5:
        tcr_vol = _n("tcr_volume_mean")

    # tcr_charge_diff: WT required; zero contribution if absent
    tcr_chg = _n("tcr_charge_diff")

    # tcr_hydrophobicity: diff preferred, mean fallback
    tcr_hyd = _n("tcr_hydrophobicity_diff")
    if tcr_hyd.isna().mean() > 0.5:
        tcr_hyd = _n("tcr_hydro_diff")
    if tcr_hyd.isna().mean() > 0.5:
        tcr_hyd = _n("tcr_hydro_mean")

    terms = [
        (present, 0.40),
        (_norm(tcr_vol.fillna(0)), 0.20),
        (_norm(tcr_chg.fillna(0)) if tcr_chg.notna().mean() > 0.1 else pd.Series(0.0, index=df.index), 0.20),
        (_norm(tcr_hyd.fillna(0)) if tcr_hyd.notna().mean() > 0.1 else pd.Series(0.0, index=df.index), 0.20),
    ]
    s = sum(v * w for v, w in terms)
    return s


def lopo_ml_scores(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """LOPO predictions: for each patient, train on all others, predict that patient."""
    try:
        from xgboost import XGBClassifier
        def _make(spw):
            return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=42, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        def _make(spw):
            return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                              learning_rate=0.1, random_state=42)

    scores = np.full(len(df), np.nan)
    patients = df["patient_id"].unique()
    for pt in patients:
        tr = df[df["patient_id"] != pt]
        te = df[df["patient_id"] == pt]
        y_tr = tr["immunogenic"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = _make(spw)
        X_tr = np.column_stack([
            pd.to_numeric(tr[f], errors="coerce").fillna(0).values if f in tr.columns
            else np.zeros(len(tr)) for f in feats])
        X_te = np.column_stack([
            pd.to_numeric(te[f], errors="coerce").fillna(0).values if f in te.columns
            else np.zeros(len(te)) for f in feats])
        for i in range(X_tr.shape[1]):
            med = np.nanmedian(X_tr[:, i])
            X_tr[np.isnan(X_tr[:, i]), i] = med if np.isfinite(med) else 0.0
            X_te[np.isnan(X_te[:, i]), i] = med if np.isfinite(med) else 0.0
        m.fit(X_tr, y_tr)
        scores[te.index - df.index[0]] = m.predict_proba(X_te)[:, 1]
    return scores


def evaluate_dataset(name: str, model, feats: list[str]) -> dict:
    df = load(name)
    df = df[df["immunogenic"].isin([0, 1])].reset_index(drop=True).copy()
    df["_bind_score"] = -pd.to_numeric(df["binding_log"], errors="coerce")
    df["_rl_score"]   = rl_tcr_v1_score(df).values
    # Use LOPO predictions to avoid in-sample leakage
    print(f"    running LOPO for {name} ({df['patient_id'].nunique()} patients)...")
    df["_ml_score"] = lopo_ml_scores(df, feats)

    results_per_pt = []
    for pt, grp in df.groupby("patient_id"):
        y = grp["immunogenic"].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        row = {"patient_id": pt, "n": len(grp), "n_pos": int(y.sum())}
        for scorer, col in [("binding_only", "_bind_score"),
                             ("rl_tcr_v1",    "_rl_score"),
                             ("melanoma_ml_v1", "_ml_score")]:
            s = grp[col].values
            row[f"{scorer}_auc"]        = round(safe_auc(y, s), 4) if safe_auc(y, s) else None
            row[f"{scorer}_recall@10"]  = round(recall_at_k(y, s, 10), 3)
            row[f"{scorer}_recall@20"]  = round(recall_at_k(y, s, 20), 3)
            row[f"{scorer}_median_rank"] = median_rank(y, s)
        results_per_pt.append(row)

    # aggregate
    agg = {}
    for scorer in ["binding_only", "rl_tcr_v1", "melanoma_ml_v1"]:
        y_all = df["immunogenic"].values.astype(int)
        col_map = {"binding_only": "_bind_score", "rl_tcr_v1": "_rl_score",
                   "melanoma_ml_v1": "_ml_score"}
        s_all = pd.to_numeric(df[col_map[scorer]], errors="coerce").fillna(0).values
        auc = safe_auc(y_all, s_all)
        ci  = bootstrap_ci(y_all, s_all)
        patient_aucs = [r[f"{scorer}_auc"] for r in results_per_pt if r[f"{scorer}_auc"] is not None]
        agg[scorer] = {
            "pooled_auc": round(auc, 4) if auc else None,
            "pooled_auc_ci": ci,
            "mean_patient_auc": round(float(np.mean(patient_aucs)), 4) if patient_aucs else None,
            "mean_recall_at_10": round(float(np.mean([r[f"{scorer}_recall@10"] for r in results_per_pt])), 3),
            "mean_recall_at_20": round(float(np.mean([r[f"{scorer}_recall@20"] for r in results_per_pt])), 3),
        }
    return {"dataset": name, "n_patients": len(results_per_pt),
            "n_rows": len(df), "n_positive": int(df["immunogenic"].sum()),
            "per_patient": results_per_pt, "aggregate": agg}


def main() -> int:
    print("=" * 60)
    print("PHASE 6 — HAYSTACK EVALUATION (melanoma)")
    print("=" * 60)

    obj = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    model, feats = obj["model"], obj["features"]
    # presentation_probability not in saved CSVs; clinical cohorts output ~0 (domain gap)
    # inject as zero column so feature count matches the trained model

    all_results = {}
    for ds in ["ott_2017", "sahin_2017"]:
        print(f"\n  [{ds}]")
        r = evaluate_dataset(ds, model, feats)
        all_results[ds] = r
        agg = r["aggregate"]
        print(f"  {'Scorer':<20} {'Pooled AUC':>10}  {'CI':>18}  "
              f"{'Mean pt AUC':>12}  {'R@10':>6}  {'R@20':>6}")
        for scorer in ["binding_only", "rl_tcr_v1", "melanoma_ml_v1"]:
            a = agg[scorer]
            ci = a["pooled_auc_ci"]
            print(f"  {scorer:<20} {str(a['pooled_auc']):>10}  "
                  f"[{ci[0]},{ci[1]}]  "
                  f"{str(a['mean_patient_auc']):>12}  "
                  f"{a['mean_recall_at_10']:>6.3f}  "
                  f"{a['mean_recall_at_20']:>6.3f}")

    out = ARTIFACTS / "haystack_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")

    # combined summary
    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY — melanoma_ml_v1 vs baselines")
    print("=" * 60)
    for ds, r in all_results.items():
        print(f"\n  {ds}  (n={r['n_rows']}, pos={r['n_positive']}, pts={r['n_patients']})")
        for scorer in ["binding_only", "rl_tcr_v1", "melanoma_ml_v1"]:
            a = r["aggregate"][scorer]
            print(f"    {scorer:<20}  AUC={a['pooled_auc']}  "
                  f"CI={a['pooled_auc_ci']}  R@10={a['mean_recall_at_10']:.3f}")

    print("\nPhase 6 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
