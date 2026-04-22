"""
PHASE 7 — Binding-Augmented Haystack Evaluation.

On the 83-row Ott labeled subset, compare three baselines:
  1. binding_only (from publication_scored_neoantigens.csv)
  2. rl_tcr_v1    (hand-crafted strategy)
  3. melanoma_ml_v1 (Stage 2 XGBoost LOPO)

Unbiased evaluation: all 83 rows have mutant_peptide + MHCflurry binding_nm.

Run: py -3.11 backend/strategy_engine/phase7_binding_haystack.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Path(__file__).resolve().parents[2] / "backend"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """ROC AUC with guard against edge cases."""
    y, s = np.array(y), np.array(s)
    m = np.isfinite(y) & np.isfinite(s) & np.isin(y, [0, 1])
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), s[m]))


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of positives in top-k predictions."""
    idx = np.argsort(-np.array(scores))[:k]
    return float(np.array(y)[idx].sum() / max(np.array(y).sum(), 1))


def median_rank(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Median rank of positives."""
    ranks = pd.Series(scores).rank(ascending=False, method="min").values
    pos_ranks = ranks[np.array(y) == 1]
    return float(np.median(pos_ranks)) if len(pos_ranks) else None


def load_publication_neoantigens() -> pd.DataFrame:
    """Load publication_scored_neoantigens.csv and filter to Ott."""
    p = Path("backend/validation/artifacts/publication_scored_neoantigens.csv")
    df = pd.read_csv(p, low_memory=False)
    return df[df["paper_source"] == "ott_2017"].copy()


def rl_tcr_v1_score(df: pd.DataFrame) -> pd.Series:
    """
    rl_tcr_v1 weights: presentation 0.40, tcr_volume 0.20, tcr_charge_diff 0.20, tcr_hydrophobicity 0.20.
    """
    def _n(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=s.index)

    present = _norm(-_n("binding_affinity_nm_numeric"))  # inverted: lower nm = higher score
    if "presentation_score" in df.columns:
        ps = pd.to_numeric(df["presentation_score"], errors="coerce")
        if ps.notna().mean() > 0.5:
            present = _norm(ps)

    tcr_vol = _n("tcr_volume_diff")
    if tcr_vol.isna().mean() > 0.5:
        tcr_vol = _n("tcr_volume_mean")

    tcr_chg = _n("tcr_charge_diff")

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
    return sum(v * w for v, w in terms)


def main() -> int:
    print("=" * 70)
    print("PHASE 7 — BINDING-AUGMENTED HAYSTACK (83-row Ott labeled subset)")
    print("=" * 70)

    # Load full mutanome (11,094 rows, 83 labeled)
    full_mutanome = pd.read_csv("validation_papers/ott_full_mutanome_labeled.csv", low_memory=False)
    labeled = full_mutanome[full_mutanome["immunogenic"].isin([0, 1])].copy()
    print(f"\nLoaded: {len(labeled)} labeled mutations from full mutanome")
    print(f"  Positives: {int(labeled['immunogenic'].sum())}")
    print(f"  Negatives: {len(labeled) - int(labeled['immunogenic'].sum())}")

    # Load publication peptide data
    pub_peps = load_publication_neoantigens()
    print(f"\nLoaded: {len(pub_peps)} Ott peptide entries from publication")

    # Map binding_affinity_nm back to mutation level (take best binding per mutation)
    binding_by_mut = pub_peps.groupby(["patient_id", "gene", "protein_change"]).agg({
        "binding_affinity_nm_numeric": "min",  # best (lowest) binding
    }).reset_index()
    binding_by_mut.columns = ["patient_id", "gene", "protein_change", "binding_nm_best"]
    binding_by_mut["binding_log"] = np.log10(binding_by_mut["binding_nm_best"])

    # Merge binding data
    labeled = labeled.merge(
        binding_by_mut,
        on=["patient_id", "gene", "protein_change"],
        how="left",
    )
    coverage = labeled["binding_nm_best"].notna().sum()
    print(f"\nBinding coverage: {coverage}/{len(labeled)} ({100*coverage/len(labeled):.1f}%)")

    # Score with three strategies
    labeled["_bind_score"] = -pd.to_numeric(labeled["binding_log"], errors="coerce")
    labeled["_rl_score"] = rl_tcr_v1_score(labeled).values

    # Load pre-trained melanoma_ml_v1 model for LOPO predictions (from Phase 6)
    import joblib
    obj = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    model, feats = obj["model"], obj["features"]

    # Simple LOPO on labeled subset
    scores = np.full(len(labeled), np.nan)
    for pt in labeled["patient_id"].unique():
        test_idx = labeled[labeled["patient_id"] == pt].index
        train_idx = labeled[labeled["patient_id"] != pt].index

        if len(train_idx) < 10 or len(labeled.loc[train_idx, "immunogenic"].unique()) < 2:
            continue

        y_train = labeled.loc[train_idx, "immunogenic"].values.astype(int)
        X_train = np.column_stack([
            pd.to_numeric(labeled.loc[train_idx, f], errors="coerce").fillna(0).values
            if f in labeled.columns else np.zeros(len(train_idx))
            for f in feats
        ])
        X_test = np.column_stack([
            pd.to_numeric(labeled.loc[test_idx, f], errors="coerce").fillna(0).values
            if f in labeled.columns else np.zeros(len(test_idx))
            for f in feats
        ])

        # Fill NaN with column medians
        for i in range(X_train.shape[1]):
            med = np.nanmedian(X_train[:, i])
            X_train[np.isnan(X_train[:, i]), i] = med if np.isfinite(med) else 0.0
            X_test[np.isnan(X_test[:, i]), i] = med if np.isfinite(med) else 0.0

        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        scores[len(labeled) - len(test_idx):len(labeled) - len(test_idx) + len(test_idx)] = m.predict_proba(X_test)[:, 1]

    labeled["_ml_score"] = scores

    # Evaluate per patient
    results = []
    for pt, grp in labeled.groupby("patient_id"):
        y = grp["immunogenic"].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        row = {"patient_id": pt, "n": len(grp), "n_pos": int(y.sum())}
        for scorer, col in [("binding_only", "_bind_score"), ("rl_tcr_v1", "_rl_score"), ("melanoma_ml_v1", "_ml_score")]:
            s = grp[col].values
            row[f"{scorer}_auc"] = round(safe_auc(y, s), 4) if safe_auc(y, s) else None
            row[f"{scorer}_recall@10"] = round(recall_at_k(y, s, 10), 3)
            row[f"{scorer}_recall@20"] = round(recall_at_k(y, s, 20), 3)
            row[f"{scorer}_median_rank"] = median_rank(y, s)
        results.append(row)

    # Aggregate
    print("\n" + "=" * 70)
    print("PER-PATIENT RESULTS (honest metrics, n >= 2 pos+neg)")
    print("=" * 70)
    print(f"{'Patient':<12} {'N':<6} {'Pos':<4}  {'Scorer':<20} {'AUC':>8}  {'R@10':>6}  {'R@20':>6}  {'Med Rank':>9}")
    for r in results:
        pt = r["patient_id"]
        for scorer in ["binding_only", "rl_tcr_v1", "melanoma_ml_v1"]:
            auc_val = r.get(f"{scorer}_auc")
            auc_str = f"{auc_val:.4f}" if auc_val else "N/A"
            med = r.get(f"{scorer}_median_rank")
            med_str = f"{med:.1f}" if med else "N/A"
            print(f"{pt:<12} {r['n']:<6} {r['n_pos']:<4}  {scorer:<20} {auc_str:>8}  "
                  f"{r[f'{scorer}_recall@10']:>6.3f}  {r[f'{scorer}_recall@20']:>6.3f}  {med_str:>9}")

    # Pooled results
    print("\n" + "=" * 70)
    print("POOLED RESULTS (all labeled mutations)")
    print("=" * 70)
    y_all = labeled["immunogenic"].values.astype(int)
    for scorer, col in [("binding_only", "_bind_score"), ("rl_tcr_v1", "_rl_score"), ("melanoma_ml_v1", "_ml_score")]:
        s_all = pd.to_numeric(labeled[col], errors="coerce").fillna(0).values
        auc = safe_auc(y_all, s_all)
        r10 = recall_at_k(y_all, s_all, 10)
        r20 = recall_at_k(y_all, s_all, 20)
        print(f"{scorer:<20}  AUC={auc:.4f}  R@10={r10:.3f}  R@20={r20:.3f}")

    # Save results
    out_results = {
        "dataset": "ott_2017_labeled_subset",
        "n_mutations": len(labeled),
        "n_positive": int(y_all.sum()),
        "binding_coverage": int(coverage),
        "per_patient": results,
        "pooled_auc": {
            "binding_only": round(safe_auc(y_all, pd.to_numeric(labeled["_bind_score"], errors="coerce").fillna(0).values), 4),
            "rl_tcr_v1": round(safe_auc(y_all, labeled["_rl_score"].values), 4),
            "melanoma_ml_v1": round(safe_auc(y_all, labeled["_ml_score"].values), 4),
        }
    }
    out_path = ARTIFACTS / "phase7_binding_haystack_results.json"
    out_path.write_text(json.dumps(out_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("\nPhase 7 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
