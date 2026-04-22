"""
PHASE 8 — True Unbiased Haystack Evaluation.

Uses generated binding_nm from Phase 7b (6,617 mutations, 59.6% of full mutanome).
All three scorers evaluated within each patient on the full set of mutations.

Scorers:
  1. binding_only: -log(binding_nm), applicable to 6,617 mutations
  2. rl_tcr_v1: only 83 labeled rows have TCR/expression features
  3. melanoma_ml_v1: Stage 2 XGBoost (with feature imputation on unlabeled rows)

Metrics: per-patient recall@K (K=10,50,100), rank of immunogenic mutations,
         AUC on 83 labeled subset with binding coverage.

Run: py -3.11 backend/strategy_engine/phase8_true_haystack.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y, s = np.array(y), np.array(s)
    m = np.isfinite(y) & np.isfinite(s) & np.isin(y, [0, 1])
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), s[m]))


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(-np.array(scores))[:k]
    return float(np.array(y)[idx].sum() / max(np.array(y).sum(), 1))


def median_rank_of_positives(y: np.ndarray, scores: np.ndarray) -> float | None:
    ranks = pd.Series(scores).rank(ascending=False, method="min").values
    pos_ranks = ranks[np.array(y) == 1]
    return float(np.median(pos_ranks)) if len(pos_ranks) else None


def rl_tcr_v1_score(df: pd.DataFrame) -> pd.Series:
    def _n(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=s.index)

    # Use generated binding_nm where available, fall back to existing binding data
    bind_nm = pd.to_numeric(df.get("binding_nm_generated", pd.Series(np.nan, index=df.index)), errors="coerce")
    bind_pub = pd.to_numeric(df.get("binding_affinity_baseline_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    best_bind = bind_nm.where(bind_nm.notna(), bind_pub)

    present = _norm(-np.log10(best_bind.replace(0, np.nan)))
    if "presentation_score" in df.columns:
        ps = pd.to_numeric(df["presentation_score"], errors="coerce")
        if ps.notna().sum() > 5:
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
        (_norm(tcr_chg.fillna(0)) if tcr_chg.notna().sum() > 5 else pd.Series(0.0, index=df.index), 0.20),
        (_norm(tcr_hyd.fillna(0)) if tcr_hyd.notna().sum() > 5 else pd.Series(0.0, index=df.index), 0.20),
    ]
    return sum(v * w for v, w in terms)


def main() -> int:
    print("=" * 70)
    print("PHASE 8 — TRUE UNBIASED HAYSTACK (Ott full mutanome + generated binding)")
    print("=" * 70)

    # Load full mutanome
    df = pd.read_csv("validation_papers/ott_full_mutanome_labeled.csv", low_memory=False)
    print(f"\nFull mutanome: {len(df):,} mutations")

    # Load generated binding predictions
    binding = pd.read_csv(ARTIFACTS / "ott_fullmutanome_binding_generated.csv")
    print(f"Generated binding: {len(binding):,} mutations ({100*len(binding)/len(df):.1f}% coverage)")

    # Merge binding predictions
    df = df.merge(
        binding[["patient_id", "gene", "protein_change", "binding_nm_generated", "best_peptide"]],
        on=["patient_id", "gene", "protein_change"],
        how="left",
    )
    df["binding_log_generated"] = np.log10(pd.to_numeric(df["binding_nm_generated"], errors="coerce").replace(0, np.nan))

    # Score 1: binding_only (generated binding)
    df["_bind_score"] = -pd.to_numeric(df["binding_log_generated"], errors="coerce")

    # Score 2: rl_tcr_v1 (hybrid: generated binding for presentation, TCR only for labeled rows)
    df["_rl_score"] = rl_tcr_v1_score(df).values

    # Score 3: melanoma_ml_v1 (XGBoost, NaN imputation for unlabeled rows)
    obj = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    model, feats = obj["model"], obj["features"]
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").fillna(0).values if f in df.columns
        else np.zeros(len(df))
        for f in feats
    ])
    for i in range(X.shape[1]):
        med = np.nanmedian(X[:, i])
        X[np.isnan(X[:, i]), i] = med if np.isfinite(med) else 0.0
    df["_ml_score"] = model.predict_proba(X)[:, 1]

    # Per-patient evaluation
    print("\n" + "=" * 70)
    print("PER-PATIENT RANKING (full mutanome, immunogenic rank within patient)")
    print("=" * 70)

    per_patient_results = []
    for pt, grp in df.groupby("patient_id"):
        n_total = len(grp)
        labeled = grp[grp["immunogenic"].isin([0, 1])]
        n_labeled = len(labeled)
        n_pos = int((labeled["immunogenic"] == 1).sum())

        print(f"\n{pt}: {n_total:,} total mutations, {n_labeled} labeled ({n_pos} immunogenic)")

        if n_pos == 0:
            continue

        # Per-patient binding coverage
        bind_cov = grp["binding_nm_generated"].notna().sum()
        print(f"  Binding coverage: {bind_cov:,}/{n_total:,} ({100*bind_cov/n_total:.1f}%)")

        # Rank immunogenic mutations within the full patient set
        for scorer, col in [("binding_only", "_bind_score"), ("rl_tcr_v1", "_rl_score"), ("melanoma_ml_v1", "_ml_score")]:
            scores = grp[col].values
            # Handle NaN by ranking NaN lowest (tied at last place)
            valid = np.isfinite(scores)
            ranked_scores = np.where(valid, scores, -np.inf)
            ranks = pd.Series(ranked_scores).rank(ascending=False, method="min").values

            imm_idx = labeled[labeled["immunogenic"] == 1].index - grp.index[0]
            imm_ranks = [int(ranks[i]) for i in imm_idx if 0 <= i < len(ranks)]
            r10 = recall_at_k(labeled["immunogenic"].values.astype(int),
                              grp.loc[labeled.index, col].values, 10)
            r50 = recall_at_k(labeled["immunogenic"].values.astype(int),
                              grp.loc[labeled.index, col].values, 50)

            genes_ranked = []
            for i in imm_idx:
                if 0 <= i < len(grp):
                    row = grp.iloc[i]
                    genes_ranked.append(f"{row['gene']}:{int(ranks[i])}")

            print(f"  {scorer:<20} R@10={r10:.3f} R@50={r50:.3f} | {', '.join(genes_ranked)}")

        per_patient_results.append({
            "patient_id": pt,
            "n_total": n_total,
            "n_labeled": n_labeled,
            "n_positive": n_pos,
            "binding_coverage": int(bind_cov),
        })

    # AUC on labeled subset with binding coverage (unbiased comparison)
    print("\n" + "=" * 70)
    print("AUC ON LABELED SUBSET (binding_nm available, unbiased comparison)")
    print("=" * 70)
    labeled_all = df[df["immunogenic"].isin([0, 1]) & df["binding_nm_generated"].notna()].copy()
    print(f"Labeled with binding: {len(labeled_all)}/83 ({int(labeled_all['immunogenic'].sum())} positive)")

    results_auc = {}
    for scorer, col in [("binding_only", "_bind_score"), ("rl_tcr_v1", "_rl_score"), ("melanoma_ml_v1", "_ml_score")]:
        y = labeled_all["immunogenic"].values.astype(int)
        s = labeled_all[col].values
        auc = safe_auc(y, s)
        r10 = recall_at_k(y, s, 10)
        r20 = recall_at_k(y, s, 20)
        print(f"  {scorer:<20} AUC={auc:.4f}  R@10={r10:.3f}  R@20={r20:.3f}")
        results_auc[scorer] = {"auc": round(auc, 4), "recall_at_10": round(r10, 3), "recall_at_20": round(r20, 3)}

    # Save
    out = {
        "dataset": "ott_2017_full_mutanome_phase8",
        "n_mutations": len(df),
        "n_labeled": int(df["immunogenic"].isin([0, 1]).sum()),
        "n_positive": int((df["immunogenic"] == 1).sum()),
        "binding_coverage": int(df["binding_nm_generated"].notna().sum()),
        "per_patient": per_patient_results,
        "labeled_auc": results_auc,
    }
    out_path = ARTIFACTS / "phase8_true_haystack_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("\nPhase 8 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
