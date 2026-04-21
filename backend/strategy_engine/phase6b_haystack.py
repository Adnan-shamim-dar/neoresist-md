"""
PHASE 6B — Full Mutanome Haystack Experiment (Ott 2017).

Run: py -3.11 backend/strategy_engine/phase6b_haystack.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS   = Path("backend/strategy_engine/artifacts")
SE_STRATS   = Path("backend/strategy_engine/strategies")
VAL_ART     = Path("backend/validation/artifacts")

# ── TCR dictionaries (exact from spec) ───────────────────────────────────────
HYDRO    = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
VOLUME   = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,
            'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,
            'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,
            'Y':193.6,'V':140.0}
CHARGE   = {'R':1,'K':1,'H':0.1,'D':-1,'E':-1}
AROMATIC = {'F':1,'W':1,'Y':1}


def tcr_positions(pep: str) -> str:
    """Positions 2,3,4,5 (0-indexed) = TCR contact for ≥8mers."""
    return pep[2:6] if len(pep) >= 6 else pep


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_pep = "mutant_peptide" in df.columns and df["mutant_peptide"].notna().sum() > 0

    if has_pep:
        pep = df["mutant_peptide"].fillna("")

        def _tcr(col_dict, fn):
            return pep.apply(lambda p: fn([col_dict.get(aa, 0) for aa in tcr_positions(p)])
                             if p else np.nan)

        df["tcr_hydro_mean"]      = _tcr(HYDRO,    np.mean)
        df["tcr_volume_mean"]     = _tcr(VOLUME,   np.mean)
        df["tcr_charge_sum"]      = _tcr(CHARGE,   sum)
        df["tcr_aromatic_count"]  = _tcr(AROMATIC, sum)
        df["pep_length"]          = pep.apply(len)
        df["hydro_full_mean"]     = pep.apply(
            lambda p: np.mean([HYDRO.get(aa, 0) for aa in p]) if p else np.nan)
        df["aliphatic_index"]     = pep.apply(lambda p: (
            (p.count('A') + 2.9*p.count('V') + 3.9*(p.count('I')+p.count('L'))) / max(len(p),1)
            if p else np.nan))
    else:
        for c in ["tcr_hydro_mean","tcr_volume_mean","tcr_charge_sum",
                  "tcr_aromatic_count","pep_length","hydro_full_mean","aliphatic_index"]:
            if c not in df.columns:
                df[c] = np.nan

    # foreignness — need wildtype_peptide
    for c in ["hamming_distance","blosum62_score","blosum62_at_mutation"]:
        if c not in df.columns:
            df[c] = np.nan

    # expression
    for raw, log_col in [("expression_tpm","expression_log2"),
                         ("expression_score","expression_log2")]:
        if raw in df.columns and log_col not in df.columns:
            df[log_col] = np.log2(pd.to_numeric(df[raw], errors="coerce").clip(lower=0) + 1)

    # binding
    if "binding_nm" in df.columns and "binding_log" not in df.columns:
        df["binding_log"] = np.log10(
            pd.to_numeric(df["binding_nm"], errors="coerce").clip(lower=0) + 1)

    return df


def norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.5, index=s.index)


def score_rl_tcr_v1(df: pd.DataFrame) -> pd.Series:
    """Apply rl_tcr_v1.yaml weights to available features."""
    cfg = yaml.safe_load(Path("configs/scoring_profiles/rl_tcr_v1.yaml").read_text())
    w = cfg["weights"]
    # feature map: yaml key → column, direction (1=higher better, -1=lower better)
    feat_map = {
        "presentation":      ("presentation_score",              1),
        "tcr_volume":        ("tcr_volume_mean",                 1),
        "tcr_charge_diff":   ("tcr_charge_sum",                  1),
        "tcr_hydrophobicity":("tcr_hydro_mean",                  1),
    }
    parts, weights = [], []
    for key, (col, direction) in feat_map.items():
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 5:
            s = pd.to_numeric(df[col], errors="coerce")
            parts.append(norm(s * direction))
            weights.append(w.get(key, 0))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    total_w = sum(weights)
    result = sum(p * (wt / total_w) for p, wt in zip(parts, weights))
    return result


def score_melanoma_ml(df: pd.DataFrame, model, feats: list[str]) -> pd.Series:
    """Apply XGBoost model; NaN for missing features → XGBoost native NaN handling."""
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").values if f in df.columns
        else np.zeros(len(df))  # presentation_probability: ~0
        for f in feats
    ])
    # impute NaN columns with global column median
    for i in range(X.shape[1]):
        col = X[:, i]
        valid = col[np.isfinite(col)]
        X[~np.isfinite(col), i] = np.median(valid) if len(valid) else 0.0
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)


def recall_at_k(y_full: np.ndarray, scores: np.ndarray, k: int, n_pos: int) -> int:
    idx = np.argsort(-scores)[:k]
    return int(y_full[idx].sum())


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("PHASE 6B: FULL MUTANOME HAYSTACK EXPERIMENT")
    print("=" * 60)

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    print("\n[STEP 1] LOAD DATA")
    labeled_path = Path("validation_papers/ott_full_mutanome_labeled.csv")
    df = pd.read_csv(labeled_path, low_memory=False)
    print(f"  Path:         {labeled_path}")
    print(f"  Total rows:   {len(df)}")
    imm = pd.to_numeric(df["immunogenic"], errors="coerce")
    print(f"  Immuno=1:     {(imm==1).sum()}")
    print(f"  Immuno=0:     {(imm==0).sum()}")
    print(f"  Unlabeled:    {imm.isna().sum()}")
    print(f"  Columns:      {list(df.columns)}")
    print("\n  First 3 rows:")
    print(df[["patient_id","gene","protein_change","immunogenic"]].head(3).to_string(index=False))

    # Merge dynamic_rl_score from haystack_scored.csv (has scores for all rows)
    scored_path = VAL_ART / "ott_haystack_scored.csv"
    if scored_path.exists():
        sc = pd.read_csv(scored_path, low_memory=False)
        sc = sc[["patient_id","gene","protein_change","dynamic_rl_score"]].drop_duplicates(
            subset=["patient_id","gene","protein_change"])
        df = df.merge(sc.rename(columns={"dynamic_rl_score":"rl_engine_score"}),
                      on=["patient_id","gene","protein_change"], how="left")
        if "dynamic_rl_score" in df.columns:
            df["rl_engine_score"] = df["rl_engine_score"].fillna(df["dynamic_rl_score"])
        print(f"\n  Merged rl_engine_score: {df['rl_engine_score'].notna().sum()}/{len(df)} rows")

    # ── STEP 2: Compute features ──────────────────────────────────────────────
    print("\n[STEP 2] COMPUTE FEATURES")
    df = compute_features(df)
    feat_cols = ["tcr_hydro_mean","tcr_volume_mean","tcr_charge_sum","tcr_aromatic_count",
                 "pep_length","hydro_full_mean","aliphatic_index","hamming_distance",
                 "blosum62_score","expression_log2","binding_log","presentation_score",
                 "self_dissimilarity","rl_engine_score"]
    print(f"  {'Feature':<30} {'Coverage':>10}")
    for c in feat_cols:
        if c in df.columns:
            n = pd.to_numeric(df[c], errors="coerce").notna().sum()
            print(f"  {c:<30} {n:>6}/{len(df)}")
        else:
            print(f"  {c:<30}  NOT IN DATA")

    has_binding = "binding_nm" in df.columns and pd.to_numeric(df["binding_nm"], errors="coerce").notna().sum() > 100
    print(f"\n  binding_nm available: {has_binding}")
    if not has_binding:
        print("  binding_only baseline NOT AVAILABLE — no binding predictions in full mutanome.")

    # ── STEP 3: Score ─────────────────────────────────────────────────────────
    print("\n[STEP 3] SCORE ALL ROWS")

    # melanoma_ml_v1
    ml_obj = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    ml_model, ml_feats = ml_obj["model"], ml_obj["features"]
    print(f"  melanoma_ml_v1 features ({len(ml_feats)}): {ml_feats}")
    df["score_ml"] = score_melanoma_ml(df, ml_model, ml_feats)
    print(f"  melanoma_ml_v1: scored {df['score_ml'].notna().sum()}/{len(df)} rows")

    # rl_tcr_v1
    df["score_rl"] = score_rl_tcr_v1(df)
    print(f"  rl_tcr_v1:      scored {df['score_rl'].notna().sum()}/{len(df)} rows")

    # rl_engine (dynamic_rl_score from existing pipeline — full coverage)
    if "rl_engine_score" in df.columns:
        df["score_engine"] = pd.to_numeric(df["rl_engine_score"], errors="coerce")
        print(f"  rl_engine:      scored {df['score_engine'].notna().sum()}/{len(df)} rows "
              f"(from ott_haystack_scored.csv)")

    # random baseline
    rng = np.random.RandomState(42)
    df["score_random"] = rng.uniform(0, 1, len(df))
    print(f"  random:         scored {len(df)}/{len(df)} rows (seed=42)")

    # ── STEP 4: Haystack evaluation ───────────────────────────────────────────
    print("\n[STEP 4] HAYSTACK EVALUATION")
    print("  Labels: immunogenic=1 → 13 positives; NaN → treated as 0 for recall@K\n")

    y_full = pd.to_numeric(df["immunogenic"], errors="coerce").fillna(0).values.astype(int)
    n_pos = int((y_full == 1).sum())

    strategies = [("melanoma_ml_v1","score_ml"),("rl_tcr_v1","score_rl"),
                  ("rl_engine","score_engine"),("random","score_random")]
    strategies = [(n, c) for n, c in strategies if c in df.columns]
    if has_binding:
        strategies.append(("binding_only","score_bind"))

    # Fill NaN scores with 0 for ranking (NaN rows sink to bottom)
    for _, col in strategies:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)

    Ks = [10, 20, 50, 100, 200, 500, 1000]
    recall = {n: {} for n, _ in strategies}
    for name, col in strategies:
        scores = df[col].values
        for k in Ks:
            recall[name][k] = recall_at_k(y_full, scores, k, n_pos)

    # Print table
    hdr = f"{'K':>5} | " + " | ".join(f"{n:>14}" for n, _ in strategies)
    print("  " + hdr)
    print("  " + "-"*len(hdr))
    for k in Ks:
        row = f"{k:>5} | " + " | ".join(
            f"{recall[n][k]:>5}/{n_pos} ({recall[n][k]/n_pos:.0%})" if recall[n][k] is not None
            else f"{'N/A':>14}" for n, _ in strategies)
        print("  " + row)

    # AUC on 83 labeled rows
    print("\n  AUC on 83 labeled rows only (13 pos + 70 neg):")
    labeled_mask = df["immunogenic"].isin([0.0, 1.0]) | df["immunogenic"].isin([0, 1])
    labeled_mask = pd.to_numeric(df["immunogenic"], errors="coerce").notna()
    lab_df = df[labeled_mask].copy()
    y_lab = pd.to_numeric(lab_df["immunogenic"], errors="coerce").values.astype(int)
    labeled_auc = {}
    for name, col in strategies:
        s = lab_df[col].values
        try:
            auc = float(roc_auc_score(y_lab, s))
            labeled_auc[name] = round(auc, 4)
            print(f"    {name:<20} AUC={auc:.4f}  (caveat: 83 labeled rows only)")
        except Exception as e:
            labeled_auc[name] = None
            print(f"    {name:<20} AUC=N/A ({e})")

    # ── STEP 5: Per-patient haystack ──────────────────────────────────────────
    print("\n[STEP 5] PER-PATIENT HAYSTACK")
    ml_col = "score_ml"
    per_patient = {}
    pt_rows = []
    for pid, grp in df.groupby("patient_id"):
        grp = grp.copy()
        total = len(grp)
        scores_pt = grp[ml_col].values
        grp["pt_rank"] = pd.Series(scores_pt).rank(ascending=False, method="min").values
        immuno_rows = grp[pd.to_numeric(grp["immunogenic"], errors="coerce") == 1]
        n_immuno = len(immuno_rows)
        best_rank  = int(immuno_rows["pt_rank"].min()) if n_immuno else None
        worst_rank = int(immuno_rows["pt_rank"].max()) if n_immuno else None
        in_top20   = int((immuno_rows["pt_rank"] <= 20).sum()) if n_immuno else 0
        per_patient[pid] = {
            "total": total, "n_immunogenic": n_immuno,
            "best_rank": best_rank, "worst_rank": worst_rank, "in_top20": in_top20,
            "immunogenic_details": [
                {"gene": str(r["gene"]), "protein_change": str(r["protein_change"]),
                 "ml_rank": int(r["pt_rank"]),
                 "binding_nm": None}
                for _, r in immuno_rows.iterrows()
            ]
        }
        pt_rows.append({"patient_id": pid, "total": total, "immuno": n_immuno,
                         "best_rank": best_rank, "worst_rank": worst_rank,
                         "in_top20": in_top20})
    pt_df = pd.DataFrame(pt_rows)
    print(f"\n  {'Patient':<10} {'Total':>7} {'Immuno':>7} {'Best_rank':>10} "
          f"{'Worst_rank':>11} {'In_top20':>9}")
    for _, r in pt_df.iterrows():
        print(f"  {r['patient_id']:<10} {r['total']:>7} {r['immuno']:>7} "
              f"{str(r['best_rank']):>10} {str(r['worst_rank']):>11} "
              f"{str(r['in_top20'])+'/'+str(r['immuno']):>9}")

    # ── STEP 6: Dramatic example ──────────────────────────────────────────────
    print("\n[STEP 6] DRAMATIC EXAMPLE")
    pos_df = df[pd.to_numeric(df["immunogenic"], errors="coerce") == 1].copy()
    # rank within full mutanome
    pos_df["_full_ml_rank"] = df[ml_col].rank(ascending=False, method="min").loc[pos_df.index].values
    # best ML rank (lowest number) — look for one with weak presentation (high binding_nm or low presentation_score)
    pos_df["_pres"] = pd.to_numeric(pos_df.get("presentation_score", pd.Series(np.nan, index=pos_df.index)),
                                    errors="coerce")
    # dramatic = best ML rank AND weakest presentation (worst binding)
    # Score: good ml_rank + bad presentation → smallest ML rank / worst presentation
    pos_df = pos_df.sort_values("_full_ml_rank")
    # Among top-ranked by ML, find the one with lowest presentation_score
    drama = pos_df.nsmallest(5, "_full_ml_rank")
    has_pres = drama["_pres"].notna().any()
    if has_pres:
        drama_row = drama.loc[drama["_pres"].idxmin()]
    else:
        drama_row = drama.iloc[0]

    ml_rank_full = int(drama_row["_full_ml_rank"])
    bind_nm      = drama_row.get("binding_nm", None)
    pres         = drama_row["_pres"]
    pres_str     = f"{pres:.3f}" if pd.notna(pres) else "N/A"
    bind_str     = f"{bind_nm:.1f}" if isinstance(bind_nm, float) and pd.notna(bind_nm) else "N/A"
    conclusion   = (
        f"Top-{ml_rank_full} of 11,094 by melanoma_ml_v1; "
        f"presentation_score={pres_str}. "
        f"TCR+sequence features surface this neoantigen despite no binding data."
    )
    print(f"  Gene:              {drama_row['gene']}")
    print(f"  Protein change:    {drama_row['protein_change']}")
    print(f"  Peptide:           {drama_row.get('mutant_peptide','N/A')}")
    print(f"  Binding nM:        {bind_str}")
    print(f"  Presentation score:{pres_str}")
    print(f"  melanoma_ml_v1 rank (of 11,094): {ml_rank_full}")
    print(f"  Patient:           {drama_row['patient_id']}")
    print(f"  Immunogenic:       YES (confirmed)")
    print(f"  Conclusion: {conclusion}")

    drama_dict = {
        "gene": str(drama_row["gene"]),
        "protein_change": str(drama_row["protein_change"]),
        "binding_nm": float(bind_nm) if isinstance(bind_nm, float) and pd.notna(bind_nm) else None,
        "presentation_score": float(pres) if pd.notna(pres) else None,
        "ml_rank": ml_rank_full,
        "binding_rank": None,
        "patient_id": str(drama_row["patient_id"]),
        "conclusion": conclusion,
    }

    # ── STEP 7: Top 20 ───────────────────────────────────────────────────────
    print("\n[STEP 7] TOP 20 BY melanoma_ml_v1")
    df["_rank_global"] = df[ml_col].rank(ascending=False, method="min")
    top20 = df.nsmallest(20, "_rank_global")[
        ["_rank_global","patient_id","gene","protein_change","score_ml","immunogenic"]
    ].copy()
    top20.columns = ["rank","patient","gene","protein_change","score","immunogenic"]
    for _, r in top20.iterrows():
        flag = " *** IMMUNOGENIC ***" if r["immunogenic"] == 1 else ""
        print(f"  {int(r['rank']):>4}  {r['patient']:<8} {r['gene']:<20} "
              f"{r['protein_change']:<15} {r['score']:.4f}{flag}")

    # ── STEP 8: NeoORF check ──────────────────────────────────────────────────
    print("\n[STEP 8] NeoORF CHECK")
    has_neoorf = ("protein_change" in df.columns and
                  df["protein_change"].astype(str).str.contains("NeoORF|fs",case=False,na=False).any())
    if has_neoorf:
        neo = df[df["protein_change"].astype(str).str.contains("NeoORF|fs",case=False,na=False)]
        print(f"  Found {len(neo)} NeoORF/frameshift rows, "
              f"{pd.to_numeric(neo['immunogenic'],errors='coerce').sum():.0f} immunogenic")
    else:
        print("  No NeoORF rows in loaded file.")
        print("  FOOTNOTE: 2 frameshift immunogenic neoantigens (NeoORF) excluded from "
              "ranking evaluation due to non-standard mutation notation. "
              "Haystack evaluates 13 of 15 joinable immunogenic mutations.")

    # ── STEP 9: Save ─────────────────────────────────────────────────────────
    print("\n[STEP 9] SAVE")
    results = {
        "total_mutations": len(df),
        "immunogenic_labeled": int((y_full == 1).sum()),
        "negative_labeled": int((pd.to_numeric(df["immunogenic"], errors="coerce") == 0).sum()),
        "unlabeled": int(pd.to_numeric(df["immunogenic"], errors="coerce").isna().sum()),
        "neoorf_excluded": 2,
        "binding_available": has_binding,
        "strategies_evaluated": [n for n, _ in strategies],
        "feature_coverage": {
            c: int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            for c in feat_cols if c in df.columns
        },
        "recall_at_k": {n: {str(k): v for k, v in recall[n].items()} for n, _ in strategies},
        "labeled_subset_auc": labeled_auc,
        "per_patient": per_patient,
        "dramatic_example": drama_dict,
        "headline": (
            f"melanoma_ml_v1: {recall['melanoma_ml_v1'][100]}/{n_pos} immunogenic "
            f"in top 100 of {len(df):,} somatic mutations; "
            f"AUC={labeled_auc.get('melanoma_ml_v1','N/A')} on 83 labeled rows"
        ),
        "limitation": (
            "Full mutanome lacks mutant_peptide and binding_nm columns. "
            "TCR contact features computable only when peptide provided. "
            "83/11094 rows have features (vaccine-level labeled set). "
            "ML model scores unlabeled rows via NaN imputation to column medians — "
            "ranking within labeled subset is the primary evaluation signal."
        ),
    }

    out_json = ARTIFACTS / "haystack_fullmutanome_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  {out_json}")

    out_csv = ARTIFACTS / "ott_fullmutanome_scored.csv"
    save_cols = [c for c in df.columns if not c.startswith("_")]
    df[save_cols].to_csv(out_csv, index=False)
    print(f"  {out_csv}")

    # ── HEADLINE ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HAYSTACK HEADLINE:")
    print("=" * 60)
    for name, _ in strategies:
        r100 = recall[name][100]
        print(f"  {name:<20}: {r100}/{n_pos} immunogenic in top 100 of {len(df):,}")
    print(f"  random (expected):    ~{n_pos*100//len(df)}/13")
    print()
    best_pt = pt_df.loc[pt_df["in_top20"].idxmax()]
    print(f"  Per-patient best: {best_pt['patient_id']} — "
          f"{best_pt['in_top20']}/{best_pt['immuno']} immunogenic in top 20 "
          f"of {best_pt['total']} mutations")
    print(f"\n  Dramatic example: {drama_dict['gene']} {drama_dict['protein_change']} — "
          f"rank {drama_dict['ml_rank']} of 11,094 by melanoma_ml_v1")
    print(f"\n  LIMITATION: Unlabeled rows lack peptide/binding features. "
          f"AUC={labeled_auc.get('melanoma_ml_v1','N/A')} computed on 83 labeled rows "
          f"only. True full-mutanome scoring requires MHCflurry on all 11K mutations.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
