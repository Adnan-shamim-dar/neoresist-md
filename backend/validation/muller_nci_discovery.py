"""
Müller NCI 2023 — Verification + Strategy Discovery.

STEP 1: Verify the 0.97 binding AUC — selection-bias check.
STEP 2: Evaluate NCI_S1–S7 strategies with LOPO AUC.
         Pre-committed directions: Score_EL +, BindStab +,
         Agretopicity − (inverted), expression +.

Run: py -3.11 backend/validation/muller_nci_discovery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from backend.validation.full_cross_validation import (
    ARTIFACTS,
    lopo_auc,
    insample_auc,
    make_serializable,
    safe_auc,
    score_rows,
)

TSV = Path("validation_papers/muller2023/muller_nci.tsv")
OUT = ARTIFACTS / "muller_nci_discovery.json"


def load() -> pd.DataFrame:
    df = pd.read_csv(TSV, sep="\t")
    df = df.rename(columns={
        "PatientID": "patient_id",
        "MT_pep_x": "mutant_peptide",
        "HLA_type_x": "hla_allele",
        "MT_BindAff": "nm_raw",
        "Quantification": "expr_raw",
        "VALIDATED": "immunogenic",
    })
    df = df[df["immunogenic"].isin([0, 1])].copy()
    for col in ["Score_EL", "BindStab", "Agretopicity", "expr_raw", "nm_raw", "ln_NumTested"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # bind_log50k for backward-compat strategies
    nm = df["nm_raw"]
    df["bind_log50k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
    return df


def dist_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "pct_above_500nm": round(float((series.dropna() > 500).mean()), 4) if "nm" in series.name.lower() else None,
    }


def per_patient_auc(df: pd.DataFrame, feat_col: str, inv: bool = False) -> dict:
    results = {}
    for pt in df["patient_id"].unique():
        sub = df[df["patient_id"] == pt]
        y = sub["immunogenic"].values.astype(float)
        if len(np.unique(y)) < 2:
            results[str(pt)] = None
            continue
        vals = sub[feat_col].values.astype(float)
        if inv:
            vals = -vals
        m = np.isfinite(vals) & np.isfinite(y)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            results[str(pt)] = None
            continue
        auc = float(roc_auc_score(y[m], vals[m]))
        results[str(pt)] = round(auc, 4)
    return results


def main() -> int:
    print(f"Loading {TSV}...")
    df = load()
    n_pos = int(df["immunogenic"].sum())
    n_pat = int(df["patient_id"].nunique())
    print(f"Rows: {len(df):,} | Positives: {n_pos} | Patients: {n_pat}")

    # ── STEP 1: Binding distribution verification ──────────────────────────
    print("\n=== STEP 1: Binding Distribution Verification ===")
    pos = df[df["immunogenic"] == 1]
    neg = df[df["immunogenic"] == 0]

    # nm_raw distribution
    neg_nm_median = float(neg["nm_raw"].median())
    pos_nm_median = float(pos["nm_raw"].median())
    neg_pct_below500 = float((neg["nm_raw"].dropna() < 500).mean())
    pos_pct_below500 = float((pos["nm_raw"].dropna() < 500).mean())
    print(f"  Negative nm_raw median: {neg_nm_median:.1f}  pct<500nM: {neg_pct_below500:.3f}")
    print(f"  Positive nm_raw median: {pos_nm_median:.1f}  pct<500nM: {pos_pct_below500:.3f}")

    # Score_EL distribution
    neg_el_median = float(neg["Score_EL"].median())
    pos_el_median = float(pos["Score_EL"].median())
    print(f"  Negative Score_EL median: {neg_el_median:.4f}")
    print(f"  Positive Score_EL median: {pos_el_median:.4f}")

    # Selection bias marker: pct of negatives already binding below 500nM
    bias_flag = neg_pct_below500 > 0.3
    print(f"  Selection-bias flag (>30% negatives <500nM): {bias_flag} ({neg_pct_below500:.1%})")

    # Per-patient AUC for Score_EL (top 10 by n_rows)
    top10_pts = (
        df.groupby("patient_id").size()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    per_pt_el = per_patient_auc(df, "Score_EL")
    per_pt_nm = per_patient_auc(df, "nm_raw", inv=True)  # lower nm = better
    print("\n  Per-patient Score_EL AUC (top-10 by n_rows):")
    for pt in top10_pts:
        el = per_pt_el.get(str(pt))
        nm_ = per_pt_nm.get(str(pt))
        el_s = f"{el:.4f}" if el is not None else "  N/A"
        nm_s = f"{nm_:.4f}" if nm_ is not None else "  N/A"
        n = int((df["patient_id"] == pt).sum())
        n_p = int(df[df["patient_id"] == pt]["immunogenic"].sum())
        print(f"    {pt}  n={n:5d}  pos={n_p}  EL_AUC={el_s}  NM_AUC={nm_s}")

    # ── STEP 2: Strategy Discovery ─────────────────────────────────────────
    print("\n=== STEP 2: Strategy Discovery NCI_S1–S7 ===")
    print("Pre-committed directions: Score_EL+ BindStab+ Agretopicity- expression+\n")

    strategies = {
        "NCI_S1_score_el_only": [
            ("Score_EL", 1.0, False),
        ],
        "NCI_S2_el_bindstab": [
            ("Score_EL", 0.6, False),
            ("BindStab", 0.4, False),
        ],
        "NCI_S3_el_agretopicity": [
            ("Score_EL", 0.6, False),
            ("Agretopicity", 0.4, True),   # inverted: lower agretopicity = more foreign
        ],
        "NCI_S4_el_expression": [
            ("Score_EL", 0.6, False),
            ("expr_raw", 0.4, False),
        ],
        "NCI_S5_el_bindstab_agret": [
            ("Score_EL", 0.4, False),
            ("BindStab", 0.3, False),
            ("Agretopicity", 0.3, True),
        ],
        "NCI_S6_el_bindstab_expr": [
            ("Score_EL", 0.4, False),
            ("BindStab", 0.3, False),
            ("expr_raw", 0.3, False),
        ],
        "NCI_S7_el_bindstab_agret_expr": [
            ("Score_EL", 0.4, False),
            ("BindStab", 0.2, False),
            ("Agretopicity", 0.2, True),
            ("expr_raw", 0.2, False),
        ],
    }

    strat_results = {}
    for sn, fl in strategies.items():
        avail = [f for f, _, _ in fl if f in df.columns and
                 pd.to_numeric(df[f], errors="coerce").notna().sum() > 10]
        missing = [f for f, _, _ in fl if f not in avail]
        is_auc = insample_auc(df, fl)
        lopo, per_pat = lopo_auc(df, fl)
        strat_results[sn] = {
            "lopo_auc": lopo,
            "insample_auc": is_auc,
            "features_used": avail,
            "features_missing": missing,
            "per_patient_lopo": per_pat,
        }
        lopo_s = f"{lopo:.4f}" if lopo is not None else "  N/A"
        is_s = f"{is_auc:.4f}" if is_auc is not None else "  N/A"
        print(f"  {sn:<45} LOPO={lopo_s}  in-sample={is_s}  missing={missing}")

    # ── Assemble output ────────────────────────────────────────────────────
    lopo_vals = {k: v["lopo_auc"] for k, v in strat_results.items() if v["lopo_auc"] is not None}
    best_strat = max(lopo_vals, key=lopo_vals.__getitem__) if lopo_vals else None

    output = {
        "dataset": "muller_nci_2023",
        "source_file": TSV.name,
        "n_rows": len(df),
        "n_immunogenic": n_pos,
        "n_patients": n_pat,
        "step1_binding_verification": {
            "selection_bias_note": (
                "NCI negatives are HLA-binding-screened. A large fraction of negatives "
                "are already strong binders — this inflates binding AUC vs unscreened cohorts."
            ),
            "negatives": {
                "nm_raw_median": round(neg_nm_median, 2),
                "pct_below_500nm": round(neg_pct_below500, 4),
                "score_el_median": round(neg_el_median, 6),
            },
            "positives": {
                "nm_raw_median": round(pos_nm_median, 2),
                "pct_below_500nm": round(pos_pct_below500, 4),
                "score_el_median": round(pos_el_median, 6),
            },
            "bias_flag_negatives_gt30pct_below500nm": bias_flag,
            "interpretation": (
                "High binding AUC is partially artifactual: screened negatives cluster "
                "near binders. LOPO AUC on per-patient basis is the publishable metric."
            ),
            "per_patient_score_el_auc": {k: v for k, v in per_pt_el.items() if v is not None},
            "per_patient_nm_auc_inv": {k: v for k, v in per_pt_nm.items() if v is not None},
            "top10_patients_by_nrows": top10_pts,
        },
        "step2_strategy_discovery": {
            "pre_committed_directions": {
                "Score_EL": "higher=better (inv=False)",
                "BindStab": "higher=better (inv=False)",
                "Agretopicity": "lower=better (inv=True — high ratio means less foreign)",
                "expr_raw": "higher=better (inv=False)",
            },
            "note": "No optimization. Directions pre-committed before AUC computation.",
            "best_strategy": best_strat,
            "best_lopo_auc": lopo_vals.get(best_strat) if best_strat else None,
            "strategies": make_serializable(strat_results),
        },
    }

    OUT.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT}")
    if best_strat:
        print(f"Best strategy: {best_strat} LOPO={lopo_vals[best_strat]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
