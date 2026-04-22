"""
PHASE 2 — Expanded Negatives: NCI Filtering.

Creates two NCI versions:
  V1 — full mutanome (presentation model training, all 292K rows)
  V2 — pre-screened equivalent (binding_nm < 500nM, for Stage 2 training)

Run: py -3.11 backend/strategy_engine/filter_nci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_nci_features() -> pd.DataFrame:
    """Load NCI from the Phase 1 feature matrix if available, else raw TSV."""
    feat_path = ARTIFACTS / "muller_nci_features.csv"
    if feat_path.exists():
        print(f"Loading from Phase 1 feature matrix: {feat_path}")
        return pd.read_csv(feat_path, low_memory=False)
    # Fallback to raw TSV
    raw = next(Path("validation_papers/muller2023").glob("*.tsv"), None)
    if raw is None:
        raise FileNotFoundError("Müller NCI data not found.")
    print(f"Loading raw TSV: {raw}")
    df = pd.read_csv(raw, sep="\t", low_memory=False)
    df = df.rename(columns={
        "PatientID": "patient_id", "MT_pep_x": "mutant_peptide",
        "HLA_type_x": "hla_allele", "MT_BindAff": "binding_nm",
        "BindStab": "binding_stability", "Agretopicity": "dai_agretopicity",
        "Quantification": "expression_tpm", "Score_EL": "presentation_score_el",
        "VALIDATED": "immunogenic",
    })
    df["immunogenic"] = pd.to_numeric(df["immunogenic"], errors="coerce")
    df = df[df["immunogenic"].isin([0, 1])].copy()
    df["immunogenic"] = df["immunogenic"].astype(int)
    return df


def report_binding_distribution(df: pd.DataFrame, label: str) -> None:
    nm = pd.to_numeric(df["binding_nm"], errors="coerce")
    pos = nm[df["immunogenic"] == 1]
    neg = nm[df["immunogenic"] == 0]
    print(f"\n  {label} binding distribution (nM):")
    print(f"    Immunogenic (n={len(pos)}): "
          f"median={pos.median():.1f}  p10={pos.quantile(0.1):.1f}  "
          f"p90={pos.quantile(0.9):.1f}")
    print(f"    Non-immuno  (n={len(neg)}): "
          f"median={neg.median():.1f}  p10={neg.quantile(0.1):.1f}  "
          f"p90={neg.quantile(0.9):.1f}")
    pct_sub500 = float((nm < 500).mean())
    print(f"    % total rows with binding_nm < 500: {pct_sub500:.1%}")


def main() -> int:
    print("=" * 60)
    print("PHASE 2 — NCI FILTERING")
    print("=" * 60)

    # Step 2A: Load NCI data
    df = load_nci_features()
    nm = pd.to_numeric(df["binding_nm"], errors="coerce")
    print(f"\n[2A] Loaded: {len(df):,} rows | "
          f"{int(df['immunogenic'].sum())} immunogenic | "
          f"{df['patient_id'].nunique()} patients")

    # Step 2B: Full statistics
    print("\n[2B] Dataset statistics:")
    if "cancer_type" in df.columns:
        print(f"  Cancer types: {df['cancer_type'].value_counts().to_dict()}")
    report_binding_distribution(df, "Full mutanome")

    # Step 2C: Create two versions

    # V1 — Full mutanome
    v1_path = ARTIFACTS / "nci_full_mutanome.csv"
    df.to_csv(v1_path, index=False)
    print(f"\n[2C] Version 1 (full mutanome): {len(df):,} rows → {v1_path}")

    # V2 — Pre-screened equivalent
    # Try 500nM first, adjust if needed
    thresholds = [500, 1000, 250]
    v2 = None
    used_threshold = None
    for thresh in thresholds:
        candidate = df[nm < thresh].copy()
        n = len(candidate)
        n_pos = int(candidate["immunogenic"].sum())
        print(f"  Filter <{thresh}nM: {n:,} rows, {n_pos} immunogenic")
        if 5_000 <= n <= 150_000 and n_pos >= 50:
            v2 = candidate
            used_threshold = thresh
            break

    if v2 is None:
        # Use 500nM regardless
        v2 = df[nm < 500].copy()
        used_threshold = 500
        print(f"  Using <500nM threshold: {len(v2):,} rows")

    v2_path = ARTIFACTS / "nci_prescreened_equivalent.csv"
    v2.to_csv(v2_path, index=False)
    print(f"\nVersion 2 (pre-screened equiv, binding<{used_threshold}nM): "
          f"{len(v2):,} rows → {v2_path}")

    # Step 2D: Report V2 stats
    n_pos_v2 = int(v2["immunogenic"].sum())
    n_neg_v2 = len(v2) - n_pos_v2
    ratio = n_neg_v2 / n_pos_v2 if n_pos_v2 > 0 else float("inf")
    print(f"\n[2D] Version 2 statistics:")
    print(f"  Rows: {len(v2):,}")
    print(f"  Immunogenic: {n_pos_v2}")
    print(f"  Non-immunogenic: {n_neg_v2:,}")
    print(f"  Positive:negative ratio: 1:{ratio:.0f}")
    report_binding_distribution(v2, "Pre-screened equiv")

    # Compare to pre-screened cohort ratios
    print("\n  Pre-screened cohort comparison:")
    comparisons = [
        ("ott_2017", 15, 82),
        ("tesla_2020", 41, 877),
        ("hilf_2019", 77, 75),
        ("sahin_2017", 109, 56),
    ]
    for name, pos, neg in comparisons:
        r = neg / pos if pos > 0 else float("inf")
        print(f"    {name:<20}: 1:{r:.1f}")

    # Step 2E: NCI pre-screened already has Phase 1 features (from feature matrix)
    # Just confirm the key features are present
    needed = ["binding_nm", "binding_log", "presentation_score_el",
              "binding_stability", "expression_tpm", "pep_length"]
    present = [c for c in needed if c in v2.columns and
               pd.to_numeric(v2[c], errors="coerce").notna().sum() > 10]
    missing = [c for c in needed if c not in present]
    print(f"\n[2E] Stage 1 features in V2:")
    print(f"  Present: {present}")
    print(f"  Missing: {missing}")

    # Save summary
    summary = {
        "v1_path": str(v1_path),
        "v1_rows": len(df),
        "v1_immunogenic": int(df["immunogenic"].sum()),
        "v2_path": str(v2_path),
        "v2_rows": len(v2),
        "v2_immunogenic": n_pos_v2,
        "v2_threshold_nm": used_threshold,
        "v2_pos_neg_ratio": f"1:{ratio:.0f}",
        "stage1_features_present": present,
        "stage1_features_missing": missing,
    }
    out_json = ARTIFACTS / "nci_filter_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved: {out_json}")
    print("\nPhase 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
