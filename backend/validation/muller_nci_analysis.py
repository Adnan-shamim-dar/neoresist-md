"""
Analyse Müller 2023 NCI dataset with pre-specified strategies.
No optimization. Pre-specified directions from Ott/TESLA locked.

Run: py -3.11 backend/validation/muller_nci_analysis.py
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

from backend.validation.full_cross_validation import (
    ARTIFACTS,
    compute_all_features,
    insample_auc,
    lopo_auc,
    make_serializable,
)

TSV = Path("validation_papers/muller2023/muller_nci.tsv")
OUT = ARTIFACTS / "muller_nci_results.json"


def load() -> pd.DataFrame:
    df = pd.read_csv(TSV, sep="\t")
    df = df.rename(columns={
        "PatientID": "patient_id",
        "MT_pep_x": "mutant_peptide",
        "HLA_type_x": "hla_allele",
        "MT_BindAff": "binding_affinity_nm_numeric",
        "Quantification": "expression_tpm_numeric",
        "VALIDATED": "immunogenic",
    })
    df = df[df["immunogenic"].isin([0, 1])].copy()
    # Score_EL is NetMHCpan EL score (0-1, higher = stronger) — keep as feature
    return df


def main() -> int:
    print(f"Loading {TSV}...")
    df = load()
    n_pos = int(df["immunogenic"].sum())
    n_pat = int(df["patient_id"].nunique())
    n_rows = len(df)
    print(f"Rows: {n_rows:,} | Positives: {n_pos} | Patients: {n_pat}")

    print("Computing features...")
    feat = compute_all_features(df, "muller_nci")
    # Expose Score_EL as a direct feature
    if "Score_EL" in df.columns:
        feat["score_el"] = pd.to_numeric(df["Score_EL"], errors="coerce")

    # Pre-specified strategies — directions locked from origin datasets
    # No optimization allowed
    strategies = {
        "binding_only": [("bind_log50k", 1.0, False)],
        "score_el_only": [("score_el", 1.0, False)],          # NetMHCpan EL baseline
        "rl_tcr_v1_from_ott": [                               # locked Ott directions
            ("bind_log50k", 0.4, False),
            ("tcr_volume", 0.2, False),
            ("tcr_charge_diff", 0.2, False),                  # NaN (no WT) → excluded
            ("tcr_hydro", 0.2, False),
        ],
        "rl_expression_v1_from_tesla": [                      # locked TESLA directions
            ("bind_log50k", 0.50, False),
            ("nm_raw", 0.25, True),                           # inverted: lower nm = better
            ("expr_raw", 0.25, False),
        ],
        "binding_plus_calis": [
            ("bind_log50k", 0.6, False),
            ("calis_immuno", 0.4, False),
        ],
    }

    results = {}
    for sn, fl in strategies.items():
        avail = [f for f, _, _ in fl if f in feat.columns and
                 pd.to_numeric(feat[f], errors="coerce").notna().sum() > 10]
        missing = [f for f, _, _ in fl if f not in avail]
        is_auc = insample_auc(feat, fl)
        lopo, per_pat = lopo_auc(feat, fl)
        results[sn] = {
            "lopo_auc": lopo,
            "insample_auc": is_auc,
            "features_used": avail,
            "features_missing": missing,
            "per_patient_lopo": per_pat,
        }
        lopo_str = f"{lopo:.4f}" if lopo is not None else "  N/A"
        is_str  = f"{is_auc:.4f}" if is_auc is not None else "  N/A"
        print(f"  {sn:<40} LOPO={lopo_str}  in-sample={is_str}  missing={missing}")

    output = {
        "dataset": "muller_nci_2023",
        "source_file": TSV.name,
        "n_rows": n_rows,
        "n_immunogenic": n_pos,
        "n_patients": n_pat,
        "n_lopo_folds": sum(1 for v in results.get("binding_only", {}).get("per_patient_lopo", {}).values() if v is not None),
        "column_map": {
            "peptide": "MT_pep_x → mutant_peptide",
            "binding_nm": "MT_BindAff → binding_affinity_nm_numeric",
            "expression": "Quantification → expression_tpm_numeric (mutation-level)",
            "label": "VALIDATED → immunogenic",
            "hla": "HLA_type_x (not normalized, not used for prediction)",
            "wildtype": "NOT AVAILABLE — tcr_charge_diff excluded from rl_tcr_v1",
            "score_el": "Score_EL = NetMHCpan EL score (0-1, higher=stronger)",
        },
        "strategies": make_serializable(results),
        "note": "No optimization. Pre-specified directions locked from Ott (tcr) and TESLA (expression).",
    }

    print(f"\nLOPO folds evaluated: {output['n_lopo_folds']}")
    OUT.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
