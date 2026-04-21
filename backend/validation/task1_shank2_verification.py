"""
TASK 1: Verify SHANK2 G486S ranking — binding_only vs rl_tcr_v1.

SHANK2 G486S (peptide SDRVKVLSI, HLA-B*08:01) is in Keskin 2019, not Ott 2017.
This script computes TCR features for all Keskin rows and ranks SHANK2 under
both strategies at mutation level.

Run with: py -3.11 backend/validation/task1_shank2_verification.py
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
)
from backend.validation.new_features_and_validate import dynamic_blend

OUT_PATH = ARTIFACTS / "shank2_verification.json"


def main() -> int:
    tm = pd.read_csv(ARTIFACTS / "training_matrix.csv")
    keskin = tm[tm["paper_source"] == "keskin_2019"].copy()
    print(f"Keskin rows: {len(keskin)}, patients: {keskin['patient_id'].nunique()}")

    # Confirm SHANK2 presence
    shank_mask = keskin["gene"].str.contains("SHANK2", case=False, na=False) | \
                 keskin["protein_change"].str.contains("G486S", case=False, na=False)
    shank_rows = keskin[shank_mask]
    print(f"SHANK2 G486S rows: {len(shank_rows)}")
    if shank_rows.empty:
        result = {
            "found": False,
            "mutation_id": None,
            "binding_affinity_nm": None,
            "binding_only_rank": None,
            "rl_tcr_v1_rank": None,
            "rank_improvement": None,
            "immunogenic": None,
            "conclusion": "not found in dataset",
            "note": "SHANK2 G486S is in Keskin 2019, not Ott 2017. "
                    "Not found in keskin data with available features.",
        }
        OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    # Compute features using the same pipeline as full_cross_validation
    keskin_feat = compute_all_features(keskin, "keskin_2019")

    # Aggregate to mutation level (take min binding across alleles)
    mut_level = (
        keskin_feat.groupby(["patient_id", "gene", "protein_change"], dropna=False)
        .agg(
            immunogenic=("immunogenic", "max"),
            nm_raw=("nm_raw", "min"),
            bind_log50k=("bind_log50k", "max"),
            bind_exp500=("bind_exp500", "max"),
            tcr_volume=("tcr_volume", "first"),
            tcr_charge_diff=("tcr_charge_diff", "first"),
            tcr_hydro=("tcr_hydro", "first"),
            tcr_surface_change=("tcr_surface_change", "first"),
            hla_allele=("hla_allele", "first"),
            mutant_peptide=("mutant_peptide", "first"),
            binding_affinity_nm_numeric=("binding_affinity_nm_numeric", "min"),
        )
        .reset_index()
    )
    mut_level = mut_level[mut_level["immunogenic"].isin([0, 1])].copy()
    n_muts = len(mut_level)
    print(f"Mutation-level rows: {n_muts}")

    # Compute rl_tcr_v1 — locked Ott directions (all positive)
    # bind_log50k 0.4, tcr_volume 0.2, tcr_charge_diff 0.2, tcr_hydro 0.2
    for col in ["bind_log50k", "tcr_volume", "tcr_charge_diff", "tcr_hydro"]:
        mut_level[col] = pd.to_numeric(mut_level[col], errors="coerce")

    rl_tcr_v1_feats = ["bind_log50k", "tcr_volume", "tcr_charge_diff", "tcr_hydro"]
    rl_tcr_v1_weights = [0.4, 0.2, 0.2, 0.2]
    mut_level["score_binding_only"] = mut_level["bind_log50k"]
    mut_level["score_rl_tcr_v1"] = dynamic_blend(mut_level, rl_tcr_v1_feats, rl_tcr_v1_weights)

    # Rank all mutations (rank 1 = highest score = best candidate)
    mut_level["rank_binding_only"] = mut_level["score_binding_only"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)
    mut_level["rank_rl_tcr_v1"] = mut_level["score_rl_tcr_v1"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)

    # Find SHANK2 G486S at mutation level
    shank_mut = mut_level[
        mut_level["gene"].str.contains("SHANK2", case=False, na=False) |
        mut_level["protein_change"].str.contains("G486S", case=False, na=False)
    ]
    print(f"\nSHANK2 at mutation level:")
    print(mut_level[["gene", "protein_change", "mutant_peptide",
                       "binding_affinity_nm_numeric", "bind_log50k",
                       "score_rl_tcr_v1", "rank_binding_only", "rank_rl_tcr_v1",
                       "immunogenic"]].sort_values("rank_rl_tcr_v1").to_string())

    if shank_mut.empty:
        conclusion = "not found in dataset"
        result = {
            "found": False,
            "mutation_id": None,
            "binding_affinity_nm": None,
            "binding_only_rank": None,
            "rl_tcr_v1_rank": None,
            "rank_improvement": None,
            "immunogenic": None,
            "conclusion": conclusion,
            "note": "SHANK2 not found at mutation level after aggregation",
        }
    else:
        row = shank_mut.iloc[0]
        binding_rank = int(row["rank_binding_only"])
        tcr_rank = int(row["rank_rl_tcr_v1"])
        rank_improvement = binding_rank - tcr_rank  # positive = improved
        n_total = n_muts
        binding_pct = round(100 * (1 - binding_rank / n_total), 1)
        tcr_pct = round(100 * (1 - tcr_rank / n_total), 1)

        if rank_improvement > 0:
            conclusion = "surfaces higher under rl_tcr_v1"
        elif rank_improvement < 0:
            conclusion = "ranks lower under rl_tcr_v1"
        else:
            conclusion = "no rank improvement"

        result = {
            "found": True,
            "dataset": "keskin_2019",
            "note": "SHANK2 G486S is NOT in Ott 2017. It is in Keskin 2019 (GBM, patient keskin_8).",
            "mutation_id": "keskin_8/SHANK2/p.G486S",
            "mutant_peptide": str(row["mutant_peptide"]),
            "hla_allele": str(row["hla_allele"]),
            "binding_affinity_nm": float(row["binding_affinity_nm_numeric"]) if pd.notna(row["binding_affinity_nm_numeric"]) else None,
            "bind_log50k": float(row["bind_log50k"]) if pd.notna(row["bind_log50k"]) else None,
            "score_rl_tcr_v1": float(row["score_rl_tcr_v1"]) if pd.notna(row["score_rl_tcr_v1"]) else None,
            "tcr_features": {
                "tcr_volume": float(row["tcr_volume"]) if pd.notna(row["tcr_volume"]) else None,
                "tcr_charge_diff": float(row["tcr_charge_diff"]) if pd.notna(row["tcr_charge_diff"]) else None,
                "tcr_hydro": float(row["tcr_hydro"]) if pd.notna(row["tcr_hydro"]) else None,
                "tcr_surface_change": float(row["tcr_surface_change"]) if pd.notna(row["tcr_surface_change"]) else None,
            },
            "total_mutations": n_total,
            "binding_only_rank": binding_rank,
            "binding_only_percentile": binding_pct,
            "rl_tcr_v1_rank": tcr_rank,
            "rl_tcr_v1_percentile": tcr_pct,
            "rank_improvement": rank_improvement,
            "immunogenic": int(row["immunogenic"]),
            "top20pct_binding": binding_rank <= round(n_total * 0.2),
            "top20pct_tcr": tcr_rank <= round(n_total * 0.2),
            "tcr_elevates_into_top20": (tcr_rank <= round(n_total * 0.2)) and (binding_rank > round(n_total * 0.2)),
            "conclusion": conclusion,
        }

    print("\n" + "=" * 60)
    print("SHANK2 VERIFICATION RESULT:")
    print(json.dumps(result, indent=2))
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
