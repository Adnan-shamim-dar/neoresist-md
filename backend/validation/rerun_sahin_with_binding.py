"""
PHASE 4/5 rerun for Sahin 2017 using MHCflurry binding predictions.

Requires: validation_papers/sahin_with_binding.csv
Generate with: py -3.11 backend/validation/add_sahin_binding.py

Run with: py -3.11 backend/validation/rerun_sahin_with_binding.py
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
    is_leaky_feature_name,
    lopo_auc,
    lopo_auc_with_train_fold_directions,
    make_serializable,
    resolve_auto_directions,
)
from backend.validation.rerun_rojas_hilf_with_binding import build_transfer_strategies, run_discovery


def main() -> int:
    sahin_path = Path("validation_papers/sahin_with_binding.csv")
    if not sahin_path.exists():
        raise FileNotFoundError(
            f"Missing: {sahin_path}\n"
            "Generate with: py -3.11 backend/validation/add_sahin_binding.py"
        )

    sahin_raw = pd.read_csv(sahin_path)
    print(f"Loaded {len(sahin_raw)} rows from {sahin_path}")

    sahin = sahin_raw[sahin_raw["immunogenic"].isin([0, 1])].copy()
    has_binding = sahin["binding_affinity_nm_numeric"].notna()
    print(f"Rows with immunogenic label: {len(sahin)}")
    print(f"Rows with binding affinity:  {has_binding.sum()}")
    print(f"Immunogenic positives:       {int(sahin['immunogenic'].sum())}")
    print(f"Patients:                    {sahin['patient_id'].nunique()}")

    patients_with_binding = sahin.loc[has_binding, "patient_id"].unique().tolist()
    print(f"Patients with binding data:  {sorted(patients_with_binding)}")

    print("\n--- Computing features ---")
    featured = compute_all_features(sahin, "sahin_2017")

    transfer_strategies = build_transfer_strategies()
    n_pat = int(featured["patient_id"].nunique())
    can_lopo = n_pat >= 3

    print(f"\nRunning transfer strategies (LOPO={'yes' if can_lopo else 'no'})...")
    transfer_results: dict[str, object] = {}
    strat_results = {}
    for sn, sd in transfer_strategies.items():
        fl = sd["features"]
        available, missing = [], []
        for feat, _, _ in fl:
            if feat in featured.columns and pd.to_numeric(featured[feat], errors="coerce").notna().sum() > 5:
                available.append(feat)
            else:
                missing.append(feat)
        if not available:
            strat_results[sn] = {
                "lopo_auc": None,
                "insample_auc": None,
                "status": f"no features available (missing: {missing})",
            }
            continue
        is_auc = insample_auc(featured, fl)
        lopo, per = lopo_auc(featured, fl) if can_lopo else (None, {})
        strat_results[sn] = {
            "lopo_auc": lopo,
            "insample_auc": is_auc,
            "per_patient": per,
            "features_available": available,
            "features_missing": missing,
            "status": "computed",
        }
        lopo_str = f"{lopo:.4f}" if lopo is not None else "N/A"
        is_str = f"{is_auc:.4f}" if is_auc is not None else "N/A"
        print(f"  {sn:<40} LOPO={lopo_str}  in-sample={is_str}")

    transfer_results["sahin_2017"] = {
        "n_rows": int(len(featured)),
        "n_immunogenic": int(featured["immunogenic"].sum()),
        "n_patients": n_pat,
        "can_lopo": can_lopo,
        "strategies": strat_results,
    }

    print("\nRunning discovery (dataset-specific hypothesis search)...")
    discovery = run_discovery("sahin_2017", featured)

    rows = []
    for sn in transfer_strategies.keys():
        rs = strat_results.get(sn, {})
        rows.append({
            "strategy": sn,
            "sahin_2017_lopo": rs.get("lopo_auc"),
            "sahin_2017_insample": rs.get("insample_auc"),
        })
    if discovery.get("best_strategy"):
        rows.append({
            "strategy": "BEST_OF_sahin_2017",
            "sahin_2017_lopo": discovery.get("best_auc") if discovery.get("can_lopo") else None,
            "sahin_2017_insample": discovery.get("best_auc") if not discovery.get("can_lopo") else None,
        })
    cross_matrix = pd.DataFrame(rows)

    output = {
        "summary": {
            "paper": "sahin_2017",
            "note": "PHASE 4/5 rerun using sahin_with_binding.csv (MHCflurry sliding window)",
            "rows_with_binding": int(has_binding.sum()),
            "total_rows": int(len(sahin)),
            "patients_with_hla": sorted(patients_with_binding),
        },
        "transfer_results": make_serializable(transfer_results),
        "discovery_results": make_serializable({"sahin_2017": discovery}),
    }

    out_json = ARTIFACTS / "sahin_cross_validation_with_binding.json"
    out_csv = ARTIFACTS / "sahin_cross_matrix_with_binding.csv"
    out_json.write_text(json.dumps(make_serializable(output), indent=2, default=str), encoding="utf-8")
    cross_matrix.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")

    print("\n=== SAHIN RESULTS SUMMARY ===")
    for _, row in cross_matrix.iterrows():
        lopo = row.get("sahin_2017_lopo")
        lopo_str = f"{lopo:.4f}" if pd.notna(lopo) else "N/A"
        print(f"  {row['strategy']:<40} LOPO={lopo_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
