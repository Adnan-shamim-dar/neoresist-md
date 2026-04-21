"""
TASK 2: Add NetMHCpan/MHCflurry %rank as comparison baseline to cross-matrix.

Data availability:
  Rojas 2023:  mhcflurry_percentile_rank available (230/230 rows)
  Hilf 2019:   mhcflurry_percentile_rank available (127/152 rows)
  Sahin 2017:  mhcflurry_percentile_rank available after add_sahin_percentile.py
  Ott 2017:    no %rank — use bind_log50k as proxy (highly correlated)
  TESLA 2020:  no %rank — use bind_log50k as proxy

Run with: py -3.11 backend/validation/task2_percentile_baseline.py
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
from backend.validation.rerun_rojas_hilf_with_binding import build_transfer_strategies


def build_baseline_strategies():
    """Baseline strategies for comparison."""
    from collections import OrderedDict
    s = build_transfer_strategies()
    # add %rank strategy — uses mhcflurry_percentile_rank column
    # score = 1 - (percentile_rank / 100); lower %rank = better binder
    s["mhcflurry_percentile_rank_only"] = {
        "features": [("mhcflurry_pr_score", 1.0, False)],
        "description": "MHCflurry presentation %rank (lower=better → inverted to score)",
    }
    return s


def add_percentile_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mhcflurry_pr_score = 1 - (percentile_rank/100) for rows that have it."""
    out = df.copy()
    if "mhcflurry_percentile_rank" in out.columns:
        pr = pd.to_numeric(out["mhcflurry_percentile_rank"], errors="coerce")
        out["mhcflurry_pr_score"] = (1.0 - pr / 100.0).clip(0, 1)
        valid = out["mhcflurry_pr_score"].notna().sum()
        print(f"  mhcflurry_pr_score: {valid}/{len(out)} rows")
    else:
        out["mhcflurry_pr_score"] = np.nan
        print("  mhcflurry_pr_score: NOT AVAILABLE (no percentile_rank column)")
    return out


def load_ott(tm: pd.DataFrame) -> pd.DataFrame:
    ott = tm[(tm["paper_source"] == "ott_2017") & (tm["immunogenic"].isin([0, 1]))].copy()
    return compute_all_features(ott, "ott_2017")


def load_tesla(tm: pd.DataFrame) -> pd.DataFrame:
    tesla_raw = Path("validation_papers/tesla_prepared.csv")
    if not tesla_raw.exists():
        raise FileNotFoundError(f"Missing {tesla_raw}")
    tesla = pd.read_csv(tesla_raw)
    tesla = tesla[tesla["immunogenic"].isin([0, 1])].copy()
    return compute_all_features(tesla, "tesla_2020")


def load_paper(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["immunogenic"].isin([0, 1])].copy()
    df = compute_all_features(df, name)
    df = add_percentile_score(df)
    return df


def run_strategies(name: str, df: pd.DataFrame, strategies: dict) -> dict:
    n_pat = int(df["patient_id"].nunique())
    can_lopo = n_pat >= 3
    results = {}
    for sn, sd in strategies.items():
        fl = sd["features"]
        available = [f for f, _, _ in fl if f in df.columns and
                     pd.to_numeric(df[f], errors="coerce").notna().sum() > 5]
        missing = [f for f, _, _ in fl if f not in available]
        if not available:
            results[sn] = {"lopo_auc": None, "insample_auc": None, "status": f"missing: {missing}"}
            continue
        is_auc = insample_auc(df, fl)
        lopo, _ = lopo_auc(df, fl) if can_lopo else (None, {})
        results[sn] = {"lopo_auc": lopo, "insample_auc": is_auc, "status": "computed"}
    return results


def main() -> int:
    tm = pd.read_csv(ARTIFACTS / "training_matrix.csv")
    strategies = build_baseline_strategies()

    print("\n=== Loading datasets ===")
    papers: dict[str, pd.DataFrame] = {}

    print("\n[Ott 2017]")
    ott = load_ott(tm)
    ott["mhcflurry_pr_score"] = np.nan  # no %rank for Ott — proxy = bind_log50k
    print("  NOTE: Ott has no mhcflurry_percentile_rank — bind_log50k used as proxy")
    papers["ott_2017"] = ott

    print("\n[TESLA 2020]")
    try:
        tesla = load_tesla(tm)
        tesla["mhcflurry_pr_score"] = np.nan  # no %rank for TESLA
        print("  NOTE: TESLA has no mhcflurry_percentile_rank — bind_log50k used as proxy")
        papers["tesla_2020"] = tesla
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")

    print("\n[Rojas 2023]")
    papers["rojas_2023"] = load_paper("validation_papers/rojas_with_binding.csv", "rojas_2023")

    print("\n[Hilf 2019]")
    papers["hilf_2019"] = load_paper("validation_papers/hilf_with_binding.csv", "hilf_2019")

    print("\n[Sahin 2017]")
    sahin_path = Path("validation_papers/sahin_with_binding.csv")
    if sahin_path.exists():
        papers["sahin_2017"] = load_paper(str(sahin_path), "sahin_2017")
    else:
        print("  SKIPPED: sahin_with_binding.csv not found")

    # Check for PRIME scores
    prime_found = any(
        any("prime" in c.lower() for c in df.columns)
        for df in papers.values()
    )
    prime_note = "PRIME not available in any dataset — future comparison needed"
    if prime_found:
        prime_note = "PRIME scores detected — add PRIME strategy if needed"
    print(f"\nPRIME: {prime_note}")

    print("\n=== Running strategies ===")
    all_results = {}
    for pname, df in papers.items():
        print(f"\n  [{pname}]")
        all_results[pname] = run_strategies(pname, df, strategies)
        for sn, r in all_results[pname].items():
            lopo = r.get("lopo_auc")
            lopo_str = f"{lopo:.4f}" if lopo is not None else "  N/A "
            print(f"    {sn:<45} LOPO={lopo_str}")

    # Build cross-matrix
    paper_order = ["ott_2017", "tesla_2020", "sahin_2017", "hilf_2019", "rojas_2023"]
    paper_order = [p for p in paper_order if p in all_results]
    rows = []
    for sn in strategies.keys():
        row = {"strategy": sn}
        lopos = []
        for pn in paper_order:
            lopo = all_results.get(pn, {}).get(sn, {}).get("lopo_auc")
            row[f"{pn}_lopo"] = lopo
            if lopo is not None:
                lopos.append(lopo)
        row["mean_lopo"] = float(np.mean(lopos)) if lopos else None
        rows.append(row)

    cross_matrix = pd.DataFrame(rows)
    print("\n=== UPDATED CROSS-MATRIX (LOPO AUC) ===")
    print(cross_matrix.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "  N/A "))

    out_csv = ARTIFACTS / "full_cross_matrix_with_baselines.csv"
    out_json = ARTIFACTS / "percentile_baseline_results.json"
    cross_matrix.to_csv(out_csv, index=False)

    output = {
        "notes": {
            "ott_2017": "No mhcflurry_percentile_rank — bind_log50k used as proxy for %rank",
            "tesla_2020": "No mhcflurry_percentile_rank — bind_log50k used as proxy for %rank",
            "rojas_2023": "mhcflurry_percentile_rank available (230/230 rows)",
            "hilf_2019": "mhcflurry_percentile_rank available (127/152 rows)",
            "sahin_2017": "mhcflurry_percentile_rank added via add_sahin_percentile.py",
            "prime": prime_note,
        },
        "results": make_serializable(all_results),
        "cross_matrix": cross_matrix.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
