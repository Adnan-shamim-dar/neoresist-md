"""
Add mhcflurry_percentile_rank to sahin_with_binding.csv
using Class1PresentationPredictor.predict_sequences().

This brings Sahin into parity with Rojas/Hilf which already have this column.
Run with: py -3.11 backend/validation/add_sahin_percentile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = Path("validation_papers/sahin_with_binding.csv")
SCAN_LENGTHS = [8, 9, 10, 11]


def main() -> int:
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {OUT_PATH} — run add_sahin_binding.py first"
        )
    df = pd.read_csv(OUT_PATH)
    print(f"Loaded {len(df)} rows from {OUT_PATH}")

    # Rows that have known HLA and a real peptide
    has_hla = df["hla_allele"].notna() & (df["hla_allele"] != "NOT_AVAILABLE")
    print(f"Rows with known HLA (need percentile): {has_hla.sum()}")

    from mhcflurry import Class1PresentationPredictor  # type: ignore

    predictor = Class1PresentationPredictor.load()

    df["mhcflurry_percentile_rank"] = np.nan
    df["mhcflurry_affinity"] = np.nan
    df["mhcflurry_best_peptide"] = None
    df["mhcflurry_mode"] = None

    cache: dict[tuple[str, str], dict] = {}
    done = 0
    total = int(has_hla.sum())

    for idx, row in df[has_hla].iterrows():
        pep = str(row.get("mutant_peptide", "")).strip().upper()
        allele = str(row.get("hla_allele", "")).strip()
        if not pep or not allele:
            continue
        key = (pep, allele)
        if key not in cache:
            try:
                plen = len(pep)
                if plen <= 15:
                    result = predictor.predict(peptides=[pep], alleles=[allele])
                    if result is not None and not result.empty:
                        best = result.iloc[0]
                        cache[key] = {
                            "affinity": float(best["affinity"]),
                            "percentile": float(best["presentation_percentile"]),
                            "best_peptide": pep,
                            "mode": "direct",
                        }
                    else:
                        cache[key] = {}
                else:
                    result = predictor.predict_sequences(
                        sequences=[pep],
                        alleles=[allele],
                        peptide_lengths=SCAN_LENGTHS,
                    )
                    if result is not None and not result.empty:
                        best = result.loc[result["affinity"].idxmin()]
                        cache[key] = {
                            "affinity": float(best["affinity"]),
                            "percentile": float(best["presentation_percentile"]),
                            "best_peptide": str(best["peptide"]),
                            "mode": "best_subpeptide",
                        }
                    else:
                        cache[key] = {}
            except Exception as e:
                cache[key] = {}
        entry = cache[key]
        if entry:
            df.at[idx, "mhcflurry_percentile_rank"] = entry.get("percentile", np.nan)
            df.at[idx, "mhcflurry_affinity"] = entry.get("affinity", np.nan)
            df.at[idx, "mhcflurry_best_peptide"] = entry.get("best_peptide")
            df.at[idx, "mhcflurry_mode"] = entry.get("mode")
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{total} done", flush=True)

    df.to_csv(OUT_PATH, index=False)
    covered = df["mhcflurry_percentile_rank"].notna().sum()
    print(f"\nUpdated {OUT_PATH}")
    print(f"Rows with mhcflurry_percentile_rank: {covered}/{len(df)}")
    print("Sample percentile values:", df["mhcflurry_percentile_rank"].dropna().head(5).round(3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
