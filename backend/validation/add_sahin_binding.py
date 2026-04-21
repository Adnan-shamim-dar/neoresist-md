"""
Generate validation_papers/sahin_with_binding.csv.

HLA alleles sourced from Sahin 2017 Nature (doi:10.1038/nature23003)
Extended Data Table 3. Per-patient alleles inferred from reported
HLA restrictions for immunogenic epitopes.

Run with: py -3.11 backend/validation/add_sahin_binding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = Path("backend/validation/artifacts")
OUT_DIR = Path("validation_papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HLA alleles from Extended Data Table 3 (Sahin 2017, Nature 547:222-226).
# Patients not shown in the table had no confirmed class-I restrictions.
SAHIN_HLA_MAP: dict[str, list[str] | None] = {
    "sahin_P01": ["HLA-A*31:01"],
    "sahin_P02": ["HLA-B*39:06"],
    "sahin_P03": None,
    "sahin_P04": ["HLA-A*02:01", "HLA-B*07:02", "HLA-B*44:02"],
    "sahin_P05": ["HLA-B*07:02"],
    "sahin_P06": ["HLA-A*11:01"],
    "sahin_P07": None,
    "sahin_P09": None,
    "sahin_P10": None,
    "sahin_P11": ["HLA-A*02:01"],
    "sahin_P12": None,
    "sahin_P17": ["HLA-A*68:01", "HLA-B*37:01"],
    "sahin_P19": ["HLA-B*57:01", "HLA-A*11:01"],
}

MAX_DIRECT_LEN = 15
SCAN_LENGTHS = [8, 9, 10, 11]


def best_binding_nm(predictor, peptide: str, hla: str) -> float:
    """Return minimum nM across sliding windows. Handles peptides of any length."""
    pep = str(peptide).strip().upper()
    if not pep:
        return np.nan
    plen = len(pep)
    if plen <= MAX_DIRECT_LEN:
        try:
            return float(predictor.predict(peptides=[pep], alleles=[hla])[0])
        except Exception:
            return np.nan
    best = np.inf
    for wlen in SCAN_LENGTHS:
        for i in range(plen - wlen + 1):
            window = pep[i : i + wlen]
            try:
                nm = float(predictor.predict(peptides=[window], alleles=[hla])[0])
                if nm < best:
                    best = nm
            except Exception:
                continue
    return best if np.isfinite(best) else np.nan


def expand_hla_rows(sahin: pd.DataFrame) -> pd.DataFrame:
    """
    For each Sahin row, create one row per known patient HLA allele.
    Patients with no known HLA keep one row with hla_allele=NOT_AVAILABLE.
    """
    expanded_rows = []
    for _, row in sahin.iterrows():
        pid = str(row["patient_id"])
        alleles = SAHIN_HLA_MAP.get(pid)
        if alleles:
            for allele in alleles:
                new_row = row.copy()
                new_row["hla_allele"] = allele
                new_row["final_hla_allele"] = allele
                new_row["binding_source"] = "mhcflurry_scan"
                new_row["binding_affinity_nm"] = "PENDING"
                new_row["binding_affinity_nm_numeric"] = np.nan
                new_row["predicted_affinity_nm"] = np.nan
                new_row["final_binding_affinity_nm"] = np.nan
                expanded_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row["hla_allele"] = "NOT_AVAILABLE"
            new_row["final_hla_allele"] = "NOT_AVAILABLE"
            new_row["binding_source"] = "not_available"
            new_row["binding_affinity_nm"] = "NOT_AVAILABLE"
            new_row["binding_affinity_nm_numeric"] = np.nan
            new_row["predicted_affinity_nm"] = np.nan
            new_row["final_binding_affinity_nm"] = np.nan
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def run_mhcflurry(df: pd.DataFrame) -> pd.DataFrame:
    from mhcflurry import Class1AffinityPredictor  # type: ignore

    predictor = Class1AffinityPredictor.load()
    cache: dict[tuple[str, str], float] = {}
    total = len(df)
    predicted = 0
    skipped = 0

    for idx, row in df.iterrows():
        allele = str(row.get("hla_allele", ""))
        if allele in ("NOT_AVAILABLE", "nan", ""):
            skipped += 1
            continue
        pep = str(row.get("mutant_peptide", "")).strip().upper()
        if not pep:
            skipped += 1
            continue
        key = (pep, allele)
        if key in cache:
            nm = cache[key]
        else:
            nm = best_binding_nm(predictor, pep, allele)
            cache[key] = nm
        df.at[idx, "binding_affinity_nm"] = str(nm) if np.isfinite(nm) else "NOT_AVAILABLE"
        df.at[idx, "binding_affinity_nm_numeric"] = nm if np.isfinite(nm) else np.nan
        df.at[idx, "predicted_affinity_nm"] = nm if np.isfinite(nm) else np.nan
        df.at[idx, "final_binding_affinity_nm"] = nm if np.isfinite(nm) else np.nan
        predicted += 1
        if predicted % 20 == 0:
            print(f"  ... {predicted}/{total - skipped} predictions done", flush=True)

    print(f"  Predicted: {predicted}, Skipped (no HLA): {skipped}", flush=True)
    return df


def main() -> int:
    tm_path = ARTIFACTS / "training_matrix.csv"
    if not tm_path.exists():
        raise FileNotFoundError(f"Missing: {tm_path}")

    tm = pd.read_csv(tm_path)
    sahin = tm[tm["paper_source"] == "sahin_2017"].copy()
    print(f"Loaded {len(sahin)} Sahin rows from training_matrix.csv")
    print(f"Patients: {sorted(sahin['patient_id'].unique().tolist())}")

    print("\nExpanding rows for multi-allele patients...")
    expanded = expand_hla_rows(sahin)

    known_hla = expanded[expanded["hla_allele"] != "NOT_AVAILABLE"]
    no_hla = expanded[expanded["hla_allele"] == "NOT_AVAILABLE"]
    print(f"Rows with known HLA: {len(known_hla)}")
    print(f"Rows without HLA:    {len(no_hla)}")
    print(f"Total expanded rows: {len(expanded)}")

    print("\nRunning MHCflurry predictions (sliding window for long peptides)...")
    expanded = run_mhcflurry(expanded)

    out_path = OUT_DIR / "sahin_with_binding.csv"
    expanded.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    has_binding = expanded["binding_affinity_nm_numeric"].notna()
    print(f"\nBinding coverage:")
    print(f"  Rows with binding affinity: {has_binding.sum()} / {len(expanded)}")
    print(f"  Unique mutations covered:   {expanded.loc[has_binding, 'mutant_peptide'].nunique()} / {sahin['mutant_peptide'].nunique()}")

    print("\nSample predictions (best binder per patient):")
    for pid in sorted(expanded["patient_id"].unique()):
        sub = expanded[(expanded["patient_id"] == pid) & has_binding]
        if sub.empty:
            print(f"  {pid}: no binding data")
        else:
            best = sub.loc[sub["binding_affinity_nm_numeric"].idxmin()]
            print(f"  {pid} | {best['hla_allele']} | {best['mutant_peptide'][:20]}... | {best['binding_affinity_nm_numeric']:.1f} nM")

    return 0


if __name__ == "__main__":
    sys.exit(main())
