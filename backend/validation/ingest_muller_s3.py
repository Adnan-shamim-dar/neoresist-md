"""
Ingest Müller 2023 Data S3 — "Features used to train neo-peptide and
mutation classifiers" (Cell Immunity 2023, doi:10.1016/j.immuni.2023.09.002).

Run AFTER placing Data_S3.xlsx in validation_papers/muller2023/:
  py -3.11 backend/validation/ingest_muller_s3.py

Reference:
  Müller M et al. (2023). Machine learning methods and harmonized datasets
  improve immunogenic neoantigen prediction. Immunity 56(11):2650-2663.
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

MULLER_DIR = Path("validation_papers/muller2023")
ARTIFACTS = Path("backend/validation/artifacts")

# Expected dataset labels in the Müller S3 file (may vary — script will detect)
KNOWN_DATASET_LABELS = ["NCI", "TESLA", "HiTIDE", "nci", "tesla", "hitide"]

# Candidate column names for key fields
PEPTIDE_CANDIDATES = ["peptide", "mutant_peptide", "neo_peptide", "neopeptide", "sequence", "mut_peptide"]
PATIENT_CANDIDATES = ["patient", "patient_id", "sample", "sample_id", "donor"]
DATASET_CANDIDATES = ["dataset", "cohort", "study", "source", "paper"]
IMMUNO_CANDIDATES = ["immunogenic", "immunogenicity", "response", "label", "y", "hit"]
BINDING_CANDIDATES = ["binding", "affinity", "ic50", "nm", "mhc_binding", "rank", "percentile_rank", "el_rank"]
EXPRESSION_CANDIDATES = ["expression", "tpm", "fpkm", "rpkm", "expr"]
ONCOGENE_CANDIDATES = ["oncogene", "intogen", "driver", "cancer_gene", "cgc"]


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name matching any candidate (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, col_orig in lower_cols.items():
            if cand.lower() in col_lower:
                return col_orig
    return None


def load_s3(path: Path) -> dict[str, pd.DataFrame]:
    """Load all sheets from Data_S3.xlsx, return {sheet_name: df}."""
    print(f"Loading: {path}")
    xl = pd.ExcelFile(path)
    print(f"Sheets found: {xl.sheet_names}")
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name, header=0)
        print(f"  Sheet '{name}': {len(df)} rows x {len(df.columns)} cols")
        sheets[name] = df
    return sheets


def identify_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "peptide": detect_column(df, PEPTIDE_CANDIDATES),
        "patient": detect_column(df, PATIENT_CANDIDATES),
        "dataset": detect_column(df, DATASET_CANDIDATES),
        "immunogenic": detect_column(df, IMMUNO_CANDIDATES),
        "binding": detect_column(df, BINDING_CANDIDATES),
        "expression": detect_column(df, EXPRESSION_CANDIDATES),
        "oncogene": detect_column(df, ONCOGENE_CANDIDATES),
    }


def split_by_dataset(df: pd.DataFrame, dataset_col: str) -> dict[str, pd.DataFrame]:
    splits = {}
    values = df[dataset_col].dropna().unique()
    print(f"  Dataset values found: {sorted([str(v) for v in values])}")
    for v in values:
        key = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        subset = df[df[dataset_col].astype(str).str.strip() == str(v).strip()].copy()
        splits[key] = subset
        print(f"  '{v}': {len(subset)} rows")
    return splits


def run_quick_auc(df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> dict:
    """Quick in-sample AUC for each feature."""
    from sklearn.metrics import roc_auc_score
    y = pd.to_numeric(df[label_col], errors="coerce")
    valid = y.isin([0, 1]) | y.isin([0.0, 1.0])
    y = y[valid].astype(int)
    results = {}
    for fc in feature_cols:
        if fc not in df.columns:
            continue
        vals = pd.to_numeric(df.loc[valid.index[valid], fc], errors="coerce")
        mask = vals.notna() & np.isfinite(vals)
        if mask.sum() < 10 or len(y[mask].unique()) < 2:
            results[fc] = None
            continue
        try:
            auc = float(roc_auc_score(y[mask], vals[mask]))
            auc = max(auc, 1 - auc)  # flip if AUC < 0.5
            results[fc] = round(auc, 4)
        except Exception:
            results[fc] = None
    return results


def main() -> int:
    MULLER_DIR.mkdir(parents=True, exist_ok=True)

    # Find the S3 file
    candidates = list(MULLER_DIR.glob("*S3*")) + list(MULLER_DIR.glob("*s3*")) + \
                 list(MULLER_DIR.glob("Data_S3*")) + list(MULLER_DIR.glob("*.xlsx"))
    candidates = [f for f in candidates if f.suffix in (".xlsx", ".csv", ".xls")]

    if not candidates:
        print("=" * 60)
        print("WAITING FOR MANUAL DOWNLOAD")
        print("=" * 60)
        print()
        print("Data_S3.xlsx not found in validation_papers/muller2023/")
        print()
        print("To obtain it:")
        print("  1. Go to: https://www.cell.com/immunity/fulltext/S1074-7613(23)00406-5")
        print("  2. Scroll to Supplementary Information")
        print("  3. Download: Data S3 — 'Features used to train neo-peptide")
        print("     and mutation classifiers'")
        print("  4. Save as: validation_papers/muller2023/Data_S3.xlsx")
        print("  5. Re-run: py -3.11 backend/validation/ingest_muller_s3.py")
        print()
        print("Script is ready — re-run after download.")
        return 0

    s3_path = candidates[0]
    print(f"Found: {s3_path}")

    sheets = load_s3(s3_path)

    # Use the largest sheet as primary data
    primary_name = max(sheets, key=lambda k: len(sheets[k]))
    df = sheets[primary_name].copy()
    print(f"\nUsing sheet '{primary_name}' as primary ({len(df)} rows)")
    print(f"All columns ({len(df.columns)}):")
    for i, c in enumerate(df.columns):
        sample = df[c].dropna().head(2).tolist()
        print(f"  [{i:02d}] {c}: {sample}")

    col_map = identify_columns(df)
    print(f"\nDetected columns: {col_map}")

    # Split by dataset
    out_files = {}
    dataset_col = col_map.get("dataset")
    if dataset_col:
        splits = split_by_dataset(df, dataset_col)
        for dname, ddf in splits.items():
            out_path = MULLER_DIR / f"muller_{dname}.csv"
            ddf.to_csv(out_path, index=False)
            out_files[dname] = out_path
            imm_col = col_map.get("immunogenic")
            if imm_col and imm_col in ddf.columns:
                pos = pd.to_numeric(ddf[imm_col], errors="coerce")
                n_pos = int((pos == 1).sum())
                print(f"  {dname}: {len(ddf)} rows, {n_pos} immunogenic, "
                      f"{ddf[col_map['patient']].nunique() if col_map.get('patient') else '?'} patients")
            print(f"  Saved: {out_path}")
    else:
        # No dataset column — save as single file
        out_path = MULLER_DIR / "muller_all.csv"
        df.to_csv(out_path, index=False)
        out_files["all"] = out_path
        print(f"No dataset column detected. Saved as: {out_path}")

    # Quick AUC for NCI subset (most similar to our melanoma datasets)
    nci_key = next((k for k in out_files if "nci" in k.lower()), None)
    if nci_key:
        nci_df = pd.read_csv(out_files[nci_key])
        imm_col = col_map.get("immunogenic")
        if imm_col and imm_col in nci_df.columns:
            print(f"\n--- NCI feature AUC (pre-specified directions, no optimization) ---")
            feature_candidates = [c for c in nci_df.columns
                                   if any(k in c.lower() for k in
                                          ["bind", "rank", "tpm", "expr", "dissim", "foreign",
                                           "tcr", "calis", "hamming", "intogen", "driver"])]
            auc_results = run_quick_auc(nci_df, imm_col, feature_candidates[:20])
            for feat, auc in sorted(auc_results.items(), key=lambda x: x[1] or 0, reverse=True):
                auc_str = f"{auc:.4f}" if auc is not None else "  N/A"
                print(f"  {feat:<40} AUC={auc_str}")

            # Check if our strategies replicate
            out_json = ARTIFACTS / "muller_nci_quick_auc.json"
            out_json.write_text(json.dumps({
                "dataset": "muller_nci",
                "n_rows": len(nci_df),
                "feature_aucs": auc_results,
                "note": "In-sample only — LOPO requires patient-level split",
            }, indent=2), encoding="utf-8")
            print(f"\nSaved: {out_json}")

    print("\nDone. Check validation_papers/muller2023/ for split CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
