"""
Run feature_factory feature computation on Tretter 2023 data.
Output: backend/strategy_engine/artifacts/tretter_2023_features.csv

Pre-compute steps before calling feature_factory:
  1. wt_binding_nm  — MHCflurry prediction on WT peptides
  2. dai_binding    — log10(wt_nm) - log10(mut_nm)
  3. self_dissimilarity — BLOSUM62 normalised per-position dissimilarity
                          (proxy; requires wt_peptide)

Usage: py -3.11 backend/strategy_engine/run_tretter_features.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from backend.strategy_engine.feature_factory import (
    EXPECTED_FEATURES,
    compute_all_features,
    coerce_immunogenic,
)

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

INPUT  = Path("validation_papers/tretter_with_binding.csv")
OUTPUT = ARTIFACTS / "tretter_2023_features.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_pep(v: object) -> str | None:
    """Return upper-case peptide if 8–11 AA and all standard residues, else None."""
    from backend.strategy_engine.feature_factory import HYDRO
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    return s if 8 <= len(s) <= 14 and all(c in HYDRO for c in s) else None


def _normalise_hla(h: object) -> str | None:
    """Convert HLA-A*02:01 → 'HLA-A0201' (MHCflurry format)."""
    import re
    if not h or (isinstance(h, float) and np.isnan(h)):
        return None
    s = str(h).strip()
    s = re.sub(r"[*:]", "", s)          # HLA-A*02:01 → HLA-A0201
    if re.match(r"HLA-[ABC]\d{4,}", s):
        return s
    return None


# ── Step 1: load ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
# Replace empty-string wt_peptide with NaN so feature_factory sees proper NaN
df["wt_peptide"] = df["wt_peptide"].replace("", np.nan)
df = coerce_immunogenic(df)
df["paper_source"] = "tretter_2023"

print(f"Loaded {len(df)} rows from {INPUT}")
wt_present = df["wt_peptide"].notna().sum()
print(f"wt_peptide present: {wt_present}/{len(df)} ({wt_present/len(df):.1%})")
print(f"Positives: {int(df['immunogenic'].sum())}/{len(df)}")


# ── Step 2: WT binding via MHCflurry ─────────────────────────────────────────
print("\n[2] Running MHCflurry on WT peptides ...")
try:
    from mhcflurry import Class1AffinityPredictor  # type: ignore
    predictor = Class1AffinityPredictor.load()
    cache: dict[tuple[str, str], float] = {}

    wt_nms: list[float | None] = []
    ok = 0
    for _, row in df.iterrows():
        wt  = _safe_pep(row.get("wt_peptide"))
        hla = _normalise_hla(row.get("hla_allele"))
        if wt is None or hla is None:
            wt_nms.append(None)
            continue
        key = (wt, hla)
        if key not in cache:
            try:
                cache[key] = float(predictor.predict(peptides=[wt], alleles=[hla])[0])
            except Exception:
                cache[key] = float("nan")
        val = cache[key]
        wt_nms.append(val if not np.isnan(val) else None)
        if val is not None and not np.isnan(val):
            ok += 1

    df["wt_binding_nm"] = [float(v) if v is not None else np.nan for v in wt_nms]
    print(f"  wt_binding_nm computed: {ok}/{len(df)}")

except Exception as exc:
    print(f"  MHCflurry unavailable — {exc}")
    df["wt_binding_nm"] = np.nan


# ── Step 3: DAI = log10(wt_nm) − log10(mut_nm) ───────────────────────────────
print("\n[3] Computing DAI ...")
mut_nm = pd.to_numeric(df["binding_nm"], errors="coerce").clip(lower=0.1)
wt_nm  = pd.to_numeric(df["wt_binding_nm"], errors="coerce").clip(lower=0.1)
dai    = np.log10(wt_nm) - np.log10(mut_nm)
df["dai_binding"] = np.where(wt_nm.notna() & mut_nm.notna(), dai, np.nan)
dai_ok = df["dai_binding"].notna().sum()
print(f"  dai_binding non-null: {dai_ok}/{len(df)} ({dai_ok/len(df):.1%})")


# ── Step 4: self_dissimilarity — BLOSUM62 per-position normalised proxy ───────
# Formula: for each position i,  d_i = max(0, BLOSUM62[wt_i][wt_i] − BLOSUM62[mut_i][wt_i])
#                                            / BLOSUM62[wt_i][wt_i]
# self_dissimilarity = mean(d_i)  clipped to [0, 1]
# Only computed where wt_peptide is available; NaN otherwise.
print("\n[4] Computing self_dissimilarity (BLOSUM62 proxy) ...")
try:
    from Bio.Align import substitution_matrices  # type: ignore
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
    alph = set(BLOSUM62.alphabet)
except Exception:
    BLOSUM62 = None
    alph = set()

sd_vals: list[float] = []
for _, row in df.iterrows():
    mut = _safe_pep(row.get("mutant_peptide"))
    wt  = _safe_pep(row.get("wt_peptide"))
    if mut is None or wt is None or len(mut) != len(wt) or BLOSUM62 is None:
        sd_vals.append(np.nan)
        continue
    dists: list[float] = []
    for m, w in zip(mut, wt):
        if m in alph and w in alph:
            wt_self = float(BLOSUM62[w][w])
            cross   = float(BLOSUM62[m][w])
            # penalty at this position, normalised to [0, 1]
            d = max(0.0, wt_self - cross) / max(wt_self, 1.0)
            dists.append(d)
    sd_vals.append(float(np.mean(dists)) if dists else np.nan)

df["self_dissimilarity"] = sd_vals
sd_ok = df["self_dissimilarity"].notna().sum()
print(f"  self_dissimilarity non-null: {sd_ok}/{len(df)} ({sd_ok/len(df):.1%})")
if sd_ok > 0:
    valid = pd.Series(sd_vals).dropna()
    print(f"  range [{valid.min():.4f}, {valid.max():.4f}]  mean={valid.mean():.4f}")


# ── Step 5: compute remaining features via feature_factory ────────────────────
print("\n[5] Computing feature_factory features ...")
df = compute_all_features(df, "tretter_2023")


# ── Step 6: save ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}  ({len(df)} rows × {len(df.columns)} cols)")


# ── Step 7: coverage report ───────────────────────────────────────────────────
FOCUS = {
    "binding_nm", "wt_binding_nm", "dai_binding",
    "tcr_hydro_mean", "tcr_volume_mean", "tcr_charge_sum",
    "self_dissimilarity", "iedb_immuno", "calis_simplified", "expression_log2",
    "presentation_score_el", "hydro_full_mean", "aliphatic_index",
    "tcr_hydro_diff", "hamming_distance",
}

NUMERIC_EXPECTED = set(EXPECTED_FEATURES) | FOCUS
STRING_COLS = {"mutant_peptide", "wt_peptide", "patient_id", "cancer_type",
               "paper_source", "hla_allele", "clonality_class"}

print("\n" + "=" * 78)
print("COVERAGE REPORT — tretter_2023_features.csv")
print("=" * 78)
print(f"{'Column':<36} {'%non-null':>10} {'%non-zero':>10}  {'note'}")
print("-" * 78)

critical_missing: list[str] = []

for col in sorted(df.columns):
    s = df[col]
    n = len(s)
    is_focus = col in FOCUS

    if col in STRING_COLS or s.dtype == object:
        # For string columns: non-null = not NaN AND not empty string
        non_null = s.apply(
            lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
                      and str(v).strip() not in ("", "nan", "None")
        ).sum()
        pct_nn  = non_null / n if n else 0.0
        pct_nz  = float("nan")
        note    = f"(string, {non_null} filled)"
    else:
        s_num   = pd.to_numeric(s, errors="coerce")
        non_null = int(s_num.notna().sum())
        pct_nn  = non_null / n if n else 0.0
        nz      = int((s_num.notna() & (s_num != 0)).sum())
        pct_nz  = nz / non_null if non_null else 0.0
        note    = f"{pct_nz:.1%} non-zero" if non_null else ""

    flag = ""
    if pct_nn < 0.50 and (col in NUMERIC_EXPECTED or is_focus):
        flag = "⚠ LOW"
        critical_missing.append(col)

    marker = "★ " if is_focus else "  "
    nn_str = f"{pct_nn:.1%}"
    nz_str = f"{pct_nz:.1%}" if not (isinstance(pct_nz, float) and np.isnan(pct_nz)) else "—"
    print(f"{marker}{col:<34} {nn_str:>10} {nz_str:>10}  {flag}")

# Focus features absent from df
for feat in sorted(FOCUS):
    if feat not in df.columns:
        print(f"  {feat:<36} {'ABSENT':>10} {'—':>10}  ⚠ LOW")
        critical_missing.append(feat)

print("=" * 78)
if critical_missing:
    print(f"\n⚠  Critical (<50% or absent): {sorted(set(critical_missing))}")
else:
    print("\n✓  All focus features have ≥50% coverage.")

print(f"\nRows: {len(df)}  |  Positives: {int(df['immunogenic'].sum())}  "
      f"|  Patients: {df['patient_id'].nunique()}  "
      f"|  Total cols: {len(df.columns)}")
