"""
PHASE 1 — Feature Factory.

Loads every dataset, standardizes column names, and computes
every possible feature. Saves per-dataset feature matrices and
a coverage inventory.

Run: py -3.11 backend/strategy_engine/feature_factory.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Amino acid property tables (canonical) ────────────────────────────────────
HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
CHARGE = {"R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
AROMATIC = {"F": 1, "W": 1, "Y": 1}

# Calis-style position weights per peptide length
CALIS_WEIGHTS = {
    8:  [0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05],
    9:  [0.05, 0.05, 0.10, 0.20, 0.20, 0.20, 0.10, 0.05, 0.05],
    10: [0.05, 0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05],
    11: [0.04, 0.04, 0.08, 0.12, 0.14, 0.14, 0.14, 0.12, 0.08, 0.05, 0.05],
}

# TCR contact positions (0-indexed) by peptide length
TCR_POS = {8: [2, 3, 4], 9: [3, 4, 5, 6], 10: [3, 4, 5, 6, 7], 11: [3, 4, 5, 6, 7]}


# ── BLOSUM62 loader ───────────────────────────────────────────────────────────
def _load_blosum():
    try:
        from Bio.Align import substitution_matrices
        return substitution_matrices.load("BLOSUM62")
    except Exception:
        return None

BLOSUM62 = _load_blosum()


# ── Column name normalization ─────────────────────────────────────────────────
COLUMN_MAP = {
    # binding nM
    "MT_BindAff": "binding_nm",
    "mhcflurry_affinity": "binding_nm",
    "binding_affinity_nm_numeric": "binding_nm",
    "binding_affinity_nm": "binding_nm",
    "final_binding_affinity_nm": "binding_nm",
    # WT binding
    "WT_BindAff": "wt_binding_nm",
    # mutant peptide
    "MT_pep_x": "mutant_peptide",
    "MT_pep": "mutant_peptide",
    "Epitope": "mutant_peptide",
    # WT peptide
    "WT_pep": "wt_peptide",
    "WT_pep_x": "wt_peptide",
    "wildtype_peptide": "wt_peptide",
    # immunogenicity
    "VALIDATED": "immunogenic",
    "Immunogenicity": "immunogenic",
    "label": "immunogenic",
    "response": "immunogenic",
    # patient
    "PatientID": "patient_id",
    "Patient": "patient_id",
    "sample": "patient_id",
    "sample_id": "patient_id",
    # expression
    "Quantification": "expression_tpm",
    "expression_tpm_numeric": "expression_tpm",
    "TPM": "expression_tpm",
    "RNAseq_expression": "expression_tpm",
    # NCI-specific features
    "Score_EL": "presentation_score_el",
    "BindStab": "binding_stability",
    "Agretopicity": "dai_agretopicity",
    # cancer type
    "CancerType": "cancer_type",
    "tumor_type": "cancer_type",
    # vaf
    "VAF": "vaf",
    "variant_allele_freq": "vaf",
    # hla
    "HLA_type_x": "hla_allele",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical column name mapping; don't clobber existing standard names.
    Only rename the FIRST matching source column to each target to avoid duplicates."""
    renames = {}
    existing = set(df.columns)
    already_targeted: set[str] = set()
    for src, tgt in COLUMN_MAP.items():
        if src in existing and tgt not in existing and tgt not in already_targeted:
            renames[src] = tgt
            already_targeted.add(tgt)
        # If tgt already in df or already renamed, skip silently
    df = df.rename(columns=renames)
    # Drop any remaining duplicate columns (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def coerce_immunogenic(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure immunogenic is 0/1 int. Drop rows with NaN immunogenic."""
    col = "immunogenic"
    if col not in df.columns:
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[col].isin([0, 1])].copy()
    df[col] = df[col].astype(int)
    return df


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    # ── Ott 2017: use mutation-level file (already aggregated + has TCR features)
    ott_path = Path("backend/validation/artifacts/ott_mutation_level_with_new_features.csv")
    if ott_path.exists():
        df = pd.read_csv(ott_path)
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        df["paper_source"] = "ott_2017"
        df["cancer_type"] = df.get("cancer_type", pd.Series("melanoma", index=df.index)).fillna("melanoma")
        datasets["ott_2017"] = df
        print(f"  ott_2017: {ott_path} → {len(df)} rows")
    else:
        print(f"  ott_2017: NOT FOUND at {ott_path}")

    # ── TESLA: use tesla_prepared.csv
    tesla_path = Path("validation_papers/tesla_prepared.csv")
    if tesla_path.exists():
        df = pd.read_csv(tesla_path)
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        if "paper_source" not in df.columns:
            df["paper_source"] = "tesla_2020"
        datasets["tesla_2020"] = df
        print(f"  tesla_2020: {tesla_path} → {len(df)} rows")
    else:
        print(f"  tesla_2020: NOT FOUND at {tesla_path}")

    # ── Sahin 2017: use sahin_with_binding.csv (has MHCflurry binding)
    sahin_path = Path("validation_papers/sahin_with_binding.csv")
    if sahin_path.exists():
        df = pd.read_csv(sahin_path)
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        if "paper_source" not in df.columns:
            df["paper_source"] = "sahin_2017"
        datasets["sahin_2017"] = df
        print(f"  sahin_2017: {sahin_path} → {len(df)} rows")

    # ── Hilf 2019: use hilf_with_binding.csv
    hilf_path = Path("validation_papers/hilf_with_binding.csv")
    if hilf_path.exists():
        df = pd.read_csv(hilf_path)
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        if "paper_source" not in df.columns:
            df["paper_source"] = "hilf_2019"
        datasets["hilf_2019"] = df
        print(f"  hilf_2019: {hilf_path} → {len(df)} rows")

    # ── Rojas 2023: use rojas_with_binding.csv
    rojas_path = Path("validation_papers/rojas_with_binding.csv")
    if rojas_path.exists():
        df = pd.read_csv(rojas_path)
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        if "paper_source" not in df.columns:
            df["paper_source"] = "rojas_2023"
        datasets["rojas_2023"] = df
        print(f"  rojas_2023: {rojas_path} → {len(df)} rows")

    # ── Keskin 2019: from training matrix
    tm_path = Path("backend/validation/artifacts/training_matrix.csv")
    if tm_path.exists():
        tm = pd.read_csv(tm_path)
        keskin = tm[tm["paper_source"] == "keskin_2019"].copy()
        if len(keskin) > 0:
            keskin = normalize_columns(keskin)
            keskin = coerce_immunogenic(keskin)
            datasets["keskin_2019"] = keskin
            print(f"  keskin_2019: training_matrix.csv → {len(keskin)} rows")

    # ── Müller NCI: full TSV
    nci_candidates = list(Path("validation_papers/muller2023").glob("*.tsv"))
    nci_path = None
    for c in nci_candidates:
        try:
            peek = pd.read_csv(c, sep="\t", nrows=3)
            cols = [x.lower() for x in peek.columns]
            if "mt_binhaff" in cols or "mt_bindaff" in cols or any("bindaff" in x for x in cols):
                nci_path = c
                break
            if "mt_pep" in " ".join(cols) and "validated" in cols:
                nci_path = c
                break
        except Exception:
            continue
    if nci_path is None and nci_candidates:
        nci_path = nci_candidates[0]

    if nci_path:
        print(f"  muller_nci: loading {nci_path} (large file, please wait)...")
        t0 = time.time()
        df = pd.read_csv(nci_path, sep="\t", low_memory=False)
        print(f"  muller_nci: {len(df):,} rows loaded in {time.time()-t0:.1f}s")
        df = normalize_columns(df)
        df = coerce_immunogenic(df)
        if "paper_source" not in df.columns:
            df["paper_source"] = "muller_nci"
        if "cancer_type" not in df.columns:
            df["cancer_type"] = "mixed_solid"
        datasets["muller_nci"] = df
        print(f"  muller_nci: {len(df):,} rows after label filter")
    else:
        print("  muller_nci: NOT FOUND in validation_papers/muller2023/")

    # ── Ott full mutanome check
    haystack_candidates = [
        Path("validation_papers/ott_2017/ott_full_mutanome.csv"),
        Path("backend/validation/artifacts/ott_full_mutanome.csv"),
        Path("validation_papers/merged_somatic_mutations.csv"),
    ]
    haystack_found = False
    for hc in haystack_candidates:
        if hc.exists():
            print(f"  ott_full_mutanome: FOUND at {hc}")
            haystack_found = True
            break
    if not haystack_found:
        print("  HAYSTACK BLOCKED: Ott full mutanome not found.")
        print("    Need ~11,092 somatic mutation rows from Ott 2017 (Nature 547:217).")

    return datasets


# ── Feature computation ───────────────────────────────────────────────────────

def _tcr_positions(pep_len: int) -> list[int]:
    return TCR_POS.get(pep_len, list(range(2, max(3, pep_len - 1))))


def _safe_pep(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    return s if 8 <= len(s) <= 11 and all(c in HYDRO for c in s) else None


def compute_binding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binding_nm-derived features."""
    nm_col = None
    for c in ["binding_nm", "mhcflurry_affinity", "binding_affinity_nm_numeric",
              "binding_affinity_nm", "final_binding_affinity_nm"]:
        if c in df.columns:
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.notna().sum() > 5:
                nm_col = c
                break

    if nm_col:
        nm = pd.to_numeric(df[nm_col], errors="coerce")
        if "binding_nm" not in df.columns:
            df["binding_nm"] = nm
        else:
            nm = pd.to_numeric(df["binding_nm"], errors="coerce")
        nm = nm.clip(lower=0.01)
        df["binding_log"] = np.log10(nm)
        df["binding_sigmoid"] = 1.0 / (1.0 + (df["binding_log"] - np.log10(500)) / 0.5)
        df["binding_rank"] = df.groupby("patient_id")["binding_nm"].rank(pct=True)
        df["binding_log50k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
    else:
        for c in ["binding_nm", "binding_log", "binding_sigmoid", "binding_rank", "binding_log50k"]:
            if c not in df.columns:
                df[c] = np.nan

    # DAI from WT binding
    wt_col = None
    for c in ["wt_binding_nm", "wt_affinity", "WT_BindAff"]:
        if c in df.columns:
            wt_col = c
            break
    if wt_col and "binding_nm" in df.columns:
        wt_nm = pd.to_numeric(df[wt_col], errors="coerce").clip(lower=0.01)
        mt_nm = pd.to_numeric(df["binding_nm"], errors="coerce").clip(lower=0.01)
        df["dai_binding"] = np.log10(wt_nm) - np.log10(mt_nm)
        df["dai_ratio"] = wt_nm / (mt_nm + 1e-6)

    # Binding stability (from Müller NCI; may already be renamed to binding_stability)
    if "binding_stability" not in df.columns and "BindStab" in df.columns:
        df["binding_stability"] = pd.to_numeric(df["BindStab"], errors="coerce")

    return df


def compute_expression_features(df: pd.DataFrame) -> pd.DataFrame:
    tpm_col = None
    for c in ["expression_tpm", "expression_tpm_numeric", "Quantification",
              "tpm", "expression_score"]:
        if c in df.columns:
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.notna().sum() > 5:
                tpm_col = c
                break

    if tpm_col:
        tpm = pd.to_numeric(df[tpm_col], errors="coerce")
        if "expression_tpm" not in df.columns:
            df["expression_tpm"] = tpm
        else:
            tpm = pd.to_numeric(df["expression_tpm"], errors="coerce")
        df["expression_log2"] = np.log2(tpm.clip(lower=0) + 1)
        df["expression_binary"] = (tpm > 1).astype(float)
        # z-score within patient (skip if <3 rows per patient)
        # Use reset-index positions to avoid stale integer indices from sliced DataFrames
        df = df.reset_index(drop=True)
        zscores = np.full(len(df), np.nan)
        for pt, grp in df.groupby("patient_id"):
            if len(grp) >= 3:
                vals = pd.to_numeric(grp["expression_tpm"], errors="coerce")
                m, s = vals.mean(), vals.std()
                if s > 0:
                    zscores[grp.index] = ((vals - m) / s).values
        df["expression_zscore"] = zscores
    else:
        for c in ["expression_tpm", "expression_log2", "expression_binary", "expression_zscore"]:
            if c not in df.columns:
                df[c] = np.nan

    return df


def compute_clonality_features(df: pd.DataFrame) -> pd.DataFrame:
    vaf_col = None
    for c in ["vaf", "VAF", "variant_allele_freq"]:
        if c in df.columns:
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.notna().sum() > 5:
                vaf_col = c
                break
    if vaf_col:
        vaf = pd.to_numeric(df[vaf_col], errors="coerce")
        if "vaf" not in df.columns:
            df["vaf"] = vaf
        classes = np.where(vaf > 0.3, "clonal",
                  np.where(vaf < 0.15, "subclonal", "intermediate"))
        classes[vaf.isna().values] = np.nan
        df["clonality_class"] = classes
    else:
        if "vaf" not in df.columns:
            df["vaf"] = np.nan
        df["clonality_class"] = np.nan

    return df


def _tcr_props(pep: str) -> tuple[float, float, float, float, int]:
    """Returns (hydro_mean, hydro_max, volume_mean, charge_sum, aromatic_count)."""
    positions = _tcr_positions(len(pep))
    hydros = [HYDRO[pep[i]] for i in positions if i < len(pep) and pep[i] in HYDRO]
    vols = [VOLUME[pep[i]] for i in positions if i < len(pep) and pep[i] in VOLUME]
    chg = sum(CHARGE.get(pep[i], 0.0) for i in positions if i < len(pep))
    arom = sum(1 for i in positions if i < len(pep) and pep[i] in AROMATIC)
    h_mean = float(np.mean(hydros)) if hydros else np.nan
    h_max = float(max(hydros)) if hydros else np.nan
    v_mean = float(np.mean(vols)) if vols else np.nan
    return h_mean, h_max, v_mean, float(chg), int(arom)


def compute_tcr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TCR contact features from mutant_peptide (and wt_peptide if available)."""
    pep_col = "mutant_peptide" if "mutant_peptide" in df.columns else None
    wt_col = "wt_peptide" if "wt_peptide" in df.columns else None

    # Check if already computed from ott_mutation_level (may use old column names)
    already = (all(c in df.columns for c in ["tcr_hydro_mean", "tcr_volume_mean"]) or
               all(c in df.columns for c in ["tcr_hydrophobicity_mut", "tcr_volume_mut"]))
    if already:
        # Just alias to new names if needed
        if "tcr_hydro_mean" not in df.columns and "tcr_hydrophobicity_mut" in df.columns:
            df["tcr_hydro_mean"] = df["tcr_hydrophobicity_mut"]
        if "tcr_volume_mean" not in df.columns and "tcr_volume_mut" in df.columns:
            df["tcr_volume_mean"] = df["tcr_volume_mut"]
        # Ensure consistent naming
        for old, new in [
            ("tcr_hydrophobicity_mut", "tcr_hydro_mean"),
            ("tcr_volume_mut", "tcr_volume_mean"),
            ("tcr_charge_mut", "tcr_charge_sum"),
            ("tcr_aromaticity_mut", "tcr_aromatic_count"),
            ("tcr_hydrophobicity_diff", "tcr_hydro_diff"),
            ("tcr_volume_diff", "tcr_volume_diff"),
            ("tcr_charge_diff", "tcr_charge_diff"),
        ]:
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        return df

    if pep_col is None:
        for c in ["tcr_hydro_mean", "tcr_hydro_max", "tcr_volume_mean",
                  "tcr_charge_sum", "tcr_aromatic_count"]:
            df[c] = np.nan
        return df

    h_means, h_maxes, v_means, c_sums, a_counts = [], [], [], [], []
    h_diffs, v_diffs, c_diffs, mut_positions = [], [], [], []

    for _, row in df.iterrows():
        pep = _safe_pep(row[pep_col])
        if pep is None:
            h_means.append(np.nan)
            h_maxes.append(np.nan)
            v_means.append(np.nan)
            c_sums.append(np.nan)
            a_counts.append(np.nan)
            h_diffs.append(np.nan)
            v_diffs.append(np.nan)
            c_diffs.append(np.nan)
            mut_positions.append(np.nan)
            continue

        hm, hx, vm, cs, ac = _tcr_props(pep)
        h_means.append(hm)
        h_maxes.append(hx)
        v_means.append(vm)
        c_sums.append(cs)
        a_counts.append(ac)

        # WT-based diff features
        if wt_col and not pd.isna(row.get(wt_col, np.nan)):
            wt = _safe_pep(row[wt_col])
            if wt is not None and len(wt) == len(pep):
                wt_hm, _, wt_vm, wt_cs, _ = _tcr_props(wt)
                h_diffs.append(hm - wt_hm if not np.isnan(hm) and not np.isnan(wt_hm) else np.nan)
                v_diffs.append(vm - wt_vm if not np.isnan(vm) and not np.isnan(wt_vm) else np.nan)
                c_diffs.append(cs - wt_cs if not np.isnan(cs) and not np.isnan(wt_cs) else np.nan)
                # mutation position
                pos = next((i for i, (m, w) in enumerate(zip(pep, wt)) if m != w), np.nan)
                mut_positions.append(pos)
            else:
                for lst in [h_diffs, v_diffs, c_diffs, mut_positions]:
                    lst.append(np.nan)
        else:
            for lst in [h_diffs, v_diffs, c_diffs, mut_positions]:
                lst.append(np.nan)

    df["tcr_hydro_mean"] = h_means
    df["tcr_hydro_max"] = h_maxes
    df["tcr_volume_mean"] = v_means
    df["tcr_charge_sum"] = c_sums
    df["tcr_aromatic_count"] = a_counts
    if any(not np.isnan(x) if not isinstance(x, float) else not np.isnan(x) for x in h_diffs):
        df["tcr_hydro_diff"] = h_diffs
        df["tcr_volume_diff"] = v_diffs
        df["tcr_charge_diff"] = c_diffs
    df["mutation_position"] = mut_positions

    return df


def compute_foreignness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hamming, BLOSUM62, self-dissimilarity passthrough."""
    # Pass through self_dissimilarity if exists
    if "self_dissimilarity" in df.columns:
        df["self_dissimilarity"] = pd.to_numeric(df["self_dissimilarity"], errors="coerce")

    pep_col = "mutant_peptide" if "mutant_peptide" in df.columns else None
    wt_col = "wt_peptide" if "wt_peptide" in df.columns else None

    if pep_col is None or wt_col is None:
        if "hamming_distance" not in df.columns:
            df["hamming_distance"] = np.nan
        if "blosum62_score" not in df.columns:
            df["blosum62_score"] = np.nan
        df["blosum62_at_mutation"] = np.nan
        return df

    hammings, blossums, blos_at_mut = [], [], []

    for _, row in df.iterrows():
        mut = row.get(pep_col)
        wt = row.get(wt_col)
        if pd.isna(mut) or pd.isna(wt) or str(mut).strip() == "" or str(wt).strip() == "":
            hammings.append(np.nan)
            blossums.append(np.nan)
            blos_at_mut.append(np.nan)
            continue
        mut, wt = str(mut).strip().upper(), str(wt).strip().upper()
        if len(mut) != len(wt) or len(mut) == 0:
            hammings.append(np.nan)
            blossums.append(np.nan)
            blos_at_mut.append(np.nan)
            continue

        # Hamming
        hammings.append(sum(m != w for m, w in zip(mut, wt)))

        # BLOSUM62
        if BLOSUM62 is not None:
            alphabet = set(BLOSUM62.alphabet) if hasattr(BLOSUM62, "alphabet") else set()
            total = 0.0
            count = 0
            mut_pos_score = np.nan
            for i, (m, w) in enumerate(zip(mut, wt)):
                if m in alphabet and w in alphabet:
                    try:
                        sc = BLOSUM62[m][w]
                        total += sc
                        count += 1
                        if m != w and np.isnan(mut_pos_score):
                            mut_pos_score = float(sc)
                    except Exception:
                        pass
            blossums.append(total if count > 0 else np.nan)
            blos_at_mut.append(mut_pos_score)
        else:
            blossums.append(np.nan)
            blos_at_mut.append(np.nan)

    if "hamming_distance" not in df.columns:
        df["hamming_distance"] = hammings
    df["blosum62_score"] = blossums
    df["blosum62_at_mutation"] = blos_at_mut

    return df


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sequence-level features from mutant_peptide."""
    pep_col = "mutant_peptide" if "mutant_peptide" in df.columns else None
    if pep_col is None:
        for c in ["pep_length", "calis_simplified", "hydro_full_mean",
                  "hydro_full_max", "aliphatic_index"]:
            if c not in df.columns:
                df[c] = np.nan
        return df

    lengths, calis_scores, hydro_means, hydro_maxes, aliph = [], [], [], [], []
    mw_list, pi_list, instab_list = [], [], []

    for _, row in df.iterrows():
        pep_raw = row.get(pep_col)
        if pd.isna(pep_raw):
            for lst in [lengths, calis_scores, hydro_means, hydro_maxes, aliph,
                        mw_list, pi_list, instab_list]:
                lst.append(np.nan)
            continue
        pep = str(pep_raw).strip().upper()
        n = len(pep)

        lengths.append(n)

        # Full-sequence hydrophobicity
        all_hydro = [HYDRO[aa] for aa in pep if aa in HYDRO]
        hydro_means.append(float(np.mean(all_hydro)) if all_hydro else np.nan)
        hydro_maxes.append(float(max(all_hydro)) if all_hydro else np.nan)

        # Aliphatic index
        cnt_A = pep.count("A")
        cnt_V = pep.count("V")
        cnt_I = pep.count("I")
        cnt_L = pep.count("L")
        aliph.append((cnt_A * 2.9 + cnt_V * 3.9 + (cnt_I + cnt_L) * 19.0) / n if n > 0 else np.nan)

        # Calis simplified
        if n in CALIS_WEIGHTS:
            ws = CALIS_WEIGHTS[n]
            sc = sum(HYDRO.get(aa, 0.0) * ws[i] for i, aa in enumerate(pep) if aa in HYDRO)
            calis_scores.append(sc)
        else:
            calis_scores.append(np.nan)

        # Biopython ProteinAnalysis
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            std_aas = set("ACDEFGHIKLMNPQRSTVWY")
            clean = "".join(aa for aa in pep if aa in std_aas)
            if len(clean) >= 4:
                pa = ProteinAnalysis(clean)
                mw_list.append(float(pa.molecular_weight()))
                pi_list.append(float(pa.isoelectric_point()))
                instab_list.append(float(pa.instability_index()))
            else:
                mw_list.append(np.nan)
                pi_list.append(np.nan)
                instab_list.append(np.nan)
        except Exception:
            mw_list.append(np.nan)
            pi_list.append(np.nan)
            instab_list.append(np.nan)

    df["pep_length"] = lengths
    df["calis_simplified"] = calis_scores
    df["hydro_full_mean"] = hydro_means
    df["hydro_full_max"] = hydro_maxes
    df["aliphatic_index"] = aliph
    df["molecular_weight"] = mw_list
    df["isoelectric_point"] = pi_list
    df["instability_index"] = instab_list

    return df


def compute_all_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Apply all feature computation steps to a dataset."""
    print(f"  → Features: {dataset_name} ({len(df)} rows)...", end="", flush=True)
    t0 = time.time()

    # Skip slow per-row computation for large NCI dataset where most features are NA
    is_nci = "nci" in dataset_name.lower()

    df = compute_binding_features(df)
    df = compute_expression_features(df)
    df = compute_clonality_features(df)

    if not is_nci:
        df = compute_tcr_features(df)
        df = compute_foreignness_features(df)
        df = compute_sequence_features(df)
    else:
        # NCI: skip per-row slow ops, just do basic ones from existing columns
        print(" [NCI: skip per-row slow features]", end="", flush=True)
        for c in ["tcr_hydro_mean", "tcr_hydro_max", "tcr_volume_mean",
                  "tcr_charge_sum", "tcr_aromatic_count", "tcr_hydro_diff",
                  "tcr_volume_diff", "tcr_charge_diff", "mutation_position",
                  "hamming_distance", "blosum62_score", "blosum62_at_mutation",
                  "pep_length", "calis_simplified", "hydro_full_mean",
                  "hydro_full_max", "aliphatic_index",
                  "molecular_weight", "isoelectric_point", "instability_index"]:
            if c not in df.columns:
                df[c] = np.nan
        # pep_length from peptide length
        if "mutant_peptide" in df.columns:
            df["pep_length"] = df["mutant_peptide"].apply(
                lambda x: len(str(x).strip()) if pd.notna(x) else np.nan
            )

    print(f" done in {time.time()-t0:.1f}s")
    return df


# ── Feature inventory ─────────────────────────────────────────────────────────
EXPECTED_FEATURES = [
    "binding_nm", "binding_log", "binding_sigmoid", "binding_rank", "binding_log50k",
    "binding_stability", "dai_binding", "dai_ratio", "dai_agretopicity",
    "presentation_score_el",
    "expression_tpm", "expression_log2", "expression_binary", "expression_zscore",
    "vaf", "clonality_class",
    "tcr_hydro_mean", "tcr_hydro_max", "tcr_volume_mean", "tcr_charge_sum", "tcr_aromatic_count",
    "tcr_hydro_diff", "tcr_volume_diff", "tcr_charge_diff",
    "self_dissimilarity", "hamming_distance", "blosum62_score", "blosum62_at_mutation",
    "pep_length", "calis_simplified", "hydro_full_mean", "hydro_full_max",
    "aliphatic_index", "molecular_weight", "isoelectric_point", "instability_index",
    "mutation_position",
]


def feature_stats(series: pd.Series) -> dict:
    # Always coerce to numeric; non-numeric (e.g. clonality_class) will yield all NaN
    s = pd.to_numeric(series, errors="coerce")
    n_valid = int(s.notna().sum())
    n_total = len(s)
    coverage = n_valid / n_total if n_total > 0 else 0.0
    result = {
        "coverage": round(coverage, 3),
        "n_valid": n_valid,
        "n_total": n_total,
        "usable": coverage >= 0.80,
    }
    if n_valid > 0:
        try:
            result["mean"] = round(float(s.mean()), 4)
            result["std"] = round(float(s.std()), 4)
        except Exception:
            pass
    return result


def build_inventory(datasets: dict[str, pd.DataFrame]) -> dict:
    inventory: dict = {}
    for name, df in datasets.items():
        n_immuno = int(df["immunogenic"].sum()) if "immunogenic" in df.columns else 0
        n_rows = len(df)
        n_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
        cancer_type = df["cancer_type"].iloc[0] if "cancer_type" in df.columns and len(df) > 0 else "unknown"

        feats: dict = {}
        for feat in EXPECTED_FEATURES:
            if feat in df.columns:
                feats[feat] = feature_stats(df[feat])
            else:
                feats[feat] = {"coverage": 0.0, "n_valid": 0, "n_total": n_rows, "usable": False}

        inventory[name] = {
            "rows": n_rows,
            "immunogenic": n_immuno,
            "patients": n_patients,
            "cancer_type": str(cancer_type),
            "features": feats,
        }
    return inventory


def print_coverage_table(inventory: dict) -> None:
    """Print feature × dataset coverage table."""
    datasets = list(inventory.keys())
    print("\n" + "=" * 90)
    print("FEATURE COVERAGE TABLE (U=USABLE >80%, P=PARTIAL 20-80%, M=MISSING <20%)")
    print("=" * 90)
    header = f"{'Feature':<35}" + "".join(f"{d[:10]:>12}" for d in datasets)
    print(header)
    print("-" * 90)

    usable_counts = {d: 0 for d in datasets}
    for feat in EXPECTED_FEATURES:
        row = f"{feat:<35}"
        for d in datasets:
            cov = inventory[d]["features"].get(feat, {}).get("coverage", 0.0)
            if cov >= 0.80:
                sym = f"U({cov:.0%})"
                usable_counts[d] += 1
            elif cov >= 0.20:
                sym = f"P({cov:.0%})"
            else:
                sym = "M"
            row += f"{sym:>12}"
        print(row)

    print("-" * 90)
    usage_row = f"{'USABLE FEATURES':<35}" + "".join(f"{usable_counts[d]:>12}" for d in datasets)
    print(usage_row)
    print("=" * 90)


def main() -> int:
    print("=" * 60)
    print("PHASE 1 — FEATURE FACTORY")
    print("=" * 60)

    # Step 1A: Load datasets
    print("\n[1A] Loading datasets...")
    datasets = load_datasets()
    print(f"\nDatasets loaded: {list(datasets.keys())}")

    # Print dataset audit
    print("\n--- Dataset Audit ---")
    for name, df in datasets.items():
        n_immuno = int(df["immunogenic"].sum()) if "immunogenic" in df.columns else "?"
        n_pts = int(df["patient_id"].nunique()) if "patient_id" in df.columns else "?"
        ct = df["cancer_type"].iloc[0] if "cancer_type" in df.columns and len(df) > 0 else "?"
        print(f"  {name:<20} rows={len(df):>7,}  immuno={str(n_immuno):>5}  "
              f"patients={str(n_pts):>3}  cancer={ct}")
        print(f"    columns: {df.columns.tolist()[:10]}...")

    # Step 1B: Compute features
    print("\n[1B] Computing features...")
    featured: dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        featured[name] = compute_all_features(df.copy(), name)

    # Step 1C: Save feature matrices
    print("\n[1C] Saving feature matrices...")
    for name, df in featured.items():
        out_path = ARTIFACTS / f"{name}_features.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df):,} rows × {len(df.columns)} cols)")

    # Build inventory
    inventory = build_inventory(featured)

    inv_path = ARTIFACTS / "feature_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"  Saved inventory: {inv_path}")

    # Step 1D: Print coverage table
    print_coverage_table(inventory)

    # Summary
    print("\n--- Phase 1 Summary ---")
    for name, inv in inventory.items():
        usable = sum(1 for f in inv["features"].values() if f["usable"])
        print(f"  {name:<20}: {inv['rows']:>7,} rows | {inv['immunogenic']:>5} pos | "
              f"{inv['patients']:>3} patients | {usable}/{len(EXPECTED_FEATURES)} features USABLE")

    print("\nPhase 1 complete.")
    time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
