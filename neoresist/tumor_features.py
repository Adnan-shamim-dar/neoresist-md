from __future__ import annotations

from typing import Any

import pandas as pd

TUMOR_TYPES = ("SKCM", "LUAD", "SARC", "PAAD", "COAD", "BLCA", "GBM", "OTHER", "UNKNOWN")
KRAS_G12_PATTERN = r"\bG12(?:C|D|V|R|A|S)\b"
KNOWN_MSI_NEOANTIGENS = {
    "RLAAMLASL",  # TGFBR2 frameshift
    "SLVRLSSSL",  # TGFBR2
    "TLSPKHALA",  # ACVR2A frameshift
    "LSALSPHAL",  # ACVR2A
}
TUMOR_TYPE_KEYWORDS = {
    "melanoma": "SKCM",
    "skcm": "SKCM",
    "lung": "LUAD",
    "luad": "LUAD",
    "lusc": "LUAD",
    "breast": "OTHER",
    "brca": "OTHER",
    "pancrea": "PAAD",
    "paad": "PAAD",
    "glioblastoma": "GBM",
    "gbm": "GBM",
    "sarc": "SARC",
    "sarcoma": "SARC",
    "colon": "COAD",
    "coad": "COAD",
    "blca": "BLCA",
    "bladder": "BLCA",
}


def normalise_tumor_type(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper().replace("-", "_")
    return text if text in TUMOR_TYPES else "OTHER"


def infer_tumor_type_from_text(*values: Any) -> str:
    joined = " ".join(str(v or "").upper().replace("-", "_") for v in values)
    for tumor_type in TUMOR_TYPES:
        if tumor_type not in {"OTHER", "UNKNOWN"} and tumor_type in joined:
            return tumor_type
    lowered = joined.lower()
    for keyword, tumor_type in TUMOR_TYPE_KEYWORDS.items():
        if keyword in lowered:
            return tumor_type
    return "UNKNOWN"


def single_tumor_type(df: pd.DataFrame) -> str | None:
    if df.empty or "tumor_type" not in df.columns:
        return None
    vals = {normalise_tumor_type(v) for v in df["tumor_type"].dropna().astype(str)}
    vals.discard("")
    return next(iter(vals)) if len(vals) == 1 else None


def _first_column(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _boolish(series: pd.Series) -> pd.Series:
    text = series.fillna(False).astype(str).str.strip().str.upper()
    return text.isin({"1", "TRUE", "T", "YES", "Y"})


def add_candidate_tumor_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["kras_g12_flag"] = []
        out["shared_neoantigen_flag"] = []
        return out

    gene = _first_column(out, ("gene", "gene_name", "Hugo_Symbol")).fillna("").astype(str).str.upper()
    protein_change = _first_column(
        out,
        ("protein_change", "source_hgvsp_short", "HGVSp_Short", "HGVSp", "hgvsp_short"),
    ).fillna("").astype(str).str.upper()
    position = pd.to_numeric(_first_column(out, ("mutation_position", "peptide_position", "pos")), errors="coerce")
    peptide = _first_column(out, ("mutant_peptide", "peptide")).fillna("").astype(str).str.upper().str.strip()
    kras_g12 = gene.eq("KRAS") & (
        protein_change.str.contains(KRAS_G12_PATTERN, regex=True, na=False) | position.eq(12)
    )

    if "kras_g12_flag" not in out.columns:
        out["kras_g12_flag"] = kras_g12
    else:
        out["kras_g12_flag"] = _boolish(out["kras_g12_flag"]) | kras_g12

    if "shared_neoantigen_flag" not in out.columns:
        out["shared_neoantigen_flag"] = peptide.isin(KNOWN_MSI_NEOANTIGENS)
    else:
        out["shared_neoantigen_flag"] = _boolish(out["shared_neoantigen_flag"]) | peptide.isin(KNOWN_MSI_NEOANTIGENS)
    return out


def detect_kras_g12(df: pd.DataFrame) -> pd.Series:
    return add_candidate_tumor_flags(df)["kras_g12_flag"]


def estimate_msi_status(df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df)) if df is not None else 0
    if total <= 0:
        return {"msi_status": "INDETERMINATE", "msi_frameshift_indel_ratio": 0.0, "msi_total_mutations": 0, "msi_frameshift_indels": 0}

    variant_class = _first_column(
        df,
        ("Variant_Classification", "variant_classification", "consequence", "mutation_type"),
    ).fillna("").astype(str)
    variant_type = _first_column(df, ("Variant_Type", "variant_type", "type")).fillna("").astype(str).str.upper()
    protein_change = _first_column(
        df,
        ("protein_change", "source_hgvsp_short", "HGVSp_Short", "HGVSp", "hgvsp_short"),
    ).fillna("").astype(str)

    class_fs = variant_class.str.contains("FRAME_SHIFT|FRAMESHIFT", case=False, regex=True, na=False)
    type_indel = variant_type.isin({"INS", "DEL", "INDEL", "INSERTION", "DELETION"})
    hgvsp_fs = protein_change.str.contains("FS|FRAMESHIFT", case=False, regex=True, na=False)
    frameshift = class_fs | (type_indel & hgvsp_fs)
    fs_count = int(frameshift.sum())
    ratio = float(fs_count / max(total, 1))
    if total < 50:
        status = "INDETERMINATE"
    elif ratio > 0.15:
        status = "MSI-H"
    else:
        status = "MSS"
    return {
        "msi_status": status,
        "msi_frameshift_indel_ratio": ratio,
        "msi_total_mutations": total,
        "msi_frameshift_indels": fs_count,
    }
