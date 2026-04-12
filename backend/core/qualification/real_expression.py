"""
Join TCGA RNA-seq TPM from cohort metadata onto candidate rows.

Expects ``rna_tpm_dict`` JSON per patient (from :mod:`backend.core.data.tcga_data`).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _parse_tpm_dict(val: Any) -> dict[str, float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return {}
    if isinstance(val, dict):
        return {str(k): float(v) for k, v in val.items() if v is not None and str(v) != "nan"}
    if isinstance(val, str) and val.strip() in ("", "{}", "null"):
        return {}
    if isinstance(val, str):
        try:
            d = json.loads(val)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def lookup_gene_tpm(tpm_dict: dict[str, float], gene: str) -> float | None:
    if not gene:
        return None
    g = str(gene).strip()
    if g in tpm_dict:
        return float(tpm_dict[g])
    g_upper = g.upper()
    for k, v in tpm_dict.items():
        if str(k).upper() == g_upper:
            return float(v)
    return None


def tpm_expression_bin(tpm: float | None) -> str:
    if tpm is None:
        return "missing"
    try:
        x = float(tpm)
    except (TypeError, ValueError):
        return "missing"
    if x != x:  # NaN
        return "missing"
    if tpm < 1.0:
        return "unexpressed"
    if tpm < 10.0:
        return "low"
    if tpm < 100.0:
        return "medium"
    return "high"


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``real_expression_tpm``, ``expression_bin``, ``RNA_data_missing``.

    Join columns must already be present (e.g. ``rna_tpm_dict`` from metadata merge).
    """
    out = df.copy()
    if out.empty:
        return out

    tpm_col: list[float | None] = []
    bin_col: list[str] = []
    miss_col: list[bool] = []

    for _, row in out.iterrows():
        d = _parse_tpm_dict(row.get("rna_tpm_dict"))
        gene = row.get("gene") or row.get("gene_name")
        tpm = lookup_gene_tpm(d, str(gene)) if gene is not None else None
        if tpm is None:
            miss_col.append(True)
            tpm_col.append(None)
            bin_col.append("missing")
        else:
            miss_col.append(False)
            tpm_col.append(float(tpm))
            bin_col.append(tpm_expression_bin(float(tpm)))

    out["real_expression_tpm"] = tpm_col
    out["expression_bin"] = bin_col
    out["RNA_data_missing"] = miss_col

    new_reasons: list[list[str]] = []
    for _, row in out.iterrows():
        reasons = row.get("exclusion_reasons", [])
        base = list(reasons) if isinstance(reasons, list) else []
        if row["RNA_data_missing"] and "expression:rna_missing" not in base:
            base.append("expression:rna_missing")
        if row["expression_bin"] == "unexpressed" and "expression:unexpressed_real" not in base:
            base.append("expression:unexpressed_real")
        new_reasons.append(base)
    out["exclusion_reasons"] = new_reasons

    return out
