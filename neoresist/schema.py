"""
Minimal canonical column contract for Dash candidate tables.

See docs/column_contract.md. Only explicit renames — no inference across MAF preview artifacts.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# Incoming synonym -> canonical (first match wins per column)
COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "patient_id": ("patient", "caseid", "case_id", "barcode", "Patient", "PATIENT_ID"),
    "hla_allele": ("hla", "allele", "HLA"),
    "gene": ("gene_name", "Gene", "Hugo_Symbol"),
    "mutant_peptide": ("peptide", "Peptide", "MT.peptide"),
}

DASH_CANDIDATE_REQUIRED = (
    "patient_id",
    "hla_allele",
    "tier",
    "rl_priority",
    "exclusion_reasons",
)


def normalize_cohort_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    lower_map = {c.lower(): c for c in out.columns}
    for canonical, aliases in COLUMN_SYNONYMS.items():
        if canonical in out.columns:
            continue
        for a in aliases:
            key = a.lower()
            if key in lower_map:
                src = lower_map[key]
                out[canonical] = out[src]
                break
    return out


def validate_dash_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (missing_required, present_columns)."""
    present = [str(c) for c in df.columns]
    missing = [c for c in DASH_CANDIDATE_REQUIRED if c not in df.columns]
    return missing, present


def format_validation_message(missing: list[str], present: list[str], *, searched: list[str]) -> str:
    lines = [
        "Missing required canonical columns after normalization:",
        ", ".join(missing) if missing else "(none)",
        "",
        "Columns present (sample):",
        ", ".join(sorted(present)[:40]) + (" …" if len(present) > 40 else ""),
        "",
        "Searched paths:",
        "\n".join(searched),
    ]
    return "\n".join(lines)


Mode = Literal["dash", "enrich_input"]
