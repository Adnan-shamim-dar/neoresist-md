from __future__ import annotations

import math

import pandas as pd


def _stub_presentation_score(peptide: str, affinity_nm: float | None) -> float:
    """
    Deterministic NetCTLpan-style placeholder in ``[0, 1]`` from peptide length and affinity.
    Lower affinity (nM) implies stronger binding; map with a smooth saturating curve.
    """
    length = max(len(peptide), 1)
    length_term = (length - 8) / 3.0
    length_term = max(0.0, min(1.0, length_term))

    aff = float(affinity_nm) if affinity_nm is not None and not math.isnan(affinity_nm) else 5000.0
    aff = max(aff, 1.0)
    aff_term = 1.0 - (math.log10(aff) / math.log10(5000.0))
    aff_term = max(0.0, min(1.0, aff_term))

    score = 0.45 * length_term + 0.55 * aff_term
    return max(0.0, min(1.0, score))


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``presentation_score`` with a deterministic stub for the qualification chain."""
    out = df.copy()
    if out.empty:
        return out

    scores = []
    for _, row in out.iterrows():
        pep = str(row.get("mutant_peptide", row.get("peptide", "")))
        aff = row.get("affinity_nm")
        try:
            aff_f = float(aff) if aff is not None and not pd.isna(aff) else None
        except (TypeError, ValueError):
            aff_f = None
        scores.append(_stub_presentation_score(pep, aff_f))
    out["presentation_score"] = scores
    return out
