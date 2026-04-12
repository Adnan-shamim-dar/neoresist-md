from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def _tpm_stub(patient_id: str, gene: str) -> float:
    """Deterministic placeholder TPM in ``(0, 120)`` for development."""
    key = f"{patient_id}|{gene}".encode("utf-8")
    h = int(hashlib.md5(key, usedforsecurity=False).hexdigest(), 16)
    unit = (h % 10_000) / 9999.0
    return 0.05 + 119.95 * unit


def _append_reason(reasons: Any, tag: str) -> list[str]:
    base = list(reasons) if isinstance(reasons, list) else []
    if tag not in base:
        base.append(tag)
    return base


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub TCGA RNA-seq join: assign placeholder TPM per ``(patient_id, gene)``.
    Flags ``expression_unexpressed`` and appends ``expression:unexpressed`` when TPM < 1.
    """
    out = df.copy()
    if out.empty:
        return out

    tpm_values = [
        _tpm_stub(str(r["patient_id"]), str(r["gene"])) for _, r in out.iterrows()
    ]
    out["expression_tpm"] = tpm_values
    out["expression_unexpressed"] = out["expression_tpm"] < 1.0

    new_reasons: list[list[str]] = []
    for _, row in out.iterrows():
        reasons = row.get("exclusion_reasons", [])
        if row["expression_unexpressed"]:
            reasons = _append_reason(reasons, "expression:unexpressed")
        new_reasons.append(reasons)
    out["exclusion_reasons"] = new_reasons
    return out
