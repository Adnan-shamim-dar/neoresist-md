from __future__ import annotations

import hashlib

import pandas as pd


def _ccf_stub(patient_id: str, gene: str) -> float:
    """Deterministic PyClone-VI placeholder CCF in ``[0, 1]``."""
    key = f"ccf|{patient_id}|{gene}".encode("utf-8")
    h = int(hashlib.sha256(key, usedforsecurity=False).hexdigest(), 16)
    return (h % 10_000) / 9999.0


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Stub PyClone-VI CCF per ``(patient_id, gene)``."""
    out = df.copy()
    if out.empty:
        return out
    out["ccf"] = [_ccf_stub(str(r["patient_id"]), str(r["gene"])) for _, r in out.iterrows()]
    return out
