from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd

LOHState = Literal["intact", "uncertain", "lost"]


def _loh_stub(patient_id: str, hla_allele: str) -> LOHState:
    key = f"loh|{patient_id}|{hla_allele}".encode("utf-8")
    h = int(hashlib.md5(key, usedforsecurity=False).hexdigest(), 16) % 3
    return ("intact", "uncertain", "lost")[h]


def escape_penalty(flag: str | None) -> float:
    if flag == "lost":
        return 1.0
    if flag == "uncertain":
        return 0.5
    return 0.0


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Stub HLA LOH status per ``(patient_id, hla_allele)`` and derive ``escape_penalty``."""
    out = df.copy()
    if out.empty:
        return out

    flags = [_loh_stub(str(r["patient_id"]), str(r["hla_allele"])) for _, r in out.iterrows()]
    out["hla_loh_flag"] = flags
    out["escape_penalty"] = [escape_penalty(f) for f in flags]

    new_reasons: list[list[str]] = []
    for _, row in out.iterrows():
        reasons = row.get("exclusion_reasons", [])
        reasons = list(reasons) if isinstance(reasons, list) else []
        flag = row["hla_loh_flag"]
        if flag == "lost" and "escape:hla_loh_lost" not in reasons:
            reasons.append("escape:hla_loh_lost")
        elif flag == "uncertain" and "escape:hla_loh_uncertain" not in reasons:
            reasons.append("escape:hla_loh_uncertain")
        new_reasons.append(reasons)
    out["exclusion_reasons"] = new_reasons
    return out
