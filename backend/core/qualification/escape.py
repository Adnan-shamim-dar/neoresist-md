from __future__ import annotations

from typing import Literal

import pandas as pd

LOHState = Literal["intact", "uncertain", "lost", "unknown"]


def escape_penalty(flag: str | None) -> float:
    if flag == "lost":
        return 1.0
    return 0.0


def apply_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve explicit LOH calls and default to zero penalty when no escape evidence exists."""
    out = df.copy()
    if out.empty:
        return out

    existing_flags = out.get("hla_loh_flag", out.get("hla_loh_status", pd.Series([pd.NA] * len(out), index=out.index)))
    flags = existing_flags.fillna("unknown").astype(str).str.lower()
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
