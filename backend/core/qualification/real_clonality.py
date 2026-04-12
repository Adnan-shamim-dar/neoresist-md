"""
Naive CCF from VAF and tumor purity + PyClone-VI-style stub inputs.

Uses ``purity_value_used`` when present (Phase 5), else ``purity``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

EPS = 1e-6


def naive_ccf_from_vaf_purity(vaf: Any, purity: Any) -> float | None:
    try:
        v = float(vaf)
        p = float(purity)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isnan(p) or p <= 0.0:
        return None
    return max(0.0, min(1.0, v / max(p, EPS)))


def _purity_for_row(row: pd.Series) -> tuple[float | None, str]:
    """Return (purity scalar, label for provenance)."""
    if "purity_value_used" in row.index:
        pv = row.get("purity_value_used")
        if pv is not None and not pd.isna(pv):
            try:
                fp = float(pv)
                if math.isnan(fp):
                    raise ValueError
                return fp, "purity_value_used"
            except (TypeError, ValueError):
                pass
    p = row.get("purity")
    if p is not None and not pd.isna(p):
        try:
            fp = float(p)
            if not math.isnan(fp):
                return fp, "purity_legacy"
        except (TypeError, ValueError):
            pass
    return None, "missing"


def write_pyclone_stub_tsv(out_dir: Path, patient_id: str, rows: pd.DataFrame) -> Path:
    """
    Write a minimal ``mutations.tsv`` per patient for PyClone-VI-style pipelines.

    Columns are illustrative; extend when wiring real PyClone-VI.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(patient_id))
    pdir = out_dir / safe
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / "mutations.tsv"

    lines = ["mutation_id\tref_counts\talt_counts\ttotal_depth\ttumour_content\tnormal_cn\tmajor_cn\tminor_cn\n"]
    for i, (_, r) in enumerate(rows.iterrows()):
        gene = str(r.get("gene") or r.get("gene_name") or f"mut{i}")
        vaf = r.get("vaf")
        pu_val, _src = _purity_for_row(r)
        try:
            vf = float(vaf) if vaf is not None and not pd.isna(vaf) else 0.5
        except (TypeError, ValueError):
            vf = 0.5
        if pu_val is not None and not math.isnan(pu_val):
            pu = float(pu_val)
        else:
            pu = 0.8
        ref_c = max(10, int(round(100 * (1 - vf))))
        alt_c = max(1, int(round(100 * vf)))
        tot = ref_c + alt_c
        lines.append(f"{gene}\t{ref_c}\t{alt_c}\t{tot}\t{pu:.4f}\t2\t1\t1\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def apply_layer(df: pd.DataFrame, *, pyclone_out_dir: Path | None = None) -> pd.DataFrame:
    """
    Add ``real_ccf``, ``ccf_naive``, ``clonal_fraction_placeholder``, ``clonality_data_missing``,
    and Phase 5 clonality provenance columns.
    """
    out = df.copy()
    if out.empty:
        return out

    real_ccf: list[float | None] = []
    naive: list[float | None] = []
    miss: list[bool] = []
    frac_ph: list[float] = []
    src_used: list[str] = []
    conf: list[float] = []
    reason: list[str] = []
    cnv_note: list[str] = []

    for _, row in out.iterrows():
        vaf = row.get("vaf")
        pu, pu_lab = _purity_for_row(row)
        c = naive_ccf_from_vaf_purity(vaf, pu)
        naive.append(c)

        has_cnv = False
        raw_cnv = row.get("cnv_segments")
        if isinstance(raw_cnv, str) and raw_cnv not in ("", "[]", "null"):
            has_cnv = len(raw_cnv) > 2
        elif isinstance(raw_cnv, list) and len(raw_cnv) > 0:
            has_cnv = True

        if c is None:
            real_ccf.append(None)
            miss.append(True)
            frac_ph.append(0.5)
            src_used.append("unresolved")
            conf.append(0.0)
            reason.append(
                "ccf_not_computed: missing VAF or non-positive purity; "
                f"purity_field={pu_lab}"
            )
        else:
            real_ccf.append(float(c))
            miss.append(False)
            frac_ph.append(float(c))
            src_used.append("vaf_over_purity_naive")
            pconf = float(row.get("purity_confidence_used") or 0.75)
            conf.append(max(0.0, min(1.0, pconf * 0.9)))
            reason.append(
                f"naive_ccf=vaf/purity using {pu_lab}; "
                f"cnv_segments={'present' if has_cnv else 'absent_placeholder'}"
            )
        cnv_note.append("absent" if not has_cnv else "present")

    out["ccf_naive"] = naive
    out["real_ccf"] = real_ccf
    out["clonal_fraction_placeholder"] = frac_ph
    out["clonality_data_missing"] = miss
    out["clonality_source_used"] = src_used
    out["clonality_confidence"] = conf
    out["clonality_resolution_reason"] = reason
    out["cnv_presence_note"] = cnv_note

    if pyclone_out_dir is not None:
        pyclone_out_dir.mkdir(parents=True, exist_ok=True)
        for pid, grp in out.groupby(out["patient_id"].astype(str)):
            write_pyclone_stub_tsv(pyclone_out_dir, pid, grp)

    return out
