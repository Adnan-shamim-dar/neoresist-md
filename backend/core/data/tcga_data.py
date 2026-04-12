"""
TCGA-SARC metadata orchestration.

Full pulls use ``backend/scripts/pull_tcga_sarc.R`` (TCGAbiolinks) via ``Rscript``.
For CI and offline work, use :func:`write_stub_tcga_metadata` or ``--stub`` on the CLI.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_metadata_path() -> Path:
    return _repo_root() / "data" / "intermediate" / "tcga_sarc_metadata.parquet"


def case_barcode_from_aliquot(patient_id: str) -> str:
    """
    Map a TCGA aliquot / sample id to a case-style key for cohort joins.

    Uses the first three hyphen-separated tokens (``TCGA-PROJECT-CASE``) when possible,
    otherwise returns the trimmed input unchanged.
    """
    pid = str(patient_id).strip()
    parts = pid.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3]).upper()
    return pid.upper()


def sample_barcode_prefix16(patient_id: str) -> str:
    """First 16 characters of barcode (common GDC sample id length), hyphens preserved."""
    pid = str(patient_id).strip()
    return pid[:16] if len(pid) >= 16 else pid


def load_tcga_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def join_tcga_metadata(candidates: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Left-join TCGA metadata onto candidates (``patient_id`` or ``case_barcode``)."""
    if candidates.empty:
        return candidates.copy()
    if meta.empty:
        out = candidates.copy()
        out["tcga_join_status"] = "no_metadata"
        return out

    m = meta.copy()
    out = candidates.copy()

    if "patient_id" in m.columns:
        merged = out.merge(m, on="patient_id", how="left", suffixes=("", "_tcga"))
    elif "barcode" in m.columns:
        merged = out.merge(m, left_on="patient_id", right_on="barcode", how="left", suffixes=("", "_tcga"))
    else:
        m = m.copy()
        m["patient_id"] = m.get("case_barcode", m.iloc[:, 0]).astype(str)
        merged = out.merge(m, on="patient_id", how="left", suffixes=("", "_tcga"))

    def _status(row: pd.Series) -> str:
        if pd.notna(row.get("purity")):
            return "matched"
        raw = row.get("rna_tpm_dict")
        if isinstance(raw, str) and len(raw) > 2 and raw not in ("{}", "null"):
            return "matched"
        if isinstance(raw, dict) and raw:
            return "matched"
        return "missing"

    merged["tcga_join_status"] = merged.apply(_status, axis=1)
    return merged


def write_stub_tcga_metadata(
    patient_ids: list[str],
    out_path: Path,
    *,
    genes: list[str] | None = None,
) -> Path:
    """
    Write deterministic stub metadata for testing (TPM dict per patient, purity, empty CNV).

    TPM values are stable hashes of ``patient_id|gene`` mapped to ``(0.05, 200)``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genes = sorted(set(genes or ["TP53", "TTN", "MUC16", "BRCA1"]))
    rows: list[dict[str, Any]] = []
    for pid in patient_ids:
        tpm: dict[str, float] = {}
        for g in genes:
            key = f"{pid}|{g}".encode("utf-8")
            h = int(hashlib.md5(key, usedforsecurity=False).hexdigest(), 16)
            tpm[g] = round(0.05 + (h % 10000) / 9999.0 * 199.95, 6)
        rows.append(
            {
                "patient_id": pid,
                "barcode": pid,
                "case_barcode": case_barcode_from_aliquot(pid),
                "sample_prefix16": sample_barcode_prefix16(pid),
                "rna_tpm_dict": json.dumps(tpm),
                "rna_long_path": "",
                "purity": 0.82,
                "purity_method": "stub_python_hash_tpm",
                "purity_confidence": 0.55,
                "cnv_segments": json.dumps([]),
                "normal_bam_path": "",
                "source": "stub_python",
            }
        )
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def run_tcga_pull_subprocess(
    *,
    out_dir: Path,
    patient_list_file: Path | None = None,
    stub: bool = False,
    rscript_executable: str = "Rscript",
) -> int:
    """
    Run ``pull_tcga_sarc.R``. Returns process exit code (0 = success).

    With ``--stub`` the R script writes minimal parquet without hitting GDC.
    """
    script = _repo_root() / "backend" / "scripts" / "pull_tcga_sarc.R"
    if not script.is_file():
        print(f"ERROR: Missing R script at {script}", file=sys.stderr)
        return 127
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        rscript_executable,
        str(script),
        "--out-dir",
        str(out_dir),
    ]
    if patient_list_file is not None:
        cmd.extend(["--patient-list", str(patient_list_file)])
    if stub:
        cmd.append("--stub")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return int(proc.returncode)
