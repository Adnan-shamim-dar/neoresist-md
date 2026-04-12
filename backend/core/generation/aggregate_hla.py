from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def discover_run_dirs(hla_runs_root: Path) -> list[Path]:
    """Return leaf run directories that contain ``candidates.csv``."""
    if not hla_runs_root.is_dir():
        return []
    out: list[Path] = []
    for hla_dir in sorted(hla_runs_root.iterdir()):
        if not hla_dir.is_dir():
            continue
        for patient_dir in sorted(hla_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            if (patient_dir / "candidates.csv").is_file():
                out.append(patient_dir)
    return out


def parse_run_metadata(path: Path) -> dict[str, Any]:
    """Load ``run_metadata.json``; return empty dict if missing or invalid."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def parse_case_report(path: Path) -> dict[str, Any]:
    """Parse ``case_report.md`` for input/candidate counts and optional run_id."""
    out: dict[str, Any] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m_in = re.search(r"number_of_input_rows:\s*(\d+)", line, re.I)
        if m_in:
            out["number_of_input_rows"] = int(m_in.group(1))
        m_out = re.search(r"number_of_output_candidates:\s*(\d+)", line, re.I)
        if m_out:
            out["number_of_output_candidates"] = int(m_out.group(1))
        m_rid = re.search(r"run_id:\s*(\S+)", line, re.I)
        if m_rid:
            out["run_id"] = m_rid.group(1).strip()
    return out


def normalize_patient_id_from_dir(dir_name: str) -> str:
    """Map folder names like ``sarc_TCGA_DX_...`` to TCGA barcodes."""
    s = dir_name
    if s.lower().startswith("sarc_"):
        s = s[5:]
    return s.replace("_", "-")


def _self_dissimilarity_stub(mutant_peptide: str, gene: str) -> float:
    """Deterministic placeholder in ``[0, 1]`` until a real metric exists."""
    h = hash((gene or "", mutant_peptide or "")) % (2**32)
    return h / (2**32 - 1)


def load_supported_annotation(supported_csv: Path) -> pd.DataFrame | None:
    """Load key columns from ``supported.csv`` for merge onto candidates."""
    if not supported_csv.is_file():
        return None
    try:
        df = pd.read_csv(supported_csv)
    except (OSError, pd.errors.EmptyDataError):
        return None
    cols = ["gene_name", "mutation_position", "source_hgvsp_short", "source_sample_id"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df[cols].drop_duplicates(subset=["gene_name", "mutation_position"])


def aggregate_runs(
    hla_runs_root: Path,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Crawl HLA run directories, merge supported mutation context, keep top-N
    candidates per ``(patient_id, hla_allele)`` by ``priority_score``.
    """
    run_dirs = discover_run_dirs(hla_runs_root)
    frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        hla_allele = run_dir.parent.name
        run_name = run_dir.name
        meta_path = run_dir / "run_metadata.json"
        report_path = run_dir / "case_report.md"
        candidates_path = run_dir / "candidates.csv"
        supported_path = run_dir / "supported.csv"

        meta = parse_run_metadata(meta_path)
        report = parse_case_report(report_path)
        n_in = int(meta.get("number_of_input_rows", report.get("number_of_input_rows", 0)) or 0)
        n_out = int(
            meta.get("number_of_output_candidates", report.get("number_of_output_candidates", 0)) or 0
        )
        run_timestamp = meta.get("timestamp") or None
        run_id = meta.get("run_id") or report.get("run_id")

        try:
            cand = pd.read_csv(candidates_path)
        except (OSError, pd.errors.EmptyDataError):
            continue

        if "priority_score" in cand.columns:
            cand = cand.sort_values("priority_score", ascending=False, kind="mergesort")
        elif "presentation_score" in cand.columns:
            cand = cand.sort_values("presentation_score", ascending=False, kind="mergesort")

        sup = load_supported_annotation(supported_path)
        if sup is not None:
            cand = cand.merge(sup, on=["gene_name", "mutation_position"], how="left")
        else:
            cand = cand.copy()
            cand["source_hgvsp_short"] = pd.NA
            cand["source_sample_id"] = pd.NA

        fallback_pid = normalize_patient_id_from_dir(run_name)
        cand["patient_id"] = cand["source_sample_id"].where(
            cand["source_sample_id"].notna(), other=fallback_pid
        )
        cand["patient_id"] = cand["patient_id"].astype(str)

        cand["hla_allele"] = hla_allele
        cand["run_timestamp"] = run_timestamp
        cand["run_id"] = run_id
        cand["run_number_of_input_rows"] = n_in
        cand["run_number_of_output_candidates"] = n_out
        cand["run_dir"] = str(run_dir)

        cand["best_allele_mismatch"] = False
        if "best_allele" in cand.columns:
            mismatch = cand["best_allele"].notna() & (cand["best_allele"].astype(str) != hla_allele)
            cand.loc[mismatch, "best_allele_mismatch"] = True

        group_cols = ["patient_id", "hla_allele"]
        cand = cand.groupby(group_cols, group_keys=False).head(top_n)

        gene = cand["gene_name"].astype(str)
        peptide = cand["peptide"].astype(str)
        cand["gene"] = gene
        cand["mutant_peptide"] = peptide
        cand["peptide_length"] = peptide.str.len()
        cand["protein_change"] = cand["source_hgvsp_short"].where(cand["source_hgvsp_short"].notna(), other=pd.NA)
        cand["wt_peptide"] = pd.NA
        cand["vaf"] = pd.NA
        cand["ref_count"] = pd.NA
        cand["alt_count"] = pd.NA
        cand["affinity_nm"] = pd.to_numeric(cand.get("affinity"), errors="coerce")
        cand["percentile_rank"] = pd.to_numeric(cand.get("presentation_percentile"), errors="coerce")
        cand["presentation_score"] = pd.to_numeric(cand.get("presentation_score"), errors="coerce").clip(0.0, 1.0)
        cand["expression_tpm"] = pd.NA
        cand["ccf"] = pd.NA
        cand["hla_loh_flag"] = pd.NA
        cand["self_dissimilarity"] = [
            _self_dissimilarity_stub(p, g) for p, g in zip(peptide.tolist(), gene.tolist(), strict=False)
        ]
        cand["tier"] = 3
        cand["rl_priority"] = math.nan
        cand["exclusion_reasons"] = [[] for _ in range(len(cand))]

        frames.append(cand)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def write_multi_hla_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
