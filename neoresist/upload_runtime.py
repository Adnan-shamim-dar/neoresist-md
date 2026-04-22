from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import entropy as scipy_entropy

from neoresist_md.backend.normalisers.maf_normaliser import MAFNormaliser
from neoresist_md.backend.normalisers.tsv_normaliser import TSVNormaliser
from neoresist_md.backend.normalisers.vcf_normaliser import VCFNormaliser
from neoresist.paths import repo_root, upload_runs_dir
from neoresist.tumor_features import estimate_msi_status

NEOVAX_CLI_CWD = repo_root() / "neoresist-md"
FIXED_HLA = "HLA-A0201"
CSV_REQUIRED_COLUMNS = {
    "gene_name",
    "protein_sequence",
    "mutation_position",
    "normal_amino_acid",
    "mutant_amino_acid",
}
MAF_HINT_COLUMNS = {"Hugo_Symbol", "Variant_Classification", "HGVSp_Short"}
MAF_REQUIRED_COLUMNS = (
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
)
TSV_REQUIRED_COLUMNS = ("chrom", "pos", "ref", "alt")

PREDICTOR_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "gene_name": ("gene", "gene_name", "hugo_symbol", "symbol"),
    "protein_sequence": ("protein_sequence", "sequence", "protein", "aa_sequence"),
    "mutation_position": ("mutation_position", "position", "mut_position", "aa_position"),
    "normal_amino_acid": ("normal_amino_acid", "ref_aa", "wt_aa", "wildtype_amino_acid"),
    "mutant_amino_acid": ("mutant_amino_acid", "alt_aa", "mut_aa", "variant_amino_acid"),
}
MAF_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "Hugo_Symbol": ("hugo_symbol", "gene", "gene_name", "symbol"),
    "Chromosome": ("chromosome", "chrom", "chr"),
    "Start_Position": ("start_position", "position", "pos", "start"),
    "Reference_Allele": ("reference_allele", "ref", "reference", "ref_allele"),
    "Tumor_Seq_Allele2": ("tumor_seq_allele2", "alt", "variant_allele", "alt_allele"),
    "Tumor_Sample_Barcode": ("tumor_sample_barcode", "sample", "sample_id", "patient_id"),
    "HGVSp_Short": ("hgvsp_short", "hgvsp", "protein_change", "aa_change"),
    "t_ref_count": ("t_ref_count", "ref_count", "ref_reads", "ref_depth"),
    "t_alt_count": ("t_alt_count", "alt_count", "alt_reads", "alt_depth"),
    "VAF": ("vaf", "af", "allele_fraction", "tumor_af"),
    "HLA_Allele": ("hla_allele", "hla", "hla_type"),
}
TSV_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "gene": ("gene", "gene_name", "hugo_symbol", "symbol"),
    "protein_change": ("protein_change", "hgvsp_short", "hgvsp", "aa_change"),
    "chrom": ("chrom", "chromosome", "chr"),
    "pos": ("pos", "position", "start_position", "start"),
    "ref": ("ref", "reference_allele", "reference", "ref_allele"),
    "alt": ("alt", "tumor_seq_allele2", "variant_allele", "alt_allele"),
    "ref_count": ("ref_count", "t_ref_count", "ref_reads", "ref_depth"),
    "alt_count": ("alt_count", "t_alt_count", "alt_reads", "alt_depth"),
    "vaf": ("vaf", "af", "allele_fraction", "tumor_af"),
    "sample_id": ("sample_id", "sample", "tumor_sample_barcode", "patient_id"),
    "hla_allele": ("hla_allele", "hla", "hla_type"),
}


@dataclass(frozen=True)
class UploadValidation:
    input_mode: str
    sample_id: str
    columns: list[str]
    row_count: int
    message: str
    normalized_bytes: bytes | None = None
    normalized_filename: str | None = None
    normaliser_report: dict[str, Any] | None = None
    germline_contamination_suspected: bool = False
    ith_entropy: float | None = None
    hla_format_invalid_count: int = 0
    hla_missing_or_unknown: bool = False
    msi_status: str = "INDETERMINATE"
    msi_frameshift_indel_ratio: float = 0.0
    msi_total_mutations: int = 0
    msi_frameshift_indels: int = 0
    detected_column_mapping: dict[str, str] | None = None


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _decode_upload(contents: str) -> bytes:
    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def _infer_separator(name: str, text: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".tsv", ".maf", ".txt"}:
        return "\t"
    first = text.splitlines()[0] if text.splitlines() else ""
    return "\t" if first.count("\t") > first.count(",") else ","


def _normalize_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _detect_mapping(
    columns: list[str],
    alias_map: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    norm_to_original: dict[str, list[str]] = {}
    for col in columns:
        norm_to_original.setdefault(_normalize_col_name(col), []).append(col)
    used: set[str] = set()
    mapped: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        candidates = (canonical, *aliases)
        for alias in candidates:
            hits = norm_to_original.get(_normalize_col_name(alias), [])
            picked = next((h for h in hits if h not in used), None)
            if picked:
                mapped[canonical] = picked
                used.add(picked)
                break
    return mapped


def _materialize_mapped_columns(df: pd.DataFrame, canonical_to_source: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for canonical, source in canonical_to_source.items():
        if canonical not in out.columns and source in out.columns:
            out[canonical] = out[source]
    return out


def _mapping_for_display(canonical_to_source: dict[str, str], target_columns: set[str]) -> dict[str, str]:
    return {
        source: canonical
        for canonical, source in canonical_to_source.items()
        if source != canonical and canonical in target_columns
    }


def _format_missing_columns_error(
    detected_label: str,
    missing_columns: list[str],
    expected_label: str,
    expected_required: tuple[str, ...],
) -> str:
    return (
        f"{detected_label} upload is missing critical columns after auto-mapping: {', '.join(missing_columns)}. "
        f"Expected format: {expected_label}. Required columns: {', '.join(expected_required)}."
    )


def validate_uploaded_table(file_bytes: bytes, filename: str) -> UploadValidation:
    sample_id = Path(filename).stem.replace(" ", "_")
    suffix = Path(filename).suffix.lower()

    # Handle formats that need dedicated normalization before generic table parsing.
    if suffix in {".vcf", ".gz"} and filename.lower().endswith((".vcf", ".vcf.gz")):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        norm = VCFNormaliser()
        vr = norm.validate(tmp_path)
        can = norm.normalise(tmp_path, patient_id=sample_id)
        flags = _scientific_flags_from_canonical(can)
        predictor_like = _canonical_to_predictor_csv(can)
        predictor_bytes = predictor_like.to_csv(index=False).encode("utf-8")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return UploadValidation(
            input_mode="csv",
            sample_id=sample_id,
            columns=list(predictor_like.columns),
            row_count=int(len(predictor_like)),
            message="VCF detected and normalized to predictor-ready CSV for demo pipeline.",
            normalized_bytes=predictor_bytes,
            normalized_filename=f"{sample_id}.normalized.csv",
            normaliser_report=vr.to_dict(),
            germline_contamination_suspected=bool(flags["germline_contamination_suspected"]),
            ith_entropy=flags["ith_entropy"],
            hla_format_invalid_count=int(flags["hla_format_invalid_count"]),
            hla_missing_or_unknown=bool(flags["hla_missing_or_unknown"]),
            msi_status=str(flags["msi_status"]),
            msi_frameshift_indel_ratio=float(flags["msi_frameshift_indel_ratio"]),
            msi_total_mutations=int(flags["msi_total_mutations"]),
            msi_frameshift_indels=int(flags["msi_frameshift_indels"]),
            detected_column_mapping=None,
        )

    text = file_bytes.decode("utf-8", errors="replace")
    sep = _infer_separator(filename, text)
    try:
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=sep, comment="#" if sep == "\t" else None)
    except Exception:
        df = pd.DataFrame()
    cols = [str(c) for c in df.columns]
    raw_msi = estimate_msi_status(df)

    predictor_mapping = _detect_mapping(cols, PREDICTOR_ALIAS_MAP)
    predictor_df = _materialize_mapped_columns(df, predictor_mapping)
    predictor_columns = set(str(c) for c in predictor_df.columns)
    predictor_missing = sorted(CSV_REQUIRED_COLUMNS.difference(predictor_columns))
    predictor_display_mapping = _mapping_for_display(predictor_mapping, set(CSV_REQUIRED_COLUMNS))
    if not predictor_missing:
        normalized_bytes = None
        normalized_filename = None
        if predictor_display_mapping or sep != ",":
            normalized_bytes = predictor_df.to_csv(index=False).encode("utf-8")
            normalized_filename = f"{sample_id}.normalized.csv"
        return UploadValidation(
            input_mode="csv",
            sample_id=sample_id,
            columns=list(str(c) for c in predictor_df.columns),
            row_count=int(len(predictor_df)),
            message="Predictor-ready table detected."
            if not predictor_display_mapping
            else "Predictor-ready table detected with auto-mapped column aliases.",
            normalized_bytes=normalized_bytes,
            normalized_filename=normalized_filename,
            hla_format_invalid_count=0,
            hla_missing_or_unknown=True,
            msi_status=str(raw_msi["msi_status"]),
            msi_frameshift_indel_ratio=float(raw_msi["msi_frameshift_indel_ratio"]),
            msi_total_mutations=int(raw_msi["msi_total_mutations"]),
            msi_frameshift_indels=int(raw_msi["msi_frameshift_indels"]),
            detected_column_mapping=predictor_display_mapping,
        )
    # Normalize MAF and generic mutation tables into predictor-ready CSV for the existing NeoVax CLI path.
    if suffix in {".maf", ".tsv", ".csv", ".txt"} or MAF_HINT_COLUMNS.intersection(set(cols)):
        maf_mapping = _detect_mapping(cols, MAF_ALIAS_MAP)
        maf_df = _materialize_mapped_columns(df, maf_mapping)
        maf_missing = sorted(c for c in MAF_REQUIRED_COLUMNS if c not in set(str(col) for col in maf_df.columns))
        detected_mapping = _mapping_for_display(maf_mapping, set(MAF_ALIAS_MAP.keys()))
        # Try MAF normaliser first for TCGA-like tables, fallback to generic mapped table.
        if not maf_missing:
            if "VAF" in maf_df.columns and ("t_ref_count" not in maf_df.columns or "t_alt_count" not in maf_df.columns):
                vaf_series = pd.to_numeric(maf_df.get("VAF"), errors="coerce").clip(0, 1)
                pseudo_depth = 100.0
                if "t_alt_count" not in maf_df.columns:
                    maf_df["t_alt_count"] = (vaf_series * pseudo_depth).round()
                if "t_ref_count" not in maf_df.columns:
                    maf_df["t_ref_count"] = ((1.0 - vaf_series) * pseudo_depth).round()
            with tempfile.NamedTemporaryFile(suffix=".maf", delete=False, mode="w", encoding="utf-8", newline="") as tmp:
                maf_df.to_csv(tmp, sep="\t", index=False)
                tmp_path = Path(tmp.name)
            maf_norm = MAFNormaliser()
            maf_vr = maf_norm.validate(tmp_path)
            maf_can = maf_norm.normalise(tmp_path, run_id="upload")
            flags = _scientific_flags_from_canonical(maf_can)
            flags.update(raw_msi)
            predictor_like = _canonical_to_predictor_csv(maf_can)
            predictor_bytes = predictor_like.to_csv(index=False).encode("utf-8")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return UploadValidation(
                input_mode="csv",
                sample_id=sample_id,
                columns=list(predictor_like.columns),
                row_count=int(len(predictor_like)),
                message="Mutation table normalized to predictor-ready CSV for demo pipeline.",
                normalized_bytes=predictor_bytes,
                normalized_filename=f"{sample_id}.normalized.csv",
                normaliser_report=maf_vr.to_dict(),
                germline_contamination_suspected=bool(flags["germline_contamination_suspected"]),
                ith_entropy=flags["ith_entropy"],
                hla_format_invalid_count=int(flags["hla_format_invalid_count"]),
                hla_missing_or_unknown=bool(flags["hla_missing_or_unknown"]),
                msi_status=str(flags["msi_status"]),
                msi_frameshift_indel_ratio=float(flags["msi_frameshift_indel_ratio"]),
                msi_total_mutations=int(flags["msi_total_mutations"]),
                msi_frameshift_indels=int(flags["msi_frameshift_indels"]),
                detected_column_mapping=detected_mapping,
            )

        tsv_norm = TSVNormaliser()
        tsv_mapping = _detect_mapping(cols, TSV_ALIAS_MAP)
        tsv_df = _materialize_mapped_columns(df, tsv_mapping)
        tsv_missing = sorted(c for c in TSV_REQUIRED_COLUMNS if c not in set(str(col) for col in tsv_df.columns))
        tsv_detected_mapping = _mapping_for_display(tsv_mapping, set(TSV_ALIAS_MAP.keys()))
        detected_mapping = {**detected_mapping, **tsv_detected_mapping}
        if tsv_missing:
            detected_label = "TSV/CSV"
            expected_label = "TSV/CSV mutation table with genomic columns"
            expected_required = TSV_REQUIRED_COLUMNS
            missing_for_error = tsv_missing
            if suffix == ".maf":
                detected_label = "MAF"
                expected_label = "MAF-style table"
                expected_required = MAF_REQUIRED_COLUMNS
                missing_for_error = maf_missing if maf_missing else tsv_missing
            raise ValueError(
                _format_missing_columns_error(
                    detected_label=detected_label,
                    missing_columns=missing_for_error,
                    expected_label=expected_label,
                    expected_required=expected_required,
                )
            )
        with tempfile.NamedTemporaryFile(suffix=suffix or ".tsv", delete=False, mode="w", encoding="utf-8", newline="") as tmp:
            tsv_df.to_csv(tmp, sep=sep, index=False)
            tmp_path = Path(tmp.name)
        tsv_vr = tsv_norm.validate(tmp_path)
        can = tsv_norm.normalise(
            tmp_path,
            patient_id=sample_id,
            column_mapping={
                "gene": tsv_mapping.get("gene", "gene"),
                "protein_change": tsv_mapping.get("protein_change", "protein_change"),
                "chrom": tsv_mapping.get("chrom", "chrom"),
                "pos": tsv_mapping.get("pos", "pos"),
                "ref": tsv_mapping.get("ref", "ref"),
                "alt": tsv_mapping.get("alt", "alt"),
                "ref_count": tsv_mapping.get("ref_count", "ref_count"),
                "alt_count": tsv_mapping.get("alt_count", "alt_count"),
            },
        )
        vaf_source = tsv_mapping.get("vaf")
        if vaf_source and vaf_source in tsv_df.columns:
            parsed_vaf = pd.to_numeric(tsv_df[vaf_source], errors="coerce").clip(0, 1)
            if "vaf" not in can.columns:
                can["vaf"] = parsed_vaf
            else:
                can["vaf"] = pd.to_numeric(can["vaf"], errors="coerce")
                can["vaf"] = can["vaf"].where(can["vaf"].notna(), parsed_vaf)
        flags = _scientific_flags_from_canonical(can)
        flags.update(raw_msi)
        predictor_like = _canonical_to_predictor_csv(can)
        predictor_bytes = predictor_like.to_csv(index=False).encode("utf-8")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return UploadValidation(
            input_mode="csv",
            sample_id=sample_id,
            columns=list(predictor_like.columns),
            row_count=int(len(predictor_like)),
            message="Delimited file normalized via generic mapper to predictor-ready CSV for demo pipeline.",
            normalized_bytes=predictor_bytes,
            normalized_filename=f"{sample_id}.normalized.csv",
            normaliser_report=tsv_vr.to_dict(),
            germline_contamination_suspected=bool(flags["germline_contamination_suspected"]),
            ith_entropy=flags["ith_entropy"],
            hla_format_invalid_count=int(flags["hla_format_invalid_count"]),
            hla_missing_or_unknown=bool(flags["hla_missing_or_unknown"]),
            msi_status=str(flags["msi_status"]),
            msi_frameshift_indel_ratio=float(flags["msi_frameshift_indel_ratio"]),
            msi_total_mutations=int(flags["msi_total_mutations"]),
            msi_frameshift_indels=int(flags["msi_frameshift_indels"]),
            detected_column_mapping=detected_mapping,
        )
    raise ValueError(
        "Uploaded file is not a supported predictor CSV/TSV or MAF-style table. "
        f"Columns seen: {', '.join(cols[:12])}"
    )


def _canonical_to_predictor_csv(canonical_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_name",
                "protein_sequence",
                "mutation_position",
                "normal_amino_acid",
                "mutant_amino_acid",
            ]
        )
    df = canonical_df.copy()
    aa_letters = "ACDEFGHIKLMNPQRSTVWY"

    def _row_seed(row: pd.Series) -> int:
        key = "|".join(
            [
                str(row.get("patient_id") or ""),
                str(row.get("gene") or ""),
                str(row.get("chrom") or ""),
                str(row.get("pos") or ""),
                str(row.get("ref") or ""),
                str(row.get("alt") or ""),
            ]
        )
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)

    def _build_row(row: pd.Series) -> dict[str, object]:
        seed = _row_seed(row)
        protein_change = str(row.get("protein_change") or row.get("source_hgvsp_short") or row.get("HGVSp_Short") or "")
        parsed = re.search(r"p\.?([A-Z])(\d+)([A-Z])", protein_change, flags=re.IGNORECASE)
        pos_raw = pd.to_numeric(pd.Series([row.get("pos")]), errors="coerce").iloc[0]
        mutation_position = int(pos_raw) % 150 + 10 if pd.notna(pos_raw) else seed % 150 + 10
        normal_aa = None
        mutant_aa = None
        if parsed:
            normal_aa = parsed.group(1).upper()
            mutation_position = int(parsed.group(2))
            mutant_aa = parsed.group(3).upper()
        protein_length = max(220, mutation_position + 20)
        normal_idx = seed % len(aa_letters)
        mutant_idx = (normal_idx + 7) % len(aa_letters)
        normal_aa = normal_aa or aa_letters[normal_idx]
        mutant_aa = mutant_aa or aa_letters[mutant_idx]
        sequence_chars = [aa_letters[(seed + i) % len(aa_letters)] for i in range(protein_length)]
        sequence_chars[mutation_position - 1] = normal_aa
        protein_sequence = "".join(sequence_chars)
        return {
            "gene_name": str(row.get("gene") or "TP53"),
            "protein_sequence": protein_sequence,
            "mutation_position": mutation_position,
            "normal_amino_acid": normal_aa,
            "mutant_amino_acid": mutant_aa,
        }

    out = pd.DataFrame([_build_row(row) for _, row in df.iterrows()])
    return out.drop_duplicates().reset_index(drop=True)


def _scientific_flags_from_canonical(canonical_df: pd.DataFrame) -> dict[str, Any]:
    if canonical_df.empty:
        return {
            "germline_contamination_suspected": False,
            "ith_entropy": None,
            "hla_format_invalid_count": 0,
            "hla_missing_or_unknown": True,
            **estimate_msi_status(canonical_df),
        }
    vaf = pd.to_numeric(canonical_df.get("vaf"), errors="coerce").dropna()
    suspicious_ratio = float(((vaf >= 0.45) & (vaf <= 0.55)).mean()) if not vaf.empty else 0.0
    germline = suspicious_ratio > 0.15

    ith_entropy = None
    if len(vaf) > 50:
        bins = pd.cut(vaf.clip(0, 1), bins=20, labels=False, include_lowest=True)
        counts = bins.value_counts().sort_index()
        probs = counts / max(float(counts.sum()), 1.0)
        if not probs.empty:
            ith_entropy = float(scipy_entropy(probs))

    hla_invalid_count = 0
    hla_missing_or_unknown = True
    if "hla_allele" in canonical_df.columns:
        hla = canonical_df["hla_allele"].fillna("").astype(str).str.strip()
        invalid = ~hla.str.fullmatch(r"HLA-[ABC]\*\d{2}:\d{2}")
        # Missing is also invalid for binding predictions.
        hla_invalid_count = int(invalid.sum())
        resolved = int((~invalid).sum())
        hla_missing_or_unknown = resolved == 0

    return {
        "germline_contamination_suspected": bool(germline),
        "ith_entropy": ith_entropy if ith_entropy is None or not math.isnan(float(ith_entropy)) else None,
        "hla_format_invalid_count": int(hla_invalid_count),
        "hla_missing_or_unknown": bool(hla_missing_or_unknown),
        **estimate_msi_status(canonical_df),
    }


def _write_upload_input(file_bytes: bytes, filename: str) -> tuple[Path, Path]:
    run_dir = upload_runs_dir() / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / Path(filename).name
    input_path.write_bytes(file_bytes)
    return run_dir, input_path


def run_uploaded_prediction(contents: str, filename: str) -> dict[str, Any]:
    file_bytes = _decode_upload(contents)
    validation = validate_uploaded_table(file_bytes, filename)
    effective_bytes = validation.normalized_bytes if validation.normalized_bytes is not None else file_bytes
    effective_name = validation.normalized_filename if validation.normalized_filename else filename
    run_dir, input_path = _write_upload_input(effective_bytes, effective_name)
    out_dir = run_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "backend.api.cli.neovax_cli",
        "--input",
        str(input_path),
        "--input-mode",
        validation.input_mode,
        "--hla",
        FIXED_HLA,
        "--sample-id",
        validation.sample_id,
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(NEOVAX_CLI_CWD),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    result: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "validation": validation,
        "run_dir": str(run_dir),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "hla": FIXED_HLA,
        "artifacts": {},
        "run_metadata": None,
        "case_report": None,
        "candidates_preview": None,
        "rejected_preview": None,
    }
    if proc.returncode != 0:
        return result

    artifact_map = {
        "candidates_csv": out_dir / "candidates.csv",
        "neo_candidates_json": out_dir / "neo_candidates.json",
        "run_metadata_json": out_dir / "run_metadata.json",
        "case_report_md": out_dir / "case_report.md",
        "supported_csv": out_dir / "supported.csv",
        "rejected_csv": out_dir / "rejected.csv",
    }
    result["artifacts"] = {k: str(v) for k, v in artifact_map.items() if v.is_file()}
    meta_path = artifact_map["run_metadata_json"]
    if meta_path.is_file():
        result["run_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
    report_path = artifact_map["case_report_md"]
    if report_path.is_file():
        result["case_report"] = report_path.read_text(encoding="utf-8", errors="replace")
    cand_path = artifact_map["candidates_csv"]
    if cand_path.is_file():
        cdf = pd.read_csv(cand_path).head(12)
        result["candidates_preview"] = cdf.to_dict("records")
    rej_path = artifact_map["rejected_csv"]
    if rej_path.is_file():
        rdf = pd.read_csv(rej_path).head(12)
        result["rejected_preview"] = rdf.to_dict("records")
    return result
