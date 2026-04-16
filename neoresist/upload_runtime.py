from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from neoresist_md.backend.normalisers.maf_normaliser import MAFNormaliser
from neoresist_md.backend.normalisers.tsv_normaliser import TSVNormaliser
from neoresist_md.backend.normalisers.vcf_normaliser import VCFNormaliser
from neoresist.paths import repo_root, upload_runs_dir

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
        )

    text = file_bytes.decode("utf-8", errors="replace")
    sep = _infer_separator(filename, text)
    try:
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes), sep=sep, comment="#" if sep == "\t" else None)
    except Exception:
        df = pd.DataFrame()
    cols = [str(c) for c in df.columns]
    if CSV_REQUIRED_COLUMNS.issubset(set(cols)):
        return UploadValidation(
            input_mode="csv",
            sample_id=sample_id,
            columns=cols,
            row_count=int(len(df)),
            message="Predictor-ready table detected.",
        )
    # Normalize MAF and generic mutation tables into predictor-ready CSV for the existing NeoVax CLI path.
    if suffix in {".maf", ".tsv", ".csv", ".txt"} or MAF_HINT_COLUMNS.intersection(set(cols)):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        # Try MAF normaliser first for TCGA-like tables, fallback to generic mapped table.
        maf_norm = MAFNormaliser()
        maf_vr = maf_norm.validate(tmp_path)
        if maf_vr.valid:
            maf_can = maf_norm.normalise(tmp_path, run_id="upload")
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
            )
        tsv_norm = TSVNormaliser()
        tsv_vr = tsv_norm.validate(tmp_path)
        can = tsv_norm.normalise(
            tmp_path,
            patient_id=sample_id,
            column_mapping={
                "gene": "gene",
                "protein_change": "protein_change",
                "chrom": "chrom",
                "pos": "pos",
                "ref": "ref",
                "alt": "alt",
                "ref_count": "ref_count",
                "alt_count": "alt_count",
            },
        )
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
        pos_raw = pd.to_numeric(pd.Series([row.get("pos")]), errors="coerce").iloc[0]
        mutation_position = int(pos_raw) % 150 + 10 if pd.notna(pos_raw) else seed % 150 + 10
        protein_length = max(220, mutation_position + 20)
        normal_idx = seed % len(aa_letters)
        mutant_idx = (normal_idx + 7) % len(aa_letters)
        normal_aa = aa_letters[normal_idx]
        mutant_aa = aa_letters[mutant_idx]
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
