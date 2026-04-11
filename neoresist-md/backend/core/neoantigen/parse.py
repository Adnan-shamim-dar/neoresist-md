from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd


HGVSP_SHORT_PATTERN = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")
CSV_REQUIRED_COLUMNS = {
    "gene_name",
    "protein_sequence",
    "mutation_position",
    "normal_amino_acid",
    "mutant_amino_acid",
}


def _parse_hgvsp_short(value: str) -> tuple[str, int, str] | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    m = HGVSP_SHORT_PATTERN.match(value)
    if not m:
        return None
    normal_aa, pos, mutant_aa = m.groups()
    return normal_aa, int(pos), mutant_aa


def read_table(file_obj_or_bytes_or_path: Any, sep: str = ",") -> pd.DataFrame:
    if isinstance(file_obj_or_bytes_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_obj_or_bytes_or_path), sep=sep)
    return pd.read_csv(file_obj_or_bytes_or_path, sep=sep)


def read_maf(file_obj_or_bytes_or_path: Any) -> pd.DataFrame:
    if isinstance(file_obj_or_bytes_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_obj_or_bytes_or_path), sep="\t", comment="#")
    return pd.read_csv(file_obj_or_bytes_or_path, sep="\t", comment="#")


def parse_maf_dataframe(maf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    sample_col = (
        "Tumor_Sample_Barcode"
        if "Tumor_Sample_Barcode" in maf_df.columns
        else "case_id"
        if "case_id" in maf_df.columns
        else None
    )

    missense_rows = 0
    parseable_rows = 0
    for idx, row in maf_df.iterrows():
        gene = row.get("Hugo_Symbol")
        variant_class = row.get("Variant_Classification")
        hgvsp_short = row.get("HGVSp_Short")
        transcript_id = row.get("Transcript_ID") if "Transcript_ID" in maf_df.columns else None
        sample_id = row.get(sample_col) if sample_col else None

        if variant_class != "Missense_Mutation":
            rejected.append(
                {
                    "original_row_index": int(idx),
                    "rejection_reason": "Unsupported mutation class (only Missense_Mutation allowed).",
                    "source_gene": gene,
                    "source_hgvsp_short": hgvsp_short,
                    "source_transcript_id": transcript_id,
                    "source_sample_id": sample_id,
                }
            )
            continue

        missense_rows += 1
        parsed = _parse_hgvsp_short(hgvsp_short)
        if parsed is None:
            rejected.append(
                {
                    "original_row_index": int(idx),
                    "rejection_reason": "Unparseable HGVSp_Short (expected simple form like p.R175H).",
                    "source_gene": gene,
                    "source_hgvsp_short": hgvsp_short,
                    "source_transcript_id": transcript_id,
                    "source_sample_id": sample_id,
                }
            )
            continue

        parseable_rows += 1
        normal_aa, mutation_position, mutant_aa = parsed
        rows.append(
            {
                "original_row_index": int(idx),
                "gene_name": gene,
                "mutation_position": mutation_position,
                "normal_amino_acid": normal_aa,
                "mutant_amino_acid": mutant_aa,
                "source_gene": gene,
                "source_hgvsp_short": hgvsp_short,
                "source_transcript_id": transcript_id,
                "source_sample_id": sample_id,
            }
        )

    parsed_df = pd.DataFrame(rows)
    rejected_df = pd.DataFrame(rejected)
    stats = {
        "total_maf_rows": int(len(maf_df)),
        "missense_rows": int(missense_rows),
        "parseable_rows": int(parseable_rows),
    }
    return parsed_df, rejected_df, stats


def detect_input_mode(input_data: Any, input_mode: str) -> str:
    if input_mode != "auto":
        return input_mode
    if isinstance(input_data, pd.DataFrame):
        return "dataframe"
    if isinstance(input_data, (bytes, bytearray)):
        return "csv"
    suffix = Path(str(input_data)).suffix.lower()
    if suffix == ".maf":
        return "maf"
    return "csv"
