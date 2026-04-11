from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .filter import fetch_sequences_for_rows, validate_sequence_matches
from .parse import CSV_REQUIRED_COLUMNS, detect_input_mode, parse_maf_dataframe, read_maf, read_table
from .rank import PEPTIDE_LENGTHS, predict_candidates
from .serialize import (
    build_run_metadata,
    bytes_sha256,
    file_sha256,
    generate_run_id,
    save_run_metadata,
    to_neo_candidates_json,
    utc_timestamp,
    write_json,
)


def _prepare_maf_predictor_rows(
    maf_input: Any,
    cache_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    maf_df = read_maf(maf_input)
    parsed_df, rejected_df, stats = parse_maf_dataframe(maf_df)
    with_seq_df = fetch_sequences_for_rows(parsed_df, cache_path=cache_path)
    matched_df, seq_rejected_df = validate_sequence_matches(with_seq_df)
    all_rejected = pd.concat([rejected_df, seq_rejected_df], ignore_index=True)

    predictor_df = (
        matched_df[
            [
                "gene_name",
                "protein_sequence",
                "mutation_position",
                "normal_amino_acid",
                "mutant_amino_acid",
                "source_gene",
                "source_hgvsp_short",
                "source_transcript_id",
                "source_sample_id",
                "fetch_source",
                "fetch_status",
            ]
        ].copy()
        if not matched_df.empty
        else pd.DataFrame(
            columns=[
                "gene_name",
                "protein_sequence",
                "mutation_position",
                "normal_amino_acid",
                "mutant_amino_acid",
                "source_gene",
                "source_hgvsp_short",
                "source_transcript_id",
                "source_sample_id",
                "fetch_source",
                "fetch_status",
            ]
        )
    )
    stats.update(
        {
            "sequence_matched_rows": int(len(predictor_df)),
            "rejected_rows": int(len(all_rejected)),
        }
    )
    return predictor_df, with_seq_df, all_rejected, stats


def run_neovax(
    *,
    input_data: str | bytes | pd.DataFrame,
    input_mode: str = "auto",
    hla_alleles: list[str],
    sample_id: str = "patient_001",
    output_dir: str | None = None,
    cache_path: str | Path = "data/cache/protein_sequences.json",
) -> dict[str, Any]:
    mode = detect_input_mode(input_data, input_mode)
    parsed_rows_df = None
    rejected_rows_df = None
    stats: dict[str, Any] = {}

    if mode == "dataframe":
        predictor_input_df = input_data.copy()  # type: ignore[union-attr]
        missing = CSV_REQUIRED_COLUMNS - set(predictor_input_df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {', '.join(sorted(missing))}")
        stats["number_of_input_rows"] = int(len(predictor_input_df))
        input_sha = "dataframe_input"
        input_ref = "in_memory_dataframe"
    elif mode == "csv":
        predictor_input_df = read_table(input_data, sep=",")
        missing = CSV_REQUIRED_COLUMNS - set(predictor_input_df.columns)
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")
        stats["number_of_input_rows"] = int(len(predictor_input_df))
        if isinstance(input_data, (bytes, bytearray)):
            input_sha = bytes_sha256(bytes(input_data))
            input_ref = "in_memory_bytes.csv"
        else:
            input_sha = file_sha256(str(input_data))
            input_ref = str(input_data)
    elif mode == "maf":
        predictor_input_df, parsed_rows_df, rejected_rows_df, maf_stats = _prepare_maf_predictor_rows(
            input_data,
            cache_path=cache_path,
        )
        if predictor_input_df.empty:
            raise ValueError("No valid sequence-matched missense rows available for prediction.")
        stats.update(maf_stats)
        stats["number_of_input_rows"] = int(maf_stats.get("total_maf_rows", 0))
        if isinstance(input_data, (bytes, bytearray)):
            input_sha = bytes_sha256(bytes(input_data))
            input_ref = "in_memory_bytes.maf"
        else:
            input_sha = file_sha256(str(input_data))
            input_ref = str(input_data)
    else:
        raise ValueError(f"Unsupported input_mode: {mode}")

    candidates_df = predict_candidates(predictor_input_df, hla_alleles)
    timestamp = utc_timestamp()
    run_id = generate_run_id(timestamp, input_sha, hla_alleles)
    output_csv_path = "neo_candidates.csv"
    run_metadata = build_run_metadata(
        run_id=run_id,
        timestamp=timestamp,
        input_file_path=input_ref,
        input_file_sha256=input_sha,
        number_of_input_rows=int(stats.get("number_of_input_rows", len(predictor_input_df))),
        hla_alleles=hla_alleles,
        peptide_lengths_used=list(PEPTIDE_LENGTHS),
        output_csv_path=output_csv_path,
        number_of_output_candidates=int(len(candidates_df)),
        extra_fields={"input_mode": mode, **{k: v for k, v in stats.items() if k != "number_of_input_rows"}},
    )
    neo_candidates = to_neo_candidates_json(
        sample_id=sample_id,
        input_mode=mode,
        hla_alleles=hla_alleles,
        candidates_df=candidates_df,
        stats=stats,
        run_metadata=run_metadata,
    )

    artifacts: dict[str, str] = {}
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        candidates_csv = out / f"{run_id}_neo_candidates.csv"
        candidates_df.to_csv(candidates_csv, index=False)
        artifacts["neo_candidates_csv"] = str(candidates_csv)
        neo_json = out / f"{run_id}_neo_candidates.json"
        artifacts["neo_candidates_json"] = write_json(neo_candidates, str(neo_json))
        meta_path = save_run_metadata(run_metadata, output_dir=str(out / "runs"))
        artifacts["run_metadata_json"] = meta_path
        if parsed_rows_df is not None:
            p = out / f"{run_id}_supported_parsed_rows.csv"
            parsed_rows_df.to_csv(p, index=False)
            artifacts["supported_parsed_rows_csv"] = str(p)
        if rejected_rows_df is not None:
            p = out / f"{run_id}_rejected_rows.csv"
            rejected_rows_df.to_csv(p, index=False)
            artifacts["rejected_rows_csv"] = str(p)

    return {
        "sample_id": sample_id,
        "input_mode": mode,
        "candidates_df": candidates_df,
        "supported_rows_df": predictor_input_df if mode == "maf" else None,
        "parsed_rows_df": parsed_rows_df,
        "rejected_rows_df": rejected_rows_df,
        "run_metadata": run_metadata,
        "neo_candidates": neo_candidates,
        "artifacts": artifacts,
    }
