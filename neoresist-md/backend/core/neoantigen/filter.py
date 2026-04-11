from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ENSEMBL_BASE = "https://rest.ensembl.org"
DEFAULT_CACHE_PATH = Path("data/cache/protein_sequences.json")


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict[str, dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_with_retry(url: str, attempts: int = 3, timeout: int = 20) -> dict[str, Any] | None:
    headers = {"Content-Type": "application/json"}
    for i in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.ok:
                return response.json()
        except Exception:
            pass
        if i < attempts - 1:
            time.sleep(0.5 * (i + 1))
    return None


def _fetch_by_transcript(transcript_id: str) -> str | None:
    clean = transcript_id.split(".")[0]
    url = f"{ENSEMBL_BASE}/sequence/id/{clean}?type=protein"
    data = _fetch_with_retry(url)
    if data and isinstance(data.get("seq"), str):
        return data["seq"]
    return None


def _fetch_by_gene_symbol(symbol: str) -> str | None:
    url = f"{ENSEMBL_BASE}/sequence/symbol/homo_sapiens/{symbol}?type=protein"
    data = _fetch_with_retry(url)
    if data and isinstance(data.get("seq"), str):
        return data["seq"]
    return None


def fetch_sequences_for_rows(
    parsed_rows_df: pd.DataFrame,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    if parsed_rows_df.empty:
        return parsed_rows_df.copy()

    cache_file = Path(cache_path)
    cache = _load_cache(cache_file)
    out_rows: list[dict[str, Any]] = []

    for row in parsed_rows_df.itertuples(index=False):
        record = row._asdict()
        gene = str(record.get("gene_name", "")).strip()
        transcript = str(record.get("source_transcript_id", "")).strip()

        protein_sequence = None
        fetch_source = "none"
        fetch_status = "failed"

        transcript_key = f"transcript:{transcript}" if transcript and transcript.lower() != "nan" else None
        gene_key = f"gene:{gene}" if gene and gene.lower() != "nan" else None

        if transcript_key and transcript_key in cache:
            protein_sequence = cache[transcript_key]["protein_sequence"]
            fetch_source = cache[transcript_key]["fetch_source"]
            fetch_status = "ok_cached"
        elif gene_key and gene_key in cache:
            protein_sequence = cache[gene_key]["protein_sequence"]
            fetch_source = cache[gene_key]["fetch_source"]
            fetch_status = "ok_cached"
        else:
            if transcript_key:
                seq = _fetch_by_transcript(transcript)
                if seq:
                    protein_sequence = seq
                    fetch_source = "ensembl_transcript"
                    fetch_status = "ok_fetched"
            if protein_sequence is None and gene_key:
                seq = _fetch_by_gene_symbol(gene)
                if seq:
                    protein_sequence = seq
                    fetch_source = "ensembl_gene_symbol"
                    fetch_status = "ok_fetched"

            if protein_sequence is not None:
                payload = {
                    "protein_sequence": protein_sequence,
                    "fetch_source": fetch_source,
                }
                if transcript_key:
                    cache[transcript_key] = payload
                if gene_key:
                    cache[gene_key] = payload

        record["protein_sequence"] = protein_sequence
        record["fetch_source"] = fetch_source
        record["fetch_status"] = fetch_status
        out_rows.append(record)

    _save_cache(cache, cache_file)
    return pd.DataFrame(out_rows)


def validate_sequence_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        record = row._asdict()
        seq = record.get("protein_sequence")
        pos = int(record["mutation_position"])
        ref = str(record["normal_amino_acid"])

        if not isinstance(seq, str) or not seq:
            rejected.append(
                {
                    "original_row_index": record.get("original_row_index"),
                    "rejection_reason": "Protein sequence fetch failed.",
                    "source_gene": record.get("source_gene"),
                    "source_hgvsp_short": record.get("source_hgvsp_short"),
                    "source_transcript_id": record.get("source_transcript_id"),
                    "source_sample_id": record.get("source_sample_id"),
                }
            )
            continue

        if pos < 1 or pos > len(seq):
            rejected.append(
                {
                    "original_row_index": record.get("original_row_index"),
                    "rejection_reason": "Mutation position out of range for fetched protein sequence.",
                    "source_gene": record.get("source_gene"),
                    "source_hgvsp_short": record.get("source_hgvsp_short"),
                    "source_transcript_id": record.get("source_transcript_id"),
                    "source_sample_id": record.get("source_sample_id"),
                }
            )
            continue

        if seq[pos - 1] != ref:
            rejected.append(
                {
                    "original_row_index": record.get("original_row_index"),
                    "rejection_reason": (
                        f"Reference amino acid mismatch at position {pos}: "
                        f"expected {ref}, found {seq[pos - 1]}."
                    ),
                    "source_gene": record.get("source_gene"),
                    "source_hgvsp_short": record.get("source_hgvsp_short"),
                    "source_transcript_id": record.get("source_transcript_id"),
                    "source_sample_id": record.get("source_sample_id"),
                }
            )
            continue

        supported.append(record)

    return pd.DataFrame(supported), pd.DataFrame(rejected)
