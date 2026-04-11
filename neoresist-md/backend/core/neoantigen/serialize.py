from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREDICTOR_NAME = "Class1PresentationPredictor"
RANKING_FORMULA_VERSION = "v1_auditable_0.6_0.4"


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_run_id(timestamp: str, input_file_sha256: str, hla_alleles: list[str]) -> str:
    compact_ts = timestamp.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    token = f"{timestamp}|{input_file_sha256}|{','.join(sorted(hla_alleles))}"
    short_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    return f"{compact_ts}_{short_hash}"


def build_run_metadata(
    *,
    run_id: str,
    timestamp: str,
    input_file_path: str,
    input_file_sha256: str,
    number_of_input_rows: int,
    hla_alleles: list[str],
    peptide_lengths_used: list[int],
    output_csv_path: str,
    number_of_output_candidates: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "input_file_path": input_file_path,
        "input_file_sha256": input_file_sha256,
        "number_of_input_rows": number_of_input_rows,
        "hla_alleles_used": hla_alleles,
        "peptide_lengths_used": peptide_lengths_used,
        "predictor_name": PREDICTOR_NAME,
        "ranking_formula_version": RANKING_FORMULA_VERSION,
        "output_csv_path": output_csv_path,
        "number_of_output_candidates": number_of_output_candidates,
    }
    if extra_fields:
        metadata.update(extra_fields)
    return metadata


def save_run_metadata(metadata: dict[str, Any], output_dir: str = "results/runs") -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{metadata['run_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return str(path)


def verify_reproducible_metadata(metadata_a: dict[str, Any], metadata_b: dict[str, Any]) -> tuple[bool, list[str]]:
    ignored = {"run_id", "timestamp"}
    keys = sorted(set(metadata_a.keys()) | set(metadata_b.keys()))
    differences: list[str] = []
    for key in keys:
        if key in ignored:
            continue
        if metadata_a.get(key) != metadata_b.get(key):
            differences.append(key)
    return (len(differences) == 0, differences)


def to_neo_candidates_json(
    *,
    sample_id: str,
    input_mode: str,
    hla_alleles: list[str],
    candidates_df,
    stats: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "neo_candidates",
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "input_mode": input_mode,
        "hla_alleles": hla_alleles,
        "stats": stats,
        "run_metadata": run_metadata,
        "candidates": candidates_df.to_dict(orient="records"),
    }


def write_json(payload: dict[str, Any], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)
