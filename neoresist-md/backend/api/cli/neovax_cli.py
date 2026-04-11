from __future__ import annotations

"""
Minimal NeoVax CLI spine.

This CLI is intentionally thin: it only parses arguments, calls
backend.core.neoantigen.service.run_neovax(), and writes deterministic
artifact filenames. Report format options (for example, --report-format)
are expected to be added later without moving scientific logic into CLI.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.core.neoantigen.service import run_neovax


def _parse_hla_values(values: list[str]) -> list[str]:
    alleles: list[str] = []
    for value in values:
        for token in value.split(","):
            clean = token.strip()
            if clean:
                alleles.append(clean)
    if not alleles:
        raise ValueError("At least one non-empty HLA allele is required via --hla.")
    return alleles


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_case_report(path: Path, result: dict[str, Any]) -> None:
    metadata = result["run_metadata"]
    lines = [
        "# Case Report",
        "",
        "## Run Summary",
        f"- sample_id: {result['sample_id']}",
        f"- input_mode: {result['input_mode']}",
        f"- run_id: {metadata.get('run_id', 'unknown')}",
        f"- number_of_input_rows: {metadata.get('number_of_input_rows', 0)}",
        f"- number_of_output_candidates: {metadata.get('number_of_output_candidates', 0)}",
        "",
        "## Deterministic Artifacts",
        "- neo_candidates.json",
        "- candidates.csv",
        "- run_metadata.json",
        "- case_report.md",
    ]
    if result["input_mode"] == "maf":
        lines.extend(["- supported.csv", "- rejected.csv"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeoVax Module 1 from CLI.")
    parser.add_argument("--input", required=True, help="Path to input CSV or MAF.")
    parser.add_argument(
        "--input-mode",
        default="auto",
        choices=["auto", "csv", "maf"],
        help="Input mode. Default: auto.",
    )
    parser.add_argument(
        "--hla",
        action="append",
        required=True,
        help="HLA allele(s). Repeat or comma-separate values.",
    )
    parser.add_argument("--sample-id", default="patient_001", help="Sample identifier.")
    parser.add_argument("--output-dir", required=True, help="Directory for deterministic outputs.")
    parser.add_argument(
        "--cache-path",
        default="data/cache/protein_sequences.json",
        help="Sequence cache path for MAF mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        hla_alleles = _parse_hla_values(args.hla)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_neovax(
            input_data=args.input,
            input_mode=args.input_mode,
            hla_alleles=hla_alleles,
            sample_id=args.sample_id,
            output_dir=None,
            cache_path=args.cache_path,
        )

        _write_json(output_dir / "neo_candidates.json", result["neo_candidates"])
        result["candidates_df"].to_csv(output_dir / "candidates.csv", index=False)
        _write_json(output_dir / "run_metadata.json", result["run_metadata"])
        _write_case_report(output_dir / "case_report.md", result)

        if result["input_mode"] == "maf":
            supported_df = result["supported_rows_df"]
            rejected_df = result["rejected_rows_df"]
            if supported_df is None or rejected_df is None:
                raise ValueError("MAF mode expected supported/rejected rows but none were returned.")
            supported_df.to_csv(output_dir / "supported.csv", index=False)
            rejected_df.to_csv(output_dir / "rejected.csv", index=False)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
