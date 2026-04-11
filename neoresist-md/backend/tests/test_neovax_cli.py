from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from backend.api.cli import neovax_cli


class TestNeoVaxCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cls.neovax_builder_root = repo_root.parent / "neovax-builder"
        cls.gold_csv = cls.neovax_builder_root / "data" / "example_mutations_gold.csv"
        cls.maf_sample = cls.neovax_builder_root / "data" / "tcga_brca_sample.maf"

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            "-m",
            "backend.api.cli.neovax_cli",
            *args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_cli_csv_outputs_expected_files(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self._run_cli(
                [
                    "--input",
                    str(self.gold_csv),
                    "--input-mode",
                    "csv",
                    "--hla",
                    "HLA-A*02:01",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            out = Path(tmp)
            expected = {
                "neo_candidates.json",
                "candidates.csv",
                "run_metadata.json",
                "case_report.md",
            }
            present = {p.name for p in out.iterdir() if p.is_file()}
            self.assertTrue(expected.issubset(present))
            self.assertNotIn("supported.csv", present)
            self.assertNotIn("rejected.csv", present)

    def test_cli_maf_outputs_expected_files(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self._run_cli(
                [
                    "--input",
                    str(self.maf_sample),
                    "--input-mode",
                    "maf",
                    "--hla",
                    "HLA-A*02:01",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            out = Path(tmp)
            expected = {
                "neo_candidates.json",
                "candidates.csv",
                "run_metadata.json",
                "case_report.md",
                "supported.csv",
                "rejected.csv",
            }
            present = {p.name for p in out.iterdir() if p.is_file()}
            self.assertTrue(expected.issubset(present))

    def test_cli_hla_parsing_repeat_and_comma_forms(self) -> None:
        parsed_repeat = neovax_cli._parse_hla_values(["HLA-A*02:01", "HLA-B*07:02"])
        parsed_comma = neovax_cli._parse_hla_values(["HLA-A*02:01,HLA-B*07:02"])
        self.assertEqual(parsed_repeat, ["HLA-A*02:01", "HLA-B*07:02"])
        self.assertEqual(parsed_comma, ["HLA-A*02:01", "HLA-B*07:02"])

    def test_cli_invalid_input_mode_or_missing_hla_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            # argparse should fail for missing --hla
            result = self._run_cli(
                [
                    "--input",
                    str(self.gold_csv),
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--hla", result.stderr)

    def test_cli_routes_through_service(self) -> None:
        fake_df = pd.DataFrame(
            [
                {
                    "gene_name": "TP53",
                    "mutation_position": 175,
                    "peptide": "SLMEQSDHF",
                    "best_allele": "HLA-A*02:01",
                    "presentation_score": 0.5,
                    "presentation_percentile": 1.0,
                    "affinity": 100.0,
                    "presentation_component": 1.0,
                    "affinity_component": 1.0,
                    "priority_score": 1.0,
                    "ranking_reason": "strong presentation + strong affinity",
                    "triage_label": "High Priority",
                    "triage_note": "Heuristic high-priority candidate.",
                }
            ]
        )
        fake_result = {
            "sample_id": "patient_001",
            "input_mode": "csv",
            "candidates_df": fake_df,
            "supported_rows_df": None,
            "rejected_rows_df": None,
            "run_metadata": {"run_id": "r1", "number_of_input_rows": 1, "number_of_output_candidates": 1},
            "neo_candidates": {"schema_name": "neo_candidates", "schema_version": "1.0.0"},
            "artifacts": {},
        }
        with TemporaryDirectory() as tmp:
            with mock.patch("backend.api.cli.neovax_cli.run_neovax", return_value=fake_result) as mocked:
                rc = neovax_cli.main(
                    [
                        "--input",
                        "x.csv",
                        "--input-mode",
                        "csv",
                        "--hla",
                        "HLA-A*02:01",
                        "--output-dir",
                        tmp,
                    ]
                )
                self.assertEqual(rc, 0)
                mocked.assert_called_once_with(
                    input_data="x.csv",
                    input_mode="csv",
                    hla_alleles=["HLA-A*02:01"],
                    sample_id="patient_001",
                    output_dir=None,
                    cache_path="data/cache/protein_sequences.json",
                )


if __name__ == "__main__":
    unittest.main()
