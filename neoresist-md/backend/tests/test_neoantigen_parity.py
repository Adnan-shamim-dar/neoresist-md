from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

from backend.core.neoantigen.rank import validate_mutation_rows
from backend.core.neoantigen.service import run_neovax


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestNeoantigenParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cls.neovax_builder_root = repo_root.parent / "neovax-builder"
        if str(cls.neovax_builder_root) not in sys.path:
            sys.path.insert(0, str(cls.neovax_builder_root))
        cls.gold_csv = cls.neovax_builder_root / "data" / "example_mutations_gold.csv"
        cls.maf_sample = cls.neovax_builder_root / "data" / "tcga_brca_sample.maf"
        cls.hlas = ["HLA-A*02:01"]

        cls.old_predictor = _load_module(
            "old_predictor_module",
            cls.neovax_builder_root / "core" / "predictor.py",
        )
        cls.old_maf_pipeline = _load_module(
            "old_maf_pipeline_module",
            cls.neovax_builder_root / "input" / "maf_pipeline.py",
        )

    def test_csv_parity_top_rows(self) -> None:
        old_df = pd.read_csv(self.gold_csv)
        old_result = self.old_predictor.predict_candidates(old_df, self.hlas)

        new_result = run_neovax(input_data=str(self.gold_csv), input_mode="csv", hla_alleles=self.hlas)
        new_df = new_result["candidates_df"]

        compare_cols = [
            "gene_name",
            "peptide",
            "best_allele",
            "presentation_score",
            "affinity",
            "presentation_component",
            "affinity_component",
            "priority_score",
            "triage_label",
            "ranking_reason",
        ]
        old_top = old_result[compare_cols].head(10).reset_index(drop=True)
        new_top = new_df[compare_cols].head(10).reset_index(drop=True)

        for col in compare_cols:
            if pd.api.types.is_numeric_dtype(old_top[col]):
                diff = (old_top[col] - new_top[col]).abs().max()
                self.assertLessEqual(float(diff), 1e-6, msg=f"Numeric mismatch in column {col}")
            else:
                self.assertTrue((old_top[col] == new_top[col]).all(), msg=f"Value mismatch in column {col}")

    def test_maf_ingest_count_parity(self) -> None:
        old = self.old_maf_pipeline.ingest_maf_to_predictor_rows(str(self.maf_sample))
        new = run_neovax(input_data=str(self.maf_sample), input_mode="maf", hla_alleles=self.hlas)

        self.assertEqual(len(old["predictor_rows_df"]), len(new["supported_rows_df"]))
        self.assertEqual(len(old["rejected_rows_df"]), len(new["rejected_rows_df"]))
        self.assertEqual(old["stats"]["total_maf_rows"], new["run_metadata"]["total_maf_rows"])
        self.assertEqual(old["stats"]["missense_rows"], new["run_metadata"]["missense_rows"])
        self.assertEqual(old["stats"]["parseable_rows"], new["run_metadata"]["parseable_rows"])
        self.assertEqual(old["stats"]["sequence_matched_rows"], new["run_metadata"]["sequence_matched_rows"])
        self.assertEqual(old["stats"]["rejected_rows"], new["run_metadata"]["rejected_rows"])

    def test_validation_error_parity_shape(self) -> None:
        bad_df = pd.DataFrame(
            [
                {
                    "gene_name": "TP53",
                    "protein_sequence": "MEEPQSDPSV",
                    "mutation_position": 50,
                    "normal_amino_acid": "R",
                    "mutant_amino_acid": "H",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, r"Row 1:"):
            validate_mutation_rows(bad_df)

    def test_metadata_required_keys(self) -> None:
        result = run_neovax(input_data=str(self.gold_csv), input_mode="csv", hla_alleles=self.hlas)
        metadata = result["run_metadata"]
        required = {
            "run_id",
            "timestamp",
            "input_file_path",
            "input_file_sha256",
            "number_of_input_rows",
            "hla_alleles_used",
            "peptide_lengths_used",
            "predictor_name",
            "ranking_formula_version",
            "output_csv_path",
            "number_of_output_candidates",
            "input_mode",
        }
        self.assertTrue(required.issubset(set(metadata.keys())))


if __name__ == "__main__":
    unittest.main()
