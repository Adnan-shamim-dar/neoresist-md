from __future__ import annotations

import unittest

import pandas as pd

from neoresist.canonical_schema import build_canonical_from_candidates, ensure_canonical_columns, validate_canonical_df


class CanonicalSchemaTest(unittest.TestCase):
    def test_ensure_canonical_columns_adds_missing_defaults(self) -> None:
        df = pd.DataFrame([{"patient_id": "P1", "mutant_peptide": "AAAA", "hla_allele": "HLA-A0101"}])
        out = ensure_canonical_columns(df, source_module="unit_test", run_id="run-1", sample_barcode="P1")
        self.assertIn("expression_confidence", out.columns)
        self.assertIn("schema_version", out.columns)
        self.assertEqual(out.loc[0, "schema_version"], "1.0.0")
        self.assertEqual(out.loc[0, "source_module"], "unit_test")
        self.assertEqual(out.loc[0, "run_id"], "run-1")
        self.assertEqual(out.loc[0, "sample_barcode"], "P1")
        self.assertEqual(out.loc[0, "expression_confidence"], "GREY")

    def test_build_canonical_from_candidates_maps_existing_fields(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "gene_name": "TP53",
                    "peptide": "AAAAAA",
                    "best_allele": "HLA-A0201",
                    "presentation_score": 0.81,
                    "rl_priority": 0.74,
                    "tier": 1,
                }
            ]
        )
        out = build_canonical_from_candidates(df, source_module="neoantigen_generation", run_id="case-1", sample_barcode="P1")
        self.assertEqual(out.loc[0, "gene"], "TP53")
        self.assertEqual(out.loc[0, "mutant_peptide"], "AAAAAA")
        self.assertEqual(out.loc[0, "hla_allele"], "HLA-A0201")
        self.assertEqual(out.loc[0, "presentation_composite"], 0.81)
        self.assertEqual(out.loc[0, "composite_priority"], 0.74)
        result = validate_canonical_df(out)
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()
