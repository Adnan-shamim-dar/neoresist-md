from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from neoresist.loaders import CohortLoadError, load_cohort_for_dash


class TestNeoresistLoaders(unittest.TestCase):
    def test_cohort_load_error_lists_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.csv"
            bad.write_text("only_one_col\n1\n", encoding="utf-8")
            try:
                load_cohort_for_dash(alt_path=str(bad))
            except CohortLoadError as e:
                self.assertTrue(e.searched)
                self.assertTrue(e.missing_columns)
                self.assertIn(str(bad.resolve()), e.searched[0])
            else:
                self.fail("expected CohortLoadError")

    def test_valid_minimal_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ok.parquet"
            df = pd.DataFrame(
                [
                    {
                        "patient_id": "P",
                        "hla_allele": "HLA-A0101",
                        "gene": "G",
                        "mutant_peptide": "AA",
                        "tier": 2,
                        "rl_priority": 0.5,
                        "expression_tpm": 1.0,
                        "ccf": 0.5,
                        "exclusion_reasons": [],
                    }
                ]
            )
            df.to_parquet(p, index=False)
            out, path, searched = load_cohort_for_dash(alt_path=str(p))
            self.assertEqual(path, str(p.resolve()))
            self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
