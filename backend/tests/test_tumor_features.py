from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from neoresist.tumor_features import add_candidate_tumor_flags, estimate_msi_status


class TumorFeatureTest(unittest.TestCase):
    def test_kras_msi_fixture_flags(self) -> None:
        fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "kras_msi_test.tsv"
        df = pd.read_csv(fixture, sep="\t")
        flagged = add_candidate_tumor_flags(df)
        self.assertTrue(bool(flagged.loc[0, "kras_g12_flag"]))
        self.assertTrue(bool(flagged.loc[0, "shared_neoantigen_flag"]))

        msi = estimate_msi_status(df)
        self.assertEqual(msi["msi_status"], "MSI-H")
        self.assertGreater(msi["msi_frameshift_indel_ratio"], 0.15)


if __name__ == "__main__":
    unittest.main()
