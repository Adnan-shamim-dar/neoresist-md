from __future__ import annotations

import unittest

import pandas as pd

from backend.tests.rl_v1_reference import apply_resistance_loop_reference
from neoresist.scoring import apply_resistance_loop_engine


class TestRlV1Parity(unittest.TestCase):
    def _fixture(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "hla_allele": "HLA-A0101",
                    "gene": "TP53",
                    "mutant_peptide": "AAAA",
                    "expression_tpm": 50.0,
                    "presentation_score": 0.6,
                    "ccf": 0.45,
                    "self_dissimilarity": 0.3,
                    "escape_penalty": 0.05,
                    "real_expression_tpm": float("nan"),
                    "real_ccf": float("nan"),
                },
                {
                    "patient_id": "P2",
                    "hla_allele": "HLA-A0101",
                    "gene": "BRCA1",
                    "mutant_peptide": "BBBB",
                    "expression_tpm": 10.0,
                    "presentation_score": 0.2,
                    "ccf": 0.9,
                    "escape_penalty": 0.0,
                    "real_expression_tpm": 200.0,
                    "real_ccf": 0.5,
                },
            ]
        )

    def test_parity_stub_only(self) -> None:
        df = self._fixture()
        ref = apply_resistance_loop_reference(df, prefer_real_evidence=False)
        eng = apply_resistance_loop_engine(df, prefer_real_evidence=False, profile_id="rl_v1")
        for col in (
            "rl_priority",
            "tier",
            "expression_norm",
            "evidence_expression_source",
            "evidence_ccf_source",
            "self_dissimilarity",
        ):
            pd.testing.assert_series_equal(
                ref[col].reset_index(drop=True),
                eng[col].reset_index(drop=True),
                rtol=1e-9,
                atol=1e-9,
                obj=f"column {col}",
            )

    def test_parity_prefer_real(self) -> None:
        df = self._fixture()
        ref = apply_resistance_loop_reference(df, prefer_real_evidence=True)
        eng = apply_resistance_loop_engine(df, prefer_real_evidence=True, profile_id="rl_v1")
        for col in (
            "rl_priority",
            "tier",
            "expression_norm",
            "evidence_expression_source",
            "evidence_ccf_source",
            "self_dissimilarity",
        ):
            pd.testing.assert_series_equal(
                ref[col].reset_index(drop=True),
                eng[col].reset_index(drop=True),
                rtol=1e-9,
                atol=1e-9,
                obj=f"column {col}",
            )


if __name__ == "__main__":
    unittest.main()
