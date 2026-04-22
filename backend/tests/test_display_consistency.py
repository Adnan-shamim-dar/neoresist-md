from __future__ import annotations

import unittest

import pandas as pd

from neoresist.dash_app.display_utils import (
    apply_display_policy,
    display_score_terms,
    ensure_display_columns,
    normalize_hla_display,
)
from neoresist.scoring import apply_resistance_loop_engine


class TestDisplayConsistency(unittest.TestCase):
    def test_display_aliases_fill_gene_and_peptide(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "gene_name": "TP53",
                    "peptide": "YLQPRTFLL",
                    "source_hgvsp_short": "p.R175H",
                    "rl_priority": 0.58,
                    "hla_allele": "HLA-A*02:01",
                }
            ]
        )
        out = ensure_display_columns(df)
        self.assertEqual(str(out.loc[0, "gene"]), "TP53")
        self.assertEqual(str(out.loc[0, "mutant_peptide"]), "YLQPRTFLL")
        self.assertEqual(str(out.loc[0, "protein_change"]), "p.R175H")

    def test_evidence_panel_math_consistency(self) -> None:
        scored = apply_resistance_loop_engine(
            pd.DataFrame(
                [
                    {
                        "patient_id": "CASE1",
                        "hla_allele": "HLA-A*02:01",
                        "gene": "TP53",
                        "mutant_peptide": "YLQPRTFLL",
                        "expression_tpm": 12.0,
                        "presentation_score": 0.8,
                        "ccf": 0.6,
                        "self_dissimilarity": 0.4,
                        "escape_penalty": 0.0,
                    }
                ]
            ),
            prefer_real_evidence=False,
            profile_id="rl_v1",
            rule_profile_id="default_rules",
        )
        row = scored.iloc[0]
        immunogenicity, resistance_penalty, rl_priority = display_score_terms(row)
        self.assertAlmostEqual(immunogenicity * (1.0 - resistance_penalty), rl_priority, places=3)
        self.assertAlmostEqual(rl_priority, float(row["rl_priority"]), places=3)

    def test_display_score_uses_weighted_resistance_penalty(self) -> None:
        row = pd.Series(
            {
                "scoring_profile": "rl_v1",
                "expression_norm": 0.7,
                "presentation_score": 0.8,
                "ccf": 0.6,
                "self_dissimilarity": 0.5,
                "escape_penalty": 0.5,
                "rl_priority": 0.61,
            }
        )
        _immunogenicity, resistance_penalty, _rl_priority = display_score_terms(row)
        self.assertAlmostEqual(resistance_penalty, 0.1, places=3)

    def test_hla_unknown_tier_gate_caps_tier_one(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "gene": "KRAS",
                    "mutant_peptide": "GADGVGKSA",
                    "rl_priority": 0.92,
                    "hla_allele": "HLA-UNK",
                    "expression_tpm": 12.0,
                    "presentation_score": 0.88,
                    "ccf": 0.91,
                    "self_dissimilarity": 0.44,
                    "escape_penalty": 0.0,
                }
            ]
        )
        out = apply_display_policy(df, tier1_above=0.43, tier2_above=0.34, exclusion_rules={})
        self.assertEqual(out.loc[0, "tier"], 2)
        self.assertEqual(out.loc[0, "hla_confidence"], "pan-allele estimate")

    def test_hla_test_normalizes_to_unknown(self) -> None:
        self.assertEqual(normalize_hla_display("hla_test"), "HLA-UNK")


if __name__ == "__main__":
    unittest.main()
