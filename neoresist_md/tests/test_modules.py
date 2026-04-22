from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from neoresist_md.backend.core.generation.mhcflurry_module import MHCFlurryModule
from neoresist_md.backend.core.recognition.foreignness_module import ForeignnessModule


def test_generation_module_runs():
    df = pd.DataFrame([{"patient_id": "P1", "gene": "TP53"}])
    out = MHCFlurryModule().safe_run(df)
    assert "binding_affinity" in out.columns
    assert "binding_confidence" in out.columns


def test_foreignness_scores_are_bounded():
    df = pd.DataFrame(
        [
            {
                "mutant_peptide": "GILGFVFTL",
                "wildtype_peptide": "GILGFVFAL",
            }
        ]
    )
    out = ForeignnessModule().safe_run(df)
    assert 0.0 <= float(out.loc[0, "mutant_wt_distance"]) <= 1.0
    assert 0.0 <= float(out.loc[0, "self_dissimilarity"]) <= 1.0
    assert 0.0 <= float(out.loc[0, "recognition_score"]) <= 1.0
    assert out.loc[0, "recognition_tool"] == "blosum62_alignment"
    assert out.loc[0, "recognition_confidence"] == "HIGH"


def test_foreignness_mutant_only_is_medium_confidence():
    df = pd.DataFrame([{"mutant_peptide": "SYFPEITHI", "wildtype_peptide": None}])
    out = ForeignnessModule().safe_run(df)
    assert pd.isna(out.loc[0, "mutant_wt_distance"])
    assert 0.0 <= float(out.loc[0, "self_dissimilarity"]) <= 1.0
    assert out.loc[0, "recognition_confidence"] == "MEDIUM"
    assert out.loc[0, "genome_build"] == "GRCh38"


def test_generation_stub_sets_low_confidence_and_warning():
    df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "gene": "TP53",
                "mutant_peptide": "GILGFVFTL",
                "hla_allele": "HLA-A*02:01",
            }
        ]
    )
    with patch.object(MHCFlurryModule, "_mhcflurry_available", return_value=False):
        out = MHCFlurryModule().safe_run(df)
    assert out.loc[0, "binding_confidence"] == "LOW"
    assert out.attrs.get("stub_warning")
    assert out.loc[0, "diploid_assumption_flag"] is True


def test_generation_invalid_hla_sets_unavailable():
    df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "gene": "TP53",
                "mutant_peptide": "GILGFVFTL",
                "hla_allele": "BADALLELE",
            }
        ]
    )
    with patch.object(MHCFlurryModule, "_mhcflurry_available", return_value=False):
        out = MHCFlurryModule().safe_run(df)
    assert out.loc[0, "binding_confidence"] == "UNAVAILABLE"


def test_generation_does_not_recompute_existing_binding():
    df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "gene": "TP53",
                "mutant_peptide": "GILGFVFTL",
                "hla_allele": "HLA-A*02:01",
                "binding_affinity": 123.0,
                "binding_rank": 0.5,
            }
        ]
    )
    with patch.object(MHCFlurryModule, "_mhcflurry_available", return_value=False):
        out = MHCFlurryModule().safe_run(df)
    assert float(out.loc[0, "binding_affinity"]) == 123.0
    assert float(out.loc[0, "binding_rank"]) == 0.5


def test_generation_real_path_populates_presentation_columns():
    class FakePredictor:
        @staticmethod
        def load():
            return FakePredictor()

        def predict(self, peptides, alleles):
            return pd.DataFrame(
                {
                    "affinity": [42.0 for _ in peptides],
                    "presentation_percentile": [0.4 for _ in peptides],
                    "processing_score": [0.7 for _ in peptides],
                    "presentation_score": [0.8 for _ in peptides],
                }
            )

    df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "gene": "TP53",
                "mutant_peptide": "GILGFVFTL",
                "hla_allele": "HLA-A*02:01",
            }
        ]
    )
    with patch.object(MHCFlurryModule, "_mhcflurry_available", return_value=True):
        with patch("mhcflurry.Class1PresentationPredictor", FakePredictor):
            with patch("mhcflurry.__version__", "2.0-test"):
                out = MHCFlurryModule().safe_run(df)
    assert float(out.loc[0, "binding_affinity"]) == 42.0
    assert float(out.loc[0, "binding_rank"]) == 0.4
    assert float(out.loc[0, "cleavage_score"]) == 0.7
    assert float(out.loc[0, "tap_score"]) == 0.8
    assert out.loc[0, "presentation_tool"] == "mhcflurry_2.0"
    assert out.loc[0, "presentation_confidence"] == "HIGH"
