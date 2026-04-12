from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.cli import enrich_cohort as enrich_mod
from backend.core.data.tcga_data import join_tcga_metadata, write_stub_tcga_metadata
from backend.core.prioritization.resistance_loop import apply_resistance_loop
from backend.core.qualification.purity_resolution import apply_evidence_provenance_columns, apply_purity_resolution
from backend.core.qualification.real_clonality import apply_layer as apply_real_clonality_layer
from backend.core.qualification.real_expression import apply_layer as apply_real_expression_layer


class TestTcgaIntegration(unittest.TestCase):
    def test_stub_metadata_join_expression_clonality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            meta_path = tdir / "tcga_sarc_metadata.parquet"
            pids = ["TCGA-DX-P1-01A", "TCGA-DX-P2-01A"]
            write_stub_tcga_metadata(pids, meta_path, genes=["GENE1", "TP53"])

            cand = pd.DataFrame(
                [
                    {
                        "patient_id": "TCGA-DX-P1-01A",
                        "hla_allele": "HLA-A0101",
                        "gene": "TP53",
                        "mutant_peptide": "AAAA",
                        "tier": 2,
                        "rl_priority": 0.5,
                        "expression_tpm": 10.0,
                        "presentation_score": 0.5,
                        "ccf": 0.4,
                        "hla_loh_status": "intact",
                        "self_dissimilarity": 0.5,
                        "exclusion_reasons": [],
                        "escape_penalty": 0.0,
                        "vaf": 0.2,
                    },
                    {
                        "patient_id": "TCGA-DX-P2-01A",
                        "hla_allele": "HLA-A0101",
                        "gene": "GENE1",
                        "mutant_peptide": "BBBB",
                        "tier": 3,
                        "rl_priority": 0.3,
                        "expression_tpm": 5.0,
                        "presentation_score": 0.4,
                        "ccf": 0.3,
                        "hla_loh_status": "intact",
                        "self_dissimilarity": 0.4,
                        "exclusion_reasons": [],
                        "escape_penalty": 0.0,
                        "vaf": 0.15,
                    },
                ]
            )

            meta = pd.read_parquet(meta_path, engine="pyarrow")
            merged = join_tcga_metadata(cand, meta)
            self.assertTrue((merged["tcga_join_status"] == "matched").all())

            merged = apply_purity_resolution(merged, stub_fallback=False)
            self.assertIn("purity_value_used", merged.columns)
            self.assertIn("purity_source_used", merged.columns)

            merged = apply_real_expression_layer(merged)
            self.assertIn("real_expression_tpm", merged.columns)
            self.assertIn("expression_bin", merged.columns)
            tpm0 = merged.loc[0, "real_expression_tpm"]
            self.assertIsNotNone(tpm0)
            self.assertGreater(float(tpm0), 0.0)

            merged = apply_real_clonality_layer(merged, pyclone_out_dir=tdir / "py")
            self.assertIn("real_ccf", merged.columns)
            ccf0 = merged.loc[0, "real_ccf"]
            self.assertIsNotNone(ccf0)
            self.assertGreaterEqual(float(ccf0), 0.0)
            self.assertLessEqual(float(ccf0), 1.0)

            out = apply_resistance_loop(merged, prefer_real_evidence=True)
            self.assertIn("rl_priority", out.columns)
            self.assertIn("evidence_expression_source", out.columns)
            self.assertTrue(out["rl_priority"].between(0.0, 1.0).all())

            out = apply_evidence_provenance_columns(out)
            for col in (
                "expression_source",
                "ccf_source",
                "purity_source",
                "metadata_source",
                "resolution_status",
                "evidence_notes",
            ):
                self.assertIn(col, out.columns)

    def test_enrich_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            qual = tdir / "qualified_candidates.parquet"
            df = pd.DataFrame(
                [
                    {
                        "patient_id": "TCGA-DX-SMOKE-01A",
                        "hla_allele": "HLA-A0201",
                        "gene": "TP53",
                        "gene_name": "TP53",
                        "mutant_peptide": "LLL",
                        "tier": 2,
                        "rl_priority": 0.5,
                        "expression_tpm": 8.0,
                        "presentation_score": 0.6,
                        "ccf": 0.5,
                        "hla_loh_status": "intact",
                        "self_dissimilarity": 0.5,
                        "exclusion_reasons": [],
                        "escape_penalty": 0.0,
                        "vaf": 0.12,
                    }
                ]
            )
            df.to_parquet(qual, index=False)
            out = tdir / "enriched_candidates.parquet"
            meta = tdir / "meta.parquet"
            rc = enrich_mod.main(
                [
                    "--input",
                    str(qual),
                    "--output",
                    str(out),
                    "--metadata",
                    str(meta),
                    "--intermediate-dir",
                    str(tdir / "inter"),
                    "--stub",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            edf = pd.read_parquet(out, engine="pyarrow")
            self.assertIn("real_expression_tpm", edf.columns)
            self.assertIn("real_ccf", edf.columns)
            self.assertIn("purity_value_used", edf.columns)
            self.assertIn("expression_source", edf.columns)

    def test_enrich_with_purity_file_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            qual = tdir / "qualified_candidates.parquet"
            df = pd.DataFrame(
                [
                    {
                        "patient_id": "TCGA-DX-OV-01A",
                        "hla_allele": "HLA-A0201",
                        "gene": "TP53",
                        "gene_name": "TP53",
                        "mutant_peptide": "LLL",
                        "tier": 2,
                        "rl_priority": 0.5,
                        "expression_tpm": 8.0,
                        "presentation_score": 0.6,
                        "ccf": 0.5,
                        "hla_loh_status": "intact",
                        "self_dissimilarity": 0.5,
                        "exclusion_reasons": [],
                        "escape_penalty": 0.0,
                        "vaf": 0.12,
                    }
                ]
            )
            df.to_parquet(qual, index=False)
            pur = tdir / "purity.csv"
            pur.write_text("patient_id,purity,purity_source\nTCGA-DX-OV-01A,0.95,manual_test\n", encoding="utf-8")
            out = tdir / "enriched_candidates.parquet"
            meta = tdir / "meta.parquet"
            rc = enrich_mod.main(
                [
                    "--input",
                    str(qual),
                    "--output",
                    str(out),
                    "--metadata",
                    str(meta),
                    "--intermediate-dir",
                    str(tdir / "inter"),
                    "--stub",
                    "--purity-file",
                    str(pur),
                ]
            )
            self.assertEqual(rc, 0)
            edf = pd.read_parquet(out, engine="pyarrow")
            self.assertAlmostEqual(float(edf.loc[0, "purity_value_used"]), 0.95, places=5)
            self.assertEqual(str(edf.loc[0, "purity_source_used"]), "manual_test")


if __name__ == "__main__":
    unittest.main()
