from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.core.data.tcga_data import join_tcga_metadata, write_stub_tcga_metadata
from backend.core.qualification.purity_resolution import (
    apply_evidence_provenance_columns,
    apply_purity_resolution,
    build_user_purity_lookup,
    load_purity_table,
    normalize_purity_scalar,
    resolve_purity_for_row,
)


class TestPurityResolution(unittest.TestCase):
    def test_normalize_rejects_out_of_range(self) -> None:
        v, w = normalize_purity_scalar(1.5)
        self.assertIsNone(v)
        v2, _ = normalize_purity_scalar(0.5)
        self.assertEqual(v2, 0.5)

    def test_user_file_overrides_tcga(self) -> None:
        row = pd.Series(
            {
                "patient_id": "TCGA-XX-0001-01A-11D-A1",
                "purity": 0.4,
                "purity_method": "tcga_metadata",
                "purity_confidence": 0.75,
            }
        )
        user_df = pd.DataFrame(
            [{"patient_id_key": "TCGA-XX-0001", "purity": 0.91, "purity_source": "manual"}]
        )
        lu = build_user_purity_lookup(user_df)
        out = resolve_purity_for_row(row, user_lookup=lu, priority=("user_file", "tcga_metadata", "stub_fallback"))
        self.assertAlmostEqual(out["purity_value_used"], 0.91, places=5)
        self.assertIn("user_file", out["purity_resolution_reason"])

    def test_tcga_used_when_no_user_and_stub_off(self) -> None:
        row = pd.Series(
            {
                "patient_id": "TCGA-YY-0002-01A",
                "purity": 0.44,
                "purity_method": "absolute",
                "purity_confidence": 0.8,
            }
        )
        out = resolve_purity_for_row(row, user_lookup=None, priority=("user_file", "tcga_metadata", "stub_fallback"), stub_fallback=False)
        self.assertAlmostEqual(out["purity_value_used"], 0.44, places=5)
        self.assertEqual(out["purity_source_used"], "tcga_metadata")

    def test_stub_fallback_only_when_enabled(self) -> None:
        row = pd.Series({"patient_id": "TCGA-ZZ-0003", "purity": float("nan")})
        off = resolve_purity_for_row(
            row, user_lookup=None, priority=("user_file", "tcga_metadata", "stub_fallback"), stub_fallback=False
        )
        self.assertTrue(math_is_nan(off["purity_value_used"]))
        self.assertEqual(off["purity_source_used"], "unresolved")
        on = resolve_purity_for_row(
            row, user_lookup=None, priority=("user_file", "tcga_metadata", "stub_fallback"), stub_fallback=True
        )
        self.assertAlmostEqual(on["purity_value_used"], 0.82, places=5)
        self.assertEqual(on["purity_source_used"], "stub_fallback")

    def test_prefix_barcode_match(self) -> None:
        long_id = "TCGA-AB-1234-01A-11D-A12DE3"
        user_df = pd.DataFrame([{"barcode_key": "TCGA-AB-1234-01A", "purity": 0.77}])
        lu = build_user_purity_lookup(user_df)
        row = pd.Series({"patient_id": long_id, "purity": float("nan")})
        out = resolve_purity_for_row(row, user_lookup=lu, priority=("user_file", "stub_fallback"), stub_fallback=True)
        self.assertAlmostEqual(out["purity_value_used"], 0.77, places=5)

    def test_candidates_json_roundtrip(self) -> None:
        df = pd.DataFrame([{"patient_id": "P1", "purity": 0.3}])
        out = apply_purity_resolution(df, stub_fallback=False)
        raw = out.loc[0, "purity_candidates_json"]
        self.assertIsInstance(json.loads(raw), list)

    def test_provenance_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P",
                    "RNA_data_missing": False,
                    "real_expression_tpm": 5.0,
                    "clonality_data_missing": False,
                    "real_ccf": 0.5,
                    "purity_source_used": "tcga_metadata",
                    "source": "stub_python",
                    "tcga_join_status": "matched",
                    "evidence_expression_source": "blended_real",
                    "evidence_ccf_source": "blended_real",
                }
            ]
        )
        out = apply_evidence_provenance_columns(df)
        self.assertEqual(out.loc[0, "expression_source"], "blended_real")
        self.assertEqual(out.loc[0, "ccf_source"], "blended_real")


def math_is_nan(x: object) -> bool:
    try:
        return bool(pd.isna(x)) or (isinstance(x, float) and x != x)
    except Exception:
        return False


class TestPurityTableLoad(unittest.TestCase):
    def test_load_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "p.csv"
            p.write_text("patient_id,purity\nTCGA-X,0.61\n", encoding="utf-8")
            df = load_purity_table(p)
            self.assertIn("purity", df.columns)

    def test_integration_join_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            meta_path = tdir / "meta.parquet"
            write_stub_tcga_metadata(["TCGA-DX-P1-01A"], meta_path, genes=["TP53"])
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
                    }
                ]
            )
            meta = pd.read_parquet(meta_path, engine="pyarrow")
            merged = join_tcga_metadata(cand, meta)
            out = apply_purity_resolution(merged, stub_fallback=False)
            self.assertIn("purity_value_used", out.columns)
            self.assertGreater(float(out.loc[0, "purity_value_used"]), 0.0)


if __name__ == "__main__":
    unittest.main()
