from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from backend.core.generation.aggregate_hla import aggregate_runs
from backend.core.prioritization.resistance_loop import apply_resistance_loop
from backend.core.qualification.clonality import apply_layer as apply_clonality_layer
from backend.core.qualification.escape import apply_layer as apply_escape_layer
from backend.core.qualification.expression import apply_layer as apply_expression_layer
from backend.core.qualification.presentation import apply_layer as apply_presentation_layer
from backend.tests.tmp_workspace import temp_workspace


def _write_minimal_run(run_dir: Path) -> None:
    hla = run_dir.parent.name
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "run_id": "test_run",
                "number_of_input_rows": 2,
                "number_of_output_candidates": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "gene_name": "GENE1",
            "mutation_position": 10,
            "peptide": "ABCDEFGH",
            "best_allele": hla,
            "presentation_score": 0.9,
            "presentation_percentile": 0.1,
            "affinity": 50.0,
            "priority_score": 0.99,
            "ranking_reason": "test",
            "triage_label": "High Priority",
            "triage_note": "test",
        },
        {
            "gene_name": "GENE2",
            "mutation_position": 20,
            "peptide": "ACDEFGHK",
            "best_allele": hla,
            "presentation_score": 0.5,
            "presentation_percentile": 0.5,
            "affinity": 500.0,
            "priority_score": 0.5,
            "ranking_reason": "test",
            "triage_label": "High Priority",
            "triage_note": "test",
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "candidates.csv", index=False)
    sup = pd.DataFrame(
        [
            {
                "gene_name": "GENE1",
                "mutation_position": 10,
                "source_hgvsp_short": "p.A10B",
                "source_sample_id": "TCGA-TEST-0001",
            },
            {
                "gene_name": "GENE2",
                "mutation_position": 20,
                "source_hgvsp_short": "p.C20D",
                "source_sample_id": "TCGA-TEST-0001",
            },
        ]
    )
    sup.to_csv(run_dir / "supported.csv", index=False)


class TestQualifySmoke(unittest.TestCase):
    def test_pipeline_in_memory(self) -> None:
        with temp_workspace("qualify_mem") as root:
            hla_root = root / "hla_test" / "HLA-A0201" / "sarc_TCGA_TEST_0001"
            hla_root.mkdir(parents=True)
            _write_minimal_run(hla_root)

            agg = aggregate_runs(root / "hla_test", top_n=10)
            self.assertEqual(len(agg), 2)
            self.assertIn("patient_id", agg.columns)

            df = apply_expression_layer(agg)
            df = apply_presentation_layer(df)
            df = apply_clonality_layer(df)
            df = apply_escape_layer(df)
            self.assertTrue((pd.to_numeric(df["escape_penalty"], errors="coerce").fillna(-1.0) == 0.0).all())
            df = apply_resistance_loop(df)

            self.assertIn("rl_priority", df.columns)
            self.assertIn("tier", df.columns)
            self.assertTrue(df["tier"].isin([1, 2, 3]).all())
            self.assertTrue((df["rl_priority"] >= 0.0).all() and (df["rl_priority"] <= 1.0).all())

            out = root / "out.parquet"
            df.to_parquet(out, index=False)
            back = pd.read_parquet(out)
            self.assertEqual(len(back), 2)

    def test_cli_module_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with temp_workspace("qualify_cli") as tmp_path:
            hla_root = tmp_path / "hla_test" / "HLA-A0201" / "sarc_TCGA_TEST_0002"
            hla_root.mkdir(parents=True)
            _write_minimal_run(hla_root)

            cmd = [
                sys.executable,
                "-m",
                "backend.cli.qualify_cohort",
                "--hla-runs-root",
                str(tmp_path / "hla_test"),
                "--top-n",
                "5",
                "--out-dir",
                str(tmp_path / "final"),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue((tmp_path / "final" / "multi_hla_candidates_v1.parquet").is_file())
            self.assertTrue((tmp_path / "final" / "qualified_candidates.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
