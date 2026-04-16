from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

from neoresist.case_store import create_case_from_upload, read_case_manifest, read_module_status, reset_modules_for_rerun, synchronize_manifest, write_module_status
from neoresist.module_runner import ModuleRunner
from neoresist.module_schema import load_module_schema


def _encoded_csv() -> str:
    raw = (
        "gene_name,protein_sequence,mutation_position,normal_amino_acid,mutant_amino_acid\n"
        "TP53,MEEPQSDPSV,4,P,S\n"
    ).encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")


class CasePlatformTest(unittest.TestCase):
    def test_module_schema_and_case_creation(self) -> None:
        schema = load_module_schema()
        self.assertIn("neoantigen_generation", schema)
        self.assertIn("expression_join", schema)
        self.assertIn("clonality_pyclone_vi", schema)
        self.assertIn("resistance_loop", schema)
        self.assertIn("strategy_engine", schema)
        self.assertIn("prioritization_tiering", schema)

        runner = ModuleRunner()
        case = create_case_from_upload(
            _encoded_csv(),
            "patient_case.csv",
            install_checks=runner.installation_checks(),
        )
        manifest = read_case_manifest(case["case_id"])
        self.assertEqual(manifest["input_mode"], "csv")
        self.assertTrue(manifest["enabled_modules"]["neoantigen_generation"])

        neo_status = read_module_status(case["case_id"], "neoantigen_generation")
        expr_status = read_module_status(case["case_id"], "expression_join")
        self.assertEqual(neo_status["status"], "pending")
        self.assertEqual(expr_status["status"], "pending")

    def test_resume_case_recovers_stale_running_module(self) -> None:
        runner = ModuleRunner()
        case = create_case_from_upload(
            _encoded_csv(),
            "patient_case.csv",
            install_checks=runner.installation_checks(),
        )
        case_id = case["case_id"]
        manifest = read_case_manifest(case_id)
        manifest["run_requested"] = True
        from neoresist.case_store import write_case_manifest

        write_case_manifest(manifest)
        status = read_module_status(case_id, "neoantigen_generation")
        status["status"] = "running"
        status["pid"] = 999999
        write_module_status(case_id, "neoantigen_generation", status)

        with patch.object(ModuleRunner, "_spawn_worker") as spawn_worker:
            decisions = runner.resume_case(case_id)

        recovered = read_module_status(case_id, "neoantigen_generation")
        self.assertEqual(decisions["neoantigen_generation"], "running")
        self.assertEqual(recovered["status"], "pending")
        self.assertEqual(recovered["checkpoint_label"], "recovered_stale_run")
        spawn_worker.assert_called()

    def test_synchronize_manifest_initializes_new_modules(self) -> None:
        runner = ModuleRunner()
        case = create_case_from_upload(
            _encoded_csv(),
            "patient_case.csv",
            install_checks=runner.installation_checks(),
        )
        manifest = synchronize_manifest(case["case_id"])
        self.assertIn("resistance_loop", manifest["module_status"])
        self.assertIn("strategy_engine", manifest["module_status"])
        self.assertIn("prioritization_tiering", manifest["module_status"])

    def test_reset_modules_for_rerun_cascades_pending_state(self) -> None:
        runner = ModuleRunner()
        case = create_case_from_upload(
            _encoded_csv(),
            "patient_case.csv",
            install_checks=runner.installation_checks(),
        )
        case_id = case["case_id"]
        for module_id in ["expression_join", "resistance_loop", "strategy_engine", "prioritization_tiering"]:
            status = read_module_status(case_id, module_id)
            status["status"] = "complete"
            write_module_status(case_id, module_id, status)
        reset_modules_for_rerun(case_id, ["expression_join", "resistance_loop", "strategy_engine", "prioritization_tiering"])
        self.assertEqual(read_module_status(case_id, "expression_join")["status"], "pending")
        self.assertEqual(read_module_status(case_id, "resistance_loop")["checkpoint_label"], "reset_for_rerun")
        self.assertFalse(read_module_status(case_id, "prioritization_tiering")["resume_ready"])


if __name__ == "__main__":
    unittest.main()
