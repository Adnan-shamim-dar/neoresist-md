from __future__ import annotations

import base64
import unittest

from neoresist.case_store import create_case_from_upload
from neoresist.module_runner import ModuleRunner
from neoresist.ops_status import collect_batch_status


def _encoded_csv() -> str:
    raw = (
        "gene_name,protein_sequence,mutation_position,normal_amino_acid,mutant_amino_acid\n"
        "TP53,MEEPQSDPSV,4,P,S\n"
    ).encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")


class OpsStatusTest(unittest.TestCase):
    def test_collect_batch_status_includes_case_module_counts(self) -> None:
        runner = ModuleRunner()
        create_case_from_upload(_encoded_csv(), "ops_case.csv", install_checks=runner.installation_checks())
        status = collect_batch_status()
        self.assertIn("persisted_cases", status)
        self.assertIn("module_state_counts", status)
        self.assertGreaterEqual(status["persisted_cases"], 1)
        self.assertIn("pending", status["module_state_counts"])


if __name__ == "__main__":
    unittest.main()
