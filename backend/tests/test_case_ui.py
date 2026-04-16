from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path

from dash.development.base_component import Component

from neoresist.case_store import create_case_from_upload, read_module_status, write_module_status
from neoresist.case_ui import render_case_detail
from neoresist.module_runner import ModuleRunner


def _encoded_csv() -> str:
    raw = (
        "gene_name,protein_sequence,mutation_position,normal_amino_acid,mutant_amino_acid\n"
        "TP53,MEEPQSDPSV,4,P,S\n"
    ).encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")


def _collect_text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return " ".join(_collect_text(item) for item in node)
    if isinstance(node, Component):
        return _collect_text(getattr(node, "children", None))
    return str(node)


class CaseUiTest(unittest.TestCase):
    def test_expert_case_detail_includes_artifacts_tab_content(self) -> None:
        runner = ModuleRunner()
        case = create_case_from_upload(_encoded_csv(), "patient_case.csv", install_checks=runner.installation_checks())
        case_id = case["case_id"]
        artifact_dir = Path(case["case_dir"]) / "modules" / "neoantigen_generation" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = artifact_dir / "canonical_candidates.csv"
        canonical_path.write_text("patient_id,expression_confidence\nP1,GREEN\n", encoding="utf-8")
        status = read_module_status(case_id, "neoantigen_generation")
        status["status"] = "complete"
        status["artifacts"] = {"canonical_candidates_csv": str(canonical_path)}
        write_module_status(case_id, "neoantigen_generation", status)

        rendered = render_case_detail(case_id, "Expert")
        text = _collect_text(rendered)
        self.assertIn("Artifacts", text)
        self.assertIn("canonical_candidates.csv", text)
        self.assertIn("Resume ready", text)

    def test_unavailable_placeholder_module_shows_runtime_missing(self) -> None:
        runner = ModuleRunner()
        case = create_case_from_upload(_encoded_csv(), "patient_case.csv", install_checks=runner.installation_checks())
        case_id = case["case_id"]
        artifact_dir = Path(case["case_dir"]) / "modules" / "presentation_netctlpan" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        adapter_status = artifact_dir / "adapter_status.json"
        adapter_status.write_text(json.dumps({"available": False, "message": "runtime missing"}), encoding="utf-8")
        status = read_module_status(case_id, "presentation_netctlpan")
        status["status"] = "unavailable"
        status["installed"] = False
        status["artifacts"] = {"adapter_status_json": str(adapter_status)}
        write_module_status(case_id, "presentation_netctlpan", status)

        rendered = render_case_detail(case_id, "Expert")
        text = _collect_text(rendered)
        self.assertIn("Runtime missing", text)
        self.assertIn("presentation_netctlpan", text)


if __name__ == "__main__":
    unittest.main()
