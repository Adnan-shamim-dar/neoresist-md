from __future__ import annotations

import base64
import unittest
from pathlib import Path

from neoresist.case_store import create_case_from_upload
from neoresist.module_runner import ModuleRunner
from neoresist.upload_runtime import validate_uploaded_table


def _demo_vcf_bytes() -> bytes:
    return (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTUMOR\n"
        "1\t12345\t.\tA\tT\t.\tPASS\t.\tGT:AD:DP\t0/1:30,20:50\n"
    ).encode("utf-8")


class UploadRuntimeTest(unittest.TestCase):
    def test_vcf_normalizes_to_predictor_ready_csv(self) -> None:
        validation = validate_uploaded_table(_demo_vcf_bytes(), "demo.vcf")
        self.assertEqual(validation.input_mode, "csv")
        self.assertEqual(validation.normalized_filename, "demo.normalized.csv")
        self.assertEqual(validation.row_count, 1)
        self.assertIsNotNone(validation.normalized_bytes)
        header = validation.normalized_bytes.decode("utf-8").splitlines()[0]
        self.assertEqual(
            header,
            "gene_name,protein_sequence,mutation_position,normal_amino_acid,mutant_amino_acid",
        )

    def test_case_creation_persists_normalized_csv_input(self) -> None:
        contents = "data:text/plain;base64," + base64.b64encode(_demo_vcf_bytes()).decode("ascii")
        case = create_case_from_upload(contents, "demo.vcf", install_checks=ModuleRunner().installation_checks())
        case_dir = Path(case["case_dir"])
        inputs = list((case_dir / "input").glob("*"))
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].suffix, ".csv")
        text = inputs[0].read_text(encoding="utf-8")
        self.assertIn("gene_name,protein_sequence,mutation_position,normal_amino_acid,mutant_amino_acid", text)


if __name__ == "__main__":
    unittest.main()
