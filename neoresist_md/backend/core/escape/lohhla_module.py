from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from ..base_module import BaseModule


class LOHHLAModule(BaseModule):
    NAME = "escape"
    INPUT_COLUMNS = []
    OUTPUT_COLUMNS = [
        "hla_loh_status",
        "hla_loh_probability",
        "allele_integrity_flag",
        "processing_disruption_flag",
        "escape_tool",
        "escape_confidence",
    ]

    @staticmethod
    def _lohhla_available() -> bool:
        candidates = [["LOHHLA"], ["lohhla"]]
        for cmd in candidates:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                if proc.returncode in {0, 1, 2}:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _has_processing_mutation_from_row(row: pd.Series) -> bool:
        genes = {"B2M", "TAP1", "TAP2", "TAPBP"}
        gene = str(row.get("gene", "")).upper().strip()
        if gene in genes:
            return True
        return False

    @staticmethod
    def _scan_vcf_for_processing_mutation(vcf_path: str | None) -> bool:
        if not vcf_path:
            return False
        path = Path(str(vcf_path))
        if not path.exists() or not path.is_file():
            return False
        genes = ("B2M", "TAP1", "TAP2", "TAPBP")
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    upper = line.upper()
                    if any(g in upper for g in genes):
                        return True
        except Exception:
            return False
        return False

    def _run_stub(self, out: pd.DataFrame) -> pd.DataFrame:
        # STUB — replace when LOHHLA runtime and full allele-specific copy-number inputs are available.
        out["hla_loh_status"] = "UNKNOWN"
        out["hla_loh_probability"] = 0.5
        out["allele_integrity_flag"] = None
        out["processing_disruption_flag"] = False
        for idx, row in out.iterrows():
            vcf_hit = self._scan_vcf_for_processing_mutation(row.get("vcf_path"))
            row_hit = self._has_processing_mutation_from_row(row)
            if vcf_hit or row_hit:
                out.at[idx, "processing_disruption_flag"] = True
                out.at[idx, "allele_integrity_flag"] = False
        out["escape_tool"] = "lohhla_stub"
        out["escape_confidence"] = "UNAVAILABLE"
        return out

    def _run_lohhla_proxy(self, out: pd.DataFrame) -> pd.DataFrame:
        # Best-effort placeholder for installed runtime; replace with true LOHHLA parsing.
        out["hla_loh_status"] = "UNKNOWN"
        out["hla_loh_probability"] = 0.3
        out["allele_integrity_flag"] = True
        out["processing_disruption_flag"] = out.apply(
            lambda r: self._has_processing_mutation_from_row(r) or self._scan_vcf_for_processing_mutation(r.get("vcf_path")),
            axis=1,
        )
        out["escape_tool"] = "lohhla"
        out["escape_confidence"] = "LOW"
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None
        if not self._lohhla_available():
            return self._run_stub(out)
        return self._run_lohhla_proxy(out)
