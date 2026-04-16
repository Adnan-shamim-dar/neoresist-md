from __future__ import annotations

import random
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..base_module import BaseModule


class NetCTLpanModule(BaseModule):
    NAME = "presentation"
    INPUT_COLUMNS = []
    OUTPUT_COLUMNS = [
        "cleavage_score",
        "tap_score",
        "presentation_composite",
        "presentation_tool",
        "presentation_confidence",
    ]

    @staticmethod
    def _tool_available() -> bool:
        try:
            proc = subprocess.run(["netCTLpan"], capture_output=True, text=True, timeout=2)
            return proc.returncode in {0, 1, 2}
        except Exception:
            return False

    @staticmethod
    def _seed_from_row(row: pd.Series) -> int:
        key = f"{row.get('patient_id', '')}|{row.get('gene', '')}|{row.get('mutant_peptide', '')}|{row.get('hla_allele', '')}"
        return abs(hash(key)) % (2**31 - 1)

    def _run_stub(self, out: pd.DataFrame) -> pd.DataFrame:
        # STUB — replace when tool available.
        cleavage = []
        tap = []
        comp = []
        for _, row in out.iterrows():
            rng = random.Random(self._seed_from_row(row))
            c = round(rng.uniform(0.05, 0.95), 4)
            t = round(rng.uniform(0.05, 0.95), 4)
            cleavage.append(c)
            tap.append(t)
            comp.append(round((c + t) / 2.0, 4))
        out["cleavage_score"] = cleavage
        out["tap_score"] = tap
        out["presentation_composite"] = comp
        out["presentation_tool"] = "netctlpan_stub"
        out["presentation_confidence"] = "LOW"
        return out

    def _run_netctlpan(self, out: pd.DataFrame) -> pd.DataFrame:
        """
        Best-effort CLI execution and parse.
        Falls back to stub when CLI output is not parseable for this environment.
        """
        try:
            peptides = out.get("mutant_peptide", pd.Series(["AAAAAAAA"], index=out.index)).astype(str).fillna("AAAAAAAA")
            alleles = out.get("hla_allele", pd.Series(["HLA-A*02:01"], index=out.index)).astype(str).fillna("HLA-A*02:01")
            with tempfile.TemporaryDirectory() as tmpdir:
                pep_file = Path(tmpdir) / "peptides.txt"
                pep_file.write_text("\n".join(peptides.tolist()), encoding="utf-8")
                allele = alleles.iloc[0]
                proc = subprocess.run(
                    ["netCTLpan", "-f", str(pep_file), "-a", str(allele)],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            floats_per_line = [[float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)] for line in lines]
            parsed = [vals for vals in floats_per_line if len(vals) >= 3]
            if len(parsed) < len(out):
                return self._run_stub(out)
            cleavage = [round(max(0.0, min(1.0, vals[-3])), 4) for vals in parsed[: len(out)]]
            tap = [round(max(0.0, min(1.0, vals[-2])), 4) for vals in parsed[: len(out)]]
            comp = [round(max(0.0, min(1.0, vals[-1])), 4) for vals in parsed[: len(out)]]
            out["cleavage_score"] = cleavage
            out["tap_score"] = tap
            out["presentation_composite"] = comp
            out["presentation_tool"] = "netctlpan"
            out["presentation_confidence"] = "HIGH"
            return out
        except Exception:
            return self._run_stub(out)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None
        if self._tool_available():
            return self._run_netctlpan(out)
        return self._run_stub(out)
