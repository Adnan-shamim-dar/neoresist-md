from __future__ import annotations

import importlib.util
import math
import random
from typing import List

import pandas as pd

from ..base_module import BaseModule


class MHCflurryModule(BaseModule):
    NAME = "generation"
    INPUT_COLUMNS = ["gene"]
    OUTPUT_COLUMNS = [
        "mutant_peptide",
        "peptide_length",
        "binding_affinity",
        "binding_rank",
        "binding_tool",
        "binding_tool_version",
        "hla_allele",
        "binding_confidence",
    ]

    @staticmethod
    def _seed_from_row(row: pd.Series) -> int:
        key = f"{row.get('patient_id', '')}|{row.get('gene', '')}|{row.get('protein_change', '')}|{row.get('mutant_peptide', '')}"
        return abs(hash(key)) % (2**31 - 1)

    @staticmethod
    def _mock_peptide(seed: int, length: int = 15) -> str:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(length))

    @staticmethod
    def _generate_windows(peptide: str) -> List[str]:
        pep = (peptide or "").strip().upper()
        if not pep:
            return []
        windows: list[str] = []
        for k in (8, 9, 10, 11):
            if len(pep) >= k:
                for i in range(0, len(pep) - k + 1):
                    windows.append(pep[i : i + k])
            else:
                windows.append((pep * math.ceil(k / max(len(pep), 1)))[:k])
        unique = list(dict.fromkeys(windows))
        return unique or [pep]

    @staticmethod
    def _mhcflurry_available() -> bool:
        return importlib.util.find_spec("mhcflurry") is not None

    def _run_mhcflurry(self, out: pd.DataFrame) -> pd.DataFrame:
        # Best-effort real path. Falls back to stub if runtime/model load fails.
        # STUB — replace when tool available.
        try:
            from mhcflurry import Class1PresentationPredictor

            predictor = Class1PresentationPredictor.load()
            peptide_series = out["mutant_peptide"].astype(str).tolist()
            allele_series = out["hla_allele"].astype(str).tolist()
            pred = predictor.predict(peptides=peptide_series, alleles=allele_series)
            if "affinity" in pred.columns:
                out["binding_affinity"] = pred["affinity"].values
            if "presentation_percentile" in pred.columns:
                out["binding_rank"] = pred["presentation_percentile"].values
            elif "percentile_rank" in pred.columns:
                out["binding_rank"] = pred["percentile_rank"].values
            out["binding_tool"] = "mhcflurry"
            out["binding_tool_version"] = self.VERSION
            out["binding_confidence"] = "HIGH"
            return out
        except Exception:
            return self._run_stub(out)

    def _run_stub(self, out: pd.DataFrame) -> pd.DataFrame:
        # STUB — replace when tool available.
        affinities = []
        ranks = []
        for _, row in out.iterrows():
            rng = random.Random(self._seed_from_row(row))
            affinities.append(round(rng.uniform(25.0, 2500.0), 3))
            ranks.append(round(rng.uniform(0.1, 35.0), 3))
        out["binding_affinity"] = affinities
        out["binding_rank"] = ranks
        out["binding_tool"] = "mhcflurry_stub"
        out["binding_tool_version"] = self.VERSION
        out["binding_confidence"] = "LOW"
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        if "hla_allele" not in base.columns:
            base["hla_allele"] = None
        base["hla_allele"] = base["hla_allele"].where(base["hla_allele"].notna(), "HLA-A*02:01")

        if "mutant_peptide" not in base.columns:
            base["mutant_peptide"] = None

        expanded_rows: list[dict] = []
        for _, row in base.iterrows():
            seed = self._seed_from_row(row)
            peptide = row.get("mutant_peptide")
            if peptide is None or str(peptide).strip() == "":
                peptide = self._mock_peptide(seed, length=15)
            windows = self._generate_windows(str(peptide))
            for w in windows:
                rec = row.to_dict()
                rec["mutant_peptide"] = w
                rec["peptide_length"] = len(w)
                expanded_rows.append(rec)
        out = pd.DataFrame(expanded_rows) if expanded_rows else base
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None
        if self._mhcflurry_available():
            return self._run_mhcflurry(out)
        return self._run_stub(out)


MHCFlurryModule = MHCflurryModule
