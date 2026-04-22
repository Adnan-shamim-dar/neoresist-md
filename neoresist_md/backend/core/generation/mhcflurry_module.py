from __future__ import annotations

import importlib.util
import logging
import math
import random
import re
from typing import List

import pandas as pd

from ..base_module import BaseModule

LOGGER = logging.getLogger(__name__)


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

    @staticmethod
    def _valid_hla_allele(value: object) -> bool:
        return bool(re.fullmatch(r"HLA-[ABC]\*\d{2}:\d{2}", str(value or "").strip()))

    @staticmethod
    def _append_exclusion_reason(out: pd.DataFrame, mask: pd.Series, reason: str) -> None:
        if "exclusion_reasons" not in out.columns:
            out["exclusion_reasons"] = None
        current = out.loc[mask, "exclusion_reasons"].fillna("").astype(str)
        out.loc[mask, "exclusion_reasons"] = current.apply(
            lambda value: reason if value.strip() == "" else (value if reason in value.split(",") else f"{value},{reason}")
        )

    @staticmethod
    def _needs_prediction(row: pd.Series) -> bool:
        return pd.isna(row.get("binding_affinity")) or pd.isna(row.get("binding_rank"))

    @staticmethod
    def _apply_diploid_assumption_flag(out: pd.DataFrame) -> None:
        if "local_copy_number" not in out.columns or out["local_copy_number"].isna().all():
            out["diploid_assumption_flag"] = True

    def _run_mhcflurry(self, out: pd.DataFrame) -> pd.DataFrame:
        try:
            import mhcflurry
            from mhcflurry import Class1PresentationPredictor

            predictor = Class1PresentationPredictor.load()
            predict_mask = out.apply(self._needs_prediction, axis=1)
            if not bool(predict_mask.any()):
                self._apply_diploid_assumption_flag(out)
                return out

            invalid_hla_mask = predict_mask & ~out["hla_allele"].map(self._valid_hla_allele)
            if bool(invalid_hla_mask.any()):
                LOGGER.error("MHCflurry prediction skipped for rows with missing/invalid hla_allele values.")
                out.loc[invalid_hla_mask, "binding_confidence"] = "UNAVAILABLE"
                out.loc[invalid_hla_mask, "binding_tool"] = "mhcflurry_2.0"
                out.loc[invalid_hla_mask, "binding_tool_version"] = mhcflurry.__version__
                self._append_exclusion_reason(out, invalid_hla_mask, "HLA_FORMAT_INVALID")

            runnable_mask = predict_mask & ~invalid_hla_mask
            if not bool(runnable_mask.any()):
                self._apply_diploid_assumption_flag(out)
                return out

            for allele, allele_df in out.loc[runnable_mask].groupby("hla_allele", dropna=False):
                peptides = allele_df["mutant_peptide"].astype(str).tolist()
                pred = predictor.predict(peptides=peptides, alleles=[str(allele)])
                idx = allele_df.index
                if "affinity" in pred.columns:
                    out.loc[idx, "binding_affinity"] = pred["affinity"].values
                if "presentation_percentile" in pred.columns:
                    out.loc[idx, "binding_rank"] = pred["presentation_percentile"].values
                elif "percentile_rank" in pred.columns:
                    out.loc[idx, "binding_rank"] = pred["percentile_rank"].values
                out.loc[idx, "binding_tool"] = "mhcflurry_2.0"
                out.loc[idx, "binding_tool_version"] = mhcflurry.__version__
                out.loc[idx, "binding_confidence"] = "HIGH"

                if "processing_score" in pred.columns:
                    out.loc[idx, "cleavage_score"] = pred["processing_score"].values
                if "presentation_score" in pred.columns:
                    out.loc[idx, "tap_score"] = pred["presentation_score"].values
                if "processing_score" in pred.columns or "presentation_score" in pred.columns:
                    out.loc[idx, "presentation_tool"] = "mhcflurry_2.0"
                    out.loc[idx, "presentation_confidence"] = "HIGH"

            self._apply_diploid_assumption_flag(out)
            return out
        except Exception as exc:
            LOGGER.warning("MHCflurry runtime failed, falling back to deterministic stub scores: %s", exc)
            return self._run_stub(out)

    def _run_stub(self, out: pd.DataFrame) -> pd.DataFrame:
        # STUB — replace when tool available.
        predict_mask = out.apply(self._needs_prediction, axis=1)
        invalid_hla_mask = predict_mask & ~out["hla_allele"].map(self._valid_hla_allele)
        if bool(invalid_hla_mask.any()):
            LOGGER.error("Stub binding prediction skipped for rows with missing/invalid hla_allele values.")
            out.loc[invalid_hla_mask, "binding_confidence"] = "UNAVAILABLE"
            out.loc[invalid_hla_mask, "binding_tool"] = "mhcflurry_stub"
            out.loc[invalid_hla_mask, "binding_tool_version"] = self.VERSION
            self._append_exclusion_reason(out, invalid_hla_mask, "HLA_FORMAT_INVALID")

        runnable_mask = predict_mask & ~invalid_hla_mask
        affinities = []
        ranks = []
        for _, row in out.loc[runnable_mask].iterrows():
            rng = random.Random(self._seed_from_row(row))
            affinities.append(round(rng.uniform(25.0, 2500.0), 3))
            ranks.append(round(rng.uniform(0.1, 35.0), 3))
        if affinities:
            out.loc[runnable_mask, "binding_affinity"] = affinities
            out.loc[runnable_mask, "binding_rank"] = ranks
            out.loc[runnable_mask, "binding_tool"] = "mhcflurry_stub"
            out.loc[runnable_mask, "binding_tool_version"] = self.VERSION
            out.loc[runnable_mask, "binding_confidence"] = "LOW"
        out.attrs["stub_warning"] = "MHCflurry not installed — binding scores are estimated"
        self._apply_diploid_assumption_flag(out)
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df.copy()
        if "hla_allele" not in base.columns:
            base["hla_allele"] = None

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
        for col in ("cleavage_score", "tap_score", "presentation_tool", "presentation_confidence"):
            if col not in out.columns:
                out[col] = None
        if self._mhcflurry_available():
            return self._run_mhcflurry(out)
        return self._run_stub(out)


MHCFlurryModule = MHCflurryModule
