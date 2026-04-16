from __future__ import annotations

import pandas as pd

from ..base_module import BaseModule


class ForeignnessModule(BaseModule):
    NAME = "recognition"
    INPUT_COLUMNS = ["mutant_peptide", "wildtype_peptide"]
    OUTPUT_COLUMNS = [
        "self_dissimilarity",
        "mutant_wt_distance",
        "recognition_score",
        "recognition_tool",
        "recognition_confidence",
    ]

    @staticmethod
    def _safe_peptide(value: object) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _blosum62():
        from Bio.Align import substitution_matrices

        return substitution_matrices.load("BLOSUM62")

    @staticmethod
    def _pair_score(mutant: str, wildtype: str, matrix) -> float:
        if not mutant or not wildtype:
            return 0.0
        L = max(len(mutant), len(wildtype))
        gap_penalty = -4.0
        score = 0.0
        for i in range(L):
            a = mutant[i] if i < len(mutant) else "-"
            b = wildtype[i] if i < len(wildtype) else "-"
            if a == "-" or b == "-":
                score += gap_penalty
                continue
            try:
                score += float(matrix[(a, b)])
            except Exception:
                score += gap_penalty
        return score

    @staticmethod
    def _norm_similarity_to_dissimilarity(score: float, length: int) -> float:
        L = max(int(length), 1)
        min_s = -4.0 * L
        max_s = 11.0 * L
        sim = (score - min_s) / (max_s - min_s)
        sim = max(0.0, min(1.0, sim))
        return round(1.0 - sim, 6)

    @staticmethod
    def _self_dissimilarity_demo(mutant: str, wildtype: str, matrix) -> float:
        # Simplified "closest self peptide" proxy based on AA change profile.
        if not mutant or not wildtype:
            return 0.0
        L = max(len(mutant), len(wildtype))
        mismatches = 0
        non_conservative = 0
        for i in range(L):
            a = mutant[i] if i < len(mutant) else "-"
            b = wildtype[i] if i < len(wildtype) else "-"
            if a != b:
                mismatches += 1
            if a == "-" or b == "-":
                non_conservative += 1
                continue
            try:
                s = float(matrix[(a, b)])
            except Exception:
                s = -4.0
            if s < 0:
                non_conservative += 1
        mismatch_ratio = mismatches / max(L, 1)
        non_conservative_ratio = non_conservative / max(L, 1)
        return round(max(0.0, min(1.0, 0.5 * mismatch_ratio + 0.5 * non_conservative_ratio)), 6)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None

        try:
            matrix = self._blosum62()
            tool = "biopython_blosum62"
        except Exception:
            # STUB — replace when Biopython/BLOSUM62 is available.
            matrix = None
            tool = "foreignness_stub_no_biopython"

        mt_vals = []
        self_vals = []
        rec_vals = []
        conf_vals = []
        for _, row in out.iterrows():
            m = self._safe_peptide(row.get("mutant_peptide"))
            w = self._safe_peptide(row.get("wildtype_peptide"))
            if not m or not w:
                mt_vals.append(None)
                self_vals.append(None)
                rec_vals.append(None)
                conf_vals.append("UNAVAILABLE")
                continue
            if matrix is None:
                L = max(len(m), len(w), 1)
                mismatch = sum(1 for i in range(L) if (m[i] if i < len(m) else "-") != (w[i] if i < len(w) else "-")) / L
                mt_dist = round(max(0.0, min(1.0, mismatch)), 6)
                self_dist = round(max(0.0, min(1.0, mismatch * 0.9 + 0.05)), 6)
                confidence = "LOW"
            else:
                score = self._pair_score(m, w, matrix)
                mt_dist = self._norm_similarity_to_dissimilarity(score, max(len(m), len(w)))
                self_dist = self._self_dissimilarity_demo(m, w, matrix)
                confidence = "HIGH"
            rec = round(max(0.0, min(1.0, 0.4 * mt_dist + 0.6 * self_dist)), 6)
            mt_vals.append(mt_dist)
            self_vals.append(self_dist)
            rec_vals.append(rec)
            conf_vals.append(confidence)

        out["mutant_wt_distance"] = mt_vals
        out["self_dissimilarity"] = self_vals
        out["recognition_score"] = rec_vals
        out["recognition_tool"] = tool
        out["recognition_confidence"] = conf_vals
        return out
