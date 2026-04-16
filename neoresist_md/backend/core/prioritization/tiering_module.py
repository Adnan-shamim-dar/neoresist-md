from __future__ import annotations

import pandas as pd

from ..base_module import BaseModule

class TieringModule(BaseModule):
    NAME = "tiering"
    INPUT_COLUMNS = ["composite_priority", "clonality_class", "expression_flag", "hla_loh_status", "binding_rank"]
    OUTPUT_COLUMNS = ["exclusion_flags", "exclusion_reasons", "tier", "composite_priority", "report_summary"]

    @staticmethod
    def _to_num(value, fallback=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(fallback)

    @staticmethod
    def _clip01(value, fallback=0.0) -> float:
        v = TieringModule._to_num(value, fallback=fallback)
        return max(0.0, min(1.0, v))

    def _compute_fallback_priority(self, row: pd.Series) -> float:
        # Fallback only if composite_priority is missing.
        bind_rank = self._to_num(row.get("binding_rank"), 100.0)
        bind_inv = max(0.0, min(1.0, 1.0 - (bind_rank / 100.0)))
        presentation = self._clip01(row.get("presentation_composite"), 0.3)
        recognition = self._clip01(row.get("recognition_score"), 0.3)
        expr_tpm = self._to_num(row.get("expression_tpm"), 0.0)
        expr_norm = max(0.0, min(1.0, expr_tpm / (expr_tpm + 1.0)))
        resistance = self._clip01(row.get("resistance_composite"), 0.5)
        immunogenicity = 0.35 * bind_inv + 0.25 * presentation + 0.25 * recognition + 0.15 * expr_norm
        return round(max(0.0, min(1.0, immunogenicity * (1.0 - resistance))), 6)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None

        exclusion_flags = []
        exclusion_reasons = []
        tiers = []
        priorities = []
        summaries = []

        for _, row in out.iterrows():
            expr_flag = str(row.get("expression_flag") or "UNKNOWN").upper()
            loh_status = str(row.get("hla_loh_status") or "UNKNOWN").upper()
            loh_penalty = self._to_num(row.get("resistance_loh_penalty"), 0.0)
            binding_rank = self._to_num(row.get("binding_rank"), 999.0)
            clonality = str(row.get("clonality_class") or "UNKNOWN").upper()

            cp_raw = row.get("composite_priority")
            cp = self._clip01(cp_raw, fallback=self._compute_fallback_priority(row))

            flags = []
            reasons = []

            # Hard exclusions
            if expr_flag == "ABSENT":
                flags.append("HARD_EXPRESSION_ABSENT")
                reasons.append("Expression absent")
            if loh_status == "LOH_DETECTED" and loh_penalty > 0.8:
                flags.append("HARD_LOH_DETECTED")
                reasons.append("HLA LOH detected with high LOH penalty")
            if binding_rank > 2.0:
                flags.append("HARD_WEAK_BINDER")
                reasons.append("Binding rank > 2.0 (weak binder)")

            if flags:
                tier = "EXCLUDED"
            else:
                if cp >= 0.7 and clonality == "CLONAL":
                    tier = "TIER_1"
                elif cp >= 0.4 or clonality == "CLONAL":
                    tier = "TIER_2"
                else:
                    tier = "TIER_3"

            if not reasons:
                reasons.append("Passed hard exclusions")

            exclusion_flags.append(",".join(flags))
            exclusion_reasons.append(",".join(reasons))
            tiers.append(tier)
            priorities.append(cp)
            summaries.append(
                f"Tier={tier}; priority={cp:.3f}; clonality={clonality}; expression={expr_flag}; "
                f"binding_rank={binding_rank:.3f}; loh_status={loh_status}"
            )

        out["exclusion_flags"] = exclusion_flags
        out["exclusion_reasons"] = exclusion_reasons
        out["tier"] = tiers
        out["composite_priority"] = priorities
        out["report_summary"] = summaries
        return out
