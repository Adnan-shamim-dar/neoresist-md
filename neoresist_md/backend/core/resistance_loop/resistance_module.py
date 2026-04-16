from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ..base_module import BaseModule


class ResistanceLoopModule(BaseModule):
    NAME = "resistance"
    INPUT_COLUMNS = ["expression_tpm", "ccf", "recognition_score"]
    OUTPUT_COLUMNS = [
        "resistance_loh_penalty",
        "resistance_volatility_flag",
        "resistance_expression_instability",
        "resistance_processing_disruption",
        "resistance_composite",
        "resistance_confidence",
        "composite_priority",
    ]

    @staticmethod
    def _confidence_rank(value: object) -> int:
        mapping = {"UNAVAILABLE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        return mapping.get(str(value or "UNAVAILABLE").upper(), 0)

    @staticmethod
    def _confidence_from_rank(rank: int) -> str:
        rev = {0: "UNAVAILABLE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        return rev.get(int(rank), "UNAVAILABLE")

    @staticmethod
    def _clip01(value: object, fallback: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return fallback

    @staticmethod
    def _to_num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _load_weights(self) -> dict:
        cfg_path = Path(__file__).resolve().parents[3] / "config" / "module_weights.yaml"
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            return {}
        if "immunogenicity" in data or "resistance" in data:
            out = {}
            imm = data.get("immunogenicity") or {}
            res = data.get("resistance") or {}
            out["immunogenicity_binding_rank"] = float(imm.get("binding", 0.35))
            out["immunogenicity_presentation"] = float(imm.get("presentation", 0.25))
            out["immunogenicity_expression"] = float(imm.get("expression", 0.20))
            out["immunogenicity_recognition"] = float(imm.get("recognition", 0.25))
            out["resistance_loh"] = float(res.get("resistance_loh", 0.35))
            out["resistance_volatility"] = float(res.get("resistance_volatility", 0.25))
            out["resistance_expression"] = float(res.get("resistance_expression", 0.25))
            out["resistance_processing"] = float(res.get("resistance_processing", 0.15))
            return out
        return data.get("weights", data)

    @staticmethod
    def _loh_penalty(status: object) -> float:
        s = str(status or "UNKNOWN").upper()
        if s == "LOH_DETECTED":
            return 1.0
        if s == "LOH_SUSPECTED":
            return 0.5
        if s == "INTACT":
            return 0.0
        return 0.3

    @staticmethod
    def _volatility_score(clonality: object) -> float:
        c = str(clonality or "UNKNOWN").upper()
        if c == "SUBCLONAL":
            return 0.8
        if c == "CLONAL":
            return 0.2
        return 0.4

    @staticmethod
    def _expr_instability(flag: object) -> float:
        f = str(flag or "UNKNOWN").upper()
        if f == "ABSENT":
            return 1.0
        if f == "LOW":
            return 0.5
        if f == "EXPRESSED":
            return 0.1
        return 0.3

    @staticmethod
    def _processing_disruption(v: object) -> float:
        if v is True:
            return 1.0
        if v is False:
            return 0.0
        return 0.2

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None
        w = self._load_weights()
        w_res_loh = float(w.get("resistance_loh", 0.35))
        w_res_vol = float(w.get("resistance_volatility", 0.25))
        w_res_expr = float(w.get("resistance_expression", 0.25))
        w_res_proc = float(w.get("resistance_processing", 0.15))
        s_res = max(w_res_loh + w_res_vol + w_res_expr + w_res_proc, 1e-9)
        w_res_loh, w_res_vol, w_res_expr, w_res_proc = (
            w_res_loh / s_res,
            w_res_vol / s_res,
            w_res_expr / s_res,
            w_res_proc / s_res,
        )

        w_bind = float(w.get("immunogenicity_binding_rank", 0.35))
        w_pres = float(w.get("immunogenicity_presentation", 0.25))
        w_rec = float(w.get("immunogenicity_recognition", 0.25))
        w_expr = float(w.get("immunogenicity_expression", 0.15))
        s_imm = max(w_bind + w_pres + w_rec + w_expr, 1e-9)
        w_bind, w_pres, w_rec, w_expr = w_bind / s_imm, w_pres / s_imm, w_rec / s_imm, w_expr / s_imm

        binding_rank = self._to_num(out.get("binding_rank", pd.Series([None] * len(out), index=out.index)))
        binding_inverted = (1.0 - (binding_rank.fillna(50.0) / 100.0)).clip(lower=0.0, upper=1.0)
        presentation = self._to_num(out.get("presentation_composite", pd.Series([None] * len(out), index=out.index))).fillna(0.3).clip(0, 1)
        recognition = self._to_num(out.get("recognition_score", pd.Series([None] * len(out), index=out.index))).fillna(0.3).clip(0, 1)
        expr_tpm = self._to_num(out.get("expression_tpm", pd.Series([None] * len(out), index=out.index))).fillna(0.0)
        expr_norm = (expr_tpm / (expr_tpm + 1.0)).clip(lower=0.0, upper=1.0)
        immunogenicity = (w_bind * binding_inverted + w_pres * presentation + w_rec * recognition + w_expr * expr_norm).clip(0, 1)

        loh = out.get("hla_loh_status", pd.Series(["UNKNOWN"] * len(out), index=out.index)).apply(self._loh_penalty)
        vol = out.get("clonality_class", pd.Series(["UNKNOWN"] * len(out), index=out.index)).apply(self._volatility_score)
        expr_inst = out.get("expression_flag", pd.Series(["UNKNOWN"] * len(out), index=out.index)).apply(self._expr_instability)
        proc = out.get("processing_disruption_flag", pd.Series([None] * len(out), index=out.index)).apply(self._processing_disruption)

        resistance = (
            w_res_loh * loh.astype(float)
            + w_res_vol * vol.astype(float)
            + w_res_expr * expr_inst.astype(float)
            + w_res_proc * proc.astype(float)
        ).clip(0, 1)

        out["resistance_loh_penalty"] = loh.astype(float).round(6)
        out["resistance_volatility_flag"] = vol.astype(float).round(6)
        out["resistance_expression_instability"] = expr_inst.astype(float).round(6)
        out["resistance_processing_disruption"] = proc.astype(float).round(6)
        out["resistance_composite"] = resistance.astype(float).round(6)
        out["composite_priority"] = (immunogenicity * (1.0 - resistance)).clip(0, 1).round(6)

        input_conf_cols = [
            "escape_confidence",
            "clonality_confidence",
            "expression_confidence",
            "recognition_confidence",
            "presentation_confidence",
            "binding_confidence",
        ]
        conf_vals = []
        for _, row in out.iterrows():
            ranks = [self._confidence_rank(row.get(col)) for col in input_conf_cols if col in out.columns]
            conf_vals.append(self._confidence_from_rank(min(ranks) if ranks else 1))
        out["resistance_confidence"] = conf_vals
        return out


ResistanceModule = ResistanceLoopModule
