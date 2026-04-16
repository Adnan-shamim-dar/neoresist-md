from __future__ import annotations

import subprocess

import pandas as pd

from ..base_module import BaseModule


class ClonalityModule(BaseModule):
    NAME = "clonality"
    INPUT_COLUMNS = ["vaf"]
    OUTPUT_COLUMNS = [
        "tumour_purity",
        "local_copy_number",
        "ccf",
        "ccf_cluster",
        "clonality_class",
        "clonality_confidence",
        "clonality_tool",
    ]

    @staticmethod
    def _pyclone_available() -> bool:
        try:
            proc = subprocess.run(["pyclone-vi"], capture_output=True, text=True, timeout=3)
            return proc.returncode in {0, 1, 2}
        except Exception:
            return False

    @staticmethod
    def _to_num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _run_stub_vaf_proxy(self, out: pd.DataFrame, reason: str) -> pd.DataFrame:
        # STUB — replace when PyClone-VI and full copy-number context are available.
        vaf = self._to_num(out.get("vaf", pd.Series([None] * len(out), index=out.index)))
        out["ccf"] = vaf.where(vaf.notna(), 0.5)
        out["clonality_class"] = out["ccf"].apply(lambda x: "CLONAL" if float(x) >= 0.4 else "SUBCLONAL")
        out["ccf_cluster"] = out["clonality_class"].map({"CLONAL": "cluster_1", "SUBCLONAL": "cluster_2"}).fillna("cluster_2")
        out["tumour_purity"] = out.get("tumour_purity", pd.Series([None] * len(out), index=out.index))
        out["tumour_purity"] = self._to_num(out["tumour_purity"]).where(self._to_num(out["tumour_purity"]).notna(), 0.7)
        out["local_copy_number"] = out.get("local_copy_number", pd.Series([None] * len(out), index=out.index))
        out["local_copy_number"] = self._to_num(out["local_copy_number"]).where(self._to_num(out["local_copy_number"]).notna(), 2.0)
        out["clonality_confidence"] = "LOW"
        out["clonality_tool"] = f"pyclone_vi_stub_vaf_proxy ({reason})"
        return out

    def _run_pyclone_proxy(self, out: pd.DataFrame) -> pd.DataFrame:
        # Best-effort local proxy when tool is present; replace with real PyClone-VI pipeline output parsing.
        vaf = self._to_num(out.get("vaf", pd.Series([None] * len(out), index=out.index))).fillna(0.0)
        cn = self._to_num(out.get("local_copy_number", pd.Series([2.0] * len(out), index=out.index))).fillna(2.0)
        purity = self._to_num(out.get("tumour_purity", pd.Series([0.7] * len(out), index=out.index))).fillna(0.7)
        ccf = ((2.0 * vaf) / (cn * purity.clip(lower=0.2))).clip(lower=0.0, upper=1.0)
        out["ccf"] = ccf
        out["clonality_class"] = ccf.apply(lambda x: "CLONAL" if float(x) >= 0.5 else "SUBCLONAL")
        out["ccf_cluster"] = out["clonality_class"].map({"CLONAL": "cluster_1", "SUBCLONAL": "cluster_2"}).fillna("cluster_2")
        out["tumour_purity"] = purity
        out["local_copy_number"] = cn
        out["clonality_confidence"] = "MEDIUM"
        out["clonality_tool"] = "pyclone-vi"
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None

        has_cn = "local_copy_number" in df.columns and self._to_num(df["local_copy_number"]).notna().any()
        if not self._pyclone_available() or not has_cn:
            reason = "pyclone-vi missing" if not self._pyclone_available() else "copy-number missing"
            return self._run_stub_vaf_proxy(out, reason)
        return self._run_pyclone_proxy(out)


PyCloneModule = ClonalityModule
