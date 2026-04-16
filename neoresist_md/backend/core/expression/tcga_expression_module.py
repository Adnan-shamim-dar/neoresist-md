from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..base_module import BaseModule


class ExpressionModule(BaseModule):
    NAME = "expression"
    INPUT_COLUMNS = ["gene"]
    OUTPUT_COLUMNS = ["expression_tpm", "expression_flag", "expression_confidence", "expression_tool"]

    @staticmethod
    def _candidate_reference_dirs() -> list[Path]:
        here = Path(__file__).resolve()
        project_root = here.parents[4]
        repo_root = project_root.parent
        return [
            project_root / "data" / "cohort" / "tcga_sarc",
            repo_root / "data" / "cohort" / "tcga_sarc",
        ]

    @staticmethod
    def _find_expression_file() -> Path | None:
        exts = ("*.csv", "*.tsv", "*.txt", "*.parquet")
        for base in ExpressionModule._candidate_reference_dirs():
            if not base.exists():
                continue
            for ext in exts:
                files = sorted(base.glob(ext))
                if files:
                    return files[0]
        return None

    @staticmethod
    def _load_reference(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep)

    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    @staticmethod
    def _flag_from_tpm(v: float | None) -> str:
        if v is None or pd.isna(v):
            return "UNKNOWN"
        if float(v) >= 1.0:
            return "EXPRESSED"
        if float(v) >= 0.1:
            return "LOW"
        return "ABSENT"

    @staticmethod
    def _seed_from_row(row: pd.Series) -> int:
        key = f"{row.get('patient_id', '')}|{row.get('gene', '')}|{row.get('mutant_peptide', '')}"
        return abs(hash(key)) % (2**31 - 1)

    def _run_stub(self, out: pd.DataFrame) -> pd.DataFrame:
        # STUB — replace when TCGA expression reference is available.
        tpm = []
        flag = []
        for _, row in out.iterrows():
            rng = random.Random(self._seed_from_row(row))
            p = rng.random()
            if p < 0.70:
                val = round(rng.uniform(1.0, 25.0), 4)
                lab = "EXPRESSED"
            elif p < 0.90:
                val = round(rng.uniform(0.1, 0.999), 4)
                lab = "LOW"
            else:
                val = None
                lab = "UNKNOWN"
            tpm.append(val)
            flag.append(lab)
        out["expression_tpm"] = tpm
        out["expression_flag"] = flag
        out["expression_confidence"] = "LOW"
        out["expression_tool"] = "tcga_expression_stub"
        return out

    def _run_join(self, out: pd.DataFrame, ref_path: Path) -> pd.DataFrame:
        ref = self._load_reference(ref_path)
        gene_col = self._pick_col(ref, ["gene", "gene_symbol", "hugo_symbol", "gene_name"])
        tpm_col = self._pick_col(ref, ["tpm", "expression_tpm", "fpkm", "median_tpm"])
        if gene_col is None or tpm_col is None:
            return self._run_stub(out)
        slim = ref[[gene_col, tpm_col]].copy()
        slim.columns = ["gene_ref", "tpm_ref"]
        slim["gene_ref"] = slim["gene_ref"].astype(str)
        slim["tpm_ref"] = pd.to_numeric(slim["tpm_ref"], errors="coerce")
        slim = slim.groupby("gene_ref", dropna=False)["tpm_ref"].mean().reset_index()

        merged = out.copy()
        merged["gene_key"] = merged["gene"].astype(str)
        merged = merged.merge(slim, how="left", left_on="gene_key", right_on="gene_ref")
        merged["expression_tpm"] = merged["expression_tpm"].where(merged["expression_tpm"].notna(), merged["tpm_ref"])
        merged["expression_flag"] = merged["expression_tpm"].apply(self._flag_from_tpm)
        merged["expression_confidence"] = "HIGH"
        merged["expression_tool"] = f"tcga_expression:{ref_path.name}"
        merged = merged.drop(columns=[c for c in ["gene_key", "gene_ref", "tpm_ref"] if c in merged.columns])
        return merged

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None
        ref_path = self._find_expression_file()
        if ref_path is None:
            return self._run_stub(out)
        return self._run_join(out, ref_path)


TCGAExpressionModule = ExpressionModule
