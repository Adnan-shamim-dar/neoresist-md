from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base_normaliser import BaseNormaliser
from .base_normaliser import ValidationIssue, ValidationResult


class MAFNormaliser(BaseNormaliser):
    name = "maf"
    REQUIRED = [
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]

    def validate(self, path: str | Path) -> ValidationResult:
        p = Path(path)
        issues: list[ValidationIssue] = []
        if not p.is_file():
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="missing_data",
                    message=f"File not found: {p}",
                )
            )
            return ValidationResult(False, self.name, required_columns=self.REQUIRED, issues=issues)
        try:
            df = pd.read_csv(p, sep="\t", comment="#")
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="format_mismatch",
                    message=f"Could not parse MAF/TSV: {exc}",
                    expected="Tab-separated MAF with standard TCGA headers",
                )
            )
            return ValidationResult(False, self.name, required_columns=self.REQUIRED, issues=issues)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="missing_data",
                    message="Required MAF columns missing.",
                    expected=", ".join(self.REQUIRED),
                )
            )
        # Step 2 classify types/quality
        for col in ("Start_Position", "t_ref_count", "t_alt_count"):
            if col in df.columns:
                bad = pd.to_numeric(df[col], errors="coerce").isna()
                for idx in df.index[bad].tolist()[:20]:
                    issues.append(
                        ValidationIssue(
                            step="CLASSIFY",
                            issue_type="quality_problem",
                            message=f"Non-numeric value in {col}.",
                            expected="numeric",
                            row_index=int(idx),
                            column=col,
                        )
                    )
        return ValidationResult(len(missing) == 0, self.name, required_columns=self.REQUIRED, missing_columns=missing, issues=issues)

    def normalise(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        p = Path(path)
        df = pd.read_csv(p, sep="\t", comment="#")
        out = self._empty_frame()
        if df.empty:
            return out

        out = pd.DataFrame(index=df.index)
        out["patient_id"] = (
            df["Tumor_Sample_Barcode"].astype(str)
            if "Tumor_Sample_Barcode" in df.columns
            else p.stem
        )
        out["sample_barcode"] = out["patient_id"]
        out["run_id"] = kwargs.get("run_id")
        out["timestamp"] = pd.Timestamp.now("UTC").isoformat()
        out["schema_version"] = "1.0.0"
        out["gene"] = df.get("Hugo_Symbol")
        out["protein_change"] = df.get("HGVSp_Short")
        out["chrom"] = df.get("Chromosome")
        out["pos"] = pd.to_numeric(df.get("Start_Position"), errors="coerce")
        out["ref"] = df.get("Reference_Allele")
        out["alt"] = df.get("Tumor_Seq_Allele2")
        out["genomic_locus"] = (
            out["chrom"].astype(str).fillna("")
            + ":"
            + out["pos"].astype("Int64").astype(str).replace("<NA>", "")
        )
        out["ref_count"] = pd.to_numeric(df.get("t_ref_count"), errors="coerce")
        out["alt_count"] = pd.to_numeric(df.get("t_alt_count"), errors="coerce")
        total = out["ref_count"].fillna(0) + out["alt_count"].fillna(0)
        out["vaf"] = (out["alt_count"] / total.where(total > 0)).clip(0, 1)

        # Step 4 degrade: keep malformed rows, flag missing as NaN instead of blocking.
        for col in self._canonical_columns():
            if col not in out.columns:
                out[col] = None
        return out[self._canonical_columns()]
