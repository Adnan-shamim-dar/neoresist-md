from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base_normaliser import BaseNormaliser
from .base_normaliser import ValidationIssue, ValidationResult


class TSVNormaliser(BaseNormaliser):
    name = "tsv"
    REQUIRED_CANONICAL = ["chrom", "pos", "ref", "alt"]

    def validate(self, path: str | Path) -> ValidationResult:
        p = Path(path)
        issues: list[ValidationIssue] = []
        if not p.is_file():
            issues.append(ValidationIssue(step="DETECT", issue_type="missing_data", message=f"File not found: {p}"))
            return ValidationResult(False, self.name, required_columns=self.REQUIRED_CANONICAL, issues=issues)
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        try:
            df = pd.read_csv(p, sep=sep)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="format_mismatch",
                    message=f"Could not parse delimited file: {exc}",
                    expected="CSV/TSV with mutation columns mapped in UI",
                )
            )
            return ValidationResult(False, self.name, required_columns=self.REQUIRED_CANONICAL, issues=issues)
        if df.empty:
            issues.append(ValidationIssue(step="CLASSIFY", issue_type="missing_data", message="Input file has zero rows."))
        return ValidationResult(valid=True, detected_format=self.name, required_columns=self.REQUIRED_CANONICAL, issues=issues)

    def normalise(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        p = Path(path)
        mapping: dict[str, str] = kwargs.get("column_mapping") or {}
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        src = pd.read_csv(p, sep=sep)
        if src.empty:
            return self._empty_frame()

        out = pd.DataFrame(index=src.index)
        patient_id = kwargs.get("patient_id") or p.stem
        out["patient_id"] = patient_id
        out["sample_barcode"] = patient_id
        out["run_id"] = kwargs.get("run_id")
        out["timestamp"] = pd.Timestamp.now("UTC").isoformat()
        out["schema_version"] = "1.0.0"

        for canonical in ("gene", "protein_change", "chrom", "pos", "ref", "alt", "ref_count", "alt_count"):
            source_col = mapping.get(canonical)
            if source_col and source_col in src.columns:
                out[canonical] = src[source_col]
            else:
                out[canonical] = None

        out["pos"] = pd.to_numeric(out["pos"], errors="coerce")
        out["ref_count"] = pd.to_numeric(out["ref_count"], errors="coerce")
        out["alt_count"] = pd.to_numeric(out["alt_count"], errors="coerce")
        total = out["ref_count"].fillna(0) + out["alt_count"].fillna(0)
        out["vaf"] = (out["alt_count"] / total.where(total > 0)).clip(0, 1)
        out["genomic_locus"] = (
            out["chrom"].astype(str).fillna("")
            + ":"
            + out["pos"].astype("Int64").astype(str).replace("<NA>", "")
        )

        # Step 4 degrade: keep bad rows, let values be NaN/None instead of failing whole file.
        for col in self._canonical_columns():
            if col not in out.columns:
                out[col] = None
        return out[self._canonical_columns()]
