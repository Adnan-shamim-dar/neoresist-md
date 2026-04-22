from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base_normaliser import BaseNormaliser
from .base_normaliser import ValidationIssue, ValidationResult


class VCFNormaliser(BaseNormaliser):
    name = "vcf"

    def validate(self, path: str | Path) -> ValidationResult:
        p = Path(path)
        issues: list[ValidationIssue] = []
        if not p.is_file():
            issues.append(ValidationIssue(step="DETECT", issue_type="missing_data", message=f"File not found: {p}"))
            return ValidationResult(False, self.name, issues=issues)
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="format_mismatch",
                    message=f"Could not read VCF file: {exc}",
                )
            )
            return ValidationResult(False, self.name, issues=issues)
        if not any(line.startswith("#CHROM") for line in lines):
            issues.append(
                ValidationIssue(
                    step="DETECT",
                    issue_type="format_mismatch",
                    message="Missing #CHROM header line.",
                    expected="Valid VCF with #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT SAMPLE",
                )
            )
        return ValidationResult(valid=len(issues) == 0, detected_format=self.name, issues=issues)

    @staticmethod
    def _parse_ad(format_keys: list[str], sample_values: list[str], alt_index: int) -> tuple[float | None, float | None]:
        if "AD" not in format_keys:
            return None, None
        ad_i = format_keys.index("AD")
        if ad_i >= len(sample_values):
            return None, None
        ad_val = sample_values[ad_i]
        parts = ad_val.split(",")
        if len(parts) <= alt_index + 1:
            return None, None
        try:
            ref_count = float(parts[0]) if parts[0] != "." else None
            alt_count = float(parts[alt_index + 1]) if parts[alt_index + 1] != "." else None
            return ref_count, alt_count
        except Exception:
            return None, None

    @staticmethod
    def _parse_dp(format_keys: list[str], sample_values: list[str]) -> float | None:
        if "DP" not in format_keys:
            return None
        dp_i = format_keys.index("DP")
        if dp_i >= len(sample_values):
            return None
        try:
            return float(sample_values[dp_i]) if sample_values[dp_i] != "." else None
        except Exception:
            return None

    def normalise(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        p = Path(path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        header = None
        records: list[dict[str, Any]] = []
        patient_id = kwargs.get("patient_id") or p.stem
        for raw in lines:
            if not raw:
                continue
            if raw.startswith("##"):
                continue
            if raw.startswith("#CHROM"):
                header = raw.lstrip("#").split("\t")
                continue
            if raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _vid, ref, alt_field, _qual, _filt, _info = parts[:8]
            fmt = parts[8] if len(parts) > 8 else ""
            sample = parts[9] if len(parts) > 9 else ""
            format_keys = fmt.split(":") if fmt else []
            sample_values = sample.split(":") if sample else []
            alts = [a.strip() for a in alt_field.split(",") if a.strip()]
            for alt_i, alt in enumerate(alts):
                ref_count, alt_count = self._parse_ad(format_keys, sample_values, alt_i)
                dp = self._parse_dp(format_keys, sample_values)
                if ref_count is None and alt_count is None and dp is not None:
                    ref_count = None
                    alt_count = None
                total = None
                if ref_count is not None or alt_count is not None:
                    total = (ref_count or 0.0) + (alt_count or 0.0)
                elif dp is not None:
                    total = dp
                vaf = None
                if total and alt_count is not None and total > 0:
                    vaf = max(0.0, min(1.0, float(alt_count) / float(total)))
                rec = {
                    "patient_id": patient_id,
                    "sample_barcode": patient_id,
                    "run_id": kwargs.get("run_id"),
                    "timestamp": pd.Timestamp.now("UTC").isoformat(),
                    "schema_version": "1.0.0",
                    "gene": kwargs.get("default_gene"),
                    "protein_change": kwargs.get("default_protein_change"),
                    "genomic_locus": f"{chrom}:{pos}",
                    "chrom": chrom,
                    "pos": pd.to_numeric(pos, errors="coerce"),
                    "ref": ref,
                    "alt": alt,
                    "ref_count": ref_count,
                    "alt_count": alt_count,
                    "vaf": vaf,
                }
                records.append(rec)

        out = pd.DataFrame(records)
        if out.empty:
            return self._empty_frame()
        vaf = pd.to_numeric(out.get("vaf"), errors="coerce").dropna()
        suspicious_ratio = float(((vaf >= 0.45) & (vaf <= 0.55)).mean()) if not vaf.empty else 0.0
        out.attrs["germline_contamination_suspected"] = suspicious_ratio > 0.15
        for col in self._canonical_columns():
            if col not in out.columns:
                out[col] = None
        return out[self._canonical_columns()]
