from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ValidationIssue:
    step: str
    issue_type: str
    message: str
    expected: str | None = None
    row_index: int | None = None
    column: str | None = None


@dataclass
class ValidationResult:
    valid: bool
    detected_format: str
    required_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "detected_format": self.detected_format,
            "required_columns": self.required_columns,
            "missing_columns": self.missing_columns,
            "issues": [
                {
                    "step": item.step,
                    "issue_type": item.issue_type,
                    "message": item.message,
                    "expected": item.expected,
                    "row_index": item.row_index,
                    "column": item.column,
                }
                for item in self.issues
            ],
        }


class BaseNormaliser(ABC):
    name: str = "base_normaliser"

    @abstractmethod
    def validate(self, path: str | Path) -> ValidationResult:
        raise NotImplementedError

    @abstractmethod
    def normalise(self, path: str | Path, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _canonical_columns() -> list[str]:
        return [
            "patient_id",
            "sample_barcode",
            "run_id",
            "timestamp",
            "schema_version",
            "gene",
            "protein_change",
            "genomic_locus",
            "chrom",
            "pos",
            "ref",
            "alt",
            "ref_count",
            "alt_count",
            "vaf",
        ]

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=BaseNormaliser._canonical_columns())
