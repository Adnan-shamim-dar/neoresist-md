from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from neoresist.paths import config_dir


@dataclass(frozen=True)
class CanonicalSchema:
    schema_version: str
    column_groups: dict[str, list[str]]
    defaults: dict[str, Any]

    @property
    def ordered_columns(self) -> list[str]:
        columns: list[str] = []
        for group in self.column_groups.values():
            for column in group:
                if column not in columns:
                    columns.append(column)
        return columns


@dataclass(frozen=True)
class CanonicalValidationResult:
    schema_version: str
    required_columns: list[str]
    missing_columns: list[str]

    @property
    def valid(self) -> bool:
        return not self.missing_columns


def canonical_schema_path() -> Path:
    return config_dir() / "canonical_schema.yaml"


def load_canonical_schema() -> CanonicalSchema:
    raw = yaml.safe_load(canonical_schema_path().read_text(encoding="utf-8")) or {}
    groups = {
        str(group): [str(column) for column in (columns or [])]
        for group, columns in (raw.get("column_groups") or {}).items()
    }
    defaults = {str(key): value for key, value in (raw.get("defaults") or {}).items()}
    return CanonicalSchema(
        schema_version=str(raw.get("schema_version") or "1.0.0"),
        column_groups=groups,
        defaults=defaults,
    )


def validate_canonical_df(df: pd.DataFrame) -> CanonicalValidationResult:
    schema = load_canonical_schema()
    required = schema.ordered_columns
    missing = [column for column in required if column not in df.columns]
    return CanonicalValidationResult(
        schema_version=schema.schema_version,
        required_columns=required,
        missing_columns=missing,
    )


def ensure_canonical_columns(
    df: pd.DataFrame,
    *,
    source_module: str,
    run_id: str | None = None,
    sample_barcode: str | None = None,
) -> pd.DataFrame:
    schema = load_canonical_schema()
    out = df.copy()
    for column in schema.ordered_columns:
        if column not in out.columns:
            out[column] = schema.defaults.get(column)

    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].where(out["timestamp"].notna(), datetime.now(UTC).replace(microsecond=0).isoformat())
    if "schema_version" in out.columns:
        out["schema_version"] = out["schema_version"].where(out["schema_version"].notna(), schema.schema_version)
    if "source_module" in out.columns:
        out["source_module"] = out["source_module"].where(out["source_module"].notna(), source_module)
    if "run_id" in out.columns and run_id is not None:
        out["run_id"] = out["run_id"].where(out["run_id"].notna(), run_id)
    if "sample_barcode" in out.columns and sample_barcode is not None:
        out["sample_barcode"] = out["sample_barcode"].where(out["sample_barcode"].notna(), sample_barcode)

    ordered = schema.ordered_columns
    remaining = [column for column in out.columns if column not in ordered]
    return out[ordered + remaining]


def build_canonical_from_candidates(
    df: pd.DataFrame,
    *,
    source_module: str,
    run_id: str,
    sample_barcode: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mappings = {
        "patient_id": ("patient_id", "sample_id"),
        "sample_barcode": ("sample_barcode",),
        "gene": ("gene", "gene_name", "Hugo_Symbol"),
        "protein_change": ("protein_change", "source_hgvsp_short"),
        "mutant_peptide": ("mutant_peptide", "peptide"),
        "wildtype_peptide": ("wildtype_peptide",),
        "peptide_length": ("peptide_length",),
        "peptide_position": ("peptide_position", "mutation_position"),
        "peptide_source": ("peptide_source", "input_mode"),
        "binding_affinity": ("binding_affinity", "affinity_nm"),
        "binding_rank": ("binding_rank",),
        "binding_tool": ("binding_tool",),
        "binding_tool_version": ("binding_tool_version",),
        "hla_allele": ("hla_allele", "best_allele"),
        "presentation_composite": ("presentation_composite", "presentation_score", "priority_score"),
        "expression_tpm": ("expression_tpm", "real_expression_tpm"),
        "ccf": ("ccf", "real_ccf"),
        "self_dissimilarity": ("self_dissimilarity",),
        "hla_loh_status": ("hla_loh_status",),
        "exclusion_reasons": ("exclusion_reasons",),
        "tier": ("tier", "triage_label"),
        "composite_priority": ("composite_priority", "rl_priority", "priority_score"),
        "report_summary": ("report_summary",),
    }
    for canonical, candidates in mappings.items():
        for source in candidates:
            if source in df.columns:
                out[canonical] = df[source]
                break

    return ensure_canonical_columns(
        out,
        source_module=source_module,
        run_id=run_id,
        sample_barcode=sample_barcode,
    )
