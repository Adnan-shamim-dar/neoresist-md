from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from neoresist.config import get_app_config, resolved_cohort_search_paths
from neoresist.dash_app.data import (
    _empty_candidates_frame,
    _normalize_loaded_frame,
    load_qualified_candidates,
    seed_cache,
)
from neoresist.schema import format_validation_message, normalize_cohort_df, validate_dash_columns
from neoresist.tumor_features import add_candidate_tumor_flags, infer_tumor_type_from_text, normalise_tumor_type


class CohortLoadError(Exception):
    """Rich error for UI: paths tried + validation details."""

    def __init__(self, message: str, *, searched: list[str], missing_columns: list[str], present_columns: list[str]):
        super().__init__(message)
        self.searched = searched
        self.missing_columns = missing_columns
        self.present_columns = present_columns


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if suf in (".csv",):
        return pd.read_csv(path)
    if suf in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported cohort file type: {path}")


def _adapt_summary_cohort_for_dash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept lightweight cohort summary tables (patient/mutations/candidates/fanout)
    and adapt them to the minimal Dash candidate contract.
    """
    if df.empty:
        return df
    out = df.copy()
    cols = {str(c).lower(): str(c) for c in out.columns}
    if "patient_id" not in out.columns:
        pcol = cols.get("patient")
        if pcol:
            out["patient_id"] = out[pcol]
    has_summary = all(k in cols for k in ("mutations", "candidates", "fanout"))
    if "patient_id" not in out.columns or not has_summary:
        return out

    # Single-row synthetic candidate context per patient from summary-level cohort.
    if "hla_allele" in out.columns:
        out["hla_allele"] = out["hla_allele"].astype(str)
    else:
        out["hla_allele"] = "HLA-UNK"
    out["mutations"] = pd.to_numeric(out["mutations"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["candidates"] = pd.to_numeric(out["candidates"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["fanout"] = pd.to_numeric(out["fanout"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fan = out["fanout"]
    fmax = float(fan.max()) if len(fan) else 0.0
    if fmax > 0:
        out["rl_priority"] = (fan / fmax).clip(0.0, 1.0)
    else:
        out["rl_priority"] = 0.0
    if "tier" in out.columns:
        out["tier"] = pd.to_numeric(out["tier"], errors="coerce").fillna(3).astype(int).clip(1, 3)
    else:
        out["tier"] = 3
        out.loc[out["rl_priority"] > 0.70, "tier"] = 1
        out.loc[(out["rl_priority"] >= 0.40) & (out["rl_priority"] <= 0.70), "tier"] = 2
    if "exclusion_reasons" in out.columns:
        out["exclusion_reasons"] = out["exclusion_reasons"].apply(
            lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)])
        )
    else:
        out["exclusion_reasons"] = [[] for _ in range(len(out))]

    # Populate optional candidate-level fields used by Dash panels.
    if "expression_tpm" not in out.columns:
        out["expression_tpm"] = out["fanout"] * 10.0
    if "ccf" not in out.columns:
        out["ccf"] = 0.5
    if "presentation_score" not in out.columns:
        out["presentation_score"] = out["rl_priority"]
    if "self_dissimilarity" not in out.columns:
        out["self_dissimilarity"] = 0.5
    if "affinity_nm" not in out.columns:
        out["affinity_nm"] = pd.NA
    if "gene" not in out.columns:
        out["gene"] = "NA"
    if "mutant_peptide" not in out.columns:
        out["mutant_peptide"] = "NA"
    if "protein_change" not in out.columns:
        out["protein_change"] = "NA"
    if "mutation_position" not in out.columns:
        out["mutation_position"] = out.index.astype(int)
    out["_summary_mode"] = True
    return add_candidate_tumor_flags(out)


def _apply_cohort_tumor_type(df: pd.DataFrame, source: Path | str | None) -> pd.DataFrame:
    out = df.copy()
    if "tumor_type" in out.columns:
        out["tumor_type"] = out["tumor_type"].map(normalise_tumor_type)
        return add_candidate_tumor_flags(out)
    try:
        dataset_id = get_app_config().defaults.dataset_id
    except Exception:
        dataset_id = ""
    out["tumor_type"] = infer_tumor_type_from_text(source, dataset_id)
    return add_candidate_tumor_flags(out)


def load_cohort_for_dash(*, alt_path: str | None = None) -> tuple[pd.DataFrame, str | None, list[str]]:
    """
    Resolve cohort file, normalize columns, validate minimum Dash contract.

    Returns (dataframe, resolved_path_or_none, list_of_searched_paths_strings).
    On missing file: empty frame, None, searched paths.
    """
    env_path = os.environ.get("NEORESIST_COHORT_PATH", "").strip()
    searched: list[str] = []

    if alt_path:
        p = Path(alt_path)
        searched.append(str(p.resolve()))
        if p.is_file():
            raw = _read_table(p)
            norm = normalize_cohort_df(raw)
            norm = _adapt_summary_cohort_for_dash(norm)
            norm = _normalize_loaded_frame(norm)
            norm = _apply_cohort_tumor_type(norm, p)
            miss, pres = validate_dash_columns(norm)
            if miss:
                raise CohortLoadError(
                    format_validation_message(miss, pres, searched=searched),
                    searched=searched,
                    missing_columns=miss,
                    present_columns=pres,
                )
            seed_cache(norm, str(p.resolve()))
            return norm, str(p.resolve()), searched

    if env_path:
        p = Path(env_path)
        searched.append(str(p.resolve()))
        if p.is_file():
            raw = _read_table(p)
            norm = normalize_cohort_df(raw)
            norm = _adapt_summary_cohort_for_dash(norm)
            norm = _normalize_loaded_frame(norm)
            norm = _apply_cohort_tumor_type(norm, p)
            miss, pres = validate_dash_columns(norm)
            if miss:
                raise CohortLoadError(
                    format_validation_message(miss, pres, searched=searched),
                    searched=searched,
                    missing_columns=miss,
                    present_columns=pres,
                )
            seed_cache(norm, str(p.resolve()))
            return norm, str(p.resolve()), searched

    for rel in resolved_cohort_search_paths():
        rp = Path(rel)
        searched.append(str(rp.resolve()))
        if not rp.is_file():
            continue
        try:
            raw = _read_table(rp)
        except Exception:
            continue
        norm = normalize_cohort_df(raw)
        norm = _adapt_summary_cohort_for_dash(norm)
        norm = _normalize_loaded_frame(norm)
        norm = _apply_cohort_tumor_type(norm, rp)
        miss, pres = validate_dash_columns(norm)
        if miss:
            # Wrong shape: keep searching
            continue
        seed_cache(norm, str(rp.resolve()))
        return norm, str(rp.resolve()), searched

    # Parity with legacy: use in-process parquet cache path
    df, legacy_path = load_qualified_candidates(alt_path=None)
    if legacy_path:
        return _apply_cohort_tumor_type(df, legacy_path), legacy_path, searched

    return _empty_candidates_frame().copy(), None, searched
