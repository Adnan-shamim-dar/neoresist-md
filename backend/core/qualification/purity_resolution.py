"""
Phase 5: deterministic purity arbitration and optional user-file ingestion.

Priority (default): ``user_file`` > ``tcga_metadata`` > ``computed_estimate`` > ``stub_fallback``.

Pure functions are safe to call from CLI or Dash.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from backend.core.data.tcga_data import case_barcode_from_aliquot, sample_barcode_prefix16

SourceKey = Literal["user_file", "tcga_metadata", "computed_estimate", "stub_fallback"]

DEFAULT_PRIORITY: tuple[SourceKey, ...] = (
    "user_file",
    "tcga_metadata",
    "computed_estimate",
    "stub_fallback",
)

STUB_FALLBACK_PURITY = 0.82
STUB_FALLBACK_CONFIDENCE = 0.35

PHASE5_PURITY_OUTPUT_COLS: tuple[str, ...] = (
    "purity_value_used",
    "purity_source_used",
    "purity_method_used",
    "purity_confidence_used",
    "purity_candidates_json",
    "purity_resolution_reason",
)


def prepare_dataframe_for_purity_resolution_rerun(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop a prior Phase 5 purity arbitration pass so :func:`apply_purity_resolution` can run again
    (e.g. Dash preview after uploading a sidecar table). Restores ``purity`` from ``purity_tcga_merged``.
    """
    b = df.copy()
    for c in PHASE5_PURITY_OUTPUT_COLS:
        if c in b.columns:
            b = b.drop(columns=[c])
    if "purity_tcga_merged" in b.columns:
        b["purity"] = b["purity_tcga_merged"]
    return b


def parse_purity_key_arg(s: str | None) -> str:
    """``auto`` | ``patient_id`` | ``barcode`` — key order for cohort ↔ purity table matching."""
    if not s or not str(s).strip():
        return "auto"
    v = str(s).strip().lower()
    if v in ("auto", "patient_id", "barcode"):
        return v
    raise ValueError(f"Invalid --purity-key {s!r}; use auto, patient_id, or barcode")


def parse_priority_arg(s: str | None) -> tuple[SourceKey, ...]:
    if not s or not str(s).strip():
        return DEFAULT_PRIORITY
    keys: list[SourceKey] = []
    allowed = set(DEFAULT_PRIORITY)
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        if t not in allowed:
            raise ValueError(f"Unknown purity priority token: {t!r}; allowed: {sorted(allowed)}")
        keys.append(t)  # type: ignore[arg-type]
    if not keys:
        return DEFAULT_PRIORITY
    # append any missing keys at end for determinism
    for k in DEFAULT_PRIORITY:
        if k not in keys:
            keys.append(k)
    return tuple(dict.fromkeys(keys))  # type: ignore[return-value]


def normalize_purity_scalar(raw: Any) -> tuple[float | None, str | None]:
    """
    Return ``(value, warning)``. Reject values outside ``[0, 1]`` or non-numeric.
    Warn (do not reject) on edge cases outside ``[0.05, 0.99]``.
    """
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return None, None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(x):
        return None, None
    if x < 0.0 or x > 1.0:
        return None, f"rejected_purity_out_of_range:{x}"
    warn = None
    if x < 0.05 or x > 0.99:
        warn = f"implausible_purity:{x}"
    return x, warn


def load_purity_table(path: Path) -> pd.DataFrame:
    """Load CSV, TSV, Parquet, or JSON (array of objects or object of records)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif suf in (".csv",):
        df = pd.read_csv(path)
    elif suf in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif suf in (".json",):
        raw = path.read_text(encoding="utf-8")
        df = _parse_json_purity(raw)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_parquet(path, engine="pyarrow")
    return normalize_purity_column_names(df)


def load_purity_table_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file content (Dash)."""
    name = (filename or "upload").lower()
    if name.endswith(".parquet"):
        import io

        return normalize_purity_column_names(pd.read_parquet(io.BytesIO(data), engine="pyarrow"))
    if name.endswith(".json"):
        return normalize_purity_column_names(_parse_json_purity(data.decode("utf-8", errors="replace")))
    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    import io

    return normalize_purity_column_names(pd.read_csv(io.BytesIO(data), sep=sep))


def _parse_json_purity(raw: str) -> pd.DataFrame:
    obj = json.loads(raw)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if "records" in obj and isinstance(obj["records"], list):
            return pd.DataFrame(obj["records"])
        return pd.DataFrame([obj])
    raise ValueError("JSON purity file must be a list of objects or a single object")


def normalize_purity_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cmap: dict[str, str] = {}
    for c in out.columns:
        cl = str(c).strip().lower().replace(" ", "_")
        if cl in (
            "patient_id",
            "patient",
            "case_id",
            "submitter_id",
            "case_submitter_id",
        ):
            cmap[c] = "patient_id_key"
        elif cl in ("sample_id", "sample_barcode", "barcode", "aliquot_barcode", "aliquot"):
            cmap[c] = "barcode_key"
        elif cl in ("purity", "tumor_purity", "estimated_purity", "cellularity", "absccn_purity"):
            cmap[c] = "purity"
        elif cl in ("purity_source", "source"):
            cmap[c] = "purity_source"
        elif cl in ("purity_method", "method"):
            cmap[c] = "purity_method"
        elif cl in ("purity_confidence", "confidence"):
            cmap[c] = "purity_confidence"
    out = out.rename(columns={k: v for k, v in cmap.items() if k in out.columns})
    if "patient_id_key" not in out.columns and "barcode_key" not in out.columns:
        raise ValueError("Purity table needs patient_id/barcode-like column or barcode column")
    if "purity" not in out.columns:
        raise ValueError("Purity table needs a purity column")
    return out


def build_user_purity_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Build lookup keyed by multiple normalized id strings per row.

    Each value: value, purity_source, purity_method, purity_confidence, match_keys
    """
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        pv, _warn = normalize_purity_scalar(row.get("purity"))
        if pv is None:
            continue
        src = str(row.get("purity_source") or "user_file").strip() or "user_file"
        meth = str(row.get("purity_method") or "user_supplied").strip() or "user_supplied"
        try:
            conf = float(row.get("purity_confidence"))
            if math.isnan(conf):
                conf = 0.9
        except (TypeError, ValueError):
            conf = 0.9
        conf = max(0.0, min(1.0, conf))
        keys: list[str] = []
        pid = row.get("patient_id_key")
        if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
            keys.append(str(pid).strip())
            keys.append(case_barcode_from_aliquot(str(pid)))
            keys.append(sample_barcode_prefix16(str(pid)))
        bc = row.get("barcode_key")
        if bc is not None and not (isinstance(bc, float) and pd.isna(bc)):
            b = str(bc).strip()
            keys.append(b)
            keys.append(case_barcode_from_aliquot(b))
            keys.append(sample_barcode_prefix16(b))
        keys = list(dict.fromkeys([k for k in keys if k]))
        payload = {
            "value": pv,
            "purity_source": src,
            "purity_method": meth,
            "purity_confidence": conf,
            "match_keys": keys,
        }
        for k in keys:
            lookup.setdefault(k, payload)
    return lookup


def _candidate_tcga(row: pd.Series) -> dict[str, Any] | None:
    pv, warn = normalize_purity_scalar(row.get("purity"))
    if pv is None:
        return None
    meth = row.get("purity_method")
    if meth is None or (isinstance(meth, float) and pd.isna(meth)):
        meth = "tcga_metadata"
    conf = row.get("purity_confidence")
    try:
        cf = float(conf) if conf is not None and not pd.isna(conf) else 0.75
    except (TypeError, ValueError):
        cf = 0.75
    if warn:
        warnings.warn(f"TCGA purity note for row: {warn}", stacklevel=3)
    return {
        "value": pv,
        "purity_source": "tcga_metadata",
        "purity_method": "tcga_metadata" if str(meth) == "stub_python_hash_tpm" else str(meth),
        "purity_confidence": max(0.0, min(1.0, cf)),
        "extra_warn": warn,
    }

def _candidate_computed(row: pd.Series) -> dict[str, Any] | None:
    pv, warn = normalize_purity_scalar(row.get("computed_purity_estimate"))
    if pv is None:
        return None
    if warn:
        warnings.warn(f"Computed purity note: {warn}", stacklevel=3)
    return {
        "value": pv,
        "purity_source": "computed_estimate",
        "purity_method": str(row.get("computed_purity_method") or "computed"),
        "purity_confidence": 0.55,
        "extra_warn": warn,
    }


def _candidate_stub() -> dict[str, Any]:
    return {
        "value": STUB_FALLBACK_PURITY,
        "purity_source": "stub_fallback",
        "purity_method": "default_stub",
        "purity_confidence": STUB_FALLBACK_CONFIDENCE,
        "extra_warn": None,
    }


def row_keys_for_match(patient_id: str, *, key_mode: str = "auto") -> list[str]:
    """
    Ordered keys to match user purity tables / metadata (deterministic).

    ``key_mode``: ``auto`` (patient id, case barcode, 16-mer prefix),
    ``patient_id`` (prefer full id first), ``barcode`` (prefer short prefix first).
    """
    pid = str(patient_id).strip()
    case = case_barcode_from_aliquot(pid)
    p16 = sample_barcode_prefix16(pid)
    if key_mode == "barcode":
        return list(dict.fromkeys([p16, pid, case]))
    if key_mode == "patient_id":
        return list(dict.fromkeys([pid, case, p16]))
    return list(dict.fromkeys([pid, case, p16]))


def resolve_purity_for_row(
    row: pd.Series,
    *,
    user_lookup: dict[str, dict[str, Any]] | None,
    priority: tuple[SourceKey, ...],
    stub_fallback: bool = False,
    purity_key: str = "auto",
) -> dict[str, Any]:
    """
    Return dict with keys: purity_value_used, purity_source_used, purity_method_used,
    purity_confidence_used, purity_candidates_json, purity_resolution_reason, and optional warnings list.
    """
    pid = str(row.get("patient_id", "")).strip()
    keys = row_keys_for_match(pid, key_mode=purity_key)

    user_hit: dict[str, Any] | None = None
    if user_lookup:
        for k in keys:
            if k in user_lookup:
                user_hit = dict(user_lookup[k])
                break

    tc = _candidate_tcga(row)
    ce = _candidate_computed(row)
    st = _candidate_stub()

    candidates: list[dict[str, Any]] = []
    if user_hit:
        candidates.append({**user_hit, "rank": "user_file"})
    if tc:
        candidates.append({**tc, "rank": "tcga_metadata"})
    if ce:
        candidates.append({**ce, "rank": "computed_estimate"})
    if stub_fallback:
        candidates.append({**st, "rank": "stub_fallback"})

    chosen: dict[str, Any] | None = None
    reason_parts: list[str] = []
    for src in priority:
        if src == "user_file" and user_hit:
            chosen = {**user_hit, "rank": "user_file"}
            reason_parts.append("selected=user_file (explicit table)")
            break
        if src == "tcga_metadata" and tc:
            chosen = {**tc, "rank": "tcga_metadata"}
            reason_parts.append("selected=tcga_metadata")
            break
        if src == "computed_estimate" and ce:
            chosen = {**ce, "rank": "computed_estimate"}
            reason_parts.append("selected=computed_estimate")
            break
        if src == "stub_fallback" and stub_fallback:
            chosen = {**st, "rank": "stub_fallback"}
            reason_parts.append("selected=stub_fallback")
            break

    if chosen is None:
        cand_json = [
            {
                "rank": c.get("rank"),
                "value": c.get("value"),
                "purity_source": c.get("purity_source"),
                "purity_method": c.get("purity_method"),
                "purity_confidence": c.get("purity_confidence"),
            }
            for c in candidates
        ]
        return {
            "purity_value_used": float("nan"),
            "purity_source_used": "unresolved",
            "purity_method_used": "",
            "purity_confidence_used": 0.0,
            "purity_candidates_json": json.dumps(cand_json, sort_keys=True),
            "purity_resolution_reason": "no_resolvable_purity_source (enable --purity-stub-fallback for default stub)",
        }

    cand_json = [
        {
            "rank": c.get("rank"),
            "value": c.get("value"),
            "purity_source": c.get("purity_source"),
            "purity_method": c.get("purity_method"),
            "purity_confidence": c.get("purity_confidence"),
        }
        for c in candidates
    ]

    rk = str(chosen.get("rank") or "")
    return {
        "purity_value_used": float(chosen["value"]),
        "purity_source_used": str(chosen.get("purity_source") or rk),
        "purity_method_used": str(chosen.get("purity_method") or ""),
        "purity_confidence_used": float(chosen.get("purity_confidence") or 0.0),
        "purity_candidates_json": json.dumps(cand_json, sort_keys=True),
        "purity_resolution_reason": "; ".join(reason_parts),
    }


def apply_purity_resolution(
    df: pd.DataFrame,
    *,
    user_purity_df: pd.DataFrame | None = None,
    priority: tuple[SourceKey, ...] | None = None,
    stub_fallback: bool = False,
    purity_key: str = "auto",
) -> pd.DataFrame:
    """
    Add Phase 5 purity arbitration columns.

    Preserves pre-resolution TCGA merge in ``purity_tcga_merged`` when ``purity`` was present,
    then sets ``purity`` to ``purity_value_used`` (canonical).
    """
    if df.empty:
        return df.copy()
    pr = priority or DEFAULT_PRIORITY
    lookup = build_user_purity_lookup(user_purity_df) if user_purity_df is not None and not user_purity_df.empty else None

    base = df.reset_index(drop=True)
    if "purity" in base.columns:
        base = base.copy()
        base["purity_tcga_merged"] = base["purity"]

    rows_out = []
    for _, row in base.iterrows():
        r = resolve_purity_for_row(
            row,
            user_lookup=lookup,
            priority=pr,
            stub_fallback=stub_fallback,
            purity_key=purity_key,
        )
        rows_out.append(r)
    add = pd.DataFrame(rows_out)
    out = pd.concat([base, add], axis=1)
    out["purity"] = out["purity_value_used"]
    return out


def apply_evidence_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate unified provenance fields required by Phase 5 schema.

    Safe to call after resistance loop and real layers.
    """
    out = df.copy()
    if out.empty:
        return _ensure_provenance_schema(out)

    expr: list[str] = []
    ccf: list[str] = []
    pur: list[str] = []
    meta: list[str] = []
    status: list[str] = []
    notes: list[str] = []

    for _, row in out.iterrows():
        ev_expr = row.get("evidence_expression_source")
        if ev_expr is not None and str(ev_expr) not in ("nan", "None", ""):
            es = str(ev_expr)
        elif pd.notna(row.get("real_expression_tpm")) and not row.get("RNA_data_missing", True):
            es = "tcga_rnaseq"
        elif row.get("RNA_data_missing"):
            es = "missing_or_stub"
        else:
            es = str(row.get("expression_bin") or "stub")
        expr.append(es)

        ev_ccf = row.get("evidence_ccf_source")
        if ev_ccf is not None and str(ev_ccf) not in ("nan", "None", ""):
            cs = str(ev_ccf)
        elif pd.notna(row.get("real_ccf")) and not row.get("clonality_data_missing", False):
            cs = "vaf_purity_naive"
        else:
            cs = "missing_or_degraded"
        ccf.append(cs)

        psu = row.get("purity_source_used")
        if psu is None or (isinstance(psu, float) and pd.isna(psu)):
            pur.append("unknown")
        else:
            psu_s = str(psu)
            pur.append(psu_s if psu_s not in ("nan", "None", "") else "unknown")

        ms = row.get("source") or row.get("source_tcga")
        if ms is None or (isinstance(ms, float) and pd.isna(ms)):
            ms = row.get("tcga_join_status")
        meta.append(str(ms or "unknown"))

        miss_rna = bool(row.get("RNA_data_missing"))
        miss_clon = bool(row.get("clonality_data_missing"))
        ps_raw = row.get("purity_source_used")
        if ps_raw is None or (isinstance(ps_raw, float) and pd.isna(ps_raw)):
            psu_s = ""
        else:
            psu_s = str(ps_raw)
        miss_pur = psu_s in ("stub_fallback", "unresolved")
        if not miss_rna and not miss_clon and not miss_pur:
            st = "complete"
        elif miss_rna and miss_clon:
            st = "minimal_stub"
        else:
            st = "partial"
        status.append(st)

        note_parts = [
            f"expr:{es}",
            f"ccf:{cs}",
            f"purity:{pur[-1]}",
        ]
        notes.append("|".join(note_parts))

    out["expression_source"] = expr
    out["ccf_source"] = ccf
    out["purity_source"] = pur
    out["metadata_source"] = meta
    out["resolution_status"] = status
    out["evidence_notes"] = notes
    return _ensure_provenance_schema(out)


def _ensure_provenance_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = (
        "expression_source",
        "ccf_source",
        "purity_source",
        "metadata_source",
        "resolution_status",
        "evidence_notes",
    )
    for c in required:
        if c not in df.columns:
            df[c] = None
    return df
