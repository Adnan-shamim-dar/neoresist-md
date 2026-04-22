from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from neoresist.dash_app.display_utils import ensure_display_columns
from neoresist.paths import repo_root

_CACHE_MT: float | None = None
_CACHE_DF: pd.DataFrame | None = None
_CACHE_PATH_KEY: str | None = None

# Bounds for aggregate filters (mutations / fanout / candidates); set at layout init.
_slider_bounds: dict[str, int | float] = {"mut": 100, "fan": 50.0, "cand": 5000}


def qualified_parquet_search_paths() -> list[Path]:
    root = repo_root()
    return [
        root / "data" / "final" / "enriched_candidates.parquet",
        root / "neoresist-md" / "data" / "final" / "enriched_candidates.parquet",
        root / "data" / "final" / "qualified_candidates.parquet",
        root / "neoresist-md" / "data" / "final" / "qualified_candidates.parquet",
    ]


def _resolve_qualified_path() -> Path | None:
    for p in qualified_parquet_search_paths():
        if p.is_file():
            return p
    return None


def _empty_candidates_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "hla_allele",
            "gene",
            "mutant_peptide",
            "tier",
            "rl_priority",
            "expression_tpm",
            "ccf",
            "hla_loh_status",
            "exclusion_reasons",
        ]
    )


def seed_cache(df: pd.DataFrame, path: str) -> None:
    """Align loader path with parquet cache for filter callbacks."""
    global _CACHE_MT, _CACHE_DF, _CACHE_PATH_KEY
    p = Path(path)
    if not p.is_file():
        return
    _CACHE_PATH_KEY = str(p.resolve())
    _CACHE_MT = p.stat().st_mtime
    _CACHE_DF = df.copy()


def load_qualified_candidates(alt_path: str | None = None) -> tuple[pd.DataFrame, str | None]:
    global _CACHE_MT, _CACHE_DF, _CACHE_PATH_KEY
    if alt_path:
        p = Path(alt_path)
        if p.is_file():
            key = str(p.resolve())
            mtime = p.stat().st_mtime
            if _CACHE_PATH_KEY == key and _CACHE_MT == mtime and _CACHE_DF is not None:
                return _CACHE_DF.copy(), key
            raw = pd.read_parquet(p, engine="pyarrow")
            df = _normalize_loaded_frame(raw)
            _CACHE_MT, _CACHE_DF, _CACHE_PATH_KEY = mtime, df, key
            return df.copy(), key
    path = _resolve_qualified_path()
    if path is None:
        _CACHE_MT, _CACHE_DF, _CACHE_PATH_KEY = None, None, None
        return _empty_candidates_frame().copy(), None
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    if _CACHE_PATH_KEY == key and _CACHE_MT == mtime and _CACHE_DF is not None:
        return _CACHE_DF.copy(), str(path)
    raw = pd.read_parquet(path, engine="pyarrow")
    df = _normalize_loaded_frame(raw)
    _CACHE_MT, _CACHE_DF, _CACHE_PATH_KEY = mtime, df, key
    return df.copy(), str(path)


def _normalize_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_candidates_frame()
    out = ensure_display_columns(df)
    if "exclusion_reasons" not in out.columns:
        out["exclusion_reasons"] = [[] for _ in range(len(out))]
    if "hla_loh_status" not in out.columns:
        if "hla_loh_flag" in out.columns:
            out["hla_loh_status"] = out["hla_loh_flag"].astype(str)
        else:
            out["hla_loh_status"] = pd.NA
    tier_s = out["tier"] if "tier" in out.columns else pd.Series(pd.NA, index=out.index, dtype=object)
    out["tier"] = pd.to_numeric(tier_s, errors="coerce").fillna(3).astype(int).clip(1, 3)
    rl_s = out["rl_priority"] if "rl_priority" in out.columns else pd.Series(pd.NA, index=out.index, dtype=object)
    out["rl_priority"] = pd.to_numeric(rl_s, errors="coerce").fillna(0.0).clip(0, 1)
    out["exclusion_reasons"] = out["exclusion_reasons"].apply(_coerce_exclusion_list)
    for col in ("expression_tpm", "ccf", "presentation_score", "self_dissimilarity", "affinity_nm"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _coerce_exclusion_list(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if hasattr(val, "tolist"):
        try:
            return [str(x) for x in val.tolist()]
        except Exception:
            return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [val] if val else []
    return [str(val)]


def _expression_norm(tpm: float) -> float:
    cap = 1000.0
    t = max(float(tpm), 0.0)
    return max(0.0, min(1.0, math.log1p(t) / math.log1p(cap)))


def _rl_breakdown_terms(row: pd.Series) -> dict[str, float]:
    tpm = float(row.get("expression_tpm") or 0.0)
    en = float(row.get("expression_norm") or _expression_norm(tpm))
    ps = float(row.get("presentation_score") or 0.0)
    if math.isnan(ps):
        ps = 0.0
    ccf = float(row.get("ccf") or 0.0)
    if math.isnan(ccf):
        ccf = 0.0
    sd = float(row.get("self_dissimilarity") or 0.0)
    if math.isnan(sd):
        sd = 0.0
    ep = float(row.get("escape_penalty") or 0.0)
    if math.isnan(ep):
        ep = 0.0
    return {
        "expression_norm": en,
        "presentation": max(0.0, min(1.0, ps)),
        "ccf": max(0.0, min(1.0, ccf)),
        "self_dissimilarity": max(0.0, min(1.0, sd)),
        "escape_penalty": max(0.0, min(1.0, ep)),
    }


def _aggregate_patient_hla(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "hla_allele",
                "mutations",
                "candidates",
                "fanout",
                "mean_rl_priority",
                "best_tier",
                "mean_expression_tpm",
                "mean_ccf",
                "mean_real_expr",
                "mean_real_ccf",
                "mean_purity_used",
                "purity_source_mode",
                "hla_loh_status",
                "top_exclusions",
            ]
        )

    def mut_key(g: pd.DataFrame) -> int:
        if "gene" in g.columns and "mutation_position" in g.columns:
            return g.groupby(["gene", "mutation_position"], dropna=False).ngroups
        if "gene" in g.columns and "protein_change" in g.columns:
            return g.groupby(["gene", "protein_change"], dropna=False).ngroups
        return int(g["gene"].nunique()) if "gene" in g.columns else len(g)

    rows: list[dict[str, Any]] = []
    for (pid, hla), g in cand.groupby(["patient_id", "hla_allele"], sort=False):
        summary_mode = bool(g.get("_summary_mode", pd.Series(False, index=g.index)).astype(bool).any())
        if summary_mode and {"mutations", "candidates", "fanout"}.issubset(set(g.columns)):
            n_mut = int(pd.to_numeric(g["mutations"], errors="coerce").fillna(0).max())
            n_cand = int(pd.to_numeric(g["candidates"], errors="coerce").fillna(0).max())
            fan = float(pd.to_numeric(g["fanout"], errors="coerce").fillna(0.0).max())
            n_mut = max(n_mut, 0)
            n_cand = max(n_cand, 0)
            fan = max(fan, 0.0)
        else:
            n_mut = mut_key(g)
            n_cand = len(g)
            fan = n_cand / max(n_mut, 1)
        mean_rl = float(pd.to_numeric(g["rl_priority"], errors="coerce").mean())
        if math.isnan(mean_rl):
            mean_rl = 0.0
        best_tier = int(pd.to_numeric(g["tier"], errors="coerce").min())
        mean_expr = float(pd.to_numeric(g.get("expression_tpm"), errors="coerce").mean())
        if math.isnan(mean_expr):
            mean_expr = 0.0
        mean_ccf = float(pd.to_numeric(g.get("ccf"), errors="coerce").mean())
        if math.isnan(mean_ccf):
            mean_ccf = 0.0
        mean_real_expr = float("nan")
        if "real_expression_tpm" in g.columns:
            mean_real_expr = float(pd.to_numeric(g["real_expression_tpm"], errors="coerce").mean())
            if math.isnan(mean_real_expr):
                mean_real_expr = 0.0
        mean_real_ccf = float("nan")
        if "real_ccf" in g.columns:
            mean_real_ccf = float(pd.to_numeric(g["real_ccf"], errors="coerce").mean())
            if math.isnan(mean_real_ccf):
                mean_real_ccf = 0.0
        mean_purity_u = float("nan")
        purity_src_mode = "—"
        if "purity_value_used" in g.columns:
            pu = pd.to_numeric(g["purity_value_used"], errors="coerce")
            mean_purity_u = float(pu.mean()) if len(pu) else float("nan")
            if math.isnan(mean_purity_u):
                mean_purity_u = float("nan")
        if "purity_source_used" in g.columns:
            pm = g["purity_source_used"].dropna().astype(str).mode()
            purity_src_mode = str(pm.iloc[0]) if len(pm) else "—"
        loh_mode = (
            g["hla_loh_status"].dropna().astype(str).mode().iloc[0]
            if "hla_loh_status" in g.columns and not g["hla_loh_status"].dropna().empty
            else "—"
        )
        evidence_availability = pd.DataFrame(
            {
                "presentation": pd.to_numeric(g.get("presentation_score"), errors="coerce"),
                "expression": pd.to_numeric(g.get("expression_tpm", g.get("expression_norm")), errors="coerce"),
                "clonality": pd.to_numeric(g.get("ccf"), errors="coerce"),
                "recognition": pd.to_numeric(g.get("self_dissimilarity"), errors="coerce"),
            }
        ).notna().sum(axis=1)
        mean_availability = float(evidence_availability.mean()) if len(evidence_availability) else 0.0
        if mean_availability >= 4:
            score_confidence = "full"
        elif mean_availability >= 2:
            score_confidence = "partial"
        else:
            score_confidence = "minimal"
        top_exc = _top_exclusion_strings(g["exclusion_reasons"], 5)
        rows.append(
            {
                "patient_id": pid,
                "hla_allele": hla,
                "mutations": int(n_mut),
                "candidates": int(n_cand),
                "fanout": float(fan),
                "mean_rl_priority": mean_rl,
                "best_tier": best_tier,
                "mean_expression_tpm": mean_expr,
                "mean_ccf": mean_ccf,
                "mean_real_expr": mean_real_expr if not math.isnan(mean_real_expr) else None,
                "mean_real_ccf": mean_real_ccf if not math.isnan(mean_real_ccf) else None,
                "mean_purity_used": mean_purity_u if not math.isnan(mean_purity_u) else None,
                "purity_source_mode": purity_src_mode,
                "hla_loh_status": loh_mode,
                "score_confidence": score_confidence,
                "data_quality_zero": bool(mean_rl == 0.0 and mean_availability <= 1.0),
                "top_exclusions": "; ".join(top_exc) if top_exc else "",
            }
        )
    return pd.DataFrame(rows)


def _top_exclusion_strings(series: pd.Series, n: int) -> list[str]:
    tags: list[str] = []
    for v in series:
        tags.extend(_coerce_exclusion_list(v))
    return [t for t, _ in Counter(tags).most_common(n)]


def top_exclusion_options(cand: pd.DataFrame, k: int = 10) -> list[dict[str, str]]:
    tags = _top_exclusion_strings(cand["exclusion_reasons"], 500)
    counts = Counter(tags).most_common(k)
    return [{"label": f"{t} ({c})", "value": t} for t, c in counts]


def filter_candidates(
    cand: pd.DataFrame,
    *,
    tiers: list[int] | None,
    hlas: list[str] | None,
    exclusion_any: list[str] | None,
    rl_range: list[float],
    expr_range: list[float],
    ccf_range: list[float],
    search: str,
) -> pd.DataFrame:
    if cand.empty:
        return cand
    f = cand.copy()
    if tiers:
        f = f[f["tier"].isin([int(x) for x in tiers])]
    if hlas:
        f = f[f["hla_allele"].astype(str).isin(hlas)]
    if exclusion_any:

        def has_exc(row: pd.Series) -> bool:
            ex = set(_coerce_exclusion_list(row.get("exclusion_reasons")))
            return bool(ex.intersection(set(exclusion_any)))

        mask = f.apply(has_exc, axis=1)
        f = f[mask]
    rl_lo, rl_hi = rl_range[0], rl_range[1]
    f = f[(f["rl_priority"] >= rl_lo) & (f["rl_priority"] <= rl_hi)]
    if "expression_tpm" in f.columns:
        et = pd.to_numeric(f["expression_tpm"], errors="coerce").fillna(0.0)
        f = f[(et >= expr_range[0]) & (et <= expr_range[1])]
    if "ccf" in f.columns:
        cc = pd.to_numeric(f["ccf"], errors="coerce").fillna(0.0)
        f = f[(cc >= ccf_range[0]) & (cc <= ccf_range[1])]
    if search:
        q = search.lower().strip()

        def row_match(r: pd.Series) -> bool:
            parts = [
                str(r.get("patient_id", "")),
                str(r.get("hla_allele", "")),
                str(r.get("gene", "")),
                str(r.get("mutant_peptide", "")),
                str(r.get("protein_change", "")),
                " ".join(_coerce_exclusion_list(r.get("exclusion_reasons"))),
            ]
            blob = " ".join(parts).lower()
            return q in blob

        f = f[f.apply(row_match, axis=1)]
    return f


def filter_agg(
    agg: pd.DataFrame,
    mut_range: list[float],
    fan_range: list[float],
    cand_range: list[float],
) -> pd.DataFrame:
    if agg.empty:
        return agg
    f = agg.copy()
    f = f[
        (f["mutations"] >= mut_range[0])
        & (f["mutations"] <= mut_range[1])
        & (f["fanout"] >= fan_range[0])
        & (f["fanout"] <= fan_range[1])
        & (f["candidates"] >= cand_range[0])
        & (f["candidates"] <= cand_range[1])
    ]
    return f


def sort_agg(agg: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if agg.empty:
        return agg
    sort_map = {
        "Mutations ↓": ("mutations", False),
        "Mutations ↑": ("mutations", True),
        "Candidates ↓": ("candidates", False),
        "Candidates ↑": ("candidates", True),
        "Fanout ↓": ("fanout", False),
        "Fanout ↑": ("fanout", True),
        "RL priority ↓": ("mean_rl_priority", False),
        "RL priority ↑": ("mean_rl_priority", True),
        "Tier ↑ (best first)": ("best_tier", True),
        "Tier ↓": ("best_tier", False),
    }
    col, asc = sort_map.get(sort_by, ("mutations", False))
    if col not in agg.columns:
        return agg
    return agg.sort_values(col, ascending=asc, kind="mergesort")


def set_slider_bounds_from_df(df: pd.DataFrame) -> None:
    agg = _aggregate_patient_hla(df)
    _slider_bounds["mut"] = int(agg["mutations"].max()) if not agg.empty else 100
    _slider_bounds["fan"] = round(float(agg["fanout"].max()), 1) if not agg.empty else 50.0
    _slider_bounds["cand"] = int(agg["candidates"].max()) if not agg.empty else 5000
    _slider_bounds["mut"] = max(int(_slider_bounds["mut"]), 1)
    _slider_bounds["fan"] = max(float(_slider_bounds["fan"]), 0.1)
    _slider_bounds["cand"] = max(int(_slider_bounds["cand"]), 1)


def slider_mut_max() -> int:
    return int(_slider_bounds["mut"])


def slider_fan_max() -> float:
    return float(_slider_bounds["fan"])


def slider_cand_max() -> int:
    return int(_slider_bounds["cand"])
