"""
NeoResist-MD — ResistanceLoop v1 evidence-aware cohort console.

Run from repo root: python app.py
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

# --- Tier theme (GitHub-dark accents) ---
TIER_COLORS = {1: "#238636", 2: "#d29922", 3: "#6e7681"}
TIER_LABELS = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}

BASE_DIR = Path(__file__).resolve().parent

QUALIFIED_CANDIDATES = [
    BASE_DIR / "data" / "final" / "enriched_candidates.parquet",
    BASE_DIR / "neoresist-md" / "data" / "final" / "enriched_candidates.parquet",
    BASE_DIR / "data" / "final" / "qualified_candidates.parquet",
    BASE_DIR / "neoresist-md" / "data" / "final" / "qualified_candidates.parquet",
]

_CACHE_MT: float | None = None
_CACHE_DF: pd.DataFrame | None = None
_CACHE_PATH_KEY: str | None = None


def _resolve_qualified_path() -> Path | None:
    for p in QUALIFIED_CANDIDATES:
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


def load_qualified_candidates(alt_path: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """
    Load ``enriched_candidates.parquet`` (preferred) or ``qualified_candidates.parquet``
    with pyarrow; cache by file mtime. Returns (dataframe, resolved_path_or_none).

    If ``alt_path`` is set and points to a file, load that cohort parquet instead.
    """
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
    out = df.copy()
    if "hla_loh_status" not in out.columns:
        if "hla_loh_flag" in out.columns:
            out["hla_loh_status"] = out["hla_loh_flag"].astype(str)
        else:
            out["hla_loh_status"] = pd.NA
    out["tier"] = pd.to_numeric(out.get("tier"), errors="coerce").fillna(3).astype(int).clip(1, 3)
    out["rl_priority"] = pd.to_numeric(out.get("rl_priority"), errors="coerce").clip(0, 1)
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


def _ccf_clonality_class(ccf: float | None) -> str:
    if ccf is None or (isinstance(ccf, float) and math.isnan(ccf)):
        return "Unknown"
    if ccf >= 0.66:
        return "High (stub)"
    if ccf >= 0.33:
        return "Medium (stub)"
    return "Low (stub)"


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
    """One row per (patient_id, hla_allele) for scatter + grid."""
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


def make_scatter_fig(agg: pd.DataFrame) -> go.Figure:
    if agg.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=12),
            title="No data — add enriched_candidates.parquet or qualified_candidates.parquet under data/final/",
            height=420,
        )
        return fig

    plot_df = agg.copy()
    plot_df["tier_label"] = plot_df["best_tier"].map(TIER_LABELS).fillna("Tier 3")
    plot_df["size_candidates"] = plot_df["candidates"].clip(lower=1)
    plot_df["top_exclusions_hover"] = plot_df["top_exclusions"].fillna("").astype(str)

    fig = px.scatter(
        plot_df,
        x="mutations",
        y="mean_rl_priority",
        size="size_candidates",
        color="best_tier",
        color_discrete_map={1: TIER_COLORS[1], 2: TIER_COLORS[2], 3: TIER_COLORS[3]},
        hover_name="patient_id",
        custom_data=["patient_id", "hla_allele", "best_tier", "mean_rl_priority", "top_exclusions_hover"],
        category_orders={"best_tier": [1, 2, 3]},
        title="Mutations vs mean RL priority (per patient × HLA)",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>HLA: %{customdata[1]}<br>"
            "Tier (best): %{customdata[2]}<br>Mean RL: %{customdata[3]:.3f}<br>"
            "Exclusions: %{customdata[4]}<extra></extra>"
        ),
        marker=dict(line=dict(width=1, color="#30363d"), opacity=0.9),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono, monospace", color="#e6edf3", size=12),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", title="Mutations (unique loci)"),
        yaxis=dict(
            gridcolor="#21262d",
            linecolor="#30363d",
            title="Mean RL priority",
            range=[-0.02, 1.02],
        ),
        legend_title_text="Tier (best)",
        margin=dict(t=48, l=20, r=20, b=20),
        height=420,
    )
    idx = plot_df["mutations"].idxmax()
    if idx is not None and not math.isnan(float(idx)):
        hyper = plot_df.loc[idx]
        fig.add_annotation(
            x=hyper["mutations"],
            y=hyper["mean_rl_priority"],
            text=f"Hypermutator: {hyper['patient_id']}",
            showarrow=True,
            arrowcolor=TIER_COLORS[2],
            bgcolor="rgba(210,153,34,0.12)",
            bordercolor=TIER_COLORS[2],
            font=dict(color="#e6edf3", size=11),
        )
    return fig


def kpi_card(value: str, label: str, *, accent: str | None = None) -> dbc.Card:
    style = {}
    if accent:
        style["borderTop"] = f"3px solid {accent}"
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(value, className="kpi-value"),
                html.Div(label, className="kpi-label"),
            ]
        ),
        class_name="kpi-card",
        style=style,
    )


def hla_coverage_badge_row(cand: pd.DataFrame) -> html.Div:
    if cand.empty or "hla_allele" not in cand.columns:
        return html.Div("HLA coverage: —", className="text-muted small")
    vc = cand["hla_allele"].astype(str).value_counts().head(12)
    badges = [
        dbc.Badge(f"{h}: {n:,}", color="secondary", className="me-1 mb-1", pill=True)
        for h, n in vc.items()
    ]
    return html.Div([html.Span("HLA coverage · ", className="text-muted small me-1"), *badges])


def evidence_badges_row(prefix: str, items: list[tuple[str, str, str]]) -> html.Div:
    """items: (label, value_str, badge_color)"""
    children: list[Any] = [html.Span(prefix, className="text-muted small me-2")]
    for lab, val, color in items:
        children.append(
            dbc.Badge(f"{lab}: {val}", color=color, className="me-1 mb-1 evidence-badge"),
        )
    return html.Div(children, className="d-flex flex-wrap align-items-center")


def build_evidence_panel(
    cand: pd.DataFrame,
    patient_id: str | None,
    hla_allele: str | None,
) -> list[Any]:
    if not patient_id or not hla_allele or cand.empty:
        return [
            html.H5("Evidence (ResistanceLoop)", className="mb-2"),
            html.Div("Select a patient × HLA from the scatter plot or the table.", className="text-muted"),
        ]
    sub = cand[(cand["patient_id"].astype(str) == str(patient_id)) & (cand["hla_allele"].astype(str) == str(hla_allele))]
    if sub.empty:
        return [
            html.H5("Evidence (ResistanceLoop)", className="mb-2"),
            html.Div("No candidate rows for this selection under current filters.", className="text-muted"),
        ]

    rep = sub.loc[sub["rl_priority"].idxmax()]
    terms = _rl_breakdown_terms(rep)
    rl_calc = (
        f"0.2×{terms['expression_norm']:.3f} + 0.3×{terms['presentation']:.3f} + "
        f"0.3×{terms['ccf']:.3f} + 0.1×{terms['self_dissimilarity']:.3f} − "
        f"0.2×{terms['escape_penalty']:.3f} ≈ {float(rep.get('rl_priority', 0.0)):.3f}"
    )
    tpm = float(rep.get("expression_tpm") or 0.0)
    ps = float(rep.get("presentation_score") or 0.0)
    ccf = float(rep.get("ccf") or 0.0)
    loh = str(rep.get("hla_loh_status", "—"))
    sd = float(rep.get("self_dissimilarity") or 0.0)

    badge_items: list[tuple[str, str, str]] = [
        ("TPM (stub)", f"{tpm:.2f}", "info"),
        ("Presentation", f"{ps:.3f}", "primary"),
        ("CCF (stub)", f"{ccf:.3f}", "warning"),
        ("LOH", loh, "danger" if loh == "lost" else "secondary"),
        ("Self-dissim.", f"{sd:.3f}", "success"),
    ]
    rtpm = rep.get("real_expression_tpm")
    if rtpm is not None and not pd.isna(rtpm):
        badge_items.insert(1, ("Real TPM", f"{float(rtpm):.2f}", "info"))
    rccf = rep.get("real_ccf")
    if rccf is not None and not pd.isna(rccf):
        badge_items.insert(4 if len(badge_items) > 4 else len(badge_items), ("Real CCF", f"{float(rccf):.3f}", "warning"))
    ebin = rep.get("expression_bin")
    if ebin is not None and str(ebin) not in ("nan", "None"):
        badge_items.append(("Expr bin", str(ebin), "secondary"))
    pvu = rep.get("purity_value_used")
    if pvu is not None and not pd.isna(pvu):
        try:
            badge_items.append(("Purity°", f"{float(pvu):.3f}", "primary"))
        except (TypeError, ValueError):
            pass
    psu = rep.get("purity_source_used")
    if psu is not None and str(psu) not in ("nan", "None", ""):
        badge_items.append(("Purity src", str(psu), "secondary"))
    pmu = rep.get("purity_method_used")
    if pmu is not None and str(pmu) not in ("nan", "None", ""):
        badge_items.append(("Purity method", str(pmu), "secondary"))
    csu = rep.get("clonality_source_used")
    if csu is not None and str(csu) not in ("nan", "None", ""):
        badge_items.append(("Clonality src", str(csu), "info"))
    esrc = rep.get("evidence_expression_source")
    csrc = rep.get("evidence_ccf_source")
    if esrc is not None or csrc is not None:
        badge_items.append(
            ("RL blend", f"expr={esrc or '—'} ccf={csrc or '—'}", "success"),
        )

    all_exc: list[str] = []
    for v in sub["exclusion_reasons"]:
        all_exc.extend(_coerce_exclusion_list(v))
    exc_counts = Counter(all_exc).most_common(12)

    top_pep = sub.nlargest(5, "rl_priority")[["mutant_peptide", "affinity_nm", "rl_priority"]].copy()
    pep_rows = []
    for _, pr in top_pep.iterrows():
        aff = pr.get("affinity_nm")
        aff_s = f"{float(aff):.1f} nM" if aff is not None and not pd.isna(aff) else "—"
        pep_rows.append(
            html.Tr(
                [
                    html.Td(str(pr.get("mutant_peptide", "")), className="font-monospace"),
                    html.Td(aff_s),
                    html.Td(f"{float(pr.get('rl_priority', 0)):.3f}"),
                ]
            )
        )

    return [
        html.H5("Evidence (ResistanceLoop v1)", className="mb-2"),
        html.Div(
            [
                html.Div(
                    [html.Strong("Selection: "), f"{patient_id} · {hla_allele}"],
                    className="mb-2 font-monospace",
                ),
                evidence_badges_row("", badge_items),
                html.Hr(className="my-2 border-secondary"),
                html.Div(
                    [
                        html.Div("Representative row (max RL in selection)", className="text-muted small"),
                        html.Div(f"Clonality class: {_ccf_clonality_class(ccf)}", className="mt-1"),
                        html.Div(f"RL components: {rl_calc}", className="mt-1 small text-break"),
                    ],
                    className="mb-2",
                ),
                html.Div("Top exclusion tags (selection)", className="text-muted small"),
                html.Ul([html.Li(f"{k} ({v})") for k, v in exc_counts] or [html.Li("—")]),
                html.Div("Top 5 peptides (mutant · affinity · RL)", className="text-muted small mt-2"),
                dbc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Peptide"), html.Th("Affinity"), html.Th("RL")])),
                        html.Tbody(pep_rows),
                    ],
                    bordered=False,
                    size="sm",
                    className="mb-0 text-light",
                ),
            ]
        ),
    ]


def build_patient_detail_card(
    cand: pd.DataFrame,
    patient_id: str | None,
    hla_allele: str | None,
) -> list[Any]:
    if not patient_id or not hla_allele or cand.empty:
        return [html.Div("Click a bubble or select a table row for patient × HLA detail.", className="text-muted")]

    sub = cand[(cand["patient_id"].astype(str) == str(patient_id)) & (cand["hla_allele"].astype(str) == str(hla_allele))]
    if sub.empty:
        return [html.Div("No rows for this selection.", className="text-muted")]

    tier_counts = sub["tier"].value_counts().reindex([1, 2, 3], fill_value=0).fillna(0).astype(int)
    pie = go.Figure(
        data=[
            go.Pie(
                labels=[TIER_LABELS[1], TIER_LABELS[2], TIER_LABELS[3]],
                values=[int(tier_counts[1]), int(tier_counts[2]), int(tier_counts[3])],
                marker=dict(colors=[TIER_COLORS[1], TIER_COLORS[2], TIER_COLORS[3]]),
                hole=0.35,
            )
        ]
    )
    pie.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=True,
        legend_font=dict(size=10, color="#e6edf3"),
    )

    sub_hist = sub.copy()
    sub_hist["tier_str"] = sub_hist["tier"].astype(str)
    hist = px.histogram(
        sub_hist,
        x="rl_priority",
        nbins=20,
        color="tier_str",
        color_discrete_map={
            "1": TIER_COLORS[1],
            "2": TIER_COLORS[2],
            "3": TIER_COLORS[3],
        },
        category_orders={"tier_str": ["1", "2", "3"]},
    )
    hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        xaxis_title="RL priority",
        yaxis_title="Candidates",
        margin=dict(l=40, r=10, t=30, b=40),
        height=220,
        legend_title_text="Tier",
    )

    exc_all: list[str] = []
    for v in sub["exclusion_reasons"]:
        exc_all.extend(_coerce_exclusion_list(v))
    top_exc = Counter(exc_all).most_common(12)
    bar = go.Figure(
        data=[
            go.Bar(
                x=[c for _, c in top_exc][::-1],
                y=[t for t, _ in top_exc][::-1],
                orientation="h",
                marker_color=TIER_COLORS[2],
            )
        ]
    )
    bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        margin=dict(l=10, r=10, t=28, b=40),
        height=260,
        title=dict(text="Exclusion reasons", font=dict(size=12)),
        xaxis_title="Count",
    )

    return [
        html.Div(
            [
                html.Div(f"{patient_id}", className="patient-id d-inline"),
                html.Span(f" · {hla_allele}", className="text-muted font-monospace"),
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=pie, config={"displayModeBar": False}), md=4, sm=12),
                dbc.Col(dcc.Graph(figure=hist, config={"displayModeBar": False}), md=4, sm=12),
                dbc.Col(dcc.Graph(figure=bar, config={"displayModeBar": False}), md=4, sm=12),
            ],
            className="g-2 mb-2",
        ),
        dbc.Button(
            "Download patient × HLA CSV",
            id="btn-download-patient",
            color="success",
            outline=True,
            size="sm",
            className="mb-1",
        ),
    ]


# --- Bootstrap: GitHub-dark leaning ---
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "NeoResist-MD · ResistanceLoop v1"
server = app.server

# Initial load for slider maxima
_INIT_DF, _INIT_PATH = load_qualified_candidates()
_INIT_AGG = _aggregate_patient_hla(_INIT_DF)
_max_mut = int(_INIT_AGG["mutations"].max()) if not _INIT_AGG.empty else 100
_max_fan = float(_INIT_AGG["fanout"].max()) if not _INIT_AGG.empty else 50.0
_max_cand = int(_INIT_AGG["candidates"].max()) if not _INIT_AGG.empty else 5000
_max_mut = max(_max_mut, 1)
_max_fan = max(_max_fan, 0.1)
_max_cand = max(_max_cand, 1)

_unique_hlas = sorted(_INIT_DF["hla_allele"].dropna().astype(str).unique().tolist()) if not _INIT_DF.empty else []
_excl_options = top_exclusion_options(_INIT_DF, 10)

app.layout = dbc.Container(
    fluid=True,
    class_name="app-shell",
    children=[
        dcc.Store(id="selected-key-store", data=None),
        dcc.Store(id="cohort-parquet-path", data=None),
        dcc.Store(id="upload-arm-store", data={"armed": False}),
        dcc.Interval(id="upload-interval", interval=1000, n_intervals=0, max_intervals=12, disabled=True),
        dcc.Download(id="download-patient-csv"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("NeoResist-MD", className="brand"),
                                html.Div(id="data-banner"),
                                html.Label("Navigation", className="text-muted"),
                                dcc.Dropdown(
                                    id="nav",
                                    options=[
                                        {"label": "Overview", "value": "Overview"},
                                        {"label": "Patient Explorer", "value": "Patient Explorer"},
                                        {"label": "Pipeline Status", "value": "Pipeline Status"},
                                        {"label": "About", "value": "About"},
                                    ],
                                    value="Overview",
                                    clearable=False,
                                    className="dash-dropdown mb-3",
                                ),
                                html.Label("Tier", className="text-muted"),
                                dcc.Dropdown(
                                    id="tier-filter",
                                    options=[{"label": TIER_LABELS[i], "value": i} for i in (1, 2, 3)],
                                    value=[1, 2, 3],
                                    multi=True,
                                    className="dash-dropdown mb-3",
                                ),
                                html.Label("HLA allele", className="text-muted"),
                                dcc.Dropdown(
                                    id="hla-filter",
                                    options=[{"label": h, "value": h} for h in _unique_hlas],
                                    value=_unique_hlas,
                                    multi=True,
                                    className="dash-dropdown mb-3",
                                ),
                                html.Label("Exclusion reasons (any)", className="text-muted"),
                                dcc.Dropdown(
                                    id="exclusion-filter",
                                    options=_excl_options,
                                    value=[],
                                    multi=True,
                                    className="dash-dropdown mb-3",
                                ),
                                html.Label("RL priority range", className="text-muted"),
                                dcc.RangeSlider(
                                    id="rl-range",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=[0, 1],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("Expression TPM range", className="text-muted"),
                                dcc.RangeSlider(
                                    id="expr-range",
                                    min=0,
                                    max=float(_INIT_DF["expression_tpm"].max() or 100) if not _INIT_DF.empty else 100.0,
                                    value=[0, float(_INIT_DF["expression_tpm"].max() or 100) if not _INIT_DF.empty else 100.0],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("CCF range", className="text-muted"),
                                dcc.RangeSlider(
                                    id="ccf-range",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=[0, 1],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("Mutations range (TMB proxy)", className="text-muted"),
                                dcc.RangeSlider(
                                    id="tmb-range",
                                    min=0,
                                    max=_max_mut,
                                    value=[0, _max_mut],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("Fanout range", className="text-muted"),
                                dcc.RangeSlider(
                                    id="fanout-range",
                                    min=0,
                                    max=_max_fan,
                                    value=[0, _max_fan],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("Candidates range", className="text-muted"),
                                dcc.RangeSlider(
                                    id="cand-range",
                                    min=0,
                                    max=_max_cand,
                                    value=[0, _max_cand],
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="mb-3",
                                ),
                                html.Label("Search", className="text-muted"),
                                dbc.Input(
                                    id="search",
                                    type="text",
                                    placeholder="Patient, gene, peptide, ...",
                                    class_name="mb-3",
                                ),
                                html.Label("Sort by", className="text-muted"),
                                dcc.Dropdown(
                                    id="sort-by",
                                    options=[
                                        {"label": "Mutations ↓", "value": "Mutations ↓"},
                                        {"label": "Candidates ↓", "value": "Candidates ↓"},
                                        {"label": "Fanout ↓", "value": "Fanout ↓"},
                                        {"label": "RL priority ↓", "value": "RL priority ↓"},
                                        {"label": "Tier ↑ (best first)", "value": "Tier ↑ (best first)"},
                                        {"label": "Mutations ↑", "value": "Mutations ↑"},
                                        {"label": "Candidates ↑", "value": "Candidates ↑"},
                                        {"label": "Fanout ↑", "value": "Fanout ↑"},
                                        {"label": "RL priority ↑", "value": "RL priority ↑"},
                                        {"label": "Tier ↓", "value": "Tier ↓"},
                                    ],
                                    value="Mutations ↓",
                                    clearable=False,
                                    className="dash-dropdown mb-3",
                                ),
                                dbc.Button("Reset all filters", id="reset-filters", color="primary", n_clicks=0, class_name="w-100"),
                            ]
                        ),
                        class_name="surface sidebar",
                    ),
                    lg=3,
                    md=4,
                    sm=12,
                ),
                dbc.Col(
                    [
                        html.H1("NeoResist-MD", className="title"),
                        html.Div(
                            "TCGA-SARC - ResistanceLoop v1 evidence-aware neoantigen qualification",
                            className="subtitle",
                        ),
                        dbc.Row(id="kpi-cards", class_name="kpi-row"),
                        html.Div(id="hla-coverage-row", className="mb-2"),
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="scatter", config={"displayModeBar": False})),
                            class_name="surface panel",
                        ),
                        dbc.Card(dbc.CardBody(id="evidence-panel"), class_name="surface panel"),
                        dbc.Card(dbc.CardBody(id="patient-detail"), class_name="surface panel"),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Patient × HLA cohort", className="mb-2"),
                                    dag.AgGrid(
                                        id="patient-grid",
                                        className="ag-theme-alpine-dark ag-grid-rl",
                                        columnDefs=[
                                            {"field": "patient_id", "headerName": "Patient", "minWidth": 160},
                                            {"field": "hla_allele", "headerName": "HLA", "minWidth": 120},
                                            {"field": "best_tier", "headerName": "Tier (best)", "maxWidth": 110},
                                            {"field": "mean_rl_priority", "headerName": "Mean RL", "maxWidth": 110},
                                            {"field": "mutations", "headerName": "Mutations"},
                                            {"field": "candidates", "headerName": "Candidates"},
                                            {"field": "fanout", "headerName": "Fanout"},
                                            {"field": "mean_expression_tpm", "headerName": "Mean TPM"},
                                            {"field": "mean_ccf", "headerName": "Mean CCF"},
                                            {"field": "mean_real_expr", "headerName": "Mean real TPM"},
                                            {"field": "mean_real_ccf", "headerName": "Mean real CCF"},
                                            {"field": "mean_purity_used", "headerName": "Mean purity°", "maxWidth": 120},
                                            {"field": "purity_source_mode", "headerName": "Purity src", "minWidth": 130},
                                            {"field": "hla_loh_status", "headerName": "LOH"},
                                            {
                                                "field": "top_exclusions",
                                                "headerName": "Top exclusions",
                                                "flex": 1,
                                                "wrapText": True,
                                                "autoHeight": True,
                                            },
                                        ],
                                        dashGridOptions={
                                            "rowSelection": "single",
                                            "animateRows": False,
                                        },
                                        getRowId={"function": "params.data.patient_id + '|' + params.data.hla_allele"},
                                        defaultColDef={
                                            "sortable": True,
                                            "filter": True,
                                            "resizable": True,
                                            "floatingFilter": False,
                                        },
                                        rowClassRules={
                                            "tier1-row": "Number(params.data.best_tier) === 1",
                                        },
                                        rowData=[],
                                        style={"height": "420px", "width": "100%"},
                                    ),
                                ]
                            ),
                            class_name="surface panel",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Upload MAF (stub)", className="mb-2"),
                                    dcc.Upload(
                                        id="upload-maf",
                                        children=dbc.Button("Upload MAF file", color="secondary", className="w-100"),
                                        multiple=False,
                                        className="w-100",
                                    ),
                                    html.Div(id="upload-status", className="mt-2 text-muted small"),
                                    html.Hr(className="border-secondary my-3"),
                                    html.H5("Phase 5 cohort parquet (optional)", className="mb-2"),
                                    html.P(
                                        "Load an alternative enriched_candidates.parquet from this machine "
                                        "(absolute path). Default search order still applies when empty.",
                                        className="text-muted small",
                                    ),
                                    dbc.Input(
                                        id="cohort-path-input",
                                        type="text",
                                        placeholder="e.g. C:/path/to/enriched_candidates.parquet",
                                        className="mb-2 font-monospace",
                                    ),
                                    dbc.Button("Use this cohort file", id="cohort-path-apply", color="info", outline=True, className="w-100"),
                                    html.Div(id="cohort-path-status", className="mt-2 text-muted small"),
                                ]
                            ),
                            class_name="surface panel",
                        ),
                        html.Div(id="footer", className="footer"),
                    ],
                    lg=9,
                    md=8,
                    sm=12,
                ),
            ],
            class_name="g-3",
        ),
    ],
)


@app.callback(
    Output("tmb-range", "value"),
    Output("fanout-range", "value"),
    Output("cand-range", "value"),
    Output("rl-range", "value"),
    Output("expr-range", "max"),
    Output("expr-range", "value"),
    Output("ccf-range", "value"),
    Output("search", "value"),
    Output("sort-by", "value"),
    Output("tier-filter", "value"),
    Output("hla-filter", "value"),
    Output("exclusion-filter", "value"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_n):
    df, _ = load_qualified_candidates()
    agg = _aggregate_patient_hla(df)
    mm = int(agg["mutations"].max()) if not agg.empty else 100
    mf = float(agg["fanout"].max()) if not agg.empty else 50.0
    mc = int(agg["candidates"].max()) if not agg.empty else 5000
    mm, mf, mc = max(mm, 1), max(mf, 0.1), max(mc, 1)
    hlas = sorted(df["hla_allele"].dropna().astype(str).unique().tolist()) if not df.empty else []
    if not df.empty and "expression_tpm" in df.columns:
        expr_hi = float(pd.to_numeric(df["expression_tpm"], errors="coerce").max() or 100.0)
    else:
        expr_hi = 100.0
    expr_hi = max(expr_hi, 1.0)
    return (
        [0, mm],
        [0, mf],
        [0, mc],
        [0, 1],
        expr_hi,
        [0.0, expr_hi],
        [0, 1],
        "",
        "Mutations ↓",
        [1, 2, 3],
        hlas,
        [],
    )


@app.callback(
    Output("cohort-parquet-path", "data"),
    Output("cohort-path-status", "children"),
    Input("cohort-path-apply", "n_clicks"),
    State("cohort-path-input", "value"),
    prevent_initial_call=True,
)
def apply_cohort_path(n_clicks, path_val: str | None):
    if not n_clicks:
        raise PreventUpdate
    raw = (path_val or "").strip()
    if not raw:
        return None, html.Span("Cleared: using default cohort file search.", className="text-info")
    p = Path(raw)
    if not p.is_file():
        return no_update, dbc.Alert(f"File not found: {p}", color="danger", className="py-2 mb-0")
    return str(p.resolve()), html.Span(f"Loaded: {p.name}", className="text-info")


@app.callback(
    Output("selected-key-store", "data"),
    Input("scatter", "clickData"),
    Input("patient-grid", "selectedRows"),
    prevent_initial_call=True,
)
def select_from_chart_or_grid(click, rows):
    tid = callback_context.triggered_id
    if tid == "scatter":
        if not click or not click.get("points"):
            raise PreventUpdate
        pt = click["points"][0]
        cd = pt.get("customdata")
        if cd is None or len(cd) < 2:
            raise PreventUpdate
        return {"patient_id": str(cd[0]), "hla_allele": str(cd[1])}
    if tid == "patient-grid":
        if not rows:
            raise PreventUpdate
        r0 = rows[0]
        if not r0.get("patient_id") or not r0.get("hla_allele"):
            raise PreventUpdate
        return {"patient_id": str(r0["patient_id"]), "hla_allele": str(r0["hla_allele"])}
    raise PreventUpdate


@app.callback(
    Output("upload-arm-store", "data"),
    Output("upload-interval", "disabled"),
    Output("upload-interval", "n_intervals"),
    Output("upload-status", "children"),
    Output("selected-key-store", "data", allow_duplicate=True),
    Input("upload-maf", "contents"),
    Input("upload-interval", "n_intervals"),
    State("upload-arm-store", "data"),
    State("upload-maf", "filename"),
    prevent_initial_call=True,
)
def upload_stub_orchestrator(contents, n_int, arm, filename):
    tid = callback_context.triggered_id
    if tid == "upload-maf":
        if not contents:
            raise PreventUpdate
        _fn = filename or "upload.maf"
        return {"armed": True, "filename": _fn}, False, 0, "Running ResistanceLoop qualification…", no_update
    if tid == "upload-interval":
        if not arm or not arm.get("armed"):
            raise PreventUpdate
        if n_int is None or n_int < 3:
            return arm, False, no_update, f"Running ResistanceLoop qualification… ({int(n_int or 0)}/3)", no_update
        df, _ = load_qualified_candidates()
        if df.empty:
            return {"armed": False}, True, "Stub finished (no cohort data loaded).", no_update
        row = df.iloc[0]
        demo = {"patient_id": str(row["patient_id"]), "hla_allele": str(row["hla_allele"])}
        return {"armed": False}, True, "Stub complete: first cohort row loaded as demo patient × HLA.", demo
    raise PreventUpdate


@app.callback(
    Output("download-patient-csv", "data"),
    Input("btn-download-patient", "n_clicks"),
    State("selected-key-store", "data"),
    prevent_initial_call=True,
)
def download_patient_csv(n_clicks, key):
    if not n_clicks or not key:
        raise PreventUpdate
    df, _ = load_qualified_candidates()
    pid, hla = key.get("patient_id"), key.get("hla_allele")
    sub = df[(df["patient_id"].astype(str) == str(pid)) & (df["hla_allele"].astype(str) == str(hla))]
    if sub.empty:
        raise dash.exceptions.PreventUpdate
    buf = io.StringIO()
    sub.to_csv(buf, index=False)
    fname = f"resistanceloop_{pid}_{hla}.csv".replace("/", "-")
    return dict(content=buf.getvalue(), filename=fname)


@app.callback(
    Output("data-banner", "children"),
    Output("kpi-cards", "children"),
    Output("hla-coverage-row", "children"),
    Output("scatter", "figure"),
    Output("evidence-panel", "children"),
    Output("patient-detail", "children"),
    Output("patient-grid", "rowData"),
    Output("footer", "children"),
    Input("nav", "value"),
    Input("tier-filter", "value"),
    Input("hla-filter", "value"),
    Input("exclusion-filter", "value"),
    Input("rl-range", "value"),
    Input("expr-range", "value"),
    Input("ccf-range", "value"),
    Input("tmb-range", "value"),
    Input("fanout-range", "value"),
    Input("cand-range", "value"),
    Input("search", "value"),
    Input("sort-by", "value"),
    Input("selected-key-store", "data"),
    State("cohort-parquet-path", "data"),
)
def update_dashboard(
    nav,
    tiers,
    hlas,
    exclusions,
    rl_range,
    expr_range,
    ccf_range,
    tmb,
    fanout,
    cand,
    search,
    sort_by,
    selected_key,
    cohort_path_override,
):
    df, path = load_qualified_candidates(alt_path=cohort_path_override)
    banner = html.Div()
    if path is None:
        banner = dbc.Alert(
            "No cohort parquet found. Prefer enriched_candidates.parquet under data/final/ "
            "(python -m backend.cli.enrich_cohort --stub) or qualified_candidates.parquet "
            "(python -m backend.cli.qualify_cohort).",
            color="warning",
            className="py-2 mb-2",
        )
    elif df.empty:
        banner = dbc.Alert("Parquet file is empty.", color="secondary", className="py-2 mb-2")

    tiers_l = [int(x) for x in (tiers or [])] or [1, 2, 3]
    hlas_l = [str(x) for x in (hlas or [])] if hlas else []
    excl_l = [str(x) for x in (exclusions or [])] if exclusions else []

    fc = filter_candidates(
        df,
        tiers=tiers_l,
        hlas=hlas_l,
        exclusion_any=excl_l,
        rl_range=list(rl_range or [0, 1]),
        expr_range=list(expr_range or [0, 1000]),
        ccf_range=list(ccf_range or [0, 1]),
        search=search or "",
    )
    agg = _aggregate_patient_hla(fc)
    agg_f = filter_agg(agg, list(tmb or [0, _max_mut]), list(fanout or [0, _max_fan]), list(cand or [0, _max_cand]))
    agg_s = sort_agg(agg_f, sort_by or "Mutations ↓")

    tier1_n = int((fc["tier"] == 1).sum()) if not fc.empty else 0
    tier2_n = int((fc["tier"] == 2).sum()) if not fc.empty else 0
    mean_rl = float(fc["rl_priority"].mean()) if not fc.empty else 0.0
    if math.isnan(mean_rl):
        mean_rl = 0.0
    pat_tier1 = int(fc.loc[fc["tier"] == 1, "patient_id"].nunique()) if not fc.empty else 0

    pur_missing = 0
    pur_by_src: Counter[str] = Counter()
    if not fc.empty and "purity_source_used" in fc.columns:
        for v in fc["purity_source_used"].astype(str):
            pur_by_src[v] += 1
        pur_missing = int((fc["purity_source_used"].astype(str) == "unresolved").sum())

    kpis = [
        dbc.Col(kpi_card(f"{tier1_n:,}", "Tier 1 candidates", accent=TIER_COLORS[1]), lg=2, md=4, sm=6),
        dbc.Col(kpi_card(f"{tier2_n:,}", "Tier 2 candidates", accent=TIER_COLORS[2]), lg=2, md=4, sm=6),
        dbc.Col(kpi_card(f"{mean_rl:.3f}", "Mean RL priority"), lg=2, md=4, sm=6),
        dbc.Col(kpi_card(f"{pat_tier1:,}", "Patients w/ Tier 1"), lg=2, md=4, sm=6),
        dbc.Col(kpi_card(f"{fc['patient_id'].nunique():,}" if not fc.empty else "0", "Patients (filtered)"), lg=2, md=4, sm=6),
        dbc.Col(kpi_card(f"{len(fc):,}", "Candidates (filtered)"), lg=2, md=4, sm=6),
    ]
    if not fc.empty and "purity_source_used" in fc.columns:
        top_src = pur_by_src.most_common(3)
        src_lbl = " · ".join(f"{k}:{v:,}" for k, v in top_src) if top_src else "—"
        kpis.append(
            dbc.Col(
                kpi_card(src_lbl, "Purity source (top)", accent="#8957e5"),
                lg=4,
                md=6,
                sm=12,
            )
        )
        kpis.append(
            dbc.Col(
                kpi_card(f"{pur_missing:,}", "Rows w/ unresolved purity"),
                lg=2,
                md=4,
                sm=6,
            )
        )

    cov = hla_coverage_badge_row(fc)

    if nav not in {"Overview", "Patient Explorer"}:
        empty = make_scatter_fig(pd.DataFrame())
        detail = [html.Div("Switch to Overview for cohort tools.", className="text-muted")]
        return (
            banner,
            kpis,
            cov,
            empty,
            detail,
            detail,
            [],
            "ResistanceLoop v1 · NeoResist-MD",
        )

    fig = make_scatter_fig(agg_s)

    pid = selected_key.get("patient_id") if isinstance(selected_key, dict) else None
    hla = selected_key.get("hla_allele") if isinstance(selected_key, dict) else None

    evidence = build_evidence_panel(fc, pid, hla)
    detail = build_patient_detail_card(fc, pid, hla)

    grid_rows = agg_s.to_dict("records")
    n_pat = int(df["patient_id"].nunique()) if not df.empty else 0
    n_hla = int(df["hla_allele"].nunique()) if not df.empty else 0
    tier1_pct = 100.0 * float((df["tier"] == 1).mean()) if not df.empty else 0.0
    mean_rl_all = float(df["rl_priority"].mean()) if not df.empty else 0.0
    if math.isnan(mean_rl_all):
        mean_rl_all = 0.0
    src = path or "qualified_candidates.parquet (missing)"
    footer = (
        f"Live cohort (full file): {n_pat:,} patients · {n_hla:,} HLA contexts · "
        f"Tier 1 share {tier1_pct:.1f}% · mean RL {mean_rl_all:.3f} · "
        f"Data: {src} · ResistanceLoop v1: evidence-aware neoantigen qualification"
    )

    return banner, kpis, cov, fig, evidence, detail, grid_rows, footer


if __name__ == "__main__":
    app.run(debug=False, port=8050)
