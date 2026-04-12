from __future__ import annotations

import math
from collections import Counter
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from neoresist.dash_app.constants import TIER_COLORS, TIER_LABELS
from neoresist.dash_app.data import _coerce_exclusion_list, _rl_breakdown_terms


def _ccf_clonality_class(ccf: float | None) -> str:
    if ccf is None or (isinstance(ccf, float) and math.isnan(ccf)):
        return "Unknown"
    if ccf >= 0.66:
        return "High (synthetic cohort estimate)"
    if ccf >= 0.33:
        return "Medium (synthetic cohort estimate)"
    return "Low (synthetic cohort estimate)"


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
    # Compress extreme outliers so one hyper-expanded case does not hide the cohort.
    plot_df["size_metric"] = plot_df["size_candidates"].apply(lambda v: math.log10(float(v) + 1.0))
    plot_df["top_exclusions_hover"] = plot_df["top_exclusions"].fillna("").astype(str)

    fig = px.scatter(
        plot_df,
        x="mutations",
        y="mean_rl_priority",
        size="size_metric",
        size_max=24,
        color="best_tier",
        color_discrete_map={1: TIER_COLORS[1], 2: TIER_COLORS[2], 3: TIER_COLORS[3]},
        hover_name="patient_id",
        custom_data=["patient_id", "hla_allele", "best_tier", "mean_rl_priority", "size_candidates", "top_exclusions_hover"],
        category_orders={"best_tier": [1, 2, 3]},
        title="Mutations vs mean RL priority (per patient × HLA)",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>HLA: %{customdata[1]}<br>"
            "Tier (best): %{customdata[2]}<br>Mean RL: %{customdata[3]:.3f}<br>"
            "Candidates: %{customdata[4]:,}<br>"
            "Exclusions: %{customdata[5]}<extra></extra>"
        ),
        marker=dict(line=dict(width=1.1, color="rgba(255,255,255,0.22)"), opacity=0.82, sizemin=5),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#4CC9F0",
            font=dict(color="#E5E7EB", family="JetBrains Mono, monospace", size=12),
        ),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono, monospace", color="#e6edf3", size=12),
        xaxis=dict(
            type="log",
            gridcolor="#21262d",
            linecolor="#30363d",
            title="Mutations (unique loci, log scale)",
        ),
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
        ("TPM (cohort model)", f"{tpm:.2f}", "info"),
        ("Presentation", f"{ps:.3f}", "primary"),
        ("CCF (cohort model)", f"{ccf:.3f}", "warning"),
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
