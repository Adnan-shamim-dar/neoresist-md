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
from neoresist.dash_app.data import _coerce_exclusion_list
from neoresist.dash_app.display_utils import (
    display_score_terms,
    display_text,
    ensure_display_columns,
    is_unknown_hla,
    normalize_hla_display,
    tier_badge_text,
)
from neoresist.tumor_features import normalise_tumor_type


def _ccf_clonality_class(ccf: float | None) -> str:
    if ccf is None or (isinstance(ccf, float) and math.isnan(ccf)):
        return "Unknown"
    if ccf >= 0.66:
        return "High (synthetic cohort estimate)"
    if ccf >= 0.33:
        return "Medium (synthetic cohort estimate)"
    return "Low (synthetic cohort estimate)"


def _candidate_immunogenicity(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    binding = 1.0 - pd.to_numeric(df.get("binding_rank", pd.Series([50.0] * len(df), index=idx)), errors="coerce").fillna(50.0).clip(lower=0, upper=100) / 100.0
    presentation = pd.to_numeric(df.get("presentation_composite", df.get("presentation_score", pd.Series([0.0] * len(df), index=idx))), errors="coerce").fillna(0.0).clip(0, 1)
    recognition = pd.to_numeric(df.get("recognition_score", pd.Series([0.0] * len(df), index=idx)), errors="coerce").fillna(0.0).clip(0, 1)
    expr = pd.to_numeric(df.get("expression_tpm", pd.Series([0.0] * len(df), index=idx)), errors="coerce").fillna(0.0)
    expr_max = max(float(expr.max()) if not expr.empty else 1.0, 1.0)
    expr = (expr / expr_max).clip(0, 1)
    return (0.30 * binding + 0.25 * presentation + 0.20 * expr + 0.25 * recognition).clip(0, 1)


def _scatter_color_series(df: pd.DataFrame, color_dimension: str) -> tuple[pd.Series, dict | None, str | None]:
    if color_dimension == "Expression Level":
        series = df.get("expression_flag", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.upper()
        return series, {"EXPRESSED": "#3FB950", "LOW": "#D29922", "ABSENT": "#F85149", "UNKNOWN": "#484F58"}, "category"
    if color_dimension == "Clonality":
        series = df.get("clonality_class", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.upper()
        return series, {"CLONAL": "#3FB950", "SUBCLONAL": "#D29922", "UNKNOWN": "#484F58"}, "category"
    if color_dimension == "LOH Status":
        series = df.get("hla_loh_status", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.upper()
        return series, {"LOH_DETECTED": "#F85149", "LOH_SUSPECTED": "#D29922", "INTACT": "#3FB950", "UNKNOWN": "#484F58"}, "category"
    if color_dimension == "Resistance Score":
        series = pd.to_numeric(df.get("resistance_composite", df.get("escape_penalty")), errors="coerce").fillna(0.0).clip(0, 1)
        return series, None, "continuous"
    series = df.get("tier", pd.Series(["EXCLUDED"] * len(df))).astype(str).str.upper().map(
        lambda v: "TIER_1" if v in {"1", "TIER_1"} else ("TIER_2" if v in {"2", "TIER_2"} else ("TIER_3" if v in {"3", "TIER_3"} else "EXCLUDED"))
    )
    return series, {"TIER_1": "#3FB950", "TIER_2": "#D29922", "TIER_3": "#F85149", "EXCLUDED": "#484F58"}, "category"


def _expression_score_series(df: pd.DataFrame) -> pd.Series:
    expression_norm = pd.to_numeric(df.get("expression_norm"), errors="coerce")
    if expression_norm.notna().any():
        return expression_norm.fillna(0.0).clip(0, 1)
    expr = pd.to_numeric(df.get("expression_tpm", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0)
    expr_max = max(float(expr.max()) if not expr.empty else 1.0, 1.0)
    return (expr / expr_max).clip(0, 1)


def _scatter_y_axis(plot_df: pd.DataFrame, requested_axis: str) -> tuple[str, pd.Series]:
    resistance = pd.to_numeric(plot_df.get("resistance_composite", pd.Series([0.0] * len(plot_df), index=plot_df.index)), errors="coerce").fillna(0.0).clip(0, 1)
    axis = requested_axis or "auto"
    if axis == "auto":
        axis = "Expression score" if resistance.nunique(dropna=True) <= 1 else "Resistance composite"
    if axis == "Expression score":
        return axis, _expression_score_series(plot_df)
    if axis == "Clonality":
        return axis, pd.to_numeric(plot_df.get("ccf", pd.Series([0.0] * len(plot_df), index=plot_df.index)), errors="coerce").fillna(0.0).clip(0, 1)
    if axis == "Self-dissimilarity":
        return axis, pd.to_numeric(plot_df.get("self_dissimilarity", pd.Series([0.0] * len(plot_df), index=plot_df.index)), errors="coerce").fillna(0.0).clip(0, 1)
    return "Resistance composite", resistance


def make_scatter_fig(
    agg: pd.DataFrame,
    color_dimension: str = "Tier",
    dragmode: str = "lasso",
    y_axis: str = "auto",
) -> go.Figure:
    dragmode = "select" if str(dragmode) == "select" else "lasso"
    candidate_mode = any(col in agg.columns for col in ("composite_priority", "recognition_score", "resistance_composite", "gene", "mutant_peptide"))
    if agg.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#8b949e", family="JetBrains Mono, monospace", size=12),
            title="No data — add enriched_candidates.parquet or qualified_candidates.parquet under data/final/",
            height=420,
            dragmode=dragmode,
        )
        return fig

    plot_df = agg.copy()
    if candidate_mode:
        plot_df = ensure_display_columns(plot_df)
        plot_df["immunogenicity_composite"] = _candidate_immunogenicity(plot_df)
        plot_df["resistance_composite"] = pd.to_numeric(
            plot_df.get("resistance_composite", pd.Series([None] * len(plot_df), index=plot_df.index)),
            errors="coerce",
        ).fillna(
            pd.to_numeric(plot_df.get("escape_penalty", pd.Series([0.0] * len(plot_df), index=plot_df.index)), errors="coerce")
        ).fillna(0.0).clip(0, 1)
        resolved_y_axis, y_series = _scatter_y_axis(plot_df, y_axis)
        plot_df["scatter_y"] = y_series
        color_series, color_map, scale_type = _scatter_color_series(plot_df, color_dimension)
        plot_df["color_value"] = color_series
        plot_df["size_metric"] = (pd.to_numeric(plot_df.get("expression_tpm", pd.Series([0.0] * len(plot_df), index=plot_df.index)), errors="coerce").fillna(0.0).clip(lower=0) + 1.0).astype(float)
        custom_df = pd.DataFrame(
            {
                "gene": plot_df.get("gene", pd.Series(["—"] * len(plot_df))).astype(str),
                "peptide": plot_df.get("mutant_peptide", pd.Series(["—"] * len(plot_df))).astype(str),
                "priority": pd.to_numeric(plot_df.get("composite_priority", plot_df.get("rl_priority", pd.Series([0.0] * len(plot_df), index=plot_df.index))), errors="coerce").fillna(0.0),
                "tier": plot_df.get("tier", pd.Series(["—"] * len(plot_df))).astype(str),
                "resistance": pd.to_numeric(plot_df["resistance_composite"], errors="coerce").fillna(0.0),
                "y_value": pd.to_numeric(plot_df["scatter_y"], errors="coerce").fillna(0.0),
            }
        )
        fig = go.Figure()
        if scale_type == "continuous":
            fig.add_trace(
                go.Scatter(
                    x=plot_df["immunogenicity_composite"],
                    y=plot_df["scatter_y"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=plot_df["color_value"],
                        opacity=plot_df.get("marker_opacity", pd.Series([0.8] * len(plot_df))).tolist(),
                        symbol=["x" if bool(v) else "circle" for v in plot_df.get("data_quality_zero", pd.Series([False] * len(plot_df)))],
                        colorscale=[[0, "#3FB950"], [0.5, "#D29922"], [1, "#F85149"]],
                        cmin=0,
                        cmax=1,
                        showscale=True,
                        colorbar=dict(title="Resistance"),
                        line=dict(width=1, color="#30363d"),
                    ),
                    customdata=custom_df.to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Peptide: %{customdata[1]}<br>"
                        f"Priority: %{{customdata[2]:.3f}}<br>Tier: %{{customdata[3]}}<br>{resolved_y_axis}: %{{customdata[5]:.3f}}<br>Resistance: %{{customdata[4]:.3f}}<extra></extra>"
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["immunogenicity_composite"],
                    y=plot_df["scatter_y"],
                    mode="markers",
                    marker=dict(
                        size=plot_df["size_metric"].clip(6, 20),
                        color=plot_df["color_value"].map(color_map).fillna("#484F58"),
                        opacity=plot_df.get("marker_opacity", pd.Series([0.8] * len(plot_df))).tolist(),
                        symbol=["x" if bool(v) else "circle" for v in plot_df.get("data_quality_zero", pd.Series([False] * len(plot_df)))],
                        line=dict(width=1, color="#30363d"),
                    ),
                    customdata=custom_df.to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Peptide: %{customdata[1]}<br>"
                        f"Priority: %{{customdata[2]:.3f}}<br>Tier: %{{customdata[3]}}<br>{resolved_y_axis}: %{{customdata[5]:.3f}}<br>Resistance: %{{customdata[4]:.3f}}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(family="JetBrains Mono, monospace", color="#e6edf3", size=12),
            dragmode=dragmode,
            margin=dict(t=48, l=20, r=20, b=20),
            height=420,
            xaxis=dict(title="Immunogenicity composite", gridcolor="#21262d", linecolor="#30363d", range=[0, 1]),
            yaxis=dict(title=resolved_y_axis, gridcolor="#21262d", linecolor="#30363d", range=[0, 1]),
        )
        if resolved_y_axis == "Resistance composite":
            for yy, label in [(0.3, "Low-risk guide"), (0.6, "T2 threshold"), (0.7, "T1 threshold")]:
                fig.add_hline(y=yy, line=dict(color="#484F58", dash="dash", width=1))
                fig.add_annotation(x=0.99, y=yy, xref="paper", yref="y", text=label, showarrow=False, font=dict(color="#8b949e", size=10), xanchor="right")
        for xx, label in [(0.4, "T2 threshold"), (0.7, "T1 threshold")]:
            fig.add_vline(x=xx, line=dict(color="#484F58", dash="dash", width=1))
            fig.add_annotation(x=xx, y=0.99, xref="x", yref="paper", text=label, showarrow=False, font=dict(color="#8b949e", size=10), yanchor="top", textangle=0)
        fig.add_annotation(
            x=0.99,
            y=0.99,
            xref="paper",
            yref="paper",
            text="Ideal: High immunogenicity, Low resistance",
            showarrow=False,
            font=dict(color="#8b949e", size=11),
            xanchor="right",
            yanchor="top",
        )
        if y_axis == "auto" and resolved_y_axis != "Resistance composite":
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=f"Auto-selected Y axis: {resolved_y_axis}",
                showarrow=False,
                font=dict(color="#8b949e", size=10),
                xanchor="left",
                yanchor="top",
            )
        return fig

    plot_df["tier_label"] = plot_df["best_tier"].map(TIER_LABELS).fillna("Tier 3")
    plot_df["size_candidates"] = plot_df["candidates"].clip(lower=1)
    plot_df["size_metric"] = plot_df["size_candidates"].apply(lambda v: math.log10(float(v) + 1.0))
    plot_df["top_exclusions_hover"] = plot_df["top_exclusions"].fillna("").astype(str)
    fig = go.Figure()
    opacity_map = {"full": 1.0, "partial": 0.6, "minimal": 0.3}
    for tier_value in [1, 2, 3]:
        sub = plot_df[plot_df["best_tier"] == tier_value]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["mutations"],
                y=sub["mean_rl_priority"],
                mode="markers",
                name=TIER_LABELS[tier_value],
                customdata=sub[["patient_id", "hla_allele", "best_tier", "mean_rl_priority", "size_candidates", "top_exclusions_hover"]].to_numpy(),
                marker=dict(
                    size=(sub["size_metric"] * 12).clip(lower=5, upper=24),
                    color=TIER_COLORS[tier_value],
                    opacity=sub.get("score_confidence", pd.Series(["partial"] * len(sub))).map(opacity_map).fillna(0.6).tolist(),
                    symbol=["x" if bool(v) else "circle" for v in sub.get("data_quality_zero", pd.Series([False] * len(sub)))],
                    line=dict(width=1.1, color="rgba(255,255,255,0.22)"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>HLA: %{customdata[1]}<br>"
                    "Tier (best): %{customdata[2]}<br>Mean RL: %{customdata[3]:.3f}<br>"
                    "Candidates: %{customdata[4]:,}<br>"
                    "Exclusions: %{customdata[5]}<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="#111827",
                    bordercolor="#4CC9F0",
                    font=dict(color="#E5E7EB", family="JetBrains Mono, monospace", size=12),
                ),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono, monospace", color="#e6edf3", size=12),
        xaxis=dict(type="log", gridcolor="#21262d", linecolor="#30363d", title="Mutations (unique loci, log scale)"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", title="Mean RL priority", range=[-0.02, 1.02]),
        legend_title_text="Tier (best)",
        margin=dict(t=48, l=20, r=20, b=20),
        height=420,
        dragmode=dragmode,
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
    hla_values = cand["hla_allele"].map(normalize_hla_display)
    if hla_values.map(is_unknown_hla).all():
        return html.Div("HLA coverage · 0% (no HLA typing)", className="text-muted small")
    vc = hla_values.astype(str).value_counts().head(12)
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
    tumor_type = normalise_tumor_type(rep.get("tumor_type", "UNKNOWN"))
    msi_status = str(rep.get("msi_status", "") or "").upper()
    immuno, resistance, score = display_score_terms(rep)
    rl_calc = f"immunogenicity={immuno:.3f} × (1−resistance={resistance:.3f}) = {score:.3f}"
    tpm = float(rep.get("expression_tpm") or 0.0)
    ps = float(rep.get("presentation_score") or 0.0)
    ccf = float(rep.get("ccf") or 0.0)
    loh = str(rep.get("hla_loh_status", "—"))
    sd = float(rep.get("self_dissimilarity") or 0.0)

    badge_items: list[tuple[str, str, str]] = [
        ("TPM (cohort model)", f"{tpm:.2f}", "info"),
        ("Presentation", f"{ps:.3f}", "primary"),
        ("VAF-derived CCF proxy", f"{ccf:.3f}", "warning"),
        ("LOH", loh, "danger" if loh == "lost" else "secondary"),
        ("Self-dissim.", f"{sd:.3f}", "success"),
    ]
    rtpm = rep.get("real_expression_tpm")
    if rtpm is not None and not pd.isna(rtpm):
        badge_items.insert(1, ("Real TPM", f"{float(rtpm):.2f}", "info"))
    rccf = rep.get("real_ccf")
    if rccf is not None and not pd.isna(rccf):
        badge_items.insert(4 if len(badge_items) > 4 else len(badge_items), ("Real VAF-derived CCF proxy", f"{float(rccf):.3f}", "warning"))
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

    hla_allele = normalize_hla_display(hla_allele)
    hla_unknown = is_unknown_hla(hla_allele)
    tier_label = tier_badge_text(rep.get("tier")).replace("_", " ")

    return [
        html.H5("Evidence (ResistanceLoop v1)", className="mb-2"),
        html.Div(
            [
                dbc.Alert(
                    "HLA type unknown for this patient. Presentation and binding predictions use pan-allele estimates and may be less precise. Provide HLA typing data to improve candidate ranking accuracy.",
                    color="warning",
                    className="mb-3",
                )
                if hla_unknown
                else html.Div(),
                dbc.Alert(
                    [
                        html.Strong("What ResistanceLoop means: "),
                        "higher scores indicate candidates with stronger presentation/expression/clonality support and fewer obvious escape routes such as LOH or unstable subclonality.",
                    ],
                    color="info",
                    className="mb-3",
                ),
                html.Div(
                    [
                        html.Strong("Selection: "),
                        html.Span(f"{patient_id} · {hla_allele}", className="font-monospace me-2"),
                        dbc.Badge(f"Tumor type: {tumor_type}", color="info", className="me-1 mb-1"),
                        dbc.Badge(
                            f"MSI: {msi_status}",
                            color="warning" if msi_status == "MSI-H" else "success" if msi_status == "MSS" else "secondary",
                            className="me-1 mb-1",
                        )
                        if msi_status
                        else html.Span(),
                    ],
                    className="mb-2",
                ),
                evidence_badges_row("", badge_items),
                html.Hr(className="my-2 border-secondary"),
                html.Div(
                    [
                        html.Div("Representative row (max RL in selection)", className="text-muted small"),
                        html.Div(
                            f"ResistanceLoop score: {score:.3f} / 1.0 · {tier_label}",
                            className="mt-1",
                        ),
                        html.Div(f"Clonality class: {_ccf_clonality_class(ccf)}", className="mt-1"),
                        html.Div(f"RL components: {rl_calc}", className="mt-1 small text-break"),
                        html.Div(
                            f"Presentation: {ps:.3f}" + (" (pan-allele est.)" if str(rep.get("hla_confidence") or "") == "pan-allele estimate" else ""),
                            className="mt-1 small",
                        ),
                        html.Div(
                            "Biology: the score rises when the peptide looks visible and durable, and falls when escape routes look plausible.",
                            className="mt-1 text-muted small",
                        ),
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

    hla_allele = normalize_hla_display(hla_allele)
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

    candidate_preview = sub.nlargest(8, "rl_priority")[["gene", "mutant_peptide", "rl_priority", "tier"]].copy()
    show_coherence = "coherence_score" in sub.columns and pd.to_numeric(sub["coherence_score"], errors="coerce").notna().any()
    if show_coherence:
        candidate_preview["coherence_score"] = sub.nlargest(8, "rl_priority")["coherence_score"].tolist()
    preview_rows = []
    for _, row in candidate_preview.iterrows():
        row_cells = [
            html.Td(str(row.get("gene", ""))),
            html.Td(display_text(row.get("mutant_peptide"), "—"), className="font-monospace"),
            html.Td(f"{float(row.get('rl_priority', 0.0)):.3f}"),
            html.Td(tier_badge_text(row.get("tier")).replace("_", " ")),
        ]
        if show_coherence:
            row_cells.append(html.Td(f"{float(row.get('coherence_score', 0.0)):.3f}"))
        preview_rows.append(
            html.Tr(row_cells)
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
        html.Hr(className="border-secondary my-3"),
        html.H5("Top candidate rows", className="mb-2"),
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Gene"), html.Th("Peptide"), html.Th("RL score"), html.Th("Tier")]
                        + ([html.Th("Coherence")] if show_coherence else [])
                    )
                ),
                html.Tbody(preview_rows),
            ],
            bordered=False,
            size="sm",
            className="mb-0 text-light",
        ),
    ]
