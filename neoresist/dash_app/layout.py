from __future__ import annotations

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from neoresist.dash_app.constants import TIER_LABELS
from neoresist.dash_app.data import (
    load_qualified_candidates,
    set_slider_bounds_from_df,
    top_exclusion_options,
)
from neoresist.loaders import CohortLoadError, load_cohort_for_dash


def build_layout() -> dbc.Container:
    try:
        init_df, _init_path, _searched = load_cohort_for_dash(alt_path=None)
    except CohortLoadError:
        init_df, _init_path = load_qualified_candidates()
    set_slider_bounds_from_df(init_df)

    from neoresist.dash_app.data import slider_cand_max, slider_fan_max, slider_mut_max

    _max_mut = slider_mut_max()
    _max_fan = slider_fan_max()
    _max_cand = slider_cand_max()

    expr_max = float(init_df["expression_tpm"].max() or 100) if not init_df.empty else 100.0
    expr_max = round(expr_max, 1)
    expr_max = max(expr_max, 1.0)
    expr_value = [0.0, expr_max]

    unique_hlas = sorted(init_df["hla_allele"].dropna().astype(str).unique().tolist()) if not init_df.empty else []
    excl_options = top_exclusion_options(init_df, 10)

    return dbc.Container(
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
                                        options=[{"label": h, "value": h} for h in unique_hlas],
                                        value=unique_hlas,
                                        multi=True,
                                        className="dash-dropdown mb-3",
                                    ),
                                    html.Label("Exclusion reasons (any)", className="text-muted"),
                                    dcc.Dropdown(
                                        id="exclusion-filter",
                                        options=excl_options,
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
                                        max=expr_max,
                                        step=0.1,
                                        value=expr_value,
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
                                        step=0.1,
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
                                id="header-subtitle",
                                children="TCGA-SARC — ResistanceLoop v1 evidence-aware neoantigen qualification",
                                className="subtitle",
                            ),
                            dbc.Collapse(
                                dbc.Card(
                                    dbc.CardBody(
                                        html.Div(
                                            id="methodology-panel",
                                            children="Loading profile metadata…",
                                            className="small text-muted",
                                        )
                                    ),
                                    class_name="surface panel mb-2",
                                ),
                                id="methodology-collapse",
                                is_open=False,
                            ),
                            dbc.Button(
                                "Methodology / active profile",
                                id="methodology-toggle",
                                color="link",
                                className="px-0 mb-2 text-info",
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
                                        html.H5("Single-patient upload (preview)", className="mb-2"),
                                        html.P(
                                            "Upload a MAF for future single-sample scoring. "
                                            "Cohort scoring is unchanged; pipeline hook is not enabled in this build.",
                                            className="text-muted small",
                                        ),
                                        dcc.Upload(
                                            id="upload-maf",
                                            children=dbc.Button("Choose MAF file", color="secondary", className="w-100"),
                                            multiple=False,
                                            className="w-100",
                                        ),
                                        html.Div(id="upload-status", className="mt-2 text-muted small"),
                                        html.Hr(className="border-secondary my-3"),
                                        html.H5("Alternative cohort Parquet (optional)", className="mb-2"),
                                        html.P(
                                            "Absolute path to enriched_candidates.parquet on this machine. "
                                            "When empty, default search order applies.",
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
