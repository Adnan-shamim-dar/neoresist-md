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
            dcc.Store(id="strategy-edit-store", data=None),
            dcc.Store(id="strategy-refresh-store", data={"revision": 0}),
            dcc.Store(id="upload-result-store", data=None),
            dcc.Store(id="active-case-store", data=None),
            dcc.Interval(id="case-status-interval", interval=3000, n_intervals=0, disabled=True),
            dcc.Download(id="download-patient-csv"),
            dcc.Download(id="download-canonical-csv"),
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
                                            {"label": "Patients", "value": "Patients"},
                                            {"label": "Upload", "value": "Upload"},
                                            {"label": "Cases", "value": "Cases"},
                                            {"label": "Advanced Strategies", "value": "Advanced Strategies"},
                                            {"label": "Pipeline Status", "value": "Pipeline Status"},
                                            {"label": "About", "value": "About"},
                                        ],
                                        value="Overview",
                                        clearable=False,
                                        className="dash-dropdown mb-3",
                                    ),
                                    html.Label("Mode", className="text-muted"),
                                    dbc.RadioItems(
                                        id="ui-mode",
                                        options=[
                                            {"label": "Simple", "value": "Simple"},
                                            {"label": "Expert", "value": "Expert"},
                                        ],
                                        value="Simple",
                                        inline=True,
                                        class_name="mb-3 mode-toggle",
                                    ),
                                    html.Div(id="sidebar-context-note", className="small text-muted mb-3"),
                                    html.Div(
                                        id="cohort-sidebar-controls",
                                        children=[
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
                                        ],
                                    ),
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
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Answer-first workspace", className="mb-2"),
                                            html.P(
                                                "Simple mode hides machinery and focuses on the best candidates, evidence cards, and export-ready outputs.",
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                    class_name="surface panel mb-2",
                                ),
                                id="simple-hero-shell",
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Full pipeline workspace", className="mb-2"),
                                            html.P(
                                                "Expert mode exposes modules, strategy controls, diagnostics, raw artifacts, and the deeper execution workflow.",
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                    class_name="surface panel mb-2",
                                ),
                                id="expert-hero-shell",
                            ),
                            html.Div(id="demo-mode-banner"),
                            html.Div(
                                [
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
                                ],
                                id="expert-methodology-shell",
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5("Active modular engine", className="mb-2"),
                                            html.Div(id="strategy-panel", className="small text-muted"),
                                            dbc.Button(
                                                "Open Advanced Strategies",
                                                id="jump-to-advanced-btn",
                                                color="info",
                                                outline=True,
                                                class_name="mt-3",
                                            ),
                                        ]
                                    ),
                                    class_name="surface panel mb-2",
                                ),
                                id="expert-strategy-shell",
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Advanced Strategies", className="mb-2"),
                                            html.P(
                                                "This workspace belongs to Expert mode. Switch the global mode toggle to Expert to inspect strategy comparisons, consensus, audit, and controlled weight tuning.",
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                    class_name="surface panel mb-2",
                                ),
                                id="advanced-strategies-simple-note",
                                style={"display": "none"},
                            ),
                            html.Div(
                                dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4("Advanced Strategies", className="mb-2"),
                                        html.P(
                                            "This is the modular scoring workspace: strategy library, comparison, consensus, audit, and controlled tuning. "
                                            "The clinical surfaces stay simpler; the proprietary engine lives here.",
                                            className="text-muted small mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div("Level 1", className="text-uppercase small text-muted mb-2"),
                                                                html.H5("Active strategy and thesis", className="mb-2"),
                                                                html.Div(id="advanced-strategy-summary"),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div("Level 2", className="text-uppercase small text-muted mb-2"),
                                                                html.H5("Saved/default strategy library", className="mb-2"),
                                                                html.Div(id="strategy-library-panel"),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.Div("Level 3", className="text-uppercase small text-muted mb-2"),
                                                                html.H5("Why modularity is the moat", className="mb-2"),
                                                                html.P(
                                                                    "Profiles, consensus, and auditability let the ranking logic evolve without rewriting the platform. "
                                                                    "Saved strategies become durable product knowledge.",
                                                                    className="mb-0",
                                                                ),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=4,
                                                ),
                                            ],
                                            class_name="g-3 mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Active strategy", className="text-muted"),
                                                        dcc.Dropdown(
                                                            id="active-strategy-select",
                                                            options=[],
                                                            value=None,
                                                            clearable=False,
                                                            className="dash-dropdown mb-3",
                                                        ),
                                                        html.Label("Compare strategies", className="text-muted"),
                                                        dcc.Dropdown(
                                                            id="compare-strategies-select",
                                                            options=[],
                                                            value=[],
                                                            multi=True,
                                                            className="dash-dropdown mb-3",
                                                        ),
                                                        dbc.Button("Clone active strategy", id="clone-strategy-btn", color="info", class_name="me-2"),
                                                        dbc.Button("Reset editor", id="reset-strategy-editor-btn", color="secondary", outline=True),
                                                    ],
                                                    lg=5,
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(id="strategy-metadata-panel", className="small"),
                                                        html.Div(id="strategy-save-status", className="small mt-2"),
                                                    ],
                                                    lg=7,
                                                    md=12,
                                                ),
                                            ],
                                            class_name="g-3 mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Strategy name", className="text-muted"),
                                                        dbc.Input(id="strategy-name-input", type="text", class_name="mb-2"),
                                                        html.Label("Description", className="text-muted"),
                                                        dcc.Textarea(
                                                            id="strategy-description-input",
                                                            className="w-100 mb-2",
                                                            style={"minHeight": "100px", "background": "#21262d", "color": "#e6edf3", "border": "1px solid #30363d", "padding": "8px"},
                                                        ),
                                                        html.Label("Expression weight", className="text-muted"),
                                                        dbc.Input(id="weight-expression-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Presentation weight", className="text-muted"),
                                                        dbc.Input(id="weight-presentation-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("CCF weight", className="text-muted"),
                                                        dbc.Input(id="weight-ccf-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                    ],
                                                    lg=4,
                                                    md=6,
                                                    sm=12,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Self-dissimilarity weight", className="text-muted"),
                                                        dbc.Input(id="weight-self-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Escape penalty weight", className="text-muted"),
                                                        dbc.Input(id="escape-penalty-input", type="number", min=-1, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Blend real expression", className="text-muted"),
                                                        dbc.Input(id="blend-expression-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Blend real CCF", className="text-muted"),
                                                        dbc.Input(id="blend-ccf-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Expression TPM cap", className="text-muted"),
                                                        dbc.Input(id="expression-cap-input", type="number", min=1, step=1, class_name="mb-2"),
                                                    ],
                                                    lg=4,
                                                    md=6,
                                                    sm=12,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Tier 1 threshold", className="text-muted"),
                                                        dbc.Input(id="tier1-threshold-input", type="number", min=0, max=1, step=0.01, class_name="mb-2"),
                                                        html.Label("Tier 2 threshold", className="text-muted"),
                                                        dbc.Input(id="tier2-threshold-input", type="number", min=0, max=1, step=0.01, class_name="mb-3"),
                                                        dbc.Button("Save strategy", id="save-strategy-btn", color="success", class_name="me-2"),
                                                        dbc.Button("Use active defaults", id="load-active-strategy-btn", color="secondary", outline=True),
                                                        html.Hr(className="border-secondary my-3"),
                                                        html.Div(id="strategy-coherence-summary", className="small"),
                                                    ],
                                                    lg=4,
                                                    md=12,
                                                    sm=12,
                                                ),
                                            ],
                                            class_name="g-3 mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Strategy comparison", className="mb-2"),
                                                                html.Div(id="strategy-comparison-table"),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    lg=6,
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Consensus ranking", className="mb-2"),
                                                                html.Div(id="strategy-consensus-table"),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    lg=6,
                                                    md=12,
                                                ),
                                            ],
                                            class_name="g-3 mb-3",
                                        ),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H5("Audit trail", className="mb-2"),
                                                    html.Div(id="strategy-audit-table"),
                                                ]
                                            ),
                                            class_name="surface panel",
                                        ),
                                    ]
                                ),
                                class_name="surface panel mb-2",
                                ),
                                id="advanced-strategies-section",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H4("Resistance-aware neoantigen review", className="mb-2"),
                                                            html.P(
                                                                "NeoResist-MD prioritizes candidates not just by recognition, but by how likely they are to remain visible under tumor escape pressure.",
                                                                className="lead mb-2",
                                                            ),
                                                            html.Div(id="overview-modularity-summary"),
                                                            html.Div(id="overview-case-summary", className="mt-3"),
                                                        ]
                                                    ),
                                                    class_name="surface panel",
                                                ),
                                                md=12,
                                            )
                                        ],
                                        class_name="g-3",
                                    ),
                                    html.Div(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Module stack panel", className="mb-2"),
                                                                html.P(
                                                                    "Pipeline cards with tool selectors and per-module influence sliders. Save creates a versioned strategy.",
                                                                    className="small text-muted mb-3",
                                                                ),
                                                                html.Div(id="expert-module-stack-panel"),
                                                                html.Label("New strategy name", className="text-muted mt-2"),
                                                                dbc.Input(
                                                                    id="expert-strategy-name-input",
                                                                    type="text",
                                                                    placeholder="e.g. Immuno-heavy v2",
                                                                    class_name="mb-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Save versioned strategy",
                                                                    id="expert-save-strategy-version-btn",
                                                                    color="success",
                                                                    class_name="me-2",
                                                                ),
                                                                html.Div(id="expert-strategy-save-status", className="small mt-2"),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    lg=6,
                                                    md=12,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H5("Strategy library browser", className="mb-2"),
                                                                    html.Div(id="expert-strategy-library-browser"),
                                                                    html.Hr(className="border-secondary my-3"),
                                                                    dbc.Button(
                                                                        "Download raw canonical table",
                                                                        id="download-canonical-btn",
                                                                        color="info",
                                                                        outline=True,
                                                                        class_name="mb-2",
                                                                    ),
                                                                    html.Div(
                                                                        "Exports the current canonical long table (post-filter) as CSV.",
                                                                        className="small text-muted",
                                                                    ),
                                                                ]
                                                            ),
                                                            class_name="surface panel mb-3",
                                                        ),
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6("AutoResearch digest", className="mb-2"),
                                                                    html.Div(
                                                                        "Placeholder: external retrieval/ranking digest will appear here when configured.",
                                                                        className="small text-muted",
                                                                    ),
                                                                ]
                                                            ),
                                                            class_name="surface panel mb-3",
                                                        ),
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6("External tool upload wizard", className="mb-2"),
                                                                    html.Div(
                                                                        "Placeholder: guided adapter/runtime upload and validation flow.",
                                                                        className="small text-muted",
                                                                    ),
                                                                ]
                                                            ),
                                                            class_name="surface panel mb-3",
                                                        ),
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6("Module confidence cards", className="mb-2"),
                                                                    html.Div(id="expert-confidence-cards"),
                                                                ]
                                                            ),
                                                            class_name="surface panel",
                                                        ),
                                                    ],
                                                    lg=6,
                                                    md=12,
                                                ),
                                            ],
                                            class_name="g-3",
                                        ),
                                        id="expert-two-panel-shell",
                                    ),
                                    dbc.Row(id="kpi-cards", class_name="kpi-row"),
                                    html.Div(id="hla-coverage-row", className="mb-2"),
                                    html.Div(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H5("Recommended candidates", className="mb-2"),
                                                    html.P(
                                                        "Simple mode shows the answer first: the highest-priority candidates under the active filters.",
                                                        className="text-muted small mb-3",
                                                    ),
                                                    html.Div(id="simple-answer-panel"),
                                                ]
                                            ),
                                            class_name="surface panel",
                                        ),
                                        id="simple-answer-shell",
                                    ),
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("RL priority map", className="mb-2"),
                                                dcc.Graph(id="scatter", config={"displayModeBar": False}),
                                            ]
                                        ),
                                        id="scatter-card",
                                        class_name="surface panel",
                                    ),
                                    dbc.Card(dbc.CardBody(id="evidence-panel"), class_name="surface panel"),
                                    dbc.Card(dbc.CardBody(id="patient-detail"), class_name="surface panel"),
                                    html.Div(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Tier mix", className="mb-2"),
                                                                dcc.Graph(id="expert-tier-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Evidence blend", className="mb-2"),
                                                                dcc.Graph(id="expert-evidence-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Tier distribution", className="mb-2"),
                                                                dcc.Graph(id="expert-tier-pie-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            class_name="g-3",
                                        ),
                                        id="expert-diagnostics-shell",
                                    ),
                                    html.Div(
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Module status distribution", className="mb-2"),
                                                                dcc.Graph(id="expert-module-health-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Strategy weight profile", className="mb-2"),
                                                                dcc.Graph(id="expert-strategy-delta-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Pipeline waterfall", className="mb-2"),
                                                                dcc.Graph(id="expert-waterfall-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Cohort module heatmap", className="mb-2"),
                                                                dcc.Graph(id="expert-heatmap-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Confidence overview", className="mb-2"),
                                                                dcc.Graph(id="expert-confidence-overview-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Resistance breakdown", className="mb-2"),
                                                                dcc.Graph(id="expert-resistance-breakdown-fig", config={"displayModeBar": False}),
                                                            ]
                                                        ),
                                                        class_name="surface panel h-100",
                                                    ),
                                                    md=12,
                                                ),
                                            ],
                                            class_name="g-3",
                                        ),
                                        id="expert-extra-diagnostics-shell",
                                    ),
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
                                                        {"field": "mean_rl_priority", "headerName": "Mean RL score", "maxWidth": 130},
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
                                ],
                                id="overview-section",
                                style={"display": "block"},
                            ),
                            html.Div(
                                [
                                    dbc.Card(dbc.CardBody(id="patient-detail-standalone"), class_name="surface panel"),
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H5("Patient × HLA cohort", className="mb-2"),
                                                dag.AgGrid(
                                                    id="patient-grid-standalone",
                                                    className="ag-theme-alpine-dark ag-grid-rl",
                                                    columnDefs=[
                                                        {"field": "patient_id", "headerName": "Patient", "minWidth": 160},
                                                        {"field": "hla_allele", "headerName": "HLA", "minWidth": 120},
                                                        {"field": "best_tier", "headerName": "Tier (best)", "maxWidth": 110},
                                                        {"field": "mean_rl_priority", "headerName": "Mean RL score", "maxWidth": 130},
                                                        {"field": "mutations", "headerName": "Mutations"},
                                                        {"field": "candidates", "headerName": "Candidates"},
                                                        {"field": "fanout", "headerName": "Fanout"},
                                                        {"field": "mean_expression_tpm", "headerName": "Mean TPM"},
                                                        {"field": "hla_loh_status", "headerName": "LOH"},
                                                        {"field": "top_exclusions", "headerName": "Top exclusions", "flex": 1, "wrapText": True, "autoHeight": True},
                                                    ],
                                                    dashGridOptions={"rowSelection": "single", "animateRows": False},
                                                    getRowId={"function": "params.data.patient_id + '|' + params.data.hla_allele"},
                                                    defaultColDef={"sortable": True, "filter": True, "resizable": True, "floatingFilter": False},
                                                    rowClassRules={"tier1-row": "Number(params.data.best_tier) === 1"},
                                                    rowData=[],
                                                    style={"height": "520px", "width": "100%"},
                                                ),
                                            ]
                                        ),
                                        class_name="surface panel",
                                    ),
                                ],
                                id="patients-section",
                                style={"display": "none"},
                            ),
                            html.Div(
                                dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Single-patient upload", className="mb-2"),
                                        html.P(
                                            "Upload a patient mutation file to create a persistent case, review module toggles, and run enabled modules in the background.",
                                            className="text-muted small",
                                        ),
                                        dcc.Upload(
                                            id="upload-maf",
                                            children=dbc.Button("Choose patient MAF / TSV", color="secondary", className="w-100"),
                                            multiple=False,
                                            className="w-100",
                                        ),
                                        html.Div(id="upload-status", className="mt-2 text-muted small"),
                                        html.Div(id="case-create-summary", className="mt-3"),
                                        html.Label("Enabled modules for this case", className="text-muted mt-3"),
                                        dbc.Checklist(id="upload-module-checklist", options=[], value=[], class_name="mb-3"),
                                        dbc.Button("Run enabled modules", id="run-case-btn", color="success", disabled=True, class_name="w-100 mb-2"),
                                        html.Div(id="upload-result-panel", className="mt-2"),
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
                                        html.Hr(className="border-secondary my-3"),
                                        html.H5("Ranked list", className="mb-2"),
                                        html.Div(id="simple-tier-badges", className="mb-2"),
                                        html.Div(id="simple-ranked-list", className="mb-3"),
                                        html.H5("Evidence cards", className="mb-2"),
                                        html.Div(id="simple-evidence-cards"),
                                    ]
                                ),
                                class_name="surface panel",
                                ),
                                id="upload-section",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H5("Uploaded cases", className="mb-2"),
                                                            dcc.Dropdown(id="case-select", options=[], value=None, clearable=False, className="dash-dropdown mb-3"),
                                                            html.Div(id="case-list-panel", className="small text-muted"),
                                                        ]
                                                    ),
                                                    class_name="surface panel h-100",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H5("Case inputs and reruns", className="mb-2"),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                dcc.Upload(
                                                                                    id="case-rna-upload",
                                                                                    children=dbc.Button("Attach RNA sidecar", color="secondary", outline=True, className="w-100"),
                                                                                    multiple=False,
                                                                                    className="w-100",
                                                                                ),
                                                                                md=4,
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Upload(
                                                                                    id="case-purity-upload",
                                                                                    children=dbc.Button("Attach purity sidecar", color="secondary", outline=True, className="w-100"),
                                                                                    multiple=False,
                                                                                    className="w-100",
                                                                                ),
                                                                                md=4,
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Upload(
                                                                                    id="case-cnv-upload",
                                                                                    children=dbc.Button("Attach CNV sidecar", color="secondary", outline=True, className="w-100"),
                                                                                    multiple=False,
                                                                                    className="w-100",
                                                                                ),
                                                                                md=4,
                                                                            ),
                                                                        ],
                                                                        class_name="g-2 mb-3",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(dbc.Button("Rerun expression", id="rerun-expression-btn", color="info", outline=True, class_name="w-100"), md=4),
                                                                            dbc.Col(dbc.Button("Rerun clonality", id="rerun-clonality-btn", color="info", outline=True, class_name="w-100"), md=4),
                                                                            dbc.Col(dbc.Button("Rerun ResistanceLoop", id="rerun-resistance-btn", color="info", outline=True, class_name="w-100"), md=4),
                                                                        ],
                                                                        class_name="g-2 mb-2",
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(dbc.Button("Rerun strategies", id="rerun-strategy-btn", color="info", outline=True, class_name="w-100"), md=6),
                                                                            dbc.Col(dbc.Button("Rerun prioritization", id="rerun-prioritization-btn", color="info", outline=True, class_name="w-100"), md=6),
                                                                        ],
                                                                        class_name="g-2",
                                                                    ),
                                                                    html.Div(id="case-action-status", className="small text-muted mt-3"),
                                                                ]
                                                            ),
                                                            class_name="surface panel",
                                                        ),
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                html.Div(id="case-detail-panel")
                                                            ),
                                                            class_name="surface panel h-100",
                                                        ),
                                                    ]
                                                ),
                                                md=8,
                                            ),
                                        ],
                                        class_name="g-3",
                                    ),
                                ],
                                id="cases-section",
                                style={"display": "none"},
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("How NeoResist-MD works", className="mb-3"),
                                            html.Div(id="about-panel"),
                                        ]
                                    ),
                                    class_name="surface panel",
                                ),
                                id="about-section",
                                style={"display": "none"},
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Pipeline status", className="mb-3"),
                                            html.Div(id="pipeline-status-panel"),
                                        ]
                                    ),
                                    class_name="surface panel",
                                ),
                                id="pipeline-status-section",
                                style={"display": "none"},
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
