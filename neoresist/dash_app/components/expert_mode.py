from __future__ import annotations

from datetime import datetime
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from neoresist.dash_app.display_utils import COMPONENT_DISPLAY_NAMES

MODULES = [
    ("Generation", "neoantigen_generation"),
    ("Presentation", "presentation_netctlpan"),
    ("Expression", "expression_join"),
    ("Clonality", "clonality_pyclone_vi"),
    ("Escape", "escape_lohhla"),
    ("Recognition", "recognition_foreignness"),
    ("ResistanceLoop", "resistance_loop"),
    ("Tiering", "prioritization_tiering"),
]

AXES = [
    COMPONENT_DISPLAY_NAMES["expression"],
    COMPONENT_DISPLAY_NAMES["presentation"],
    COMPONENT_DISPLAY_NAMES["ccf"],
    COMPONENT_DISPLAY_NAMES["self_dissimilarity"],
    COMPONENT_DISPLAY_NAMES["escape"],
]


def derive_module_confidence(status: dict[str, Any]) -> tuple[str, str]:
    state = str((status or {}).get("status") or "pending").lower()
    confidence = str((status or {}).get("confidence_state") or (status or {}).get("confidence") or "").upper()
    if state == "pending":
        return "not yet run", "pending"
    if state == "unavailable":
        return "not available", "unavailable"
    if state == "failed":
        return "failed", "failed"
    if state == "running":
        return "running", "running"
    if confidence in {"LOW", "MEDIUM", "HIGH"}:
        return confidence, confidence.lower()
    return "HIGH", "high"


def _weight_from_strategy(current_weights: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(current_weights.get(key, default))
    except (TypeError, ValueError):
        return default


def _radar_values(current_weights: dict[str, Any]) -> list[float]:
    return [
        _weight_from_strategy(current_weights, "expression", 0.20),
        _weight_from_strategy(current_weights, "presentation", 0.30),
        _weight_from_strategy(current_weights, "ccf", 0.30),
        _weight_from_strategy(current_weights, "self_dissimilarity", 0.10),
        _weight_from_strategy(current_weights, "escape", 0.20),
    ]


def build_strategy_radar_figure(current_weights: dict[str, Any], compare_weights: list[float] | None = None, compare_name: str | None = None) -> go.Figure:
    values = _radar_values(current_weights)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=AXES + [AXES[0]],
            fill="toself",
            line=dict(color="#2F81F7", width=2),
            fillcolor="rgba(47,129,247,0.3)",
            name="Current",
        )
    )
    if compare_weights:
        comp = list(compare_weights)
        fig.add_trace(
            go.Scatterpolar(
                r=comp + [comp[0]],
                theta=AXES + [AXES[0]],
                fill="none",
                line=dict(color="#D29922", width=2, dash="dash"),
                name=compare_name or "Loaded",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161B22",
        plot_bgcolor="#161B22",
        font=dict(color="#8B949E"),
        polar=dict(
            bgcolor="#161B22",
            radialaxis=dict(range=[0, 1], visible=True, gridcolor="#30363D", color="#8B949E"),
            angularaxis=dict(gridcolor="#30363D", color="#8B949E"),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        showlegend=False,
    )
    return fig


def build_pipeline_diagram(module_statuses: dict, run_id: str) -> go.Figure:
    fig = go.Figure()
    x0 = 0
    width = 120
    gap = 40
    y_center = 0
    for i, (label, module_id) in enumerate(MODULES):
        status = module_statuses.get(module_id, {}) or {}
        state = str(status.get("status") or "PENDING").upper()
        confidence_label, confidence_state = derive_module_confidence(status)
        tool = str(status.get("tool") or status.get("message") or status.get("install_message") or "—")
        fill = "#30363D"
        border = "#30363D"
        dash = None
        if state == "COMPLETE" and confidence_state == "high":
            fill = "rgba(63,185,80,0.2)"
            border = "#3FB950"
        elif state == "COMPLETE" and confidence_state in {"low", "medium"}:
            fill = "rgba(210,153,34,0.2)"
            border = "#D29922"
        elif state == "RUNNING":
            fill = "rgba(47,129,247,0.2)"
            border = "#2F81F7"
        elif confidence_state in {"pending", "unavailable"}:
            fill = "#484F58"
            border = "#484F58"
            dash = "dash"
        elif confidence_state == "failed" or state in {"ERROR", "FAILED"}:
            fill = "#F85149"
            border = "#F85149"
        x_center = x0 + i * (width + gap)
        fig.add_shape(
            type="rect",
            x0=x_center - width / 2,
            x1=x_center + width / 2,
            y0=y_center - 30,
            y1=y_center + 30,
            line=dict(color=border, width=2, dash=dash),
            fillcolor=fill,
            layer="below",
        )
        if i < len(MODULES) - 1:
            nx = x0 + (i + 1) * (width + gap) - width / 2
            fig.add_annotation(
                x=nx,
                y=y_center,
                ax=x_center + width / 2,
                ay=y_center,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowwidth=1.5,
                arrowcolor="#58A6FF",
                text="",
            )
        fig.add_trace(
            go.Scatter(
                x=[x_center],
                y=[y_center],
                mode="markers",
                marker=dict(size=70, opacity=0.01),
                hoverinfo="none",
                hovertemplate="<extra></extra>",
                customdata=[[
                    module_id,
                    label,
                    state,
                    confidence_label,
                    tool,
                    run_id,
                    status.get("updated_at") or status.get("finished_at") or status.get("started_at") or "—",
                ]],
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=260,
        xaxis=dict(visible=False, range=[-80, len(MODULES) * (width + gap) - gap + 80]),
        yaxis=dict(visible=False, range=[-80, 80]),
        dragmode=False,
        clickmode="event",
        hovermode="closest",
        showlegend=False,
    )
    return fig


def _strategy_options(strategy_library: list[Any]) -> list[dict[str, str]]:
    options = []
    for item in strategy_library:
        strategy_id = getattr(item, "strategy_id", None) or str((item or {}).get("strategy_id"))
        label = getattr(item, "display_name", None) or str((item or {}).get("display_name") or strategy_id)
        options.append({"label": f"{label} ({strategy_id})", "value": strategy_id})
    return options


def build_strategy_editor(current_weights: dict, strategy_library: list) -> html.Div:
    values = _radar_values(current_weights)
    return html.Div(
        [
            dcc.Graph(
                id="expert-strategy-radar",
                figure=build_strategy_radar_figure(current_weights),
                config={"displayModeBar": False},
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Expression", className="text-muted text-micro"),
                            dcc.Slider(id="expert-weight-binding", min=0, max=1, step=0.05, value=values[0]),
                            html.Label("Presentation / Binding", className="text-muted text-micro mt-2"),
                            dcc.Slider(id="expert-weight-presentation", min=0, max=1, step=0.05, value=values[1]),
                            html.Label("Clonality (CCF)", className="text-muted text-micro mt-2"),
                            dcc.Slider(id="expert-weight-expression", min=0, max=1, step=0.05, value=values[2]),
                            html.Label("Foreignness", className="text-muted text-micro mt-2"),
                            dcc.Slider(id="expert-weight-recognition", min=0, max=1, step=0.05, value=values[3]),
                            html.Label("Escape resistance", className="text-muted text-micro mt-2"),
                            dcc.Slider(id="expert-weight-resistance", min=0, max=1, step=0.05, value=values[4]),
                            html.Div(
                                "Tier thresholds are configurable. Theory-based defaults. Will update from empirical calibration.",
                                className="text-muted text-micro mt-3",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            dbc.Button(
                                "Save as Strategy",
                                id="expert-save-strategy-btn",
                                color="success",
                                class_name="w-100 mb-2",
                            ),
                            dbc.Select(id="expert-strategy-load-select", options=_strategy_options(strategy_library), value=None, class_name="mb-2"),
                            dbc.Button("Apply to Current Patient", id="expert-apply-strategy-btn", color="info", class_name="w-100 mb-2"),
                            html.Div(id="expert-strategy-save-status", className="small mt-2"),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Save Strategy")),
                                    dbc.ModalBody(
                                        [
                                            dbc.Input(id="expert-strategy-modal-input", type="text", placeholder="Strategy name"),
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        [
                                            dbc.Button("Cancel", id="expert-strategy-modal-cancel", color="secondary", outline=True),
                                            dbc.Button("Confirm", id="expert-strategy-modal-confirm", color="success", class_name="ms-2"),
                                        ]
                                    ),
                                ],
                                id="expert-strategy-modal",
                                is_open=False,
                            ),
                        ],
                        md=5,
                    ),
                ],
                class_name="g-3",
            ),
        ],
        className="strategy-editor",
    )


def build_stub_banners(module_statuses: dict, tool_registry: dict) -> html.Div:
    cards: list[Any] = []
    registry_lookup: dict[str, tuple[str, str]] = {}
    for section, section_data in (tool_registry or {}).items():
        if not isinstance(section_data, dict):
            continue
        for tool_key, tool_data in section_data.items():
            if not isinstance(tool_data, dict):
                continue
            registry_lookup[f"{section}.{tool_key}"] = (
                str(tool_data.get("name") or tool_key),
                str(tool_data.get("install") or "See tool_registry.yaml"),
            )

    module_name_map = {
        "neoantigen_generation": ("Generation", "binding.mhcflurry"),
        "presentation_netctlpan": ("Presentation", "presentation.netctlpan"),
        "expression_join": ("Expression", "recognition.blosum62"),
        "clonality_pyclone_vi": ("Clonality", "clonality.pyclone_vi"),
        "escape_lohhla": ("Escape", "escape.lohhla"),
        "recognition_foreignness": ("Recognition", "recognition.blosum62"),
        "resistance_loop": ("ResistanceLoop", "recognition.blosum62"),
        "prioritization_tiering": ("Tiering", "recognition.blosum62"),
    }
    for module_id, status in (module_statuses or {}).items():
        status_obj = status if isinstance(status, dict) else {}
        state = str(status_obj.get("status") or "").lower()
        msg = str(status_obj.get("message") or "")
        if state not in {"unavailable"} and "stub" not in msg.lower():
            continue
        label, tool_ref = module_name_map.get(module_id, (module_id, ""))
        tool_name, install_cmd = registry_lookup.get(tool_ref, (tool_ref.split(".")[-1] or "tool", "See tool_registry.yaml"))
        cards.append(
            html.Div(
                f"Optional module: {label} is using cohort-level estimates. Expert setup: {tool_name} · {install_cmd}",
                className="stub-warning-banner stub-info-banner",
            )
        )
    return html.Div(cards)
