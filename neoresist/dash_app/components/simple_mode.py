from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from neoresist.dash_app.display_utils import (
    display_text,
    escape_risk_summary,
    evidence_summary_binding,
    evidence_summary_clonality,
    evidence_summary_expression,
)


def _as_series(row: Any) -> pd.Series:
    if isinstance(row, pd.Series):
        return row
    return pd.Series(row or {})


def _fmt_pending(value: Any, precision: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Pending"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _tier_label(value: Any) -> str:
    text = str(value).upper()
    if text in {"1", "TIER_1"}:
        return "TIER_1"
    if text in {"2", "TIER_2"}:
        return "TIER_2"
    if text in {"3", "TIER_3"}:
        return "TIER_3"
    return "EXCLUDED"


def _flag(value: Any) -> bool:
    return str(value or "").strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def build_kpi_hero(df: pd.DataFrame) -> dbc.Card:
    total = int(len(df)) if not df.empty else 0
    tier1 = int(df.get("tier", pd.Series(dtype=object)).map(_tier_label).eq("TIER_1").sum()) if not df.empty else 0
    rec = df.get("recognition_score")
    if rec is not None and not pd.Series(rec).dropna().empty:
        q1 = round(float(pd.Series(rec).dropna().quantile(0.25)) * 100)
        q3 = round(float(pd.Series(rec).dropna().quantile(0.75)) * 100)
        rec_text = f"{q1}% – {q3}%"
    else:
        rec_text = "Pending"

    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(str(total), className="text-kpi"),
                                html.Div("Total Candidates", className="text-secondary text-body"),
                            ],
                            md=4,
                            sm=12,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.Div(str(tier1), className="text-kpi"),
                                        html.Div("Tier 1 Candidates", className="text-secondary text-body"),
                                    ]
                                ),
                                style={"background": "rgba(63,185,80,0.15)", "border": "1px solid #3FB950"},
                            ),
                            md=4,
                            sm=12,
                        ),
                        dbc.Col(
                            [
                                html.Div(rec_text, className="text-kpi"),
                                html.Div("Est. Recognition Range", className="text-secondary text-body"),
                            ],
                            md=4,
                            sm=12,
                        ),
                    ],
                    className="g-3 align-items-center",
                ),
                html.Div(
                    "Tier thresholds are configurable. Defaults based on published MHC-I binding and clonality literature.",
                    className="text-micro",
                    style={"color": "var(--text-muted)", "marginTop": "12px"},
                ),
            ]
        ),
        style={
            "background": "var(--bg-surface)",
            "border": "1px solid var(--border-default)",
        },
        class_name="mb-3",
    )


def build_candidate_cards(df: pd.DataFrame) -> list[dbc.Card]:
    if df is None or df.empty:
        return [
            dbc.Card(
                dbc.CardBody(html.Div("No candidates available yet.", className="text-muted")),
                class_name="candidate-card mb-2",
                style={"minHeight": "80px"},
            )
        ]

    sort_col = "composite_priority" if "composite_priority" in df.columns else "rl_priority"
    work = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    cards: list[dbc.Card] = []
    for idx, row in work.iterrows():
        s = _as_series(row)
        seq = display_text(s.get("mutant_peptide"))
        short_seq = seq if len(seq) <= 15 else f"{seq[:15]}…"
        seq_id = f"candidate-seq-{idx}"
        tier_raw = _tier_label(s.get("tier"))
        tier_cls = {
            "TIER_1": "tier-1",
            "TIER_2": "tier-2",
            "TIER_3": "tier-3",
            "EXCLUDED": "excluded",
        }.get(tier_raw, "excluded")
        score = s.get("composite_priority", s.get("rl_priority"))
        score_text = "Pending" if score is None or pd.isna(score) else f"{int(round(float(score) * 100))}"
        binding_label, binding_color = evidence_summary_binding(s)
        expr_label, expr_color = evidence_summary_expression(s)
        clonality_label, clonality_color = evidence_summary_clonality(s)
        risk_label, risk_color = escape_risk_summary(s)
        badges: list[Any] = []
        if str(s.get("hla_confidence") or "") == "pan-allele estimate":
            badges.append(dbc.Badge("Pan-allele est.", color="warning", className="me-1"))
        if _flag(s.get("kras_g12_flag")):
            badge_id = f"kras-g12-badge-{idx}"
            badges.extend(
                [
                    dbc.Badge("KRAS G12", id=badge_id, color="primary", className="me-1"),
                    dbc.Tooltip(
                        "Known oncogenic driver mutation with established neoantigen potential and targeted therapy options (sotorasib, adagrasib).",
                        target=badge_id,
                    ),
                ]
            )
        if _flag(s.get("shared_neoantigen_flag")):
            badge_id = f"shared-msi-badge-{idx}"
            badges.extend(
                [
                    dbc.Badge("Shared MSI neoantigen", id=badge_id, color="info", className="me-1"),
                    dbc.Tooltip(
                        "This peptide matches a known recurrent frameshift neoantigen found across MSI-H tumors.",
                        target=badge_id,
                    ),
                ]
            )

        card = dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(html.Div(str(idx + 1), className="text-muted", style={"width": "20px"}), width="auto"),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                f"{display_text(s.get('gene'))} {display_text(s.get('protein_change'), '')}",
                                                className="text-emphasis fw-bold",
                                            ),
                                            dbc.Badge(
                                                tier_raw.replace("_", " "),
                                                class_name=f"tier-badge {tier_cls} ms-2",
                                            ),
                                            *badges,
                                        ],
                                        className="d-flex align-items-center flex-wrap gap-2",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("●", style={"color": binding_color, "marginRight": "6px"}),
                                                    html.Span(binding_label, className="text-body me-3"),
                                                    html.Span("●", style={"color": expr_color, "marginRight": "6px"}),
                                                    html.Span(expr_label, className="text-body me-3"),
                                                    html.Span("●", style={"color": clonality_color, "marginRight": "6px"}),
                                                    html.Span(clonality_label, className="text-body"),
                                                ],
                                                className="d-flex align-items-center flex-wrap",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("●", style={"color": risk_color, "marginRight": "6px"}),
                                                    html.Span(risk_label, className="text-body"),
                                                ],
                                                className="d-flex align-items-center mt-1",
                                            ),
                                        ]
                                    ),
                                ],
                                class_name="flex-grow-1",
                            ),
                            dbc.Col(
                                [
                                    html.Span(short_seq, id=seq_id, className="peptide-sequence text-body"),
                                    dbc.Tooltip(seq, target=seq_id),
                                ],
                                width="auto",
                                class_name="peptide-col",
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(score_text, className="score-value text-kpi text-end", style={"fontSize": "24px"}),
                                        html.Div("RL / 100", className="text-micro text-muted text-end"),
                                    ],
                                ),
                                width="auto",
                                class_name="text-end",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "›",
                                    id={"type": "expand-candidate", "index": idx},
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    class_name="candidate-expand-btn",
                                ),
                                width="auto",
                                class_name="text-end",
                            ),
                        ],
                        class_name="g-2 align-items-center h-100",
                    )
                ]
            ),
            class_name="candidate-card mb-2",
            style={"minHeight": "80px", "background": "var(--bg-surface)", "border": "1px solid var(--border-default)"},
        )
        cards.append(card)
    return cards


def build_stub_banners(module_statuses: dict, tool_registry: dict) -> html.Div:
    module_name_map = {
        "expression_join": "Expression",
        "clonality_pyclone_vi": "Clonality",
        "presentation_netctlpan": "Presentation",
        "escape_lohhla": "Escape",
        "recognition_foreignness": "Recognition",
    }
    optional_modules: list[str] = []
    for module_id, status in (module_statuses or {}).items():
        status_obj = status if isinstance(status, dict) else {}
        state = str(status_obj.get("status") or "").lower()
        msg = str(status_obj.get("message") or "")
        if state not in {"unavailable"} and "stub" not in msg.lower():
            continue
        label = module_name_map.get(module_id)
        if label:
            optional_modules.append(label)
    if not optional_modules:
        return html.Div()
    names = ", ".join(sorted(set(optional_modules)))
    return html.Div(
        f"Optional modules: {names} scores use cohort-level estimates. Provide RNA-seq, resolved HLA, or clonality sidecars for patient-specific values.",
        className="stub-warning-banner stub-info-banner",
    )
