from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from neoresist.dash_app.display_utils import display_score_terms, display_text, escape_risk_summary, normalize_hla_display


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, dict)):
        return value
    if isinstance(value, pd.Series):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _v(value: Any, default: str = "Pending") -> str:
    value = _normalize_value(value)
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(item) for item in value if str(item))
        return text or default
    try:
        is_missing = value is None or pd.isna(value)
    except (TypeError, ValueError):
        is_missing = value is None
    if is_missing:
        return default
    return str(value)


def _num(value: Any, digits: int = 3) -> str:
    value = _normalize_value(value)
    if isinstance(value, list):
        value = next((item for item in value if item is not None and str(item) != ""), None)
    if value is None or pd.isna(value):
        return "Pending"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _conf_class(value: Any) -> str:
    state = _v(value, "UNAVAILABLE").upper()
    legacy = {"GREEN": "HIGH", "YELLOW": "MEDIUM", "GREY": "UNAVAILABLE"}
    state = legacy.get(state, state)
    return state.lower() if state in {"HIGH", "MEDIUM", "LOW", "UNAVAILABLE"} else "unavailable"


def _conf_label(value: Any) -> str:
    state = _v(value, "UNAVAILABLE").upper()
    legacy = {"GREEN": "HIGH", "YELLOW": "MEDIUM", "GREY": "UNAVAILABLE"}
    return legacy.get(state, state)


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
    value = _normalize_value(value)
    if isinstance(value, list):
        value = next((item for item in value if item is not None and str(item).strip()), "")
    return str("" if value is None else value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def build_evidence_drawer(row: dict[str, Any] | pd.Series) -> html.Div:
    s = row if isinstance(row, pd.Series) else pd.Series(row or {})
    gene = display_text(s.get("gene"))
    protein_change = display_text(s.get("protein_change"), "")
    title = f"{gene} {protein_change}".strip()
    hla = normalize_hla_display(_v(s.get("hla_allele")))
    peptide = display_text(s.get("mutant_peptide"))
    tier = _tier_label(s.get("tier"))
    tier_class = {"TIER_1": "tier-1", "TIER_2": "tier-2", "TIER_3": "tier-3"}.get(tier, "excluded")
    resistance = s.get("resistance_composite")
    width = 0 if resistance is None or pd.isna(resistance) else max(0.0, min(1.0, float(resistance))) * 100
    if width < 30:
        bar = "#3FB950"
    elif width <= 60:
        bar = "#D29922"
    else:
        bar = "#F85149"
    special_badges: list[Any] = []
    if _flag(s.get("kras_g12_flag")):
        special_badges.append(dbc.Badge("KRAS G12", id="drawer-kras-g12-badge", color="primary", className="me-1 mb-1"))
    if _flag(s.get("shared_neoantigen_flag")):
        special_badges.append(dbc.Badge("Shared MSI neoantigen", id="drawer-shared-msi-badge", color="info", className="me-1 mb-1"))
    if str(s.get("hla_confidence") or "") == "pan-allele estimate":
        special_badges.append(dbc.Badge("Pan-allele est.", color="warning", className="me-1 mb-1"))

    expr_flag = _v(s.get("expression_flag"))
    expr_badge_class = {
        "EXPRESSED": "tier-1",
        "LOW": "tier-2",
        "ABSENT": "tier-3",
        "UNKNOWN": "excluded",
    }.get(expr_flag, "excluded")
    hla_loh = _v(s.get("hla_loh_status"))
    hla_loh_class = {
        "LOH_DETECTED": "tier-3",
        "LOH_SUSPECTED": "tier-2",
        "INTACT": "tier-1",
        "UNKNOWN": "excluded",
    }.get(hla_loh.upper(), "excluded")
    escape_conf = _v(s.get("escape_confidence"))
    clonality_conf = _v(s.get("clonality_confidence"))
    if clonality_conf.upper() == "LOW":
        clonality_note = "⚠ VAF proxy — PyClone-VI not run. Install for accurate CCF."
    else:
        clonality_note = None
    if escape_conf.upper() == "UNAVAILABLE":
        escape_note = "LOHHLA not installed — HLA LOH status unknown."
    else:
        escape_note = None
    diploid_note = "ⓘ Diploid copy number assumed — no CNV data provided." if bool(s.get("diploid_assumption_flag")) else None
    exclusion_text = _v(s.get("exclusion_reasons"), "")
    if "IFN_GAMMA_PATHWAY_DISRUPTED:" in exclusion_text:
        try:
            ifn_gene = [x.split(":", 1)[1] for x in exclusion_text.split(",") if x.startswith("IFN_GAMMA_PATHWAY_DISRUPTED:")][0]
        except Exception:
            ifn_gene = "UNKNOWN"
        ifn_note = f"⚠ IFN-γ pathway gene {ifn_gene} mutated in this sample — antigen presentation machinery may be disrupted."
    else:
        ifn_note = None

    risk_label, _risk_color = escape_risk_summary(s)
    immunogenicity, resistance_penalty, display_score = display_score_terms(s)

    return html.Div(
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(title, className="text-panel-title fw-bold"),
                            dbc.Button("×", id="evidence-drawer-close", color="secondary", outline=True, size="sm"),
                        ],
                        className="d-flex justify-content-between align-items-start mb-2",
                    ),
                    html.Div(
                        [html.Span(hla, className="me-2"), html.Span(peptide, className="peptide-sequence")],
                        className="text-body mb-3",
                    ),
                    html.Div(
                        [
                            *special_badges,
                            dbc.Tooltip(
                                "Known oncogenic driver mutation with established neoantigen potential and targeted therapy options (sotorasib, adagrasib).",
                                target="drawer-kras-g12-badge",
                            )
                            if _flag(s.get("kras_g12_flag"))
                            else html.Div(),
                            dbc.Tooltip(
                                "This peptide matches a known recurrent frameshift neoantigen found across MSI-H tumors.",
                                target="drawer-shared-msi-badge",
                            )
                            if _flag(s.get("shared_neoantigen_flag"))
                            else html.Div(),
                        ],
                        className="mb-2",
                    )
                    if special_badges
                    else html.Div(),
                    html.Div(
                        [
                            html.Div("BINDING", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(f"Affinity: {_num(s.get('binding_affinity'), 1)} nM"),
                                    html.Div(f"Rank: {_num(s.get('binding_rank'), 1)}%"),
                                    html.Div(f"Tool: {_v(s.get('binding_tool'))}"),
                                    html.Span(
                                        _conf_label(s.get("binding_confidence")),
                                        className=f"confidence-badge {_conf_class(s.get('binding_confidence'))}",
                                    ),
                                    dbc.Tooltip(
                                        "Binding affinity < 500nM indicates potential MHC-I presentation. Rank % is percentile among random peptides for this allele.",
                                        target="binding-help",
                                    ),
                                    html.Span("?", id="binding-help", className="ms-2 text-muted"),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("PRESENTATION", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(f"Cleavage score: {_num(s.get('cleavage_score'))}"),
                                    html.Div(f"Tap score: {_num(s.get('tap_score'))}"),
                                    html.Div(f"Composite: {_num(s.get('presentation_composite'))}"),
                                    html.Div(f"Tool: {_v(s.get('presentation_tool'))}"),
                                    html.Div("Pan-allele est. because HLA typing is unresolved.", className="text-micro text-muted mt-1")
                                    if str(s.get("hla_confidence") or "") == "pan-allele estimate"
                                    else html.Div(),
                                    html.Span(
                                        _conf_label(s.get("presentation_confidence")),
                                        className=f"confidence-badge {_conf_class(s.get('presentation_confidence'))}",
                                    ),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("EXPRESSION", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(f"TPM: {_num(s.get('expression_tpm'), 2)}"),
                                    html.Span(_v(expr_flag), className=f"tier-badge {expr_badge_class}"),
                                    html.Span(_conf_label(s.get("expression_confidence")), className=f"confidence-badge {_conf_class(s.get('expression_confidence'))} ms-2"),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("CLONALITY", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(f"VAF-derived CCF proxy (Low confidence — PyClone-VI not run): {_num(s.get('vaf_derived_ccf_proxy', s.get('ccf')), 3)}"),
                                    html.Div(f"Clonality class: {_v(s.get('clonality_class'))}"),
                                    html.Span(_conf_label(clonality_conf), className=f"confidence-badge {_conf_class(s.get('clonality_confidence'))}"),
                                    html.Div(clonality_note, className="text-micro text-muted mt-1") if clonality_note else html.Div(),
                                    html.Div(diploid_note, className="text-micro text-muted mt-1") if diploid_note else html.Div(),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("ESCAPE / HLA INTEGRITY", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Span("HLA LOH: ", className="me-1"),
                                    html.Span(hla_loh, className=f"tier-badge {hla_loh_class}"),
                                    html.Div(f"Processing disruption: {_v(s.get('processing_disruption_flag'))}"),
                                    html.Span(_conf_label(escape_conf), className=f"confidence-badge {_conf_class(s.get('escape_confidence'))}"),
                                    html.Div(escape_note, className="text-micro text-muted mt-1") if escape_note else html.Div(),
                                    html.Div(ifn_note, className="text-micro text-muted mt-1") if ifn_note else html.Div(),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("RECOGNITION", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(f"Self-dissimilarity: {_num(s.get('self_dissimilarity'))}"),
                                    html.Div(f"Mutant/WT distance: {_num(s.get('mutant_wt_distance'))}"),
                                    html.Div(f"Recognition score: {_num(s.get('recognition_score'))}"),
                                    html.Div("Tool: BLOSUM62 alignment"),
                                    html.Span(_conf_label(s.get("recognition_confidence")), className=f"confidence-badge {_conf_class(s.get('recognition_confidence'))}"),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("RESISTANCELOOP", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(
                                        className="progress",
                                        children=[
                                            html.Div(
                                                style={
                                                    "width": f"{width:.0f}%",
                                                    "height": "10px",
                                                    "background": bar,
                                                    "borderRadius": "999px",
                                                }
                                            )
                                        ],
                                        style={"background": "#30363d", "borderRadius": "999px", "height": "10px"},
                                    ),
                                    html.Div(f"Composite: {_num(s.get('resistance_composite'))}"),
                                    html.Div(f"Interpretation: {risk_label}", className="text-micro text-muted mt-1"),
                                    html.Table(
                                        [
                                            html.Tr([html.Td("LOH penalty"), html.Td(_num(s.get("resistance_loh_penalty"))), html.Td("LOH escape likelihood")]),
                                            html.Tr([html.Td("Volatility"), html.Td(_num(s.get("resistance_volatility_flag"))), html.Td("Subclonal instability")]),
                                            html.Tr([html.Td("Expression instability"), html.Td(_num(s.get("resistance_expression_instability"))), html.Td("Low or absent expression raises risk")]),
                                            html.Tr([html.Td("Processing disruption"), html.Td(_num(s.get("resistance_processing_disruption"))), html.Td("Machinery disruption reduces durability")]),
                                        ],
                                        className="table table-sm text-body mt-2",
                                    ),
                                ],
                                className="small",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div("FINAL PRIORITY", className="text-nav fw-bold mb-2"),
                            html.Div(
                                [
                                    html.Div(
                                        f"{int(round(display_score * 100))}",
                                        className="text-kpi fw-bold",
                                        style={
                                            "color": {
                                                "TIER_1": "#3FB950",
                                                "TIER_2": "#D29922",
                                                "TIER_3": "#F85149",
                                                "EXCLUDED": "#8B949E",
                                            }.get(tier, "#8B949E")
                                        },
                                    ),
                                    html.Div("RL / 100", className="text-micro text-muted"),
                                    html.Div(
                                        f"{immunogenicity:.3f} × (1 - {resistance_penalty:.3f}) = {display_score:.3f}",
                                        className="text-micro text-muted mt-1",
                                    ),
                                ],
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Button(
                        "Export for Tumor Board",
                        id="export-tumor-board",
                        color="primary",
                        class_name="w-100",
                        style={"background": "var(--accent-blue)", "color": "white"},
                    ),
                ]
            )
        ],
    )
