from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from neoresist.case_store import read_case_audit, read_case_manifest, read_module_status, synchronize_manifest
from neoresist.dash_app.constants import TIER_COLORS, TIER_LABELS
from neoresist.dash_app.data import _coerce_exclusion_list
from neoresist.module_schema import load_module_schema
from neoresist.strategy_registry import list_strategies
from neoresist.tumor_features import normalise_tumor_type


STATUS_COLORS = {
    "pending": "secondary",
    "running": "info",
    "complete": "success",
    "failed": "danger",
    "unavailable": "warning",
    "disabled": "dark",
}


def _msi_color(status: Any) -> str:
    text = str(status or "INDETERMINATE").upper()
    if text == "MSS":
        return "success"
    if text == "MSI-H":
        return "warning"
    return "secondary"

_CONF_CANONICAL = {"GREEN": "HIGH", "YELLOW": "MEDIUM", "GREY": "UNAVAILABLE"}
_CONF_BADGE_COLORS = {
    "HIGH": "success",
    "MEDIUM": "warning",
    "LOW": "danger",
    "UNAVAILABLE": "secondary",
    "UNKNOWN": "secondary",
}


def _normalise_confidence(value: Any) -> str:
    raw = str(value or "UNKNOWN").strip().upper()
    return _CONF_CANONICAL.get(raw, raw)


def _preview_table(records: list[dict[str, Any]], columns: list[str], empty_text: str) -> Any:
    if not records:
        return html.Div(empty_text, className="text-muted small")
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th(col.replace("_", " ").title()) for col in columns])),
            html.Tbody([html.Tr([html.Td(str(row.get(col, "—"))) for col in columns]) for row in records[:8]]),
        ],
        responsive=True,
        hover=True,
        size="sm",
        class_name="strategy-table",
    )


def _load_csv_preview(path_str: str | None, columns: list[str]) -> list[dict[str, Any]]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.is_file():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    present = [col for col in columns if col in df.columns]
    if not present:
        return []
    return df[present].head(8).to_dict("records")


def _status_badge(status: dict[str, Any]) -> dbc.Badge:
    state = str(status.get("status", "pending"))
    return dbc.Badge(state.upper(), color=STATUS_COLORS.get(state, "secondary"))


def _artifact_table_rows(artifacts: dict[str, Any]) -> list[Any]:
    rows = []
    for key, path in artifacts.items():
        rows.append(html.Tr([html.Td(str(key)), html.Td(Path(str(path)).name), html.Td(str(path))]))
    return rows


def _load_json(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _confidence_badges(status: dict[str, Any]) -> list[Any]:
    artifacts = status.get("artifacts") or {}
    canonical_candidates = [path for key, path in artifacts.items() if str(key).endswith("canonical_csv")]
    badges: list[Any] = []
    if canonical_candidates:
        path = Path(str(canonical_candidates[0]))
        try:
            df = pd.read_csv(path, nrows=50)
            confidence_cols = [col for col in df.columns if col.endswith("_confidence")]
            for col in confidence_cols[:6]:
                values = Counter(str(v) for v in df[col].dropna().astype(str))
                if not values:
                    continue
                top_value, top_count = values.most_common(1)[0]
                normalized = _normalise_confidence(top_value)
                color = _CONF_BADGE_COLORS.get(normalized, "secondary")
                badges.append(
                    dbc.Badge(
                        f"{col.replace('_confidence', '')}: {normalized} ({top_count})",
                        color=color,
                        className="me-1 mb-1",
                    )
                )
        except Exception:
            pass
    if not badges:
        state = str(status.get("status", "pending"))
        if state == "complete":
            badges.append(dbc.Badge("Confidence: available", color="success", className="me-1 mb-1"))
        elif state == "unavailable":
            badges.append(dbc.Badge("Confidence: blocked by runtime", color="warning", className="me-1 mb-1"))
        else:
            badges.append(dbc.Badge("Confidence: pending", color="secondary", className="me-1 mb-1"))
    return badges


def _module_progress_strip(statuses: dict[str, dict[str, Any]]) -> html.Div:
    chips = []
    for module_id, status in statuses.items():
        chips.append(
            dbc.Badge(
                f"{module_id.replace('_', ' ')}: {str(status.get('status', 'pending'))}",
                color=STATUS_COLORS.get(str(status.get("status", "pending")), "secondary"),
                className="me-1 mb-1",
            )
        )
    return html.Div(chips, className="mb-2")


def _case_overview(case_id: str, manifest: dict[str, Any], mode: str, statuses: dict[str, dict[str, Any]]) -> html.Div:
    counts = Counter(str(status.get("status", "pending")) for status in statuses.values())
    blocking = next(
        (
            f"{spec.display_name}: {statuses[module_id].get('message') or 'waiting'}"
            for module_id, spec in load_module_schema().items()
            if str(statuses[module_id].get("status")) in {"pending", "running", "failed", "unavailable"}
        ),
        "All enabled modules have finished or are disabled for this case.",
    )
    return html.Div(
        [
            dbc.Badge(f"Case: {case_id}", color="info", className="me-1 mb-1"),
            dbc.Badge(f"Sample: {manifest.get('sample_id', '—')}", color="secondary", className="me-1 mb-1"),
            dbc.Badge(f"Tumor: {normalise_tumor_type(manifest.get('tumor_type'))}", color="info", className="me-1 mb-1"),
            dbc.Badge(f"MSI: {manifest.get('msi_status', 'INDETERMINATE')}", color=_msi_color(manifest.get("msi_status")), className="me-1 mb-1"),
            dbc.Badge(f"Input: {manifest.get('input_mode', '—')}", color="secondary", className="me-1 mb-1"),
            dbc.Badge("HLA-A0201", color="secondary", className="me-1 mb-1"),
            dbc.Badge("Simple mode" if mode == "Simple" else "Expert mode", color="dark", className="me-1 mb-1"),
            dbc.Badge(f"Complete {counts.get('complete', 0)}", color="success", className="me-1 mb-1"),
            dbc.Badge(f"Running {counts.get('running', 0)}", color="info", className="me-1 mb-1"),
            dbc.Badge(f"Pending {counts.get('pending', 0)}", color="secondary", className="me-1 mb-1"),
            dbc.Badge(f"Unavailable {counts.get('unavailable', 0)}", color="warning", className="me-1 mb-1"),
            html.P(
                "Results are read from disk. The page never blocks on computation; it only reflects the latest stored module state.",
                className="mt-2 mb-2",
            ),
            html.Div(f"Current blocker / next step: {blocking}", className="small text-muted"),
        ]
    )


def _module_detail_card(module_id: str, spec, status: dict[str, Any]) -> dbc.Card:
    artifacts = status.get("artifacts") or {}
    artifact_list = [
        html.Li(f"{key}: {Path(str(path)).name}") for key, path in artifacts.items()
    ] or [html.Li("No artifacts yet.")]
    required_inputs = spec.required_inputs or []
    optional_inputs = spec.optional_inputs or []
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6(spec.display_name, className="mb-1"),
                        _status_badge(status),
                    ],
                    className="d-flex justify-content-between align-items-start mb-2",
                ),
                html.P(spec.tooltip.short, className="small mb-2"),
                html.Div(str(status.get("message") or ""), className="small text-muted mb-2"),
                html.Div(
                    [
                        dbc.Badge(
                            "Installed" if status.get("installed") else "Runtime missing",
                            color="success" if status.get("installed") else "warning",
                            className="me-1 mb-1",
                        ),
                        dbc.Badge(f"Artifacts: {len(artifacts)}", color="secondary", className="me-1 mb-1"),
                        dbc.Badge(f"Version {spec.version}", color="dark", className="me-1 mb-1"),
                        dbc.Badge(
                            f"Checkpoint: {status.get('checkpoint_label') or '—'}",
                            color="secondary",
                            className="me-1 mb-1",
                        ),
                    ],
                    className="mb-2",
                ),
                html.Div(_confidence_badges(status), className="mb-2"),
                html.Div(
                    [
                        html.Div(f"Started: {status.get('started_at') or '—'}", className="small text-muted"),
                        html.Div(f"Finished: {status.get('finished_at') or '—'}", className="small text-muted"),
                        html.Div(
                            f"Resume ready: {'yes' if status.get('resume_ready') else 'no'}",
                            className="small text-muted",
                        ),
                        html.Div(str(status.get("install_message") or ""), className="small text-muted mt-1"),
                        html.Div(str(status.get("error") or ""), className="small text-warning mt-1"),
                    ]
                ),
                html.Hr(className="my-2 border-secondary"),
                dbc.Alert(
                    "This module is intentionally explicit. Supported local modules write browseable artifacts; unsupported scientific runtimes remain visible as unavailable instead of being hidden.",
                    color="info" if status.get("installed") else "warning",
                    className="py-2 mb-2 small",
                ),
                html.Div("Inputs", className="text-muted small"),
                html.Ul(
                    [html.Li(inp) for inp in required_inputs] + [html.Li(f"Optional: {inp}") for inp in optional_inputs]
                    or [html.Li("No declared inputs.")],
                    className="small mb-2",
                ),
                html.Div("Artifacts", className="text-muted small"),
                html.Ul(artifact_list, className="small mb-0"),
            ]
        ),
        class_name="surface h-100",
    )


def _module_cards(statuses: dict[str, dict[str, Any]], expert: bool) -> Any:
    specs = load_module_schema()
    cards = []
    for module_id, spec in specs.items():
        status = statuses.get(module_id, {})
        if expert:
            cards.append(dbc.Col(_module_detail_card(module_id, spec, status), md=4, sm=12))
        else:
            cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [html.H6(spec.display_name, className="mb-1"), _status_badge(status)],
                                    className="d-flex justify-content-between align-items-start mb-2",
                                ),
                                html.P(spec.tooltip.short, className="small mb-2"),
                                html.Div(str(status.get("message") or ""), className="small text-muted mb-2"),
                                html.Div(
                                    [
                                        dbc.Badge(
                                            "Artifacts ready" if status.get("artifacts") else "Waiting for output",
                                            color="success" if status.get("artifacts") else "secondary",
                                            className="me-1 mb-1",
                                        ),
                                        dbc.Badge(
                                            "Installed" if status.get("installed") else "Runtime missing",
                                            color="success" if status.get("installed") else "warning",
                                            className="me-1 mb-1",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        class_name="surface h-100",
                    ),
                    md=4,
                    sm=12,
                )
            )
    return dbc.Row(cards, class_name="g-3 mb-3")


def render_case_list(cases: list[dict[str, Any]], active_case_id: str | None, expert: bool = False) -> Any:
    if not cases:
        return html.Div("No uploaded cases yet. Create one from the Upload page.", className="text-muted small")
    rows = [
        html.Div(
            [
                dbc.Badge(f"Cases ({len(cases)})", color="info", className="me-2 mb-2"),
                dcc.ConfirmDialogProvider(
                    dbc.Button("Delete all test cases", color="danger", outline=True, size="sm", className="mb-2"),
                    id="bulk-delete-test-cases-confirm",
                    message="Delete all demo/test cases except the curated keep set?",
                )
                if expert
                else html.Div(),
            ],
            className="d-flex flex-wrap align-items-center gap-2 mb-2",
        )
    ]
    for manifest in cases:
        case_id = str(manifest.get("case_id") or "unknown")
        badge_color = "info" if case_id == active_case_id else "secondary"
        rows.append(
            html.Div(
                [
                    dbc.Badge(case_id, color=badge_color, className="me-2 mb-1"),
                    html.Span(str(manifest.get("sample_id") or "sample"), className="me-2"),
                    html.Span(str(manifest.get("input_mode") or "unknown"), className="text-muted small"),
                    dcc.ConfirmDialogProvider(
                        dbc.Button("Delete case", color="danger", outline=True, size="sm", className="ms-auto"),
                        id={"type": "delete-case-confirm", "case_id": case_id},
                        message=f"Delete case {case_id}? This removes the stored case directory.",
                    )
                    if expert
                    else html.Div(),
                ],
                className="d-flex flex-wrap align-items-center gap-2 mb-2",
            )
        )
    return html.Div(rows)


def render_case_detail(case_id: str | None, mode: str, active_strategy_id: str = "rl_v1") -> Any:
    if not case_id:
        return html.Div(
            [
                html.H5("Case progress"),
                html.P("Select a case to review persisted module progress and outputs.", className="text-muted"),
            ]
        )

    manifest = synchronize_manifest(case_id)
    specs = load_module_schema()
    statuses = {module_id: read_module_status(case_id, module_id) for module_id in specs}
    neo_artifacts = statuses["neoantigen_generation"].get("artifacts", {})
    expr_artifacts = statuses["expression_join"].get("artifacts", {})
    neo_preview = statuses["neoantigen_generation"].get("preview") or _load_csv_preview(
        neo_artifacts.get("candidates_csv"),
        ["gene_name", "peptide", "best_allele", "presentation_score", "priority_score", "triage_label"],
    )
    expr_preview = statuses["expression_join"].get("preview") or _load_csv_preview(
        expr_artifacts.get("expression_candidates_csv"),
        ["gene_name", "peptide", "real_expression_tpm", "expression_bin", "RNA_data_missing"],
    )
    resistance_artifacts = statuses.get("resistance_loop", {}).get("artifacts", {})
    resistance_preview = statuses.get("resistance_loop", {}).get("preview") or _load_csv_preview(
        resistance_artifacts.get("resistance_scored_csv"),
        ["gene", "mutant_peptide", "rl_priority", "tier", "evidence_expression_source", "evidence_ccf_source"],
    )
    strategy_artifacts = statuses.get("strategy_engine", {}).get("artifacts", {})
    strategy_preview = statuses.get("strategy_engine", {}).get("preview") or _load_csv_preview(
        strategy_artifacts.get("strategy_consensus_csv"),
        ["patient_id", "hla_allele", "gene", "mutant_peptide", "consensus_score", "consensus_agreement_count"],
    )
    prioritization_artifacts = statuses.get("prioritization_tiering", {}).get("artifacts", {})
    prioritization_preview = statuses.get("prioritization_tiering", {}).get("preview") or _load_csv_preview(
        prioritization_artifacts.get("prioritized_candidates_csv"),
        ["patient_id", "hla_allele", "gene", "mutant_peptide", "tier", "composite_priority"],
    )
    top = _case_overview(case_id, manifest, mode, statuses)
    progress_strip = _module_progress_strip(statuses)
    state_counts = Counter(str(status.get("status", "pending")) for status in statuses.values())
    if state_counts.get("running", 0):
        stage_label = "Processing"
        stage_color = "info"
    elif state_counts.get("pending", 0):
        stage_label = "Queued"
        stage_color = "secondary"
    elif state_counts.get("failed", 0):
        stage_label = "Needs attention"
        stage_color = "danger"
    else:
        stage_label = "Results ready"
        stage_color = "success"
    blocking = next(
        (
            f"{spec.display_name}: {statuses[module_id].get('message') or 'waiting'}"
            for module_id, spec in load_module_schema().items()
            if str(statuses[module_id].get("status")) in {"pending", "running", "failed", "unavailable"}
        ),
        "All enabled modules have finished or are disabled for this case.",
    )

    case_hero = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(f"Patient case {case_id}", className="mb-1"),
                                html.Div(
                                    f"Sample: {manifest.get('sample_id', '—')} · Tumor: {normalise_tumor_type(manifest.get('tumor_type'))} · MSI: {manifest.get('msi_status', 'INDETERMINATE')} ({float(manifest.get('msi_frameshift_indel_ratio') or 0.0):.2f}) · Input: {manifest.get('input_mode', '—')}",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.Badge(stage_label, color=stage_color, className="ms-2"),
                    ],
                    className="d-flex justify-content-between align-items-start flex-wrap gap-2",
                ),
                html.Div(
                    [
                        dbc.Badge(f"Active strategy: {active_strategy_id}", color="info", className="me-1 mb-1"),
                        dbc.Badge(f"Complete {state_counts.get('complete', 0)}", color="success", className="me-1 mb-1"),
                        dbc.Badge(f"Running {state_counts.get('running', 0)}", color="info", className="me-1 mb-1"),
                        dbc.Badge(f"Pending {state_counts.get('pending', 0)}", color="secondary", className="me-1 mb-1"),
                        dbc.Badge(f"Unavailable {state_counts.get('unavailable', 0)}", color="warning", className="me-1 mb-1"),
                        dbc.Badge(f"Failed {state_counts.get('failed', 0)}", color="danger", className="me-1 mb-1"),
                    ],
                    className="mt-3",
                ),
                html.Div(
                    f"This patient page reads stored module outputs from disk and updates as each module finishes. "
                    f"Current blocker / next step: {blocking}.",
                    className="mt-3 small text-muted",
                ),
            ]
        ),
        class_name="surface panel mb-3",
    )
    inputs = manifest.get("input_files", {}) or {}
    audit_rows = read_case_audit(case_id)
    strategy_names = [strategy.display_name for strategy in list_strategies()[:6]]

    simple = html.Div(
        [
            case_hero,
            top,
            progress_strip,
            dbc.Alert(
                "Simple mode stays readable: the newest candidate output appears first, while slower biology layers keep their saved status or clearly show that they are unavailable.",
                color="info",
                className="mb-3",
            ),
            _module_cards(statuses, expert=False),
            html.H5("Early results", className="mb-2"),
            _preview_table(
                neo_preview,
                ["gene_name", "peptide", "best_allele", "presentation_score", "priority_score", "triage_label"],
                "Neoantigen results will appear here as soon as the first module completes.",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.P("Upload creates a stored case, background jobs write results to disk, and the page keeps reading saved outputs instead of freezing."),
                            html.Ul(
                                [
                                    html.Li("IC50 is a binding-affinity proxy; lower values often indicate stronger peptide-HLA binding."),
                                    html.Li("Expression reports measured RNA support when an RNA sidecar is available."),
                                    html.Li("CCF is a clonality proxy that helps estimate whether a target is likely clonal or subclonal."),
                                    html.Li("LOH marks loss-of-heterozygosity escape risk in HLA-aware review."),
                                ]
                            ),
                        ],
                        title="How this case is processed",
                    ),
                    dbc.AccordionItem(
                        [
                            html.P(
                                "ResistanceLoop weights presentation, expression, clonality, self-dissimilarity, and escape risk. The score rises when a target looks visible and durable, and falls when escape routes look plausible.",
                                className="mb-0",
                            )
                        ],
                        title="What ResistanceLoop means",
                    ),
                ],
                start_collapsed=True,
                class_name="mt-4",
            ),
        ]
    )

    expert_tabs = dbc.Tabs(
        [
            dbc.Tab(
                label="Modules",
                tab_id="modules",
                children=html.Div(
                    [
                        html.P(
                            "Expert mode exposes module state, install availability, dependencies, and artifact readiness without collapsing everything into one flat table.",
                            className="pt-3",
                        ),
                        _module_cards(statuses, expert=True),
                    ]
                ),
            ),
            dbc.Tab(
                label="Evidence",
                tab_id="evidence",
                children=html.Div(
                    [
                        dbc.Alert(
                            "Expert mode exposes both pillars of the product moat: asynchronous evidence modules and modular scoring strategies.",
                            color="info",
                            className="mb-3 mt-3",
                        ),
                        html.H6("Neoantigen output", className="mb-2"),
                        _preview_table(
                            neo_preview,
                            ["gene_name", "peptide", "best_allele", "presentation_score", "priority_score", "triage_label"],
                            "No neoantigen output yet.",
                        ),
                        html.H6("Expression output", className="mt-4 mb-2"),
                        _preview_table(
                            expr_preview,
                            ["gene_name", "peptide", "real_expression_tpm", "expression_bin", "RNA_data_missing"],
                            "Expression output is unavailable until the module completes or a sidecar is attached.",
                        ),
                        html.H6("ResistanceLoop output", className="mt-4 mb-2"),
                        _preview_table(
                            resistance_preview,
                            ["gene", "mutant_peptide", "rl_priority", "tier", "evidence_expression_source", "evidence_ccf_source"],
                            "ResistanceLoop output will appear once the local scoring module completes.",
                        ),
                        html.H6("Strategy consensus", className="mt-4 mb-2"),
                        _preview_table(
                            strategy_preview,
                            ["patient_id", "hla_allele", "gene", "mutant_peptide", "consensus_score", "consensus_agreement_count"],
                            "Consensus output will appear once the strategy engine completes.",
                        ),
                        html.H6("Final prioritization", className="mt-4 mb-2"),
                        _preview_table(
                            prioritization_preview,
                            ["patient_id", "hla_allele", "gene", "mutant_peptide", "tier", "composite_priority"],
                            "Final prioritization output will appear once tiering completes.",
                        ),
                    ],
                    className="pt-2",
                ),
            ),
            dbc.Tab(
                label="Inputs",
                tab_id="inputs",
                children=html.Div(
                    [
                        html.P("The case is stored on disk and can be extended with optional sidecars without recomputing the original upload.", className="pt-3"),
                        dbc.Table(
                            [
                                html.Thead(html.Tr([html.Th("Input role"), html.Th("Stored file")])),
                                html.Tbody(
                                    [
                                        html.Tr([html.Td(role.replace("_", " ").title()), html.Td(str(path) if path else "—")])
                                        for role, path in inputs.items()
                                    ]
                                ),
                            ],
                            bordered=False,
                            hover=True,
                            responsive=True,
                            size="sm",
                            class_name="strategy-table",
                        ),
                    ]
                ),
            ),
            dbc.Tab(
                label="Artifacts",
                tab_id="artifacts",
                children=html.Div(
                    [
                        html.P("Expert mode exposes raw persisted artifacts so the next session or operator can resume from exact disk outputs, not from chat context.", className="pt-3"),
                        *[
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H6(load_module_schema()[module_id].display_name, className="mb-2"),
                                        dbc.Table(
                                            [
                                                html.Thead(html.Tr([html.Th("Artifact key"), html.Th("File"), html.Th("Stored path")])),
                                                html.Tbody(_artifact_table_rows(status.get("artifacts") or {})),
                                            ] if (status.get("artifacts") or {}) else [
                                                html.Thead(html.Tr([html.Th("Artifact key"), html.Th("File"), html.Th("Stored path")])),
                                                html.Tbody([html.Tr([html.Td("—"), html.Td("No artifacts yet"), html.Td("—")])]),
                                            ],
                                            bordered=False,
                                            hover=True,
                                            responsive=True,
                                            size="sm",
                                            class_name="strategy-table",
                                        ),
                                    ]
                                ),
                                class_name="surface panel mb-3",
                            )
                            for module_id, status in statuses.items()
                        ],
                    ]
                ),
            ),
            dbc.Tab(
                label="Strategies",
                tab_id="strategies",
                children=html.Div(
                    [
                        html.P(
                            "Expert mode keeps modular scoring visible. The advanced strategy workspace stays available separately, while this case view summarizes the saved strategy library.",
                            className="pt-3",
                        ),
                        html.Div([dbc.Badge(name, color="secondary", className="me-1 mb-1") for name in strategy_names]),
                    ]
                ),
            ),
            dbc.Tab(
                label="Audit",
                tab_id="audit",
                children=html.Div(
                    [
                        html.P("Case audit is append-only and reflects module starts, reruns, sidecar attachments, and save decisions.", className="pt-3"),
                        _preview_table(
                            audit_rows,
                            ["timestamp", "event_type", "audit_version"],
                            "No case audit events yet.",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H6("Module summaries", className="mb-2"),
                                    html.Div(
                                        [
                                            dbc.Badge(f"{load_module_schema()[module_id].display_name}: {status.get('checkpoint_label', '—')}", color="secondary", className="me-1 mb-1")
                                            for module_id, status in statuses.items()
                                        ]
                                    ),
                                ]
                            ),
                            class_name="surface panel mt-3",
                        ),
                    ]
                ),
            ),
        ]
    )

    if mode == "Simple":
        return simple
    return html.Div([case_hero, top, progress_strip, expert_tabs])
