from __future__ import annotations

import io
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import yaml
from dash import ALL, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from neoresist.dash_app.data import (
    _aggregate_patient_hla,
    filter_agg,
    filter_candidates,
    load_qualified_candidates,
    qualified_parquet_search_paths,
    set_slider_bounds_from_df,
    slider_cand_max,
    slider_fan_max,
    slider_mut_max,
    sort_agg,
    top_exclusion_options,
)
from neoresist.dash_app.display_utils import (
    COMPONENT_DISPLAY_NAMES,
    apply_display_policy,
    is_unknown_hla as display_is_unknown_hla,
    load_exclusion_rules,
    normalize_hla_display,
)
from neoresist.dash_app.panels import (
    build_evidence_panel,
    build_patient_detail_card,
    hla_coverage_badge_row,
    kpi_card,
    make_scatter_fig,
)
from neoresist.dash_app.components import (
    build_candidate_cards,
    build_evidence_drawer,
    build_expert_stub_banners,
    derive_module_confidence,
    build_kpi_hero,
    build_pipeline_diagram,
    build_simple_stub_banners,
    build_strategy_editor,
    build_strategy_radar_figure,
)
from neoresist.case_store import (
    attach_case_input,
    create_case_from_upload,
    delete_case,
    list_cases,
    read_module_status,
    read_case_manifest,
    reset_module_for_rerun,
    reset_modules_for_rerun,
    set_module_enabled,
    write_case_manifest,
)
from neoresist.case_ui import render_case_detail, render_case_list
from neoresist.dash_app.constants import TIER_COLORS
from neoresist.dash_app.layout import build_nav_options
from neoresist.loaders import load_cohort_for_dash
from neoresist.module_runner import ModuleRunner
from neoresist.module_schema import load_module_schema
from neoresist.ops_status import collect_batch_status
from neoresist.paths import repo_root
from neoresist.strategy_registry import (
    clone_strategy,
    deserialize_strategy,
    get_strategy,
    list_strategies,
    log_audit_event,
    read_audit_events,
    save_strategy,
    score_candidates_for_strategy,
    serialize_strategy,
    summarize_strategy_run,
    build_consensus_table,
)
from neoresist.ui_state import EXPERT_ONLY_NAV, resolve_view_state, section_style
from neoresist.tumor_features import infer_tumor_type_from_text, normalise_tumor_type, single_tumor_type
from neoresist.upload_runtime import CSV_REQUIRED_COLUMNS
from neoresist.version import __version__
from neoresist_md.backend.core.module_runner import ModuleRunner as NeoResistModuleRunner


def _render_records_table(records: list[dict[str, Any]], columns: list[tuple[str, str]], empty_text: str) -> Any:
    if not records:
        return html.Div(empty_text, className="text-muted small")
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th(label) for _, label in columns])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                ", ".join(map(str, row.get(key, [])))
                                if isinstance(row.get(key), list)
                                else str(row.get(key, "—"))
                            )
                            for key, _ in columns
                        ]
                    )
                    for row in records
                ]
            ),
        ],
        bordered=False,
        hover=True,
        responsive=True,
        size="sm",
        class_name="strategy-table",
    )


MODULE_TOOL_OPTIONS: dict[str, list[dict[str, str]]] = {
    "neoantigen_generation": [{"label": "Local NeoVax generator", "value": "local_neovax"}],
    "expression_join": [
        {"label": "RNA sidecar join", "value": "rna_sidecar_join"},
        {"label": "Stub expression fallback", "value": "stub_expression"},
    ],
    "clonality_pyclone_vi": [
        {"label": "PyClone-VI adapter", "value": "pyclone_vi_adapter"},
        {"label": "Stub clonality fallback", "value": "stub_ccf"},
    ],
    "resistance_loop": [{"label": "ResistanceLoop v1", "value": "rl_v1"}],
    "strategy_engine": [{"label": "Consensus ranker", "value": "strategy_consensus"}],
    "prioritization_tiering": [{"label": "Tier mapper", "value": "tier_mapper"}],
    "presentation_netctlpan": [{"label": "NetCTLpan adapter (unavailable)", "value": "netctlpan_adapter"}],
    "escape_lohhla": [{"label": "LOHHLA adapter (unavailable)", "value": "lohhla_adapter"}],
    "recognition_foreignness": [{"label": "BLOSUM62 foreignness", "value": "blosum62_alignment"}],
}

MODULE_WEIGHT_TO_STRATEGY_KEY: dict[str, str] = {
    "expression_join": "expression_norm",
    "presentation_netctlpan": "presentation",
    "clonality_pyclone_vi": "ccf",
    "recognition_foreignness": "self_dissimilarity",
}


def _tool_registry_path() -> Path:
    return repo_root() / "neoresist_md" / "config" / "tool_registry.yaml"


def _load_tool_registry_state() -> dict[str, Any]:
    path = _tool_registry_path()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    state: dict[str, Any] = {}
    for section, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        state[section] = {}
        for tool_key, tool_data in section_data.items():
            if not isinstance(tool_data, dict):
                continue
            check_cmd = tool_data.get("check_command")
            available = False
            if check_cmd in (None, ""):
                available = True
            else:
                try:
                    import subprocess

                    proc = subprocess.run(str(check_cmd).split(" "), capture_output=True, text=True, timeout=4)
                    available = proc.returncode in {0, 1, 2}
                except Exception:
                    available = False
            state[section][tool_key] = {
                **tool_data,
                "available": available,
            }
    return state

def _current_case_id(active_case: Any) -> str | None:
    if isinstance(active_case, dict) and active_case.get("case_id"):
        return str(active_case["case_id"])
    recent = list_cases(limit=1)
    if recent and recent[0].get("case_id"):
        return str(recent[0]["case_id"])
    return None


def _case_has_scored_artifacts(case_id: str) -> bool:
    for module_id in ("prioritization_tiering", "strategy_engine", "resistance_loop"):
        status = read_module_status(case_id, module_id)
        if status.get("artifacts"):
            return True
    return False


def _curated_keep_case_ids(manifests: list[dict[str, Any]]) -> set[str]:
    keep: set[str] = set()
    tcga_candidates = [
        str(item.get("case_id"))
        for item in reversed(manifests)
        if item.get("case_id") and str(item.get("sample_id") or "") == "tcga_sarc_case1" and _case_has_scored_artifacts(str(item.get("case_id")))
    ]
    if tcga_candidates:
        keep.add(tcga_candidates[0])
    for sample_id in ("demo", "patient_case"):
        for item in manifests:
            if item.get("case_id") and str(item.get("sample_id") or "") == sample_id:
                keep.add(str(item.get("case_id")))
                break
    return keep


def _strategy_from_slider_weights(base_strategy, weights: dict[str, float]):
    new_weights = dict(base_strategy.weights)
    for key in ("expression_norm", "presentation", "ccf", "self_dissimilarity"):
        if key in weights:
            new_weights[key] = max(0.0, min(1.0, float(weights[key])))
    escape_weight = weights.get("escape_penalty_weight", base_strategy.escape_penalty_weight)
    return replace(base_strategy, weights=new_weights, escape_penalty_weight=float(escape_weight))


def _axis_weights_from_strategy(strategy) -> dict[str, float]:
    return {
        "expression": float(strategy.weights.get("expression_norm", 0.20)),
        "presentation": float(strategy.weights.get("presentation", 0.30)),
        "ccf": float(strategy.weights.get("ccf", 0.30)),
        "self_dissimilarity": float(strategy.weights.get("self_dissimilarity", 0.10)),
        "escape": abs(float(strategy.escape_penalty_weight)),
    }


def _is_unknown_hla(value: Any) -> bool:
    return display_is_unknown_hla(value)


def _hla_coverage_counts(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty or "patient_id" not in df.columns:
        return 0, 0
    if "hla_allele" not in df.columns:
        total = int(df["patient_id"].astype(str).nunique())
        return 0, total
    work = df[["patient_id", "hla_allele"]].copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work["resolved_hla"] = ~work["hla_allele"].map(_is_unknown_hla)
    per_patient = work.groupby("patient_id", dropna=False)["resolved_hla"].max()
    resolved = int(per_patient.sum())
    total = int(len(per_patient))
    return resolved, total


def _msi_counts(df: pd.DataFrame) -> tuple[int, int] | None:
    if df.empty or "patient_id" not in df.columns or "msi_status" not in df.columns:
        return None
    work = df[["patient_id", "msi_status"]].copy()
    work["patient_id"] = work["patient_id"].astype(str)
    work["msi_status"] = work["msi_status"].fillna("INDETERMINATE").astype(str).str.upper()
    per_patient = work.groupby("patient_id", dropna=False)["msi_status"].agg(
        lambda s: "MSI-H" if (s == "MSI-H").any() else s.iloc[0]
    )
    return int((per_patient == "MSI-H").sum()), int(len(per_patient))


def _upload_format_label(filename: str, input_mode: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".maf":
        return "MAF"
    if suffix in {".vcf", ".gz"} or str(filename or "").lower().endswith((".vcf", ".vcf.gz")):
        return "VCF"
    if suffix in {".tsv", ".txt"}:
        return "TSV"
    if suffix == ".csv":
        return "CSV"
    return str(input_mode or "TSV").upper()


def _upload_validation_checklist(validation: Any, filename: str, tumor_type: str) -> html.Div:
    columns = set(str(c) for c in (getattr(validation, "columns", []) or []))
    missing = sorted(CSV_REQUIRED_COLUMNS.difference(columns))
    mapping = getattr(validation, "detected_column_mapping", None) or {}
    mapping_text = (
        "; ".join(f"{src} → {dst}" for src, dst in sorted(mapping.items()))
        if mapping
        else "none (canonical headers already present)"
    )
    checklist = [
        (True, f"Format detected: {_upload_format_label(filename, getattr(validation, 'input_mode', ''))}"),
        (not missing, f"Required columns present: {'yes' if not missing else 'no, missing ' + ', '.join(missing)}"),
        (True, f"Auto-mapped aliases: {mapping_text}"),
        (not bool(getattr(validation, "hla_missing_or_unknown", False)), f"HLA allele found: {'yes' if not bool(getattr(validation, 'hla_missing_or_unknown', False)) else 'no'}"),
        (bool(getattr(validation, "sample_id", "")), f"Patient ID parsed: {'yes, ' + str(getattr(validation, 'sample_id', '')) if getattr(validation, 'sample_id', '') else 'no'}"),
        (True, f"Mutation count: {int(getattr(validation, 'row_count', 0) or 0)}"),
        (normalise_tumor_type(tumor_type) != "UNKNOWN", f"Tumor type selected: {normalise_tumor_type(tumor_type)}"),
    ]
    return html.Div(
        [
            html.Div(
                [html.Span("✅" if ok else "⚠", className="me-2"), html.Span(text)],
                className="small mb-1",
                style={"color": "var(--text-primary)" if ok else "var(--tier2-color)"},
            )
            for ok, text in checklist
        ],
        className="upload-validation-checklist mb-3",
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("tool-registry-store", "data"),
        Input("nav", "value"),
    )
    def load_tool_registry_store(_nav):
        return _load_tool_registry_state()

    @app.callback(
        Output("methodology-collapse", "is_open"),
        Input("methodology-toggle", "n_clicks"),
        State("methodology-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_methodology(_n, is_open):
        return not bool(is_open)

    @app.callback(
        Output("methodology-panel", "children"),
        Input("nav", "value"),
    )
    def methodology_text(_nav):
        try:
            from neoresist.config import get_app_config
            from neoresist.profiles import load_scoring_profile

            cfg = get_app_config()
            sp = load_scoring_profile(cfg.defaults.scoring_profile_id)
            parts = [
                html.P(
                    [
                        html.Strong("Dataset: "),
                        f"{cfg.defaults.dataset_id} · ",
                        html.Strong("Scoring: "),
                        f"{sp.display_name} v{sp.version} · ",
                        html.Strong("Rule profile: "),
                        cfg.defaults.rule_profile_id,
                    ],
                    className="mb-2",
                ),
                html.P(sp.description or "", className="mb-0"),
            ]
            return html.Div(parts)
        except Exception:
            return html.P(
                "Profile metadata loads from configs/ after full platform setup. "
                "ResistanceLoop v1 weights: expression 0.2, presentation 0.3, CCF 0.3, "
                "self-dissimilarity 0.1, escape penalty −0.2.",
                className="mb-0",
            )

    @app.callback(
        Output("strategy-panel", "children"),
        Input("nav", "value"),
        Input("active-strategy-select", "value"),
        Input("strategy-refresh-store", "data"),
    )
    def strategy_panel(_nav, active_strategy_id, _refresh):
        try:
            from neoresist.config import get_app_config
            from neoresist.paths import config_dir

            cfg = get_app_config()
            strategy = get_strategy(active_strategy_id or cfg.defaults.scoring_profile_id)

            prof_dir = config_dir() / "scoring_profiles"
            prof_names = sorted(p.stem for p in prof_dir.glob("*.yaml"))

            weights_table = dbc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Component"), html.Th("Weight")])),
                        html.Tbody(
                            [
                                html.Tr([html.Td("expression_norm"), html.Td(f"{strategy.weights.get('expression_norm', 0.0):.3f}")]),
                                html.Tr([html.Td("presentation"), html.Td(f"{strategy.weights.get('presentation', 0.0):.3f}")]),
                                html.Tr([html.Td("ccf"), html.Td(f"{strategy.weights.get('ccf', 0.0):.3f}")]),
                                html.Tr([html.Td("self_dissimilarity"), html.Td(f"{strategy.weights.get('self_dissimilarity', 0.0):.3f}")]),
                                html.Tr([html.Td("escape_penalty"), html.Td(f"{strategy.escape_penalty_weight:.3f}")]),
                            ]
                        ),
                    ],
                bordered=False,
                size="sm",
                class_name="mb-2",
            )

            return html.Div(
                [
                    html.Div(
                        [
                            dbc.Badge(f"Dataset: {cfg.defaults.dataset_id}", color="secondary", className="me-1"),
                            dbc.Badge(
                                f"Active strategy: {strategy.strategy_id}",
                                color="info",
                                className="me-1",
                            ),
                            dbc.Badge(f"ID: {strategy.strategy_id}", color="dark", className="me-1"),
                            dbc.Badge(
                                f"Rule profile: {strategy.rule_profile_id} v{strategy.rule_version}",
                                color="warning",
                                className="me-1",
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(strategy.description or "", className="mb-2"),
                    html.Div(
                        "The ranking engine is modular: saved strategies can reweight evidence without changing the underlying app.",
                        className="mb-2",
                    ),
                    html.Div(
                        [
                            html.Strong("Current structure: "),
                            f"expression {strategy.weights.get('expression_norm', 0.0):.2f}, "
                            f"presentation {strategy.weights.get('presentation', 0.0):.2f}, "
                            f"CCF {strategy.weights.get('ccf', 0.0):.2f}, "
                            f"self-dissimilarity {strategy.weights.get('self_dissimilarity', 0.0):.2f}, "
                            f"escape {strategy.escape_penalty_weight:.2f}",
                        ],
                        className="mb-2",
                    ),
                    dbc.Button("Inspect or edit strategies", id="jump-to-advanced-inline-btn", color="info", outline=True, class_name="mt-1"),
                    html.Div(
                        f"Saved/default profiles available: {', '.join(prof_names) if prof_names else 'none found'}.",
                        className="text-muted",
                    ),
                ]
            )
        except Exception:
            return html.Div(
                [
                    html.Div("Modular profile metadata unavailable in current environment.", className="mb-1"),
                    html.Div("Fallback structure: expression_norm, presentation, ccf, self_dissimilarity, escape_penalty.", className="text-muted"),
                ]
            )

    @app.callback(
        Output("active-strategy-select", "options"),
        Output("compare-strategies-select", "options"),
        Output("active-strategy-select", "value"),
        Output("compare-strategies-select", "value"),
        Input("strategy-refresh-store", "data"),
        State("active-strategy-select", "value"),
        State("compare-strategies-select", "value"),
    )
    def refresh_strategy_selectors(_refresh, active_value, compare_value):
        strategies = list_strategies()
        options = [
            {
                "label": f"{s.display_name} ({'built-in' if s.origin == 'builtin' else 'local'})",
                "value": s.strategy_id,
            }
            for s in strategies
        ]
        valid_ids = {s.strategy_id for s in strategies}
        active = active_value if active_value in valid_ids else (strategies[0].strategy_id if strategies else None)
        compare = [str(x) for x in (compare_value or []) if str(x) in valid_ids]
        if active and active not in compare:
            compare = [active, *compare][:6]
        return options, options, active, compare

    @app.callback(
        Output("strategy-edit-store", "data"),
        Output("strategy-save-status", "children", allow_duplicate=True),
        Input("active-strategy-select", "value"),
        Input("clone-strategy-btn", "n_clicks"),
        Input("reset-strategy-editor-btn", "n_clicks"),
        Input("load-active-strategy-btn", "n_clicks"),
        State("strategy-edit-store", "data"),
        prevent_initial_call=True,
    )
    def load_strategy_editor(active_strategy_id, clone_clicks, reset_clicks, load_clicks, existing_store):
        tid = callback_context.triggered_id
        if not active_strategy_id:
            raise PreventUpdate
        if tid in {"active-strategy-select", "reset-strategy-editor-btn", "load-active-strategy-btn"}:
            strategy = get_strategy(active_strategy_id)
            return serialize_strategy(strategy), html.Span("Editor synced to active strategy.", className="text-info")
        if tid == "clone-strategy-btn":
            clone = clone_strategy(active_strategy_id)
            log_audit_event(
                event_type="clone_strategy",
                strategy=clone,
                app_version=__version__,
                metadata={"source_strategy_id": active_strategy_id},
            )
            return serialize_strategy(clone), html.Span(
                f"Cloned {active_strategy_id} into local editor. Save to persist.", className="text-warning"
            )
        if existing_store:
            return existing_store, no_update
        raise PreventUpdate

    @app.callback(
        Output("strategy-name-input", "value"),
        Output("strategy-description-input", "value"),
        Output("weight-expression-input", "value"),
        Output("weight-presentation-input", "value"),
        Output("weight-ccf-input", "value"),
        Output("weight-self-input", "value"),
        Output("escape-penalty-input", "value"),
        Output("blend-expression-input", "value"),
        Output("blend-ccf-input", "value"),
        Output("expression-cap-input", "value"),
        Output("tier1-threshold-input", "value"),
        Output("tier2-threshold-input", "value"),
        Output("strategy-metadata-panel", "children"),
        Input("strategy-edit-store", "data"),
    )
    def hydrate_strategy_editor(strategy_payload):
        if not strategy_payload:
            blank = ""
            return blank, blank, 0.2, 0.3, 0.3, 0.1, -0.2, 0.85, 0.85, 1000, 0.7, 0.4, html.Div(
                "Select a strategy to inspect or clone.", className="text-muted small"
            )
        strategy = deserialize_strategy(strategy_payload)
        meta = html.Div(
            [
                dbc.Badge(f"Strategy id: {strategy.strategy_id}", color="secondary", className="me-1 mb-1"),
                dbc.Badge(f"Origin: {strategy.origin}", color="info", className="me-1 mb-1"),
                dbc.Badge(f"Parent: {strategy.parent_strategy_id or '—'}", color="dark", className="me-1 mb-1"),
                dbc.Badge(f"Scoring profile: {strategy.scoring_profile_id}", color="primary", className="me-1 mb-1"),
                dbc.Badge(f"Rule profile: {strategy.rule_profile_id}", color="warning", className="me-1 mb-1"),
                html.Div(
                    f"Version {strategy.strategy_version} · last modified {strategy.last_modified or 'built-in'}",
                    className="text-muted small mt-2",
                ),
            ]
        )
        return (
            strategy.display_name,
            strategy.description,
            strategy.weights.get("expression_norm", 0.0),
            strategy.weights.get("presentation", 0.0),
            strategy.weights.get("ccf", 0.0),
            strategy.weights.get("self_dissimilarity", 0.0),
            strategy.escape_penalty_weight,
            strategy.blend_expression,
            strategy.blend_ccf,
            strategy.expression_tpm_cap,
            strategy.tier1_above,
            strategy.tier2_above,
            meta,
        )

    @app.callback(
        Output("strategy-refresh-store", "data"),
        Output("strategy-edit-store", "data", allow_duplicate=True),
        Output("strategy-save-status", "children", allow_duplicate=True),
        Input("save-strategy-btn", "n_clicks"),
        State("strategy-edit-store", "data"),
        State("strategy-name-input", "value"),
        State("strategy-description-input", "value"),
        State("weight-expression-input", "value"),
        State("weight-presentation-input", "value"),
        State("weight-ccf-input", "value"),
        State("weight-self-input", "value"),
        State("escape-penalty-input", "value"),
        State("blend-expression-input", "value"),
        State("blend-ccf-input", "value"),
        State("expression-cap-input", "value"),
        State("tier1-threshold-input", "value"),
        State("tier2-threshold-input", "value"),
        State("strategy-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def persist_strategy(
        n_clicks,
        strategy_payload,
        display_name,
        description,
        w_expr,
        w_pres,
        w_ccf,
        w_self,
        escape_penalty,
        blend_expr,
        blend_ccf,
        expr_cap,
        tier1,
        tier2,
        refresh_state,
    ):
        if not n_clicks or not strategy_payload:
            raise PreventUpdate
        base = deserialize_strategy(strategy_payload)
        try:
            tier1_f = float(tier1 if tier1 is not None else base.tier1_above)
            tier2_f = float(tier2 if tier2 is not None else base.tier2_above)
        except (TypeError, ValueError):
            return no_update, no_update, dbc.Alert(
                "Strategy thresholds must be numeric.",
                color="danger",
                className="py-2 mb-0",
            )
        if tier1_f <= tier2_f:
            return no_update, no_update, dbc.Alert(
                "Tier 1 threshold must be greater than Tier 2 threshold.",
                color="danger",
                className="py-2 mb-0",
            )
        changes: dict[str, Any] = {}
        strategy_id = base.strategy_id
        origin = base.origin
        parent_id = base.parent_strategy_id
        if base.origin == "builtin":
            origin = "user_variant"
            parent_id = base.strategy_id
            from neoresist.strategy_registry import _slugify  # type: ignore

            strategy_id = f"user-{_slugify(display_name or base.display_name)}"
        updated = base.__class__(
            strategy_id=strategy_id,
            display_name=str(display_name or base.display_name).strip() or base.display_name,
            description=str(description or "").strip(),
            origin=origin,
            parent_strategy_id=parent_id,
            scoring_profile_id=base.scoring_profile_id,
            scoring_version=base.scoring_version,
            rule_profile_id=base.rule_profile_id,
            rule_version=base.rule_version,
            strategy_version=base.strategy_version,
            created_at=base.created_at,
            last_modified=base.last_modified,
            expression_tpm_cap=float(expr_cap or base.expression_tpm_cap),
            blend_expression=float(blend_expr if blend_expr is not None else base.blend_expression),
            blend_ccf=float(blend_ccf if blend_ccf is not None else base.blend_ccf),
            weights={
                "expression_norm": float(w_expr if w_expr is not None else base.weights["expression_norm"]),
                "presentation": float(w_pres if w_pres is not None else base.weights["presentation"]),
                "ccf": float(w_ccf if w_ccf is not None else base.weights["ccf"]),
                "self_dissimilarity": float(w_self if w_self is not None else base.weights["self_dissimilarity"]),
            },
            escape_penalty_weight=float(escape_penalty if escape_penalty is not None else base.escape_penalty_weight),
            tier1_above=tier1_f,
            tier2_above=tier2_f,
        )
        for key, new_val in serialize_strategy(updated).items():
            if serialize_strategy(base).get(key) != new_val:
                changes[key] = {"before": serialize_strategy(base).get(key), "after": new_val}
        saved = save_strategy(updated)
        log_audit_event(
            event_type="save_strategy",
            strategy=saved,
            app_version=__version__,
            changes=changes,
        )
        revision = int((refresh_state or {}).get("revision", 0)) + 1
        return {"revision": revision}, serialize_strategy(saved), dbc.Alert(
            f"Saved local strategy: {saved.display_name} ({saved.strategy_id}). Select it in Active strategy to use it.",
            color="success",
            className="py-2 mb-0",
        )

    @app.callback(
        Output("strategy-comparison-table", "children"),
        Output("strategy-consensus-table", "children"),
        Output("strategy-coherence-summary", "children"),
        Output("strategy-audit-table", "children"),
        Input("active-strategy-select", "value"),
        Input("compare-strategies-select", "value"),
        Input("strategy-refresh-store", "data"),
        Input("tier-filter", "value"),
        Input("hla-filter", "value"),
        Input("exclusion-filter", "value"),
        Input("rl-range", "value"),
        Input("expr-range", "value"),
        Input("ccf-range", "value"),
        Input("search", "value"),
        Input("selected-key-store", "data"),
        State("cohort-parquet-path", "data"),
    )
    def render_strategy_lab(
        active_strategy_id,
        compare_ids,
        _refresh,
        tiers,
        hlas,
        exclusions,
        rl_range,
        expr_range,
        ccf_range,
        search,
        selected_key,
        cohort_path_override,
    ):
        if not active_strategy_id:
            empty = html.Div("No strategy selected.", className="text-muted small")
            return empty, empty, empty, empty
        try:
            df, _path, _searched = load_cohort_for_dash(alt_path=cohort_path_override)
        except Exception:
            empty = html.Div("Advanced Strategies is waiting for a valid cohort file.", className="text-muted small")
            return empty, empty, empty, empty
        fc = filter_candidates(
            df,
            tiers=[int(x) for x in (tiers or [])] or [1, 2, 3],
            hlas=[str(x) for x in (hlas or [])] if hlas else [],
            exclusion_any=[str(x) for x in (exclusions or [])] if exclusions else [],
            rl_range=list(rl_range or [0, 1]),
            expr_range=list(expr_range or [0, 1000]),
            ccf_range=list(ccf_range or [0, 1]),
            search=search or "",
        )
        strategy_ids = []
        for sid in [active_strategy_id, *list(compare_ids or [])]:
            if sid and sid not in strategy_ids:
                strategy_ids.append(str(sid))
        strategy_frames: dict[str, pd.DataFrame] = {}
        strategy_objs = {sid: get_strategy(sid) for sid in strategy_ids}
        for sid, strategy in strategy_objs.items():
            strategy_frames[sid] = score_candidates_for_strategy(fc, strategy)
        active_top_keys = set(
            strategy_frames[active_strategy_id]
            .sort_values("rl_priority", ascending=False)
            .head(25)["candidate_key"]
            .astype(str)
            .tolist()
        )
        comparison_rows = []
        for sid in strategy_ids:
            summary = summarize_strategy_run(
                strategy_frames[sid],
                strategy_objs[sid],
                reference_keys=active_top_keys,
            )
            comparison_rows.append(summary)
        comparison_table = _render_records_table(
            comparison_rows,
            [
                ("display_name", "Strategy"),
                ("origin", "Origin"),
                ("tier1_candidates", "Tier 1"),
                ("tier2_candidates", "Tier 2"),
                ("tier3_candidates", "Tier 3"),
                ("mean_rl_priority", "Mean RL"),
                ("top_rl_priority", "Top RL"),
                ("mean_coherence", "Mean coherence"),
                ("overlap_with_active", "Overlap vs active"),
                ("divergence_pct", "Divergence %"),
            ],
            "No strategy comparison data yet.",
        )

        consensus = build_consensus_table(strategy_frames).head(12)
        consensus_table = _render_records_table(
            consensus.to_dict("records"),
            [
                ("patient_id", "Patient"),
                ("hla_allele", "HLA"),
                ("gene", "Gene"),
                ("mutant_peptide", "Peptide"),
                ("consensus_score", "Consensus"),
                ("consensus_agreement_count", "Agreement"),
                ("rank_dispersion", "Rank dispersion"),
                ("mean_coherence", "Mean coherence"),
            ],
            "Consensus rows will appear after selecting at least one strategy.",
        )

        active_frame = strategy_frames[active_strategy_id]
        coherence_bits: list[Any] = []
        coherence_bits.append(
            dbc.Badge(
                f"Active mean coherence: {float(active_frame['coherence_score'].mean()):.3f}" if not active_frame.empty else "Active mean coherence: —",
                color="success",
                className="me-1 mb-1",
            )
        )
        if not consensus.empty:
            coherence_bits.append(
                dbc.Badge(
                    f"Consensus top mean: {float(consensus['consensus_score'].head(5).mean()):.3f}",
                    color="primary",
                    className="me-1 mb-1",
                )
            )
        if isinstance(selected_key, dict) and selected_key.get("patient_id") and selected_key.get("hla_allele"):
            pid = str(selected_key["patient_id"])
            hla = str(selected_key["hla_allele"])
            sub = active_frame[
                (active_frame["patient_id"].astype(str) == pid)
                & (active_frame["hla_allele"].astype(str) == hla)
            ].sort_values("rl_priority", ascending=False)
            if not sub.empty:
                top = sub.iloc[0]
                reasons = top.get("coherence_reasons") or []
                coherence_bits.append(
                    html.Div(
                        [
                            html.Div(
                                f"Selected {pid} · {hla}: top candidate coherence {float(top.get('coherence_score', 0.0)):.3f}",
                                className="mt-2",
                            ),
                            html.Ul([html.Li(str(r)) for r in reasons[:4]] or [html.Li("No coherence penalties triggered.")], className="small mb-0"),
                        ]
                    )
                )
        audit_rows = read_audit_events(limit=20)
        audit_table = _render_records_table(
            audit_rows,
            [
                ("timestamp", "Timestamp"),
                ("event_type", "Event"),
                ("strategy_id", "Strategy"),
                ("strategy_origin", "Origin"),
                ("parent_strategy_id", "Parent"),
            ],
            "Audit events will appear after you clone, save, or activate strategies.",
        )
        return comparison_table, consensus_table, html.Div(coherence_bits), audit_table

    @app.callback(
        Output("strategy-save-status", "children", allow_duplicate=True),
        Input("active-strategy-select", "value"),
        Input("compare-strategies-select", "value"),
        prevent_initial_call=True,
    )
    def audit_strategy_selection(active_strategy_id, compare_ids):
        tid = callback_context.triggered_id
        if not active_strategy_id:
            raise PreventUpdate
        strategy = get_strategy(active_strategy_id)
        if tid == "active-strategy-select":
            log_audit_event(
                event_type="select_active_strategy",
                strategy=strategy,
                app_version=__version__,
            )
            return html.Span(f"Active strategy set to {strategy.strategy_id}.", className="text-info")
        if tid == "compare-strategies-select":
            log_audit_event(
                event_type="compare_strategy_set",
                strategy=strategy,
                app_version=__version__,
                metadata={"compare_ids": [str(x) for x in (compare_ids or [])]},
            )
            return html.Span("Comparison set updated.", className="text-info")
        raise PreventUpdate

    @app.callback(
        Output("header-subtitle", "children"),
        Input("nav", "value"),
    )
    def header_subtitle(_nav):
        try:
            from neoresist.config import get_app_config

            return get_app_config().branding.subtitle
        except Exception:
            return "TCGA-SARC — ResistanceLoop v1 evidence-aware neoantigen qualification"

    @app.callback(
        Output("cohort-sidebar-controls", "style"),
        Output("sidebar-context-note", "children"),
        Output("advanced-strategies-simple-note", "style"),
        Output("advanced-strategies-section", "style"),
        Output("overview-section", "style"),
        Output("patients-section", "style"),
        Output("upload-section", "style"),
        Output("cases-section", "style"),
        Output("about-section", "style"),
        Output("pipeline-status-section", "style"),
        Output("scatter-card", "style"),
        Output("simple-hero-shell", "style"),
        Output("expert-hero-shell", "style"),
        Output("expert-methodology-shell", "style"),
        Output("expert-strategy-shell", "style"),
        Output("simple-answer-shell", "style"),
        Output("expert-diagnostics-shell", "style"),
        Output("expert-two-panel-shell", "style"),
        Output("expert-extra-diagnostics-shell", "style"),
        Output("scatter-controls-shell", "style"),
        Input("nav", "value"),
        Input("ui-mode", "value"),
    )
    def toggle_sections(nav, ui_mode):
        state = resolve_view_state(nav, ui_mode)
        return (
            section_style(bool(state["show_filters"])),
            str(state["note"]),
            section_style(bool(state["advanced_simple_note"])),
            section_style(bool(state["advanced_section"])),
            section_style(bool(state["overview_section"])),
            section_style(bool(state["patients_section"])),
            section_style(bool(state["upload_section"])),
            section_style(bool(state["cases_section"])),
            section_style(bool(state["about_section"])),
            section_style(bool(state["pipeline_status_section"])),
            section_style(bool(state["scatter_card"])),
            section_style(bool(state["simple_hero_shell"])),
            section_style(bool(state["expert_hero_shell"])),
            section_style(bool(state["expert_methodology_shell"])),
            section_style(bool(state["expert_strategy_shell"])),
            section_style(bool(state["simple_answer_shell"])),
            section_style(bool(state["expert_diagnostics_shell"])),
            section_style(bool(state["expert_two_panel_shell"])),
            section_style(bool(state["expert_extra_diagnostics_shell"])),
            section_style(bool(state["scatter_controls_shell"])),
        )

    @app.callback(
        Output("upload-expert-controls", "style"),
        Input("ui-mode", "value"),
    )
    def toggle_upload_expert_controls(ui_mode):
        return section_style(str(ui_mode or "Simple") == "Expert")

    @app.callback(
        Output("tumor-type-select", "value", allow_duplicate=True),
        Input("upload-maf", "filename"),
        Input("upload-maf-global", "filename"),
        State("tumor-type-select", "value"),
        prevent_initial_call=True,
    )
    def prefill_tumor_type_from_filename(filename, filename_global, current_value):
        if str(current_value or "").strip():
            raise PreventUpdate
        detected = infer_tumor_type_from_text(filename or filename_global or "")
        if detected == "UNKNOWN":
            raise PreventUpdate
        return detected

    @app.callback(
        Output("nav", "value", allow_duplicate=True),
        Input("jump-to-advanced-btn", "n_clicks"),
        Input("jump-to-advanced-inline-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def jump_to_advanced(n1, n2):
        tid = callback_context.triggered_id
        if tid == "jump-to-advanced-btn" and (n1 or 0) < 1:
            raise PreventUpdate
        if tid == "jump-to-advanced-inline-btn" and (n2 or 0) < 1:
            raise PreventUpdate
        return "Advanced Strategies"

    @app.callback(
        Output("nav", "value", allow_duplicate=True),
        Input("nav", "value"),
        Input("ui-mode", "value"),
        prevent_initial_call=True,
    )
    def enforce_simple_mode_lands_on_overview(nav_value, ui_mode):
        # Keep Simple mode on supported routes and avoid blank expert-only views.
        if str(ui_mode or "Simple") == "Simple" and str(nav_value or "") in EXPERT_ONLY_NAV:
            return "Overview"
        raise PreventUpdate

    @app.callback(
        Output("case-status-interval", "disabled"),
        Input("nav", "value"),
        Input("active-case-store", "data"),
        Input("case-select", "value"),
        Input("upload-result-store", "data"),
    )
    def control_case_polling(nav, active_case, selected_case, upload_case):
        # Avoid constant global UI "Updating..." churn when case polling is not needed.
        if str(nav or "Overview") not in {"Upload", "Cases", "Advanced Strategies"}:
            return True

        case_id = None
        if selected_case:
            case_id = str(selected_case)
        elif isinstance(active_case, dict) and active_case.get("case_id"):
            case_id = str(active_case.get("case_id"))
        elif isinstance(upload_case, dict) and upload_case.get("case_id"):
            case_id = str(upload_case.get("case_id"))

        if not case_id:
            return True

        try:
            manifest = read_case_manifest(case_id)
        except Exception:
            return True
        if not manifest:
            return True

        enabled_map = manifest.get("enabled_modules", {}) or {}
        try:
            specs = load_module_schema()
        except Exception:
            return True
        enabled_modules = [module_id for module_id in specs if bool(enabled_map.get(module_id, True))]
        if not enabled_modules:
            return True

        for module_id in enabled_modules:
            try:
                status = str(read_module_status(case_id, module_id).get("status", "pending")).lower()
            except Exception:
                status = "pending"
            if status in {"pending", "running"}:
                return False
        return True

    @app.callback(
        Output("global-progress-shell", "style"),
        Output("global-progress-bar", "value"),
        Output("global-progress-bar", "label"),
        Output("global-progress-bar", "color"),
        Output("global-progress-bar", "animated"),
        Output("global-progress-meta", "children"),
        Input("case-status-interval", "n_intervals"),
        Input("active-case-store", "data"),
        Input("case-select", "value"),
        Input("upload-result-store", "data"),
        Input("nav", "value"),
    )
    def update_global_progress(_n, active_case, selected_case, upload_case, nav):
        show_shell = str(nav or "Overview") in {"Overview", "Upload", "Cases", "Advanced Strategies"}
        style = section_style(show_shell)

        case_id = None
        if selected_case:
            case_id = str(selected_case)
        elif isinstance(active_case, dict) and active_case.get("case_id"):
            case_id = str(active_case.get("case_id"))
        elif isinstance(upload_case, dict) and upload_case.get("case_id"):
            case_id = str(upload_case.get("case_id"))

        if not case_id:
            return style, 0, "No active case", "secondary", False, "Upload a patient file to start a tracked run."

        try:
            manifest = read_case_manifest(case_id)
            if not manifest:
                return style, 0, "Case not found", "warning", False, f"Selected case {case_id} is missing on disk."
            specs = load_module_schema()
            enabled_map = manifest.get("enabled_modules", {}) or {}
            statuses = {module_id: read_module_status(case_id, module_id) for module_id in specs}

            enabled_modules = [m for m in specs if bool(enabled_map.get(m, True))]
            if not enabled_modules:
                return style, 100, "No modules enabled", "secondary", False, f"Case {case_id}: enable at least one module to run."

            done_states = {"complete", "disabled"}
            done_count = sum(1 for m in enabled_modules if str(statuses.get(m, {}).get("status", "pending")) in done_states)
            progress = int(round((done_count / max(len(enabled_modules), 1)) * 100))

            running = [m for m in enabled_modules if str(statuses.get(m, {}).get("status", "")) == "running"]
            failed = [m for m in enabled_modules if str(statuses.get(m, {}).get("status", "")) == "failed"]
            unavailable = [m for m in enabled_modules if str(statuses.get(m, {}).get("status", "")) == "unavailable"]
            pending = [m for m in enabled_modules if str(statuses.get(m, {}).get("status", "")) == "pending"]

            if failed:
                color = "danger"
                label = f"{progress}% - action needed"
                animated = False
                next_step = statuses.get(failed[0], {}).get("message") or "A module failed. Open Cases and use rerun controls."
            elif unavailable:
                color = "info"
                label = f"Pipeline: {done_count}/{len(enabled_modules)} modules complete"
                animated = False
                next_step = f"{len(unavailable)} optional modules are available with additional data or runtime setup."
            elif running:
                color = "info"
                label = f"{progress}% - processing"
                animated = True
                next_step = statuses.get(running[0], {}).get("message") or "Modules are running. Safe to wait on this page."
            elif pending:
                color = "secondary"
                label = f"{progress}% - queued"
                animated = True
                next_step = statuses.get(pending[0], {}).get("message") or "Modules are queued. Run enabled modules if nothing starts."
            else:
                color = "success"
                label = "100% - results ready"
                animated = False
                next_step = "All enabled modules finished. You can review ranked results now."

            meta = (
                f"Case {case_id} | complete: {done_count}/{len(enabled_modules)} | "
                f"running: {len(running)} | pending: {len(pending)} | failed: {len(failed)} | "
                f"unavailable: {len(unavailable)} | next: {next_step}"
            )
            return style, progress, label, color, animated, meta
        except Exception as exc:
            return style, 0, "Progress unavailable", "warning", False, f"Could not read progress for case {case_id}: {exc}"

    @app.callback(
        Output("overview-modularity-summary", "children"),
        Output("advanced-strategy-summary", "children"),
        Output("strategy-library-panel", "children"),
        Output("overview-case-summary", "children"),
        Output("demo-mode-banner", "children"),
        Input("active-strategy-select", "value"),
        Input("strategy-refresh-store", "data"),
        Input("active-case-store", "data"),
        Input("upload-result-store", "data"),
        Input("ui-mode", "value"),
    )
    def modularity_summaries(active_strategy_id, _refresh, _active_case, _upload_case, ui_mode):
        strategies = list_strategies()
        active = get_strategy(active_strategy_id or strategies[0].strategy_id)
        saved_names = [s.display_name for s in strategies[:8]]
        recent_cases = list_cases(limit=3)
        compact = html.Div(
            [
                dbc.Badge(f"Active strategy: {active.strategy_id}", color="info", className="me-1 mb-1"),
                dbc.Badge(f"Rule thresholds: {active.tier1_above:.2f}/{active.tier2_above:.2f}", color="secondary", className="me-1 mb-1"),
                html.P(
                    "The product moat is modular scoring: the same cohort can be re-ranked through saved strategies, compared, audited, and upgraded over time.",
                    className="mb-0 mt-2",
                ),
            ]
        )
        advanced = html.Div(
            [
                dbc.Badge(f"Active: {active.display_name}", color="info", className="me-1 mb-1"),
                dbc.Badge(f"Origin: {active.origin}", color="secondary", className="me-1 mb-1"),
                dbc.Badge(f"Parent: {active.parent_strategy_id or 'built-in'}", color="dark", className="me-1 mb-1"),
                html.P(
                    "This workspace is where you make the proprietary engine visible: compare profiles, inspect consensus, preserve audited variants, and tune evidence weighting deliberately.",
                    className="mt-2 mb-0",
                ),
            ]
        )
        library = html.Div(
            [
                html.Ul([html.Li(name) for name in saved_names], className="mb-0"),
            ]
        )
        case_summary = html.Div(
            [
                html.H6("Recent uploaded cases", className="mb-2"),
                html.Div(
                    [html.Div(f"{item.get('sample_id', 'sample')} — {item.get('case_id', 'case')}", className="small mb-1") for item in recent_cases]
                    if recent_cases
                    else [html.Div("No uploaded cases yet. Use Upload to create a persistent case workflow.", className="small text-muted")]
                ),
            ]
        )
        banner = html.Div()
        if recent_cases and recent_cases[0].get("case_id"):
            cid = str(recent_cases[0]["case_id"])
            manifest = read_case_manifest(cid)
            stubbed: list[str] = []
            for module_id in load_module_schema():
                st = read_module_status(cid, module_id)
                if str(st.get("status") or "").lower() == "unavailable":
                    stubbed.append(module_id)
            banner_items: list[Any] = []
            if stubbed:
                pretty = {
                    "expression_join": "Expression",
                    "clonality_pyclone_vi": "Clonality",
                    "presentation_netctlpan": "Presentation",
                    "escape_lohhla": "Escape",
                    "recognition_foreignness": "Recognition",
                }
                labels = ", ".join(pretty.get(module_id, module_id) for module_id in stubbed)
                extra = (
                    " Open Expert mode and Pipeline Status for runtime setup details."
                    if str(ui_mode or "Simple") == "Expert"
                    else ""
                )
                banner_items.append(
                    dbc.Alert(
                        f"Optional modules: {labels} use cohort-level estimates in this demo build.{extra}",
                        color="info",
                        className="py-2 mb-2",
                    )
                )
            if bool(manifest.get("germline_contamination_suspected", False)):
                banner_items.append(
                    dbc.Alert(
                        "⚠ Possible germline contamination detected (VAF distribution bimodal). Confirm this is a somatic-only variant file.",
                        color="warning",
                        className="py-2 mb-2",
                    )
                )
            banner = html.Div(banner_items) if banner_items else html.Div()
        return compact, advanced, library, case_summary, banner

    @app.callback(
        Output("expert-module-stack-panel", "children"),
        Output("expert-module-stack-description", "children"),
        Output("expert-strategy-library-browser", "children"),
        Output("expert-confidence-cards", "children"),
        Output("expert-module-health-fig", "figure"),
        Output("expert-strategy-delta-fig", "figure"),
        Input("active-strategy-select", "value"),
        Input("active-case-store", "data"),
        Input("upload-result-store", "data"),
        Input("case-select", "value"),
        Input("tool-registry-store", "data"),
    )
    def render_expert_workspace(active_strategy_id, _active_case, _upload_case, selected_case, tool_registry_state):
        strategies = list_strategies()
        active = get_strategy(active_strategy_id or strategies[0].strategy_id)
        specs = load_module_schema()
        if selected_case:
            case_id = str(selected_case)
        else:
            recent_cases = list_cases(limit=1)
            case_id = str(recent_cases[0].get("case_id")) if recent_cases and recent_cases[0].get("case_id") else None
        statuses = {module_id: read_module_status(case_id, module_id) for module_id in specs} if case_id else {}
        try:
            hla_df, _ = load_qualified_candidates()
            hla_unknown_active = bool(hla_df.get("hla_allele", pd.Series(dtype=object)).map(_is_unknown_hla).any()) if not hla_df.empty else False
        except Exception:
            hla_unknown_active = False
        if hla_unknown_active:
            for module_id in ("neoantigen_generation", "presentation_netctlpan"):
                if module_id not in statuses:
                    continue
                status_obj = dict(statuses[module_id] or {})
                module_state = str(status_obj.get("status") or "pending").lower()
                if module_state == "complete":
                    status_obj["confidence"] = "MEDIUM"
                    status_obj["confidence_state"] = "MEDIUM"
                    status_obj["message"] = "HLA allele not provided — binding predictions are approximate."
                statuses[module_id] = status_obj
        module_stack = html.Div(
            [
                dcc.Graph(
                    id="expert-pipeline-diagram",
                    figure=build_pipeline_diagram(statuses, case_id or "no-run"),
                    config={"displayModeBar": False},
                ),
            ]
        )

        current_weights = _axis_weights_from_strategy(active)
        strategy_browser = build_strategy_editor(current_weights, strategies)

        confidence_bits: list[Any] = []
        if statuses:
            for module_id in specs:
                module_state = str(statuses.get(module_id, {}).get("status", "pending"))
                confidence_label, confidence_state = derive_module_confidence(statuses.get(module_id, {}))
                color = {
                    "high": "success",
                    "medium": "warning",
                    "low": "danger",
                    "pending": "secondary",
                    "unavailable": "secondary",
                    "failed": "danger",
                    "running": "info",
                }.get(confidence_state, "secondary")
                badge_id = f"expert-confidence-{module_id}"
                badge_text = f"{module_id.replace('_', ' ')}: {confidence_label}"
                confidence_bits.append(
                    dbc.Badge(
                        badge_text,
                        color=color,
                        className="me-1 mb-1",
                        id=badge_id,
                    )
                )
                if hla_unknown_active and module_id in {"neoantigen_generation", "presentation_netctlpan"}:
                    confidence_bits.append(
                        dbc.Tooltip(
                            "HLA allele not provided — binding predictions are approximate.",
                            target=badge_id,
                        )
                    )
        else:
            confidence_bits.append(html.Div("No persisted case yet; confidence cards appear after upload/run.", className="small text-muted"))
        confidence_cards = html.Div(confidence_bits)

        status_counts = Counter(str(status.get("status", "pending")) for status in statuses.values()) if statuses else Counter({"pending": len(specs)})
        status_df = pd.DataFrame({"status": list(status_counts.keys()), "count": list(status_counts.values())})
        module_health_fig = px.bar(status_df, x="status", y="count", color="status", title="")
        module_health_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            margin=dict(l=30, r=20, t=10, b=30),
            height=260,
            showlegend=False,
        )

        weight_df = pd.DataFrame(
            {
                "component": [
                    COMPONENT_DISPLAY_NAMES["expression"],
                    COMPONENT_DISPLAY_NAMES["presentation"],
                    COMPONENT_DISPLAY_NAMES["ccf"],
                    COMPONENT_DISPLAY_NAMES["self_dissimilarity"],
                    COMPONENT_DISPLAY_NAMES["escape"],
                ],
                "weight": [
                    current_weights["expression"],
                    current_weights["presentation"],
                    current_weights["ccf"],
                    current_weights["self_dissimilarity"],
                    current_weights["escape"],
                ],
            }
        )
        strategy_delta_fig = px.bar(weight_df, x="component", y="weight", title="")
        strategy_delta_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            margin=dict(l=30, r=20, t=10, b=30),
            height=260,
            showlegend=False,
        )
        strategy_delta_fig.update_yaxes(range=[0, 1])

        return (
            module_stack,
            "Hover a module to see details.",
            strategy_browser,
            confidence_cards,
            module_health_fig,
            strategy_delta_fig,
        )

    @app.callback(
        Output("expert-module-stack-description", "children", allow_duplicate=True),
        Input("expert-pipeline-diagram", "hoverData"),
        prevent_initial_call=True,
    )
    def update_module_stack_description(hover_data):
        points = (hover_data or {}).get("points") or []
        if not points:
            return "Hover a module to see details."
        cd = points[0].get("customdata") or []
        if len(cd) < 7:
            raise PreventUpdate
        return f"{cd[1]} | status: {cd[2]} | confidence: {cd[3]} | tool: {cd[4]} | run: {cd[5]} | updated: {cd[6]}"

    @app.callback(
        Output("expert-strategy-radar", "figure"),
        Input("expert-weight-binding", "value"),
        Input("expert-weight-presentation", "value"),
        Input("expert-weight-expression", "value"),
        Input("expert-weight-recognition", "value"),
        Input("expert-weight-resistance", "value"),
        Input("expert-strategy-load-select", "value"),
    )
    def refresh_strategy_radar(binding, presentation, expression, recognition, resistance, compare_strategy_id):
        current_weights = {
            "expression": float(binding or 0.0),
            "presentation": float(presentation or 0.0),
            "ccf": float(expression or 0.0),
            "self_dissimilarity": float(recognition or 0.0),
            "escape": float(resistance or 0.0),
        }
        compare_weights = None
        compare_name = None
        if compare_strategy_id:
            try:
                strategy = get_strategy(str(compare_strategy_id))
                compare_weights = _axis_weights_from_strategy(strategy)
                compare_name = strategy.display_name
            except Exception:
                compare_weights = None
        return build_strategy_radar_figure(current_weights, list(compare_weights.values()) if compare_weights else None, compare_name)

    @app.callback(
        Output("expert-weight-binding", "value"),
        Output("expert-weight-presentation", "value"),
        Output("expert-weight-expression", "value"),
        Output("expert-weight-recognition", "value"),
        Output("expert-weight-resistance", "value"),
        Input("expert-strategy-load-select", "value"),
        prevent_initial_call=True,
    )
    def load_strategy_into_editor(strategy_id):
        if not strategy_id:
            raise PreventUpdate
        strategy = get_strategy(str(strategy_id))
        axis = _axis_weights_from_strategy(strategy)
        return axis["expression"], axis["presentation"], axis["ccf"], axis["self_dissimilarity"], axis["escape"]

    @app.callback(
        Output("expert-strategy-modal", "is_open"),
        Output("expert-strategy-save-status", "children", allow_duplicate=True),
        Output("strategy-refresh-store", "data", allow_duplicate=True),
        Input("expert-save-strategy-btn", "n_clicks"),
        Input("expert-strategy-modal-cancel", "n_clicks"),
        Input("expert-strategy-modal-confirm", "n_clicks"),
        State("expert-strategy-modal", "is_open"),
        State("expert-strategy-modal-input", "value"),
        State("expert-weight-binding", "value"),
        State("expert-weight-presentation", "value"),
        State("expert-weight-expression", "value"),
        State("expert-weight-recognition", "value"),
        State("expert-weight-resistance", "value"),
        State("expert-strategy-load-select", "value"),
        State("strategy-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def expert_strategy_modal(save_clicks, cancel_clicks, confirm_clicks, is_open, name_value, binding, presentation, expression, recognition, resistance, load_strategy_id, refresh_state):
        tid = callback_context.triggered_id
        if not any(int(click or 0) for click in (save_clicks, cancel_clicks, confirm_clicks)):
            raise PreventUpdate
        state = dict(refresh_state or {})
        if tid == "expert-save-strategy-btn":
            if not save_clicks:
                raise PreventUpdate
            return True, no_update, no_update
        if tid == "expert-strategy-modal-cancel":
            if not cancel_clicks:
                raise PreventUpdate
            return False, no_update, no_update
        if tid != "expert-strategy-modal-confirm" or not confirm_clicks:
            raise PreventUpdate
        base = get_strategy(load_strategy_id or state.get("applied_strategy_id") or list_strategies()[0].strategy_id)
        display_name = str(name_value or f"{base.display_name} Expert Variant").strip()
        updated = replace(
            base,
            display_name=display_name,
            weights={
                "expression_norm": float(binding or 0.0),
                "presentation": float(presentation or 0.0),
                "ccf": float(expression or 0.0),
                "self_dissimilarity": float(recognition or 0.0),
            },
            escape_penalty_weight=-abs(float(resistance or 0.0)),
        )
        saved = save_strategy(updated)
        try:
            NeoResistModuleRunner(config_path=str(repo_root() / "neoresist_md" / "config")).save_strategy(
                name=saved.strategy_id,
                weights={
                    "binding": float(binding or 0.0),
                    "presentation": float(presentation or 0.0),
                    "expression": float(expression or 0.0),
                    "recognition": float(recognition or 0.0),
                    "resistance": float(resistance or 0.0),
                },
                tool_selections={
                    "binding": "mhcflurry",
                    "presentation": "netctlpan",
                    "clonality": "pyclone_vi",
                    "recognition": "blosum62",
                },
                source="user_defined",
            )
        except Exception:
            pass
        state["loaded_strategy_id"] = saved.strategy_id
        state["revision"] = int(state.get("revision", 0)) + 1
        return (
            False,
            dbc.Alert(f"Saved {saved.display_name} ({saved.strategy_id}).", color="success", className="py-2 mb-0"),
            state,
        )

    @app.callback(
        Output("strategy-refresh-store", "data", allow_duplicate=True),
        Output("expert-strategy-save-status", "children", allow_duplicate=True),
        Input("expert-apply-strategy-btn", "n_clicks"),
        State("expert-weight-binding", "value"),
        State("expert-weight-presentation", "value"),
        State("expert-weight-expression", "value"),
        State("expert-weight-recognition", "value"),
        State("expert-weight-resistance", "value"),
        State("expert-strategy-load-select", "value"),
        State("strategy-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def apply_strategy_to_patient(n_clicks, binding, presentation, expression, recognition, resistance, load_strategy_id, refresh_state):
        if not n_clicks:
            raise PreventUpdate
        state = dict(refresh_state or {})
        state["applied_strategy_id"] = str(load_strategy_id or state.get("applied_strategy_id") or "")
        state["applied_weights"] = {
            "expression_norm": float(binding or 0.0),
            "presentation": float(presentation or 0.0),
            "ccf": float(expression or 0.0),
            "self_dissimilarity": float(recognition or 0.0),
            "escape_penalty_weight": -abs(float(resistance or 0.0)),
        }
        state["revision"] = int(state.get("revision", 0)) + 1
        return state, dbc.Alert("Applied strategy weights to the current patient view.", color="info", className="py-2 mb-0")

    @app.callback(
        Output("expert-bottom-drawer", "className"),
        Output("expert-bottom-drawer-content", "children"),
        Output("expert-drawer-context-store", "data"),
        Input("expert-pipeline-diagram", "clickData"),
        Input("scatter", "selectedData"),
        Input("expert-bottom-drawer-close", "n_clicks"),
        State("selected-key-store", "data"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def expert_drawer_router(pipeline_click, scatter_selected, close_clicks, selected_key, active_case):
        tid = callback_context.triggered_id
        if tid == "expert-bottom-drawer-close":
            return "expert-bottom-drawer", no_update, None

        case_id = _current_case_id(active_case)
        if tid == "expert-pipeline-diagram" and pipeline_click and pipeline_click.get("points"):
            point = pipeline_click["points"][0]
            cd = point.get("customdata") or []
            module_id = str(cd[0]) if len(cd) else ""
            if not module_id or not case_id:
                raise PreventUpdate
            status = read_module_status(case_id, module_id)
            artifacts = status.get("artifacts") or {}
            preview_df = pd.DataFrame()
            preview_path = None
            for path_str in artifacts.values():
                p = Path(str(path_str))
                if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}:
                    preview_path = p
                    try:
                        preview_df = pd.read_csv(p, sep="\t" if p.suffix.lower() in {".tsv", ".txt"} else ",")
                    except Exception:
                        preview_df = pd.DataFrame()
                    break
            if preview_df.empty and "preview" in status and isinstance(status.get("preview"), list):
                preview_df = pd.DataFrame(status.get("preview") or [])
            if selected_key and isinstance(selected_key, dict) and {"patient_id", "hla_allele"} <= set(selected_key):
                pid = str(selected_key["patient_id"])
                hla = str(selected_key["hla_allele"])
                for col in ("patient_id", "sample_barcode", "hla_allele"):
                    if col in preview_df.columns:
                        mask = preview_df[col].astype(str) == (pid if col != "hla_allele" else hla)
                        preview_df = preview_df.loc[mask] if mask.any() else preview_df
            preview = preview_df.head(10)
            table = (
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in preview.columns[:10]],
                    data=preview.iloc[:, :10].fillna("").to_dict("records"),
                    style_table={"overflowX": "auto"},
                    style_cell={"backgroundColor": "#161B22", "color": "#E6EDF3", "fontSize": "12px", "fontFamily": "JetBrains Mono, monospace"},
                    style_header={"backgroundColor": "#1C2128", "fontWeight": "bold"},
                )
                if not preview.empty
                else html.Div("No artifact preview available.", className="text-muted")
            )
            content = html.Div(
                [
                    html.Div(
                        [
                            html.Div(f"{module_id}", className="text-panel-title fw-bold"),
                            dbc.Button("×", id="expert-bottom-drawer-close", color="secondary", outline=True, size="sm"),
                        ],
                        className="d-flex justify-content-between align-items-start mb-2",
                    ),
                    html.Div(f"Run: {case_id}", className="text-muted text-micro mb-2"),
                    html.Div(f"Status: {status.get('status', 'pending')} · Confidence: {status.get('confidence_state', status.get('confidence', '—'))}", className="mb-2"),
                    html.Div("Raw artifact preview", className="text-nav fw-bold mb-2"),
                    table,
                    html.Div("Confidence breakdown", className="text-nav fw-bold mt-3 mb-1"),
                    dbc.Table(
                        [
                            html.Tbody(
                                [
                                    html.Tr([html.Td("Installed"), html.Td(str(status.get("installed", True)))]),
                                    html.Tr([html.Td("Resume ready"), html.Td(str(status.get("resume_ready", False)))]),
                                    html.Tr([html.Td("Checkpoint"), html.Td(str(status.get("checkpoint_label") or "—"))]),
                                    html.Tr([html.Td("Message"), html.Td(str(status.get("message") or "No runtime message available."))]),
                                ]
                            )
                        ],
                        bordered=False,
                        size="sm",
                        class_name="mb-2",
                    ),
                    dbc.Button("Re-run module", id="expert-rerun-module-btn", color="success", class_name="me-2"),
                ]
            )
            return "expert-bottom-drawer open", content, {"mode": "module", "module_id": module_id, "case_id": case_id, "artifact_path": str(preview_path) if preview_path else None}

        if tid == "scatter" and scatter_selected and scatter_selected.get("points"):
            rows = []
            for pt in scatter_selected["points"]:
                cd = pt.get("customdata") or []
                if len(cd) >= 5:
                    rows.append(
                        {
                            "gene": cd[0],
                            "mutant_peptide": cd[1],
                            "composite_priority": cd[2],
                            "tier": cd[3],
                            "resistance_composite": cd[4],
                        }
                    )
            if not rows:
                raise PreventUpdate
            table = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in ["gene", "mutant_peptide", "composite_priority", "tier", "resistance_composite"]],
                data=rows,
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#161B22", "color": "#E6EDF3", "fontSize": "12px", "fontFamily": "JetBrains Mono, monospace"},
                style_header={"backgroundColor": "#1C2128", "fontWeight": "bold"},
            )
            content = html.Div(
                [
                    html.Div(
                        [
                            html.Div("Selected Candidates", className="text-panel-title fw-bold"),
                            dbc.Button("×", id="expert-bottom-drawer-close", color="secondary", outline=True, size="sm"),
                        ],
                        className="d-flex justify-content-between align-items-start mb-2",
                    ),
                    html.Div(f"{len(rows)} points selected with lasso.", className="text-muted text-micro mb-2"),
                    table,
                    dbc.Button("Export Selected", id="expert-export-selected-btn", color="primary", class_name="w-100 mt-3", style={"background": "var(--accent-blue)", "color": "white"}),
                ]
            )
            return "expert-bottom-drawer open", content, {"mode": "selection", "rows": rows, "case_id": case_id}

        raise PreventUpdate

    @app.callback(
        Output("expert-selected-download", "data"),
        Input("expert-export-selected-btn", "n_clicks"),
        State("expert-drawer-context-store", "data"),
        prevent_initial_call=True,
    )
    def export_selected_candidates(n_clicks, context):
        if not n_clicks or not isinstance(context, dict) or context.get("mode") != "selection":
            raise PreventUpdate
        rows = context.get("rows") or []
        if not rows:
            raise PreventUpdate
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return dict(content=buf.getvalue(), filename="expert_selected_candidates.csv")

    @app.callback(
        Output("expert-strategy-save-status", "children"),
        Input("expert-rerun-module-btn", "n_clicks"),
        State("expert-drawer-context-store", "data"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def rerun_module_from_drawer(n_clicks, context, active_case):
        if not n_clicks or not isinstance(context, dict) or context.get("mode") != "module":
            raise PreventUpdate
        case_id = str(context.get("case_id") or _current_case_id(active_case) or "")
        module_id = str(context.get("module_id") or "")
        if not case_id or not module_id:
            raise PreventUpdate
        reset_module_for_rerun(case_id, module_id)
        try:
            ModuleRunner().resume_case(case_id)
        except Exception:
            pass
        return dbc.Alert(f"Queued {module_id} for rerun.", color="info", className="py-2 mb-0")

    @app.callback(
        Output("expert-strategy-save-status", "children", allow_duplicate=True),
        Input({"type": "module-enable-toggle", "module_id": ALL}, "value"),
        State({"type": "module-enable-toggle", "module_id": ALL}, "id"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def sync_module_enable_toggles(values, ids, active_case):
        if not values or not ids:
            raise PreventUpdate
        case_id = None
        if isinstance(active_case, dict) and active_case.get("case_id"):
            case_id = str(active_case["case_id"])
        if case_id is None:
            recent = list_cases(limit=1)
            if recent and recent[0].get("case_id"):
                case_id = str(recent[0]["case_id"])
        if not case_id:
            raise PreventUpdate
        failed_updates: list[str] = []
        for id_obj, val in zip(ids, values):
            if not isinstance(id_obj, dict):
                continue
            module_id = str(id_obj.get("module_id") or "")
            if module_id:
                try:
                    set_module_enabled(case_id, module_id, bool(val))
                except Exception:
                    failed_updates.append(module_id)
        try:
            ModuleRunner().resume_case(case_id)
        except Exception:
            pass
        if failed_updates:
            return dbc.Alert(
                f"Some module toggles could not be saved for case {case_id}: {', '.join(failed_updates)}",
                color="warning",
                className="py-2 mb-0",
            )
        return dbc.Alert(f"Updated enabled modules for case {case_id}.", color="info", className="py-2 mb-0")

    @app.callback(
        Output("download-canonical-csv", "data"),
        Input("download-canonical-btn", "n_clicks"),
        State("tier-filter", "value"),
        State("hla-filter", "value"),
        State("exclusion-filter", "value"),
        State("rl-range", "value"),
        State("expr-range", "value"),
        State("ccf-range", "value"),
        State("search", "value"),
        State("cohort-parquet-path", "data"),
        prevent_initial_call=True,
    )
    def download_canonical_csv(
        n_clicks,
        tiers,
        hlas,
        exclusions,
        rl_range,
        expr_range,
        ccf_range,
        search,
        cohort_path_override,
    ):
        if not n_clicks:
            raise PreventUpdate
        df, _path, _searched = load_cohort_for_dash(alt_path=cohort_path_override)
        fc = filter_candidates(
            df,
            tiers=[int(x) for x in (tiers or [])] or [1, 2, 3],
            hlas=[str(x) for x in (hlas or [])] if hlas else [],
            exclusion_any=[str(x) for x in (exclusions or [])] if exclusions else [],
            rl_range=list(rl_range or [0, 1]),
            expr_range=list(expr_range or [0, 1000]),
            ccf_range=list(ccf_range or [0, 1]),
            search=search or "",
        )
        if fc.empty:
            raise PreventUpdate
        buf = io.StringIO()
        fc.to_csv(buf, index=False)
        return dict(content=buf.getvalue(), filename="canonical_candidates_filtered.csv")

    @app.callback(
        Output("about-panel", "children"),
        Input("nav", "value"),
        Input("active-strategy-select", "value"),
    )
    def about_panel(_nav, active_strategy_id):
        strategy = get_strategy(active_strategy_id or "rl_v1")
        return html.Div(
            [
                html.P(
                    "NeoResist-MD helps a clinician or translational researcher review tumor-derived neoantigen candidates "
                    "with resistance-aware ranking rather than presentation alone.",
                    className="lead text-light",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Problem", className="mb-2"),
                                        html.P(
                                            "Tumors can generate neoantigens that look immunogenic on paper but may disappear under treatment pressure through LOH, low expression, or subclonal instability.",
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("What this app does", className="mb-2"),
                                        html.P(
                                            "It aggregates cohort candidates, enriches them with evidence, computes ResistanceLoop scores, and lets you compare modular strategies side by side.",
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=6,
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
                                        html.H5("Pipeline", className="mb-2"),
                                        html.Ol(
                                            [
                                                html.Li("Generate or aggregate neoantigen candidates."),
                                                html.Li("Join expression, purity, clonality, and escape evidence."),
                                                html.Li("Score with ResistanceLoop."),
                                                html.Li("Review candidates, patient detail, and strategy comparisons."),
                                            ],
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("ResistanceLoop", className="mb-2"),
                                        html.P(
                                            "ResistanceLoop scores whether a candidate is both promising and durable. "
                                            "Higher scores mean stronger presentation/expression/clonality support with fewer obvious escape routes.",
                                            className="mb-2",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li("What it scores: expression, presentation, clonality, self-dissimilarity, and escape penalties."),
                                                html.Li("Why it matters: it penalizes candidates that may vanish biologically even if they look recognizable."),
                                                html.Li(f"Active strategy now: {strategy.display_name} ({strategy.strategy_id})."),
                                            ],
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=6,
                        ),
                    ],
                    class_name="g-3 mb-3",
                ),
                dbc.Alert(
                    "Modular scoring matters because the field moves. NeoResist-MD keeps the scoring profile visible and editable so weights can evolve without rewriting the app.",
                    color="info",
                    className="mb-0",
                ),
            ]
        )

    @app.callback(
        Output("pipeline-status-panel", "children"),
        Input("nav", "value"),
        Input("upload-result-store", "data"),
    )
    def pipeline_status_panel(_nav, upload_result):
        status = collect_batch_status()
        recent_upload = upload_result or {}
        cases = list_cases()
        module_counts = status.get("module_state_counts") or {}
        module_specs = load_module_schema()
        install_checks = ModuleRunner().installation_checks()
        module_rows = []
        for module_id, spec in module_specs.items():
            installed, reason = install_checks.get(module_id, (False, "No runtime check reported."))
            if installed:
                state = "available"
                color = "success"
                reason_text = reason or "Runtime available."
            else:
                state = "requires setup" if spec.missing_policy == "mark_unavailable" else "unavailable"
                color = "warning" if state == "requires setup" else "danger"
                reason_text = reason or spec.tooltip.short or "Additional setup required."
            module_rows.append(
                html.Tr(
                    [
                        html.Td(spec.display_name),
                        html.Td(dbc.Badge(state.upper(), color=color)),
                        html.Td(reason_text),
                    ]
                )
            )
        top = dbc.Row(
            [
                dbc.Col(kpi_card(str(status["discovered_run_dirs"]), "Discovered run dirs"), md=3, sm=6),
                dbc.Col(kpi_card(str(status["runs_with_candidates"]), "Runs with candidates"), md=3, sm=6),
                dbc.Col(kpi_card(str(status["runs_with_case_report"]), "Runs with reports"), md=3, sm=6),
                dbc.Col(kpi_card(str(status.get("persisted_cases", len(cases))), "Persisted cases"), md=3, sm=6),
            ],
            class_name="g-2 mb-3",
        )
        bundle = status.get("cohort_bundle") or {}
        bundle_bits = [
            dbc.Badge(f"Reference upload HLA: {status['reference_hla']}", color="info", className="me-1 mb-1"),
            dbc.Badge(f"Bundle patients: {bundle.get('unique_patients', 0)}", color="secondary", className="me-1 mb-1"),
            dbc.Badge(f"Bundle rows: {bundle.get('output_rows', 0)}", color="secondary", className="me-1 mb-1"),
            dbc.Badge(f"Patient metric rows: {status['patient_metric_rows']}", color="secondary", className="me-1 mb-1"),
        ]
        upload_block = html.Div("No local upload run yet in this session.", className="text-muted")
        if recent_upload:
            upload_block = dbc.Alert(
                [
                    html.Div(f"Last upload status: {'validated' if recent_upload.get('ok') else 'failed'}"),
                    html.Div(f"Case id: {recent_upload.get('case_id', '—')}"),
                    html.Div(f"Input mode: {recent_upload.get('input_mode', '—')}", className="small"),
                ],
                color="success" if recent_upload.get("ok") else "warning",
                className="mb-0",
            )
        return html.Div(
            [
                top,
                html.Div(bundle_bits, className="mb-3"),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Module availability", className="mb-2"),
                            dbc.Table(
                                [
                                    html.Thead(html.Tr([html.Th("Module"), html.Th("Status"), html.Th("Reason / setup note")])),
                                    html.Tbody(module_rows),
                                ],
                                bordered=False,
                                hover=True,
                                responsive=True,
                                size="sm",
                                class_name="strategy-table mb-0",
                            ),
                        ]
                    ),
                    class_name="surface panel mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Batch interpretation", className="mb-2"),
                                        html.P(
                                            "No active long-running batch worker is attached to the Dash UI right now. "
                                            "This panel shows the latest visible cohort/build evidence so the app does not appear frozen or ambiguous.",
                                            className="mb-2",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(f"Latest run directories: {', '.join(status['latest_runs']) if status['latest_runs'] else 'none found'}"),
                                                html.Li(f"Recent upload runs: {', '.join(status['recent_upload_runs']) if status['recent_upload_runs'] else 'none yet'}"),
                                                html.Li(f"Persisted case ids: {', '.join(status.get('recent_case_ids')[:4]) if status.get('recent_case_ids') else 'none yet'}"),
                                                html.Li(
                                                    "Module states: "
                                                    + (", ".join(f"{k}={v}" for k, v in sorted(module_counts.items())) if module_counts else "none yet")
                                                ),
                                            ],
                                            className="mb-0",
                                        ),
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=7,
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Latest upload / runtime", className="mb-2"),
                                        upload_block,
                                    ]
                                ),
                                class_name="surface h-100",
                            ),
                            md=5,
                        ),
                    ],
                    class_name="g-3",
                ),
            ]
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
        df, _path, _searched = load_cohort_for_dash(alt_path=None)
        set_slider_bounds_from_df(df)
        agg = _aggregate_patient_hla(df)
        mm = int(agg["mutations"].max()) if not agg.empty else 100
        mf = round(float(agg["fanout"].max()), 1) if not agg.empty else 50.0
        mc = int(agg["candidates"].max()) if not agg.empty else 5000
        mm, mf, mc = max(mm, 1), max(mf, 0.1), max(mc, 1)
        hlas = sorted(df["hla_allele"].dropna().astype(str).unique().tolist()) if not df.empty else []
        if not df.empty and "expression_tpm" in df.columns:
            expr_hi = float(pd.to_numeric(df["expression_tpm"], errors="coerce").max() or 100.0)
        else:
            expr_hi = 100.0
        expr_hi = round(max(expr_hi, 1.0), 1)
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
        Input("patient-grid-standalone", "selectedRows"),
        prevent_initial_call=True,
    )
    def select_from_chart_or_grid(click, rows, rows_standalone):
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
        if tid == "patient-grid-standalone":
            if not rows_standalone:
                raise PreventUpdate
            r0 = rows_standalone[0]
            if not r0.get("patient_id") or not r0.get("hla_allele"):
                raise PreventUpdate
            return {"patient_id": str(r0["patient_id"]), "hla_allele": str(r0["hla_allele"])}
        raise PreventUpdate

    @app.callback(
        Output("upload-status", "children"),
        Output("case-create-summary", "children"),
        Output("upload-module-checklist", "options"),
        Output("upload-module-checklist", "value"),
        Output("run-case-btn", "disabled"),
        Output("upload-result-store", "data"),
        Input("upload-maf", "contents"),
        Input("upload-maf-global", "contents"),
        State("upload-maf", "filename"),
        State("upload-maf-global", "filename"),
        State("ui-mode", "value"),
        State("tumor-type-select", "value"),
        prevent_initial_call=True,
    )
    def create_case_from_upload_callback(contents, contents_global, filename, filename_global, ui_mode, tumor_type_value):
        trigger = callback_context.triggered_id
        if trigger == "upload-maf-global":
            active_contents = contents_global
            active_filename = filename_global
        else:
            active_contents = contents
            active_filename = filename
        if not active_contents:
            raise PreventUpdate
        fname = active_filename or "upload.tsv"
        inferred_tumor_type = infer_tumor_type_from_text(fname)
        tumor_type = normalise_tumor_type(tumor_type_value or inferred_tumor_type)
        try:
            runner = ModuleRunner()
            module_specs = load_module_schema()
            case_info = create_case_from_upload(
                active_contents,
                fname,
                install_checks=runner.installation_checks(),
                mode_context=str(ui_mode or "Simple"),
            )
            manifest = dict(case_info.get("manifest") or {})
            manifest["tumor_type"] = tumor_type
            manifest["msi_status"] = str(getattr(case_info["validation"], "msi_status", "INDETERMINATE"))
            manifest["msi_frameshift_indel_ratio"] = float(getattr(case_info["validation"], "msi_frameshift_indel_ratio", 0.0) or 0.0)
            manifest["msi_total_mutations"] = int(getattr(case_info["validation"], "msi_total_mutations", 0) or 0)
            manifest["msi_frameshift_indels"] = int(getattr(case_info["validation"], "msi_frameshift_indels", 0) or 0)
            write_case_manifest(manifest)
            case_info["manifest"] = manifest
        except Exception as exc:
            return (
                dbc.Alert(f"Upload validation or runtime setup failed: {exc}", color="danger", className="py-2 mb-0"),
                html.Div("No case created.", className="text-muted small"),
                [],
                [],
                True,
                {"ok": False, "error": str(exc), "filename": fname},
            )
        validation = case_info["validation"]
        module_specs = load_module_schema()
        options = [
            {
                "label": f"{spec.display_name}{' (heavy)' if spec.heavy else ''}{'' if runner.installation_checks().get(module_id, (True, ''))[0] else ' (runtime unavailable)'}",
                "value": module_id,
            }
            for module_id, spec in module_specs.items()
        ]
        install_checks = runner.installation_checks()
        enabled = [module_id for module_id in module_specs if install_checks.get(module_id, (True, ""))[0] or module_id in {"expression_join", "clonality_pyclone_vi"}]
        panel = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(kpi_card(case_info["case_id"], "Case id"), md=4, sm=12),
                        dbc.Col(kpi_card(str(validation.row_count), "Input rows"), md=4, sm=6),
                        dbc.Col(kpi_card(validation.input_mode.upper(), "Input mode"), md=4, sm=6),
                        dbc.Col(kpi_card(tumor_type, "Tumor type"), md=4, sm=6),
                        dbc.Col(kpi_card(str(getattr(validation, "msi_status", "INDETERMINATE")), "MSI status"), md=4, sm=6),
                    ],
                    class_name="g-2 mb-3",
                ),
                dbc.Alert(
                    "The file has been validated and persisted as a local case. Review the enabled modules, then click Run to start background jobs and open the case progress view.",
                    color="info",
                    className="mb-2",
                ),
                html.Div(
                    [
                        dbc.Badge(f"Sample: {validation.sample_id}", color="secondary", className="me-1 mb-1"),
                        dbc.Badge(f"Tumor: {tumor_type}", color="info", className="me-1 mb-1"),
                        dbc.Badge(
                            f"MSI: {getattr(validation, 'msi_status', 'INDETERMINATE')} ({float(getattr(validation, 'msi_frameshift_indel_ratio', 0.0) or 0.0):.2f})",
                            color="warning" if str(getattr(validation, "msi_status", "")).upper() == "MSI-H" else "success" if str(getattr(validation, "msi_status", "")).upper() == "MSS" else "secondary",
                            className="me-1 mb-1",
                        ),
                        dbc.Badge(f"Columns detected: {len(validation.columns)}", color="secondary", className="me-1 mb-1"),
                        dbc.Badge(str(ui_mode or "Simple"), color="dark", className="me-1 mb-1"),
                        dbc.Badge("HLA format warnings", color="warning", className="me-1 mb-1")
                        if int(getattr(validation, "hla_format_invalid_count", 0) or 0) > 0
                        else html.Span(),
                    ]
                ),
                _upload_validation_checklist(validation, fname, tumor_type),
                dbc.Alert(
                    f"HLA format validation: {int(getattr(validation, 'hla_format_invalid_count', 0) or 0)} invalid rows (expected HLA-A*02:01 style). Binding set to UNAVAILABLE for invalid rows.",
                    color="warning",
                    className="mb-2",
                )
                if int(getattr(validation, "hla_format_invalid_count", 0) or 0) > 0
                else html.Div(),
                dbc.Alert(
                    "⚠ No HLA typing detected — pan-allele estimates will be used.",
                    color="warning",
                    className="mb-2",
                )
                if bool(getattr(validation, "hla_missing_or_unknown", False))
                else html.Div(),
            ]
        )
        return (
            dbc.Alert(f"Validated {fname} and created case {case_info['case_id']}.", color="success", className="py-2 mb-0"),
            panel,
            options,
            enabled,
            False,
            {
                "ok": True,
                "case_id": case_info["case_id"],
                "filename": fname,
                "input_mode": validation.input_mode,
                "row_count": validation.row_count,
                "sample_id": validation.sample_id,
                "tumor_type": tumor_type,
                "msi_status": str(getattr(validation, "msi_status", "INDETERMINATE")),
                "msi_frameshift_indel_ratio": float(getattr(validation, "msi_frameshift_indel_ratio", 0.0) or 0.0),
                "germline_contamination_suspected": bool(getattr(validation, "germline_contamination_suspected", False)),
                "ith_entropy": getattr(validation, "ith_entropy", None),
                "hla_format_invalid_count": int(getattr(validation, "hla_format_invalid_count", 0) or 0),
                "hla_missing_or_unknown": bool(getattr(validation, "hla_missing_or_unknown", False)),
                "detected_column_mapping": dict(getattr(validation, "detected_column_mapping", None) or {}),
            },
        )

    @app.callback(
        Output("active-case-store", "data"),
        Output("nav", "value", allow_duplicate=True),
        Output("upload-result-panel", "children"),
        Input("run-case-btn", "n_clicks"),
        State("upload-result-store", "data"),
        State("upload-module-checklist", "value"),
        prevent_initial_call=True,
    )
    def run_case_modules(n_clicks, case_payload, enabled_modules):
        if not n_clicks or not case_payload or not case_payload.get("case_id"):
            raise PreventUpdate
        case_id = str(case_payload["case_id"])
        module_specs = load_module_schema()
        enabled_set = set(str(x) for x in (enabled_modules or []))
        for module_id in module_specs:
            set_module_enabled(case_id, module_id, module_id in enabled_set)
        runner = ModuleRunner()
        decisions = runner.start_case(case_id)
        return (
            {"case_id": case_id},
            "Cases",
            dbc.Alert(
                f"Started case {case_id}. Modules now running or queued: {', '.join(f'{k}={v}' for k, v in decisions.items())}",
                color="success",
                className="mb-0",
            ),
        )

    @app.callback(
        Output("case-select", "options"),
        Output("case-select", "value"),
        Output("case-list-panel", "children"),
        Output("nav", "options"),
        Input("active-case-store", "data"),
        Input("upload-result-store", "data"),
        Input("ui-mode", "value"),
    )
    def refresh_cases(active_case, upload_case, ui_mode):
        manifests = list_cases()
        options = [
            {
                "label": f"{item.get('sample_id', 'sample')} ({item.get('case_id', 'case')})",
                "value": str(item.get("case_id")),
            }
            for item in manifests
            if item.get("case_id")
        ]
        desired = None
        if isinstance(active_case, dict):
            desired = active_case.get("case_id")
        if desired is None and isinstance(upload_case, dict):
            desired = upload_case.get("case_id")
        valid_ids = {item["value"] for item in options}
        value = str(desired) if desired in valid_ids else (options[0]["value"] if options else None)
        return (
            options,
            value,
            render_case_list(manifests, value, expert=str(ui_mode or "Simple") == "Expert"),
            build_nav_options(len(manifests)),
        )

    @app.callback(
        Output("active-case-store", "data", allow_duplicate=True),
        Output("case-action-status", "children", allow_duplicate=True),
        Input({"type": "delete-case-confirm", "case_id": ALL}, "submit_n_clicks"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def delete_single_case(_submit_clicks, active_case):
        tid = callback_context.triggered_id
        if not isinstance(tid, dict) or not tid.get("case_id"):
            raise PreventUpdate
        case_id = str(tid["case_id"])
        deleted = delete_case(case_id)
        manifests = list_cases()
        current_active = str(active_case.get("case_id")) if isinstance(active_case, dict) and active_case.get("case_id") else None
        valid_ids = [str(item.get("case_id")) for item in manifests if item.get("case_id")]
        if current_active == case_id or current_active not in set(valid_ids):
            next_case = valid_ids[0] if valid_ids else None
        else:
            next_case = current_active
        return (
            {"case_id": next_case} if next_case else None,
            dbc.Alert(
                f"{'Deleted' if deleted else 'No-op for missing'} case {case_id}.",
                color="warning" if deleted else "secondary",
                className="py-2 mb-0",
            ),
        )

    @app.callback(
        Output("active-case-store", "data", allow_duplicate=True),
        Output("case-action-status", "children", allow_duplicate=True),
        Input("bulk-delete-test-cases-confirm", "submit_n_clicks"),
        prevent_initial_call=True,
    )
    def bulk_delete_test_cases(submit_n_clicks):
        if not submit_n_clicks:
            raise PreventUpdate
        manifests = list_cases(limit=500)
        keep_ids = _curated_keep_case_ids(manifests)
        delete_ids = [
            str(item.get("case_id"))
            for item in manifests
            if item.get("case_id") and str(item.get("case_id")) not in keep_ids
        ]
        for case_id in delete_ids:
            delete_case(case_id)
        remaining = list_cases(limit=500)
        next_case = str(remaining[0].get("case_id")) if remaining and remaining[0].get("case_id") else None
        keep_text = ", ".join(sorted(keep_ids)) if keep_ids else "none"
        return (
            {"case_id": next_case} if next_case else None,
            dbc.Alert(
                f"Deleted {len(delete_ids)} cases. Kept: {keep_text}.",
                color="warning",
                className="py-2 mb-0",
            ),
        )

    @app.callback(
        Output("active-case-store", "data", allow_duplicate=True),
        Input("case-select", "value"),
        prevent_initial_call=True,
    )
    def sync_active_case(case_id):
        if not case_id:
            raise PreventUpdate
        case_id = str(case_id)
        return {"case_id": case_id}

    @app.callback(
        Output("case-detail-panel", "children"),
        Input("active-case-store", "data"),
        Input("case-select", "value"),
        Input("ui-mode", "value"),
        Input("active-strategy-select", "value"),
    )
    def render_case_progress(active_case, selected_case, ui_mode, active_strategy_id):
        case_id = selected_case
        if isinstance(active_case, dict) and active_case.get("case_id"):
            case_id = selected_case or active_case.get("case_id")
        try:
            return render_case_detail(str(case_id) if case_id else None, str(ui_mode or "Simple"), str(active_strategy_id or "rl_v1"))
        except Exception as exc:
            return dbc.Alert(
                f"Case view failed to load ({exc}). Switch back to Upload and open a valid case.",
                color="warning",
                className="mb-0",
            )

    @app.callback(
        Output("case-action-status", "children"),
        Input("case-rna-upload", "contents"),
        Input("case-purity-upload", "contents"),
        Input("case-cnv-upload", "contents"),
        State("case-rna-upload", "filename"),
        State("case-purity-upload", "filename"),
        State("case-cnv-upload", "filename"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def attach_sidecar(rna_contents, purity_contents, cnv_contents, rna_name, purity_name, cnv_name, active_case):
        case_id = str((active_case or {}).get("case_id") or "")
        if not case_id:
            raise PreventUpdate
        trigger = callback_context.triggered_id
        if trigger == "case-rna-upload" and rna_contents:
            attach_case_input(case_id, "rna_sidecar", rna_contents, rna_name or "rna.tsv")
            reset_modules_for_rerun(case_id, ["expression_join", "resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Attached RNA sidecar and queued expression plus downstream scoring modules.", color="info", className="mb-0 py-2")
        if trigger == "case-purity-upload" and purity_contents:
            attach_case_input(case_id, "purity_sidecar", purity_contents, purity_name or "purity.tsv")
            reset_modules_for_rerun(case_id, ["clonality_pyclone_vi", "resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Attached purity sidecar and queued clonality plus downstream scoring modules.", color="info", className="mb-0 py-2")
        if trigger == "case-cnv-upload" and cnv_contents:
            attach_case_input(case_id, "cnv_sidecar", cnv_contents, cnv_name or "cnv.tsv")
            reset_modules_for_rerun(case_id, ["clonality_pyclone_vi", "resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Attached CNV sidecar and queued clonality plus downstream scoring modules.", color="info", className="mb-0 py-2")
        raise PreventUpdate

    @app.callback(
        Output("case-action-status", "children", allow_duplicate=True),
        Input("rerun-expression-btn", "n_clicks"),
        Input("rerun-clonality-btn", "n_clicks"),
        Input("rerun-resistance-btn", "n_clicks"),
        Input("rerun-strategy-btn", "n_clicks"),
        Input("rerun-prioritization-btn", "n_clicks"),
        State("active-case-store", "data"),
        prevent_initial_call=True,
    )
    def rerun_case_module(expr_clicks, clon_clicks, resistance_clicks, strategy_clicks, prioritization_clicks, active_case):
        case_id = str((active_case or {}).get("case_id") or "")
        if not case_id:
            raise PreventUpdate
        trigger = callback_context.triggered_id
        if trigger == "rerun-expression-btn" and expr_clicks:
            reset_modules_for_rerun(case_id, ["expression_join", "resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Expression join and downstream modules queued for rerun.", color="info", className="mb-0 py-2")
        if trigger == "rerun-clonality-btn" and clon_clicks:
            reset_modules_for_rerun(case_id, ["clonality_pyclone_vi", "resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Clonality and downstream modules queued for rerun.", color="info", className="mb-0 py-2")
        if trigger == "rerun-resistance-btn" and resistance_clicks:
            reset_modules_for_rerun(case_id, ["resistance_loop", "strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("ResistanceLoop and downstream modules queued for rerun.", color="info", className="mb-0 py-2")
        if trigger == "rerun-strategy-btn" and strategy_clicks:
            reset_modules_for_rerun(case_id, ["strategy_engine", "prioritization_tiering"])
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Strategy engine and prioritization queued for rerun.", color="info", className="mb-0 py-2")
        if trigger == "rerun-prioritization-btn" and prioritization_clicks:
            reset_module_for_rerun(case_id, "prioritization_tiering")
            ModuleRunner().resume_case(case_id)
            return dbc.Alert("Prioritization queued for rerun.", color="info", className="mb-0 py-2")
        raise PreventUpdate

    @app.callback(
        Output("download-patient-csv", "data"),
        Input("btn-download-patient", "n_clicks"),
        Input("export-tumor-board", "n_clicks"),
        State("selected-key-store", "data"),
        State("simple-candidate-store", "data"),
        prevent_initial_call=True,
    )
    def download_patient_csv(patient_clicks, export_clicks, key, candidate_rows):
        trigger = callback_context.triggered_id
        if trigger == "export-tumor-board":
            rows = candidate_rows or []
            if not rows:
                raise PreventUpdate
            sub = pd.DataFrame(rows)
            if sub.empty:
                raise PreventUpdate
            pid = str(sub.get("patient_id", pd.Series(["patient"])).iloc[0] or "patient")
            hla = str(sub.get("hla_allele", pd.Series(["hla"])).iloc[0] or "hla")
        else:
            if not patient_clicks or not key:
                raise PreventUpdate
            df, _ = load_qualified_candidates()
            pid, hla = key.get("patient_id"), key.get("hla_allele")
            sub = df[(df["patient_id"].astype(str) == str(pid)) & (df["hla_allele"].astype(str) == str(hla))]
            if sub.empty:
                raise PreventUpdate
        buf = io.StringIO()
        sub.to_csv(buf, index=False)
        fname = f"resistanceloop_{pid}_{hla}.csv".replace("/", "-")
        return dict(content=buf.getvalue(), filename=fname)

    @app.callback(
        Output("evidence-drawer", "className"),
        Output("evidence-drawer-content", "children"),
        Input({"type": "expand-candidate", "index": ALL}, "n_clicks"),
        Input("evidence-drawer-close", "n_clicks"),
        State("simple-candidate-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_evidence_drawer(_clicks, close_clicks, candidate_store):
        trigger = callback_context.triggered_id
        if not any(int(click or 0) for click in (_clicks or [])) and not close_clicks:
            raise PreventUpdate
        if trigger == "evidence-drawer-close":
            if not close_clicks:
                raise PreventUpdate
            return "evidence-drawer", no_update
        if isinstance(trigger, dict) and trigger.get("type") == "expand-candidate":
            idx = int(trigger.get("index", -1))
            rows = candidate_store or []
            if 0 <= idx < len(rows):
                return "evidence-drawer open", build_evidence_drawer(rows[idx])
        return "evidence-drawer", no_update

    @app.callback(
        Output("data-banner", "children"),
        Output("kpi-cards", "children"),
        Output("hla-coverage-row", "children"),
        Output("scatter", "figure"),
        Output("evidence-panel", "children"),
        Output("patient-detail", "children"),
        Output("simple-answer-panel", "children"),
        Output("simple-tier-badges", "children"),
        Output("simple-ranked-list", "children"),
        Output("simple-evidence-cards", "children"),
        Output("expert-tier-fig", "figure"),
        Output("expert-tier-pie-fig", "figure"),
        Output("expert-evidence-fig", "figure"),
        Output("expert-waterfall-fig", "figure"),
        Output("expert-heatmap-fig", "figure"),
        Output("expert-confidence-overview-fig", "figure"),
        Output("expert-resistance-breakdown-fig", "figure"),
        Output("patient-grid", "rowData"),
        Output("patient-detail-standalone", "children"),
        Output("patient-grid-standalone", "rowData"),
        Output("simple-candidate-store", "data"),
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
        Input("ui-mode", "value"),
        Input("active-strategy-select", "value"),
        Input("strategy-refresh-store", "data"),
        Input("scatter-color-dimension", "value"),
        Input("scatter-y-axis", "value"),
        Input("scatter-selection-mode", "value"),
        Input("selected-key-store", "data"),
        Input("active-case-store", "data"),
        Input("tool-registry-store", "data"),
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
        ui_mode,
        active_strategy_id,
        strategy_refresh_state,
        scatter_color_dimension,
        scatter_y_axis,
        scatter_selection_mode,
        selected_key,
        active_case,
        tool_registry_state,
        cohort_path_override,
    ):
        from neoresist.dash_app.data import _empty_candidates_frame
        from neoresist.loaders import CohortLoadError, load_cohort_for_dash

        try:
            df, path, search_log = load_cohort_for_dash(alt_path=cohort_path_override)
        except CohortLoadError as e:
            df = _empty_candidates_frame().copy()
            path = None
            search_log = list(e.searched)
            miss_txt = ", ".join(e.missing_columns) if e.missing_columns else "unknown"
            pres_txt = ", ".join(sorted(e.present_columns)[:30])
            if len(e.present_columns) > 30:
                pres_txt += " …"
            banner_err = dbc.Alert(
                [
                    html.P("Cohort file loaded but failed Dash column validation.", className="mb-2"),
                    html.P([html.Strong("Missing canonical columns: "), miss_txt]),
                    html.P([html.Strong("Columns present (sample): "), pres_txt], className="small"),
                    html.P([html.Strong("Searched: "), html.Pre("\n".join(search_log), className="small mb-0")]),
                ],
                color="danger",
                className="py-2 mb-2",
            )
            empty = make_scatter_fig(pd.DataFrame(), dragmode=str(scatter_selection_mode or "lasso"), y_axis=str(scatter_y_axis or "auto"))
            detail = [html.Div("Fix columns or use enriched Parquet.", className="text-muted")]
            empty_fig = px.bar(title="No data")
            return (
                banner_err,
                [],
                html.Div(),
                empty,
                detail,
                detail,
                detail,
                detail,
                detail,
                detail,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                [],
                detail,
                [],
                [],
                "NeoResist-MD",
            )

        banner = html.Div()
        if path is None:
            tried = search_log
            paths_txt = "\n".join(str(p) for p in tried) if tried else "(no search paths)"
            banner = dbc.Alert(
                [
                    html.P(
                        "No cohort file found. Expected Parquet (enriched or qualified) under data/final/ "
                        "or paths from configs/app_config.yaml (CSV/XLSX supported if columns match).",
                        className="mb-2",
                    ),
                    html.P([html.Strong("Searched (in order): "), html.Pre(paths_txt, className="small mb-0")]),
                ],
                color="warning",
                className="py-2 mb-2",
            )
        elif df.empty:
            banner = dbc.Alert("Parquet file is empty.", color="secondary", className="py-2 mb-2")

        active_strategy = get_strategy(active_strategy_id or list_strategies()[0].strategy_id)
        base_display_df = apply_display_policy(
            df,
            tier1_above=float(active_strategy.tier1_above),
            tier2_above=float(active_strategy.tier2_above),
            exclusion_rules=load_exclusion_rules(active_strategy.strategy_id),
        )
        tiers_l = [int(x) for x in (tiers or [])] or [1, 2, 3]
        hlas_l = [str(x) for x in (hlas or [])] if hlas else []
        excl_l = [str(x) for x in (exclusions or [])] if exclusions else []

        fc = filter_candidates(
            base_display_df,
            tiers=tiers_l,
            hlas=hlas_l,
            exclusion_any=excl_l,
            rl_range=list(rl_range or [0, 1]),
            expr_range=list(expr_range or [0, 1000]),
            ccf_range=list(ccf_range or [0, 1]),
            search=search or "",
        )
        agg = _aggregate_patient_hla(fc)
        mm, mf, mc = slider_mut_max(), slider_fan_max(), slider_cand_max()
        agg_f = filter_agg(agg, list(tmb or [0, mm]), list(fanout or [0, mf]), list(cand or [0, mc]))
        agg_s = sort_agg(agg_f, sort_by or "Mutations ↓")

        display_fc = fc
        if str(ui_mode or "Simple") == "Expert" and isinstance(strategy_refresh_state, dict):
            applied_weights = strategy_refresh_state.get("applied_weights") or {}
            if applied_weights:
                try:
                    base_strategy_id = strategy_refresh_state.get("applied_strategy_id") or strategy_refresh_state.get("active_strategy_id")
                    base_strategy = get_strategy(base_strategy_id or active_strategy.strategy_id)
                    custom_strategy = _strategy_from_slider_weights(
                        base_strategy,
                        {
                            "expression_norm": applied_weights.get("expression_norm", base_strategy.weights.get("expression_norm", 0.0)),
                            "presentation": applied_weights.get("presentation", base_strategy.weights.get("presentation", 0.0)),
                            "ccf": applied_weights.get("ccf", base_strategy.weights.get("ccf", 0.0)),
                            "self_dissimilarity": applied_weights.get("self_dissimilarity", base_strategy.weights.get("self_dissimilarity", 0.0)),
                            "escape_penalty_weight": applied_weights.get("escape_penalty_weight", base_strategy.escape_penalty_weight),
                        },
                    )
                    active_strategy = custom_strategy
                    display_fc = score_candidates_for_strategy(fc, custom_strategy)
                    display_fc["composite_priority"] = display_fc.get("rl_priority", pd.Series(dtype=float))
                except Exception:
                    display_fc = fc
        display_fc = apply_display_policy(
            display_fc,
            tier1_above=float(active_strategy.tier1_above),
            tier2_above=float(active_strategy.tier2_above),
            exclusion_rules=load_exclusion_rules(active_strategy.strategy_id),
        )
        display_fc = filter_candidates(
            display_fc,
            tiers=tiers_l,
            hlas=hlas_l,
            exclusion_any=excl_l,
            rl_range=list(rl_range or [0, 1]),
            expr_range=list(expr_range or [0, 1000]),
            ccf_range=list(ccf_range or [0, 1]),
            search=search or "",
        )
        agg = _aggregate_patient_hla(display_fc)
        agg_f = filter_agg(agg, list(tmb or [0, mm]), list(fanout or [0, mf]), list(cand or [0, mc]))
        agg_s = sort_agg(agg_f, sort_by or "Mutations ↓")

        tier1_n = int((display_fc["tier"] == 1).sum()) if not display_fc.empty else 0
        tier2_n = int((display_fc["tier"] == 2).sum()) if not display_fc.empty else 0
        mean_rl = float(display_fc["rl_priority"].mean()) if not display_fc.empty else 0.0
        if math.isnan(mean_rl):
            mean_rl = 0.0
        pat_tier1 = int(display_fc.loc[display_fc["tier"] == 1, "patient_id"].nunique()) if not display_fc.empty else 0
        hla_resolved, hla_total = _hla_coverage_counts(display_fc if not display_fc.empty else df)
        hla_cov_pct = (100.0 * hla_resolved / hla_total) if hla_total else 0.0
        cohort_tumor_type = single_tumor_type(display_fc if not display_fc.empty else df)
        msi_counts = _msi_counts(display_fc if not display_fc.empty else df)

        pur_missing = 0
        pur_by_src: Counter[str] = Counter()
        if not display_fc.empty and "purity_source_used" in display_fc.columns:
            for v in display_fc["purity_source_used"].astype(str):
                pur_by_src[v] += 1
            pur_missing = int((display_fc["purity_source_used"].astype(str) == "unresolved").sum())

        kpis = [
            dbc.Col(kpi_card(f"{tier1_n:,}", "Tier 1 candidates", accent=TIER_COLORS[1]), lg=2, md=4, sm=6),
            dbc.Col(kpi_card(f"{tier2_n:,}", "Tier 2 candidates", accent=TIER_COLORS[2]), lg=2, md=4, sm=6),
            dbc.Col(kpi_card(f"{mean_rl:.3f}", "Mean RL priority"), lg=2, md=4, sm=6),
            dbc.Col(kpi_card(f"{pat_tier1:,}", "Patients w/ Tier 1"), lg=2, md=4, sm=6),
            dbc.Col(kpi_card(f"{display_fc['patient_id'].nunique():,}" if not display_fc.empty else "0", "Patients (filtered)"), lg=2, md=4, sm=6),
            dbc.Col(kpi_card(f"{len(display_fc):,}", "Candidates (filtered)"), lg=2, md=4, sm=6),
            dbc.Col(
                kpi_card(
                    f"{hla_cov_pct:.0f}% (no HLA typing)" if hla_total and hla_resolved == 0 else f"{hla_resolved}/{hla_total} ({hla_cov_pct:.0f}%)",
                    "HLA coverage",
                ),
                lg=2,
                md=4,
                sm=6,
            ),
        ]
        if cohort_tumor_type:
            kpis.append(dbc.Col(kpi_card(cohort_tumor_type, "Tumor type"), lg=2, md=4, sm=6))
        if msi_counts:
            msi_h, msi_total = msi_counts
            kpis.append(dbc.Col(kpi_card(f"{msi_h}/{msi_total}", "MSI-H patients"), lg=2, md=4, sm=6))
        if not display_fc.empty and "purity_source_used" in display_fc.columns:
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

        cov = hla_coverage_badge_row(display_fc)

        if nav not in {"Overview", "Patients", "Advanced Strategies"}:
            empty = make_scatter_fig(pd.DataFrame(), dragmode=str(scatter_selection_mode or "lasso"), y_axis=str(scatter_y_axis or "auto"))
            detail = [html.Div("Switch to Overview for cohort tools.", className="text-muted")]
            empty_fig = px.bar(title="Switch to Overview for cohort tools.")
            return (
                banner,
                kpis,
                cov,
                empty,
                detail,
                detail,
                detail,
                detail,
                detail,
                detail,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                [],
                detail,
                [],
                [],
                "ResistanceLoop v1 · NeoResist-MD",
            )

        fig = make_scatter_fig(
            display_fc if str(ui_mode or "Simple") == "Expert" else agg_s,
            color_dimension=str(scatter_color_dimension or "Tier"),
            dragmode=str(scatter_selection_mode or "lasso"),
            y_axis=str(scatter_y_axis or "auto"),
        )

        pid = selected_key.get("patient_id") if isinstance(selected_key, dict) else None
        hla = selected_key.get("hla_allele") if isinstance(selected_key, dict) else None

        evidence = build_evidence_panel(display_fc, pid, hla)
        detail = build_patient_detail_card(display_fc, pid, hla)
        standalone_detail = build_patient_detail_card(display_fc, pid, hla)
        sort_col = "composite_priority" if "composite_priority" in display_fc.columns else "rl_priority"
        simple_df = display_fc.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True) if not display_fc.empty else pd.DataFrame()
        hero = build_kpi_hero(display_fc)
        simple_stub_banners = html.Div()
        ith_note = html.Div()
        case_id_for_simple = _current_case_id(active_case)
        if case_id_for_simple:
            try:
                manifest = read_case_manifest(case_id_for_simple)
            except Exception:
                manifest = {}
            try:
                statuses_for_simple = {
                    module_id: read_module_status(case_id_for_simple, module_id)
                    for module_id in load_module_schema()
                }
            except Exception:
                statuses_for_simple = {}
            simple_stub_banners = build_simple_stub_banners(statuses_for_simple, tool_registry_state or {})
            ith_raw = manifest.get("ith_entropy")
            try:
                ith_value = float(ith_raw)
            except (TypeError, ValueError):
                ith_value = None
            if ith_value is not None and ith_value > 1.5:
                ith_note = html.Div(
                    f"ⓘ High intra-tumor heterogeneity detected (entropy {ith_value:.2f}) — subclonal neoantigens may not be present in all tumor cells.",
                    className="text-micro",
                    style={"color": "var(--text-muted)", "marginBottom": "10px"},
                )
        cards = build_candidate_cards(simple_df)
        simple_answer = html.Div(
            [
                hero,
                simple_stub_banners,
                ith_note,
                html.Div(
                    cards if cards else [html.Div("No ranked candidates under the current filters.", className="text-muted")],
                    style={"maxHeight": "680px", "overflowY": "auto", "paddingRight": "4px"},
                ),
            ]
        )
        shortlist = simple_df.head(12) if not simple_df.empty else pd.DataFrame()
        shortlist_cols = [col for col in ["gene", "mutant_peptide", "composite_priority", "tier"] if col in shortlist.columns]
        simple_ranked = (
            html.Div(cards, style={"maxHeight": "540px", "overflowY": "auto", "paddingRight": "4px"})
            if cards
            else html.Div("No ranked candidates under the current filters.", className="text-muted")
        )
        tier_badges = html.Div(
            [
                dbc.Badge(f"TIER 1: {int((display_fc['tier'] == 1).sum()) if 'tier' in display_fc.columns else 0}", color="success", className="me-1"),
                dbc.Badge(f"TIER 2: {int((display_fc['tier'] == 2).sum()) if 'tier' in display_fc.columns else 0}", color="warning", className="me-1"),
                dbc.Badge(f"TIER 3: {int((display_fc['tier'] == 3).sum()) if 'tier' in display_fc.columns else 0}", color="secondary", className="me-1"),
                dbc.Badge(
                    f"EXCLUDED: {int((display_fc['tier'].astype(str).str.contains('EXCLUDED', case=False, na=False)).sum()) if 'tier' in display_fc.columns else 0}",
                    color="danger",
                ),
            ]
        )
        evidence_cards: list[Any] = []
        for row in shortlist.head(6).to_dict("records"):
            risk = "LOW"
            score = float(row.get("rl_priority", 0.0) or 0.0)
            if score < 0.35:
                risk = "HIGH"
            elif score < 0.6:
                risk = "MEDIUM"
            evidence_cards.append(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Strong(f"{row.get('gene', 'Candidate')} {row.get('mutant_peptide', '')}"),
                                    dbc.Badge(str(row.get("tier", "—")).replace("_", " "), color="secondary", className="ms-2"),
                                ],
                                className="mb-2",
                            ),
                            html.Div(f"ResistanceLoop score: {score:.3f} / 1.0", className="small"),
                            html.Div(f"Escape risk: {risk}", className="small"),
                            html.Div(f"HLA: {normalize_hla_display(row.get('hla_allele', '—'))}", className="small text-muted"),
                        ]
                    ),
                    class_name="surface panel mb-2",
                )
            )
        simple_evidence_cards = evidence_cards if evidence_cards else [html.Div("Evidence cards appear after ranking data loads.", className="text-muted small")]
        if not display_fc.empty and "hla_allele" in display_fc.columns and display_fc["hla_allele"].map(_is_unknown_hla).any():
            simple_evidence_cards = [
                dbc.Alert(
                    "HLA type unknown for this patient. Presentation and binding predictions use pan-allele estimates and may be less precise. Provide HLA typing data to improve candidate ranking accuracy.",
                    color="warning",
                    className="mb-2",
                ),
                *simple_evidence_cards,
            ]

        tier_order = ["TIER_1", "TIER_2", "TIER_3", "EXCLUDED"]
        tier_counts = (
            display_fc["tier"].astype(str).str.upper().map(
                lambda v: "TIER_1" if v in {"1", "TIER_1"} else ("TIER_2" if v in {"2", "TIER_2"} else ("TIER_3" if v in {"3", "TIER_3"} else "EXCLUDED"))
            ).value_counts().reindex(tier_order, fill_value=0)
            if not display_fc.empty and "tier" in display_fc.columns
            else pd.Series(dtype=float)
        )
        tier_fig = px.bar(x=[label.replace("_", " ") for label in tier_counts.index], y=tier_counts.values, labels={"x": "", "y": "Candidates"}, title="")
        tier_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=30, r=20, t=10, b=30), height=260)
        tier_pie_fig = px.pie(
            names=[label.replace("_", " ") for label in tier_counts.index],
            values=tier_counts.values if len(tier_counts) else [1],
            hole=0.45,
        )
        tier_pie_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=20, r=20, t=10, b=20), height=260)
        expr_stub = int((display_fc.get("evidence_expression_source", pd.Series(dtype=str)).astype(str) == "stub").sum()) if not display_fc.empty and "evidence_expression_source" in display_fc.columns else int(len(display_fc))
        expr_real = max(int(len(display_fc) - expr_stub), 0)
        ev_fig = px.pie(names=["Real/blended evidence", "Stub evidence"], values=[expr_real, expr_stub], hole=0.45)
        ev_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=20, r=20, t=10, b=20), height=260, showlegend=True)

        # Waterfall story through major module gates.
        total_n = int(len(df))
        affinity = pd.to_numeric(df.get("affinity_nm", df.get("affinity", pd.Series([float("nan")] * len(df), index=df.index))), errors="coerce")
        min_expr_tpm = float(load_exclusion_rules(active_strategy.strategy_id).get("min_expression_tpm", 1.0))
        max_escape_penalty = float(load_exclusion_rules(active_strategy.strategy_id).get("max_escape_penalty", 0.5))
        binding_df = df[affinity.le(5000).fillna(False)] if not affinity.empty else df
        expr_values = pd.to_numeric(binding_df.get("expression_tpm", pd.Series([0.0] * len(binding_df), index=binding_df.index)), errors="coerce").fillna(0.0)
        expr_df = binding_df[expr_values.ge(min_expr_tpm)] if not binding_df.empty else binding_df
        clon_values = pd.to_numeric(expr_df.get("ccf", pd.Series([float("nan")] * len(expr_df), index=expr_df.index)), errors="coerce")
        clon_df = expr_df[clon_values.notna()] if not expr_df.empty else expr_df
        escape_values = pd.to_numeric(clon_df.get("escape_penalty", pd.Series([0.0] * len(clon_df), index=clon_df.index)), errors="coerce").fillna(0.0)
        escape_df = clon_df[escape_values.le(max_escape_penalty)] if not clon_df.empty else clon_df
        binding_n = int(len(binding_df))
        expr_n = int(len(expr_df))
        clon_n = int(len(clon_df))
        escape_n = int(len(escape_df))
        final_n = int(len(display_fc))
        wf_df = pd.DataFrame(
            {
                "stage": ["Start", "After binding", "After expression", "After clonality", "After escape", "Final ranked"],
                "count": [total_n, binding_n, expr_n, clon_n, escape_n, final_n],
                "note": [
                    "Input candidate set",
                    f"removed {max(total_n - binding_n, 0)} (affinity > 5000 nM)",
                    f"removed {max(binding_n - expr_n, 0)} (TPM < {min_expr_tpm:.1f})",
                    f"removed {max(expr_n - clon_n, 0)} (missing clonality support)",
                    f"removed {max(clon_n - escape_n, 0)} (escape penalty > {max_escape_penalty:.2f})",
                    f"removed {max(escape_n - final_n, 0)} (tiering + active filters)",
                ],
            }
        )
        waterfall_fig = px.bar(wf_df, x="stage", y="count", text="note")
        waterfall_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=30, r=20, t=10, b=50), height=280, showlegend=False)
        waterfall_fig.update_traces(textposition="outside", cliponaxis=False)

        # Cohort module heatmap.
        module_cols = {
            COMPONENT_DISPLAY_NAMES["presentation"]: pd.to_numeric(df.get("presentation_score", pd.Series([0.0] * len(df), index=df.index)), errors="coerce"),
            COMPONENT_DISPLAY_NAMES["expression"]: pd.to_numeric(df.get("expression_norm", df.get("expression_tpm", pd.Series([0.0] * len(df), index=df.index))), errors="coerce"),
            COMPONENT_DISPLAY_NAMES["ccf"]: pd.to_numeric(df.get("ccf", pd.Series([0.0] * len(df), index=df.index)), errors="coerce"),
            COMPONENT_DISPLAY_NAMES["self_dissimilarity"]: pd.to_numeric(df.get("self_dissimilarity", pd.Series([0.0] * len(df), index=df.index)), errors="coerce"),
            COMPONENT_DISPLAY_NAMES["escape"]: pd.to_numeric(df.get("escape_penalty", pd.Series([0.0] * len(df), index=df.index)), errors="coerce").fillna(0.0) * abs(float(active_strategy.escape_penalty_weight)),
        }
        hdf = pd.DataFrame(module_cols)
        if "patient_id" in df.columns and not df.empty:
            hdf["patient_id"] = df["patient_id"].astype(str)
            grp = hdf.groupby("patient_id", dropna=False).mean(numeric_only=True).fillna(0.0)
            grp = grp.sort_values(by=COMPONENT_DISPLAY_NAMES["escape"], ascending=False).head(40)
            heatmap_fig = px.imshow(grp.T, aspect="auto", color_continuous_scale="Viridis")
        else:
            heatmap_fig = px.imshow(pd.DataFrame([[0.0]], index=["module"], columns=["patient"]), aspect="auto")
        heatmap_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=30, r=20, t=10, b=30), height=320)

        # Module confidence stacked overview.
        conf_states = ["HIGH", "MEDIUM", "LOW", "UNAVAILABLE"]
        conf_map = {
            "binding": "binding_confidence",
            "presentation": "presentation_confidence",
            "expression": "expression_confidence",
            "clonality": "clonality_confidence",
            "escape": "escape_confidence",
            "recognition": "recognition_confidence",
            "resistance": "resistance_confidence",
        }
        conf_rows: list[dict[str, Any]] = []
        for module_name, col in conf_map.items():
            if col in df.columns:
                vals = df[col].astype(str).str.upper()
            else:
                vals = pd.Series(["UNAVAILABLE"] * len(df))
            for st in conf_states:
                conf_rows.append({"module": module_name, "confidence": st, "count": int((vals == st).sum())})
        conf_df = pd.DataFrame(conf_rows)
        confidence_overview_fig = px.bar(conf_df, x="module", y="count", color="confidence", barmode="stack")
        confidence_overview_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=30, r=20, t=10, b=30), height=300)

        # Resistance breakdown for selected or top candidate.
        target = shortlist.head(1).to_dict("records")[0] if not shortlist.empty else {}
        if pid is not None and hla is not None and not display_fc.empty:
            sel = display_fc[(display_fc["patient_id"].astype(str) == str(pid)) & (display_fc["hla_allele"].astype(str) == str(hla))]
            if not sel.empty:
                target = sel.sort_values(["tier", "rl_priority"], ascending=[True, False]).head(1).to_dict("records")[0]
        loh_component = target.get("resistance_loh_penalty")
        if loh_component is None and str(target.get("hla_loh_status", "")).lower() in {"lost", "loh_detected"}:
            loh_component = target.get("escape_penalty")
        rb_raw = {
            "LOH penalty": loh_component,
            "Volatility": target.get("resistance_volatility_flag"),
            "Expression instability": target.get("resistance_expression_instability"),
            "Processing disruption": target.get("resistance_processing_disruption"),
        }
        rb_rows = []
        for component, raw_value in rb_raw.items():
            value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            has_data = pd.notna(value) and float(value) > 0.0
            rb_rows.append(
                {
                    "component": component,
                    "value": float(value) if has_data else 0.0,
                    "plot_value": float(value) if has_data else 0.02,
                    "status": "Measured" if has_data else "No data",
                    "label": f"{float(value):.2f}" if has_data else "No data",
                }
            )
        rb_df = pd.DataFrame(rb_rows)
        resistance_breakdown_fig = px.bar(
            rb_df,
            x="component",
            y="plot_value",
            color="status",
            text="label",
            color_discrete_map={"Measured": "#2F81F7", "No data": "#8B949E"},
        )
        resistance_breakdown_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"), margin=dict(l=30, r=20, t=10, b=30), height=280, showlegend=False)
        resistance_breakdown_fig.update_yaxes(title="Penalty")
        resistance_breakdown_fig.update_traces(textposition="outside", cliponaxis=False)
        resistance_breakdown_fig.add_annotation(
            x=0.5,
            y=-0.22,
            xref="paper",
            yref="paper",
            text="Resistance components require LOH and expression stability data. Components without input data contribute zero penalty.",
            showarrow=False,
            font=dict(color="#8b949e", size=10),
            xanchor="center",
        )

        grid_rows = agg_s.to_dict("records")
        n_pat = int(display_fc["patient_id"].nunique()) if not display_fc.empty else 0
        n_hla = int(display_fc["hla_allele"].nunique()) if not display_fc.empty else 0
        tier1_pct = 100.0 * float((display_fc["tier"] == 1).mean()) if not display_fc.empty else 0.0
        mean_rl_all = float(display_fc["rl_priority"].mean()) if not display_fc.empty else 0.0
        if math.isnan(mean_rl_all):
            mean_rl_all = 0.0
        src = path or "qualified_candidates.parquet (missing)"
        try:
            from neoresist.version import __version__

            ver = __version__
        except Exception:
            ver = "?"
        footer = (
            f"Live cohort (full file): {n_pat:,} patients · {n_hla:,} HLA contexts · "
            f"Tier 1 share {tier1_pct:.1f}% · mean RL {mean_rl_all:.3f} · "
            f"Data: {src} · app v{ver} · ResistanceLoop v1"
        )

        return (
            banner,
            kpis,
            cov,
            fig,
            evidence,
            detail,
            simple_answer,
            tier_badges,
            simple_ranked,
            simple_evidence_cards,
            tier_fig,
            tier_pie_fig,
            ev_fig,
            waterfall_fig,
            heatmap_fig,
            confidence_overview_fig,
            resistance_breakdown_fig,
            grid_rows,
            standalone_detail,
            grid_rows,
            simple_df.to_dict("records"),
            footer,
        )
