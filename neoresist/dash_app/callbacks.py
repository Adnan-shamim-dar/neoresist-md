from __future__ import annotations

import io
import math
from collections import Counter
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback_context, html, no_update
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
from neoresist.dash_app.panels import (
    build_evidence_panel,
    build_patient_detail_card,
    hla_coverage_badge_row,
    kpi_card,
    make_scatter_fig,
)
from neoresist.dash_app.constants import TIER_COLORS
from neoresist.loaders import load_cohort_for_dash


def register_callbacks(app) -> None:
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
            return {"armed": True, "filename": _fn}, False, 0, "Armed: single-patient pipeline not enabled; timer demo only.", no_update
        if tid == "upload-interval":
            if not arm or not arm.get("armed"):
                raise PreventUpdate
            if n_int is None or n_int < 3:
                return arm, False, no_update, f"Waiting… ({int(n_int or 0)}/3)", no_update
            df, _ = load_qualified_candidates()
            if df.empty:
                return {"armed": False}, True, "No cohort loaded; cannot select demo row.", no_update
            row = df.iloc[0]
            demo = {"patient_id": str(row["patient_id"]), "hla_allele": str(row["hla_allele"])}
            return {"armed": False}, True, "Demo: first cohort row selected (upload pipeline pending).", demo
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
            raise PreventUpdate
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
            empty = make_scatter_fig(pd.DataFrame())
            detail = [html.Div("Fix columns or use enriched Parquet.", className="text-muted")]
            return (
                banner_err,
                [],
                html.Div(),
                empty,
                detail,
                detail,
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
        mm, mf, mc = slider_mut_max(), slider_fan_max(), slider_cand_max()
        agg_f = filter_agg(agg, list(tmb or [0, mm]), list(fanout or [0, mf]), list(cand or [0, mc]))
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

        return banner, kpis, cov, fig, evidence, detail, grid_rows, footer
