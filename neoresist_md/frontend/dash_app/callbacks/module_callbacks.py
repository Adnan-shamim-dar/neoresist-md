from __future__ import annotations

from dash import Input, Output


def register_module_callbacks(app) -> None:
    @app.callback(Output("module-panel-status", "children"), Input("module-run-btn", "n_clicks"))
    def module_panel_status(n_clicks):
        return "Ready." if not n_clicks else "Module run requested."

