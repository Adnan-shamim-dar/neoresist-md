from __future__ import annotations

from dash import Input, Output
from dash.exceptions import PreventUpdate


def register_upload_callbacks(app) -> None:
    @app.callback(Output("upload-status", "children"), Input("upload-input", "contents"), prevent_initial_call=True)
    def upload_status(contents):
        if not contents:
            raise PreventUpdate
        return "Upload received."

