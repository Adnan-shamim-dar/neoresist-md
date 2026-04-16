from __future__ import annotations

from dash import Input, Output


def register_mode_callbacks(app) -> None:
    @app.callback(
        Output("mode-shell", "style"),
        Output("expert-shell", "style"),
        Input("ui-mode-toggle", "value"),
    )
    def toggle_mode(mode):
        is_expert = str(mode) == "Expert"
        return ({"display": "none"}, {"display": "block"}) if is_expert else ({"display": "block"}, {"display": "none"})

