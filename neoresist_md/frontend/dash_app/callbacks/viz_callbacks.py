from __future__ import annotations

import plotly.express as px
from dash import Input, Output


def register_viz_callbacks(app) -> None:
    @app.callback(Output("simple-scatter", "figure"), Input("ui-mode-toggle", "value"))
    def render_scatter(_mode):
        return px.scatter(x=[0, 1], y=[0, 1], title="Scatter placeholder")

