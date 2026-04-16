from __future__ import annotations

from dash import html


def module_card(module_name: str):
    return html.Div([html.Strong(module_name), html.Div("Status: placeholder")], className="module-card")

