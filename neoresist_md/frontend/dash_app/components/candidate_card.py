from __future__ import annotations

from dash import html


def candidate_card(title: str):
    return html.Div([html.Strong(title), html.Div("Evidence placeholder")], className="candidate-card")

