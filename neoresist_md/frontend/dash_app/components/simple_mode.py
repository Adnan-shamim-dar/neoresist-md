from __future__ import annotations

from dash import dcc, html

from .candidate_card import candidate_card


def simple_mode_layout():
    return html.Div(
        [
            html.H4("Simple Mode"),
            html.Div(id="simple-ranked-list", children=[candidate_card("Candidate placeholder")]),
            dcc.Graph(id="simple-scatter"),
        ]
    )

