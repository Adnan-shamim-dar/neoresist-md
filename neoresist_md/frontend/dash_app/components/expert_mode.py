from __future__ import annotations

from dash import html

from .module_card import module_card


def expert_mode_layout():
    return html.Div(
        [
            html.H4("Expert Mode"),
            html.Button("Run", id="module-run-btn"),
            html.Div(id="module-panel-status"),
            html.Div([module_card("generation"), module_card("resistance_loop")]),
        ]
    )

