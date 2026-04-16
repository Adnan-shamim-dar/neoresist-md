from __future__ import annotations

from dash import dcc, html

from .components.expert_mode import expert_mode_layout
from .components.simple_mode import simple_mode_layout
from .components.upload_wizard import upload_wizard


def build_layout():
    return html.Div(
        [
            dcc.Store(id="ui-mode-store", data="Simple"),
            html.H2("NeoResist-MD"),
            dcc.RadioItems(
                id="ui-mode-toggle",
                options=[{"label": "Simple", "value": "Simple"}, {"label": "Expert", "value": "Expert"}],
                value="Simple",
                inline=True,
            ),
            upload_wizard(),
            html.Div(id="mode-shell", children=simple_mode_layout()),
            html.Div(id="expert-shell", children=expert_mode_layout(), style={"display": "none"}),
        ]
    )

