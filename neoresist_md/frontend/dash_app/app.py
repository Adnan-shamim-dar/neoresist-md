from __future__ import annotations

from dash import Dash

from .callbacks import register_callbacks
from .layout import build_layout

app = Dash(__name__)
app.layout = build_layout()
register_callbacks(app)

server = app.server

if __name__ == "__main__":
    app.run(debug=True)

