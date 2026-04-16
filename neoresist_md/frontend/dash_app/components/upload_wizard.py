from __future__ import annotations

from dash import dcc, html


def upload_wizard():
    return html.Div(
        [
            html.H5("Upload Wizard"),
            dcc.Upload(id="upload-input", children=html.Button("Upload file")),
            html.Div(id="upload-status"),
        ]
    )

