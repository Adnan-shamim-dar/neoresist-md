"""
NeoResist-MD — Dash entrypoint. UI lives under ``neoresist.dash_app``.

Run from repo root::

    python app.py

Or::

    set PORT=8050
    python app.py
"""

from __future__ import annotations

import os

from neoresist.dash_app import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DASH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(debug=debug, host="0.0.0.0", port=port)
