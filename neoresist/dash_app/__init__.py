from __future__ import annotations

from dash import Dash

from neoresist.dash_app.constants import APP_TITLE, ASSET_VERSION, EXTERNAL_STYLESHEETS
from neoresist.paths import repo_root
from neoresist_md.config_validation import validate_config_contract


def create_app() -> Dash:
    """Build Dash app (lazy imports avoid circular import with ``neoresist.loaders``)."""
    from neoresist.dash_app.callbacks import register_callbacks
    from neoresist.dash_app.layout import build_layout

    validate_config_contract(repo_root() / "neoresist_md" / "config")
    assets = str(repo_root() / "assets")
    app = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        assets_folder=assets,
        assets_url_path=f"assets-{ASSET_VERSION}",
    )
    app.title = APP_TITLE
    app.layout = build_layout()
    register_callbacks(app)
    return app
