from __future__ import annotations

from .mode_callbacks import register_mode_callbacks
from .module_callbacks import register_module_callbacks
from .upload_callbacks import register_upload_callbacks
from .viz_callbacks import register_viz_callbacks


def register_callbacks(app) -> None:
    register_mode_callbacks(app)
    register_upload_callbacks(app)
    register_module_callbacks(app)
    register_viz_callbacks(app)

