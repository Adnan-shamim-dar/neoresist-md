from __future__ import annotations

from .base_adapter import BaseAdapter


class PvacToolsAdapter(BaseAdapter):
    tool_id = "pvactools"

    def is_available(self) -> tuple[bool, str]:
        return False, "pVACtools not configured in this environment."

