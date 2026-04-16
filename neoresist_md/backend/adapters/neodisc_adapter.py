from __future__ import annotations

from .base_adapter import BaseAdapter


class NeoDiscAdapter(BaseAdapter):
    tool_id = "neodisc"

    def is_available(self) -> tuple[bool, str]:
        return False, "NeoDisc not configured in this environment."

