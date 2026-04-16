from __future__ import annotations

from .base_adapter import BaseAdapter


class AdapterTemplate(BaseAdapter):
    tool_id = "template"

    def is_available(self) -> tuple[bool, str]:
        return False, "Implement adapter availability check."

