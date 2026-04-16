from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    tool_id: str = "base_adapter"

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        raise NotImplementedError

