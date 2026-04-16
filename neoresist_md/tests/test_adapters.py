from __future__ import annotations

from neoresist_md.backend.adapters.pvactools_adapter import PvacToolsAdapter


def test_adapter_contract():
    available, message = PvacToolsAdapter().is_available()
    assert isinstance(available, bool)
    assert isinstance(message, str)

