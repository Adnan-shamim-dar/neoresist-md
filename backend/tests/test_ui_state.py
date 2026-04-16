from __future__ import annotations

import unittest

from neoresist.ui_state import resolve_view_state


class UiStateTest(unittest.TestCase):
    def test_expert_mode_keeps_scatter_visible_on_overview(self) -> None:
        state = resolve_view_state("Overview", "Expert")
        self.assertTrue(state["overview_section"])
        self.assertTrue(state["scatter_card"])

    def test_advanced_strategies_simple_mode_shows_note_not_workspace(self) -> None:
        state = resolve_view_state("Advanced Strategies", "Simple")
        self.assertFalse(state["advanced_simple_note"])
        self.assertFalse(state["advanced_section"])
        self.assertTrue(state["overview_section"])
        self.assertTrue(state["scatter_card"])

    def test_advanced_strategies_expert_mode_shows_workspace(self) -> None:
        state = resolve_view_state("Advanced Strategies", "Expert")
        self.assertFalse(state["advanced_simple_note"])
        self.assertTrue(state["advanced_section"])


if __name__ == "__main__":
    unittest.main()
