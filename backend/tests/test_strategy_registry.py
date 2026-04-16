from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from neoresist.paths import audit_log_path, strategy_store_dir
from neoresist.strategy_registry import (
    build_consensus_table,
    clone_strategy,
    list_strategies,
    log_audit_event,
    read_audit_events,
    save_strategy,
    score_candidates_for_strategy,
)
from neoresist.version import __version__


class TestStrategyRegistry(unittest.TestCase):
    def tearDown(self) -> None:
        for path in strategy_store_dir().glob("user-test-*.json"):
            path.unlink(missing_ok=True)
        ap = audit_log_path()
        if ap.is_file():
            txt = ap.read_text(encoding="utf-8")
            kept = [line for line in txt.splitlines() if '"strategy_id":"user-test-' not in line]
            if kept:
                ap.write_text("\n".join(kept) + "\n", encoding="utf-8")
            else:
                ap.unlink(missing_ok=True)

    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "hla_allele": "HLA-A0101",
                    "gene": "TP53",
                    "mutant_peptide": "AAAAAAA",
                    "protein_change": "p.R1H",
                    "expression_tpm": 12.0,
                    "real_expression_tpm": 18.0,
                    "presentation_score": 0.82,
                    "ccf": 0.74,
                    "real_ccf": 0.71,
                    "self_dissimilarity": 0.55,
                    "escape_penalty": 0.05,
                    "hla_loh_status": "intact",
                    "tier": 2,
                    "rl_priority": 0.5,
                    "exclusion_reasons": [],
                },
                {
                    "patient_id": "P1",
                    "hla_allele": "HLA-A0101",
                    "gene": "GENE2",
                    "mutant_peptide": "BBBBBBB",
                    "protein_change": "p.R2H",
                    "expression_tpm": 0.8,
                    "real_expression_tpm": 0.4,
                    "presentation_score": 0.73,
                    "ccf": 0.19,
                    "real_ccf": 0.16,
                    "self_dissimilarity": 0.48,
                    "escape_penalty": 0.62,
                    "hla_loh_status": "lost",
                    "tier": 3,
                    "rl_priority": 0.3,
                    "exclusion_reasons": ["loh_risk"],
                },
            ]
        )

    def test_clone_save_reload_and_audit(self) -> None:
        base = list_strategies()[0]
        clone = clone_strategy(base.strategy_id, display_name="Test Strategy Clone")
        clone = clone.__class__(**{**clone.__dict__, "strategy_id": "user-test-strategy-clone"})
        saved = save_strategy(clone)
        self.assertTrue((strategy_store_dir() / "user-test-strategy-clone.json").is_file())
        self.assertTrue(any(s.strategy_id == saved.strategy_id for s in list_strategies()))

        log_audit_event(
            event_type="save_strategy",
            strategy=saved,
            app_version=__version__,
            changes={"display_name": {"before": base.display_name, "after": saved.display_name}},
        )
        events = read_audit_events(limit=10)
        self.assertTrue(any(e["strategy_id"] == "user-test-strategy-clone" for e in events))

    def test_score_and_consensus_outputs(self) -> None:
        strategies = list_strategies()[:2]
        frames = {}
        for strategy in strategies:
            scored = score_candidates_for_strategy(self._sample_df(), strategy)
            self.assertIn("strategy_id", scored.columns)
            self.assertIn("coherence_score", scored.columns)
            self.assertTrue(scored["coherence_score"].between(0.0, 1.0).all())
            frames[strategy.strategy_id] = scored

        consensus = build_consensus_table(frames, top_n=10)
        self.assertFalse(consensus.empty)
        self.assertIn("consensus_score", consensus.columns)
        self.assertIn("consensus_agreement_count", consensus.columns)


if __name__ == "__main__":
    unittest.main()
