from __future__ import annotations

from backend.strategy_engine.generate_peptides import generate_frameshift_peptides


def test_frameshift_peptide_generation():
    result = generate_frameshift_peptides(
        {"gene_name": "DHX40", "aa_change": "p.K123fs", "mutation_position": 123},
        lengths=[9],
    )
    assert len(result) > 0
    assert all(r["variant_type"] == "FRAMESHIFT" for r in result)
    assert all(r["is_frameshift"] is True for r in result)
    assert all(r["wt_pep"] is None for r in result)
    assert all(len(r["mt_pep"]) == 9 for r in result)


def test_frameshift_not_dropped():
    result = generate_frameshift_peptides(
        {"gene_name": "CASP1", "aa_change": "p.G183fs", "mutation_position": 183},
        lengths=[9],
    )
    assert len(result) > 0, "Frameshift peptides must not be silently dropped"
