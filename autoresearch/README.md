# autoresearch/

Core strategy discovery engine for NeoResist-MD.

## Files

- `autoresearch_loop.py` — Main search loop. 7 operator families, adaptive family weighting, promotion criteria, crystallisation protocol. 3,087 lines.
- `paper_evidence.py` — Evaluation pipeline. Dataset loading, score_strategy, evaluate_reranking, bootstrap CI, permutation tests.

## Usage

```bash
# Run the discovery loop (from repo root)
python -m backend.strategy_engine.autoresearch_loop --help

# Or directly
python autoresearch/autoresearch_loop.py --config configs/autoresearch_loop.yaml
```

## Configuration

All loop parameters are in `configs/autoresearch_loop.yaml`. Key settings:
- `max_rounds`: iterations per session
- `ci_margin`: minimum CI overlap allowed for promotion
- `seed_stability_runs`: number of seed repetitions for stability gate
- `min_contexts_for_phase_advance`: contexts that must promote before advancing

## Operator families

| Family | Description |
|---|---|
| existing | Pre-specified library seeds |
| mutation | Perturbation of surviving strategies |
| recombination | Crossover between survivors |
| random_blend | Dirichlet-sampled combinations |
| toolstack | Tool-output-tied feature sets |
| ml_importance | Cross-validated feature importances |
| gated_ranking | Binding-gated TCR reranking |
