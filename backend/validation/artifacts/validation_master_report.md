# ResistanceLoop Validation Master Report

## Scope
- Objective: test whether multi-feature neoantigen scoring can outperform binding-only ranking.
- Dataset focus: Ott 2017 mutation-level table with engineered features.
- Strict rule in this step: pre-specified hypotheses only, no fitting/optimization in LOPO evaluation.

## What Was Completed
1. Infrastructure build for publication scoring and matrix generation.
2. Expression scoring bug diagnosis and correction in validation pipeline.
3. Multi-stage diagnostics (coverage, per-paper behavior, distribution checks).
4. Broad feature sweep and nested CV to quantify overfitting risk.
5. New biological feature generation (DAI, IEDB-like, foreignness proxy, TCR physicochemical, anchor, mutation position, hamming).
6. Targeted pre-specified hypothesis testing (this run).

## Testing Integrity
- Test suite before targeted run: `python -m unittest discover backend/tests -v` -> 40/40 pass.
- Test suite after targeted run: `python -m unittest discover backend/tests -v` -> 40/40 pass.

## Data Snapshot
- Mutations analyzed: 97
- Immunogenic: 15
- Patients: 6 (ott_1, ott_2, ott_3, ott_4, ott_5, ott_6)

## Prior Stage Signals (Context)
- Earlier nested CV baseline (binding-only mean LOPO): 0.6493686868686869
- Earlier nested CV sweep-winner fixed strategy: 0.725378787878788
- Earlier nested CV optimized weighted sum: 0.5752946127946128
- Earlier nested CV optimized logistic: 0.6034511784511785
- Broad sweep baseline AUC (in-sample reference): 0.633739837398374

## Single-Feature Verification (This Run)
- Binding baseline (bind_log50k) best AUC: 0.633739837398374
- TCR volume (mut): 0.6943089430894309
- TCR charge diff: 0.6743589743589744
- TCR hydrophobicity mut: 0.6686991869918699

## Targeted Hypotheses Tested
- Total hypotheses: 21
- Baseline hypothesis: H0_binding_only
- Combinations include pair and triple biologically motivated models with fixed weights.
- AUTO direction only resolves sign from feature-group means, not weights.

## Ranked LOPO Results (Top 10)
1. H14_bind_plus_3tcr: LOPO mean 0.7577, in-sample 0.6545, delta vs binding +0.1083
2. H20_kitchen_sink: LOPO mean 0.7535, in-sample 0.6488, delta vs binding +0.1042
3. H13_bind_vol_hydro: LOPO mean 0.7405, in-sample 0.7049, delta vs binding +0.0912
4. H15_tcr_only: LOPO mean 0.7350, in-sample 0.6374, delta vs binding +0.0856
5. H2_bind_plus_charge_diff: LOPO mean 0.7189, in-sample 0.6630, delta vs binding +0.0695
6. H12_bind_vol_charge: LOPO mean 0.7170, in-sample 0.6398, delta vs binding +0.0676
7. H17_bind_heavy_tcr_volume: LOPO mean 0.7101, in-sample 0.7000, delta vs binding +0.0607
8. H1_bind_plus_tcr_volume: LOPO mean 0.7073, in-sample 0.6951, delta vs binding +0.0579
9. H18_equal_bind_tcr_volume: LOPO mean 0.7073, in-sample 0.6951, delta vs binding +0.0579
10. H19_tcr_volume_heavy: LOPO mean 0.7073, in-sample 0.6959, delta vs binding +0.0579

## Best vs Baseline
- Best hypothesis: H14_bind_plus_3tcr
- Best LOPO mean AUC: 0.7577
- Baseline LOPO mean AUC: 0.6494
- Delta: +0.1083
- Interpretation class: STRONG

## Statistical Check (Best vs Baseline)
- Paired stats computed across patients where both AUCs exist.
- See JSON `paired_stats_best_vs_baseline` for t-test, Wilcoxon, and bootstrap CI.

## Overfitting Audit
- For top models, compare in-sample vs LOPO gap.
- Large gap (>0.10) indicates overfit risk and should be treated as exploratory only.

## Strategy Summary Across Project
- Binding-only remains robust and stable.
- Wide unconstrained search can inflate in-sample AUC on small N.
- Pre-specified targeted strategies provide more honest external estimates.
- TCR feature families show real univariate signal, but multivariate lift must be judged on LOPO.

## Publication-Ready Claims You Can Defend
1. We implemented and validated multiple orthogonal feature families beyond binding.
2. We explicitly separated exploratory optimization from confirmatory testing.
3. We used leave-one-patient-out evaluation for honest patient-level generalization.
4. We provide full artifacts and per-patient outcomes for reproducibility.

## Limitations
- Small cohort (n=6 patients, 97 mutation-level rows) makes CIs wide.
- Statistical power is constrained for paired tests.
- Some advanced features depend on approximate proxies due data limitations.

## Recommended Next Steps
1. External validation on independent cohorts with same pre-specified hypotheses.
2. Pre-register one confirmatory model family before next run.
3. Aggregate multi-study mutation-level dataset with harmonized feature generation.

## Artifacts
- `targeted_hypothesis_results.json`
- `targeted_hypothesis_ranked.csv`
- `ott_mutation_level_with_new_features.csv`
- `new_features_validation_results.json`
- `nested_cv_results.json`
- `feature_sweep_results.json`
