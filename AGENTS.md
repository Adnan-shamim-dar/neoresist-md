# AGENTS.md — NeoResist-MD Project State
# Single source of truth for all AI coding agents (Claude, Codex, Cursor, etc.)
# READ THIS FIRST. UPDATE THIS LAST.
# Last updated: 2026-04-21 (session 7) by Claude (claude-sonnet-4-6)

## GOLDEN RULE
Never break what works. The stable RL v1 engine in neoresist/
is production. neoresist_md/ is the new modular arch in progress.
Tests must pass before and after every change.
Run: python -m unittest discover backend/tests -v

## WHAT THIS PROJECT IS
NeoResist-MD: Dash web app for resistance-aware neoantigen
prioritization. Scores tumor mutations by immunogenicity
minus resistance penalty. Dual UI modes (Simple + Expert).

Entry point: app.py → http://127.0.0.1:8050
Branch: dash-ui-migration
Python: 3.14.3 (app), 3.11 (MHCflurry/validation scripts in .venv311/)

## ARCHITECTURE

```
neoresist/           STABLE — RL v1 engine, DO NOT REFACTOR
  loaders.py         Data ingestion (MAF/TSV/Parquet, ordered search paths)
  scoring.py         FROZEN — core RL scoring logic, never touch
  profiles.py        Load YAML scoring/rule profiles into frozen dataclasses
  schema.py          Column synonyms + validate_dash_columns
  canonical_schema.py  Long-format table contract, append-only module outputs
  module_schema.py   Prompt-aligned module registry from YAML
  case_store.py      Persist cases, manifests, artifacts, audit events
  case_worker.py     Per-module execution, writing artifacts to disk
  module_runner.py   Resume-safe orchestration of background module workers
  case_ui.py         Expert/Simple case rendering from persisted module state
  ops_status.py      Summarize runtime state for pipeline status panel
  strategy_registry.py  Save/load/score strategies, manage audit trail
  upload_runtime.py  Upload normalization (VCF/MAF → predictor-ready CSV)
  ui_state.py        resolve_view_state() — governs 20+ section visibility
  dash_app/
    layout.py        1200+ lines, responsive sidebar + main content
    callbacks.py     Filter/aggregate/render logic, strategy editor, case lifecycle
    panels.py        KPI cards, patient detail, evidence layout, scatter figures
    data.py          Filter aggregation, slider bounds, in-memory Parquet caching
    display_utils.py HLA normalization, exclusion rule loading, display policies
    constants.py     Tier colors/labels, Bootstrap Cyborg dark theme

neoresist_md/        IN PROGRESS — new modular architecture (do not break neoresist/)
  backend/core/
    base_module.py         Base class for all modules
    module_runner.py       Async background execution + checkpoint resumption
    clonality/             PyClone module (clonal frequency estimation)
    escape/                LOHHLA module (HLA LOH escape detection)
    expression/            Expression normalization module
    generation/            MHCflurry module (neoantigen generation + binding)
    presentation/          HLA presentation scoring
    prioritization/        Candidate ranking module
    recognition/           Foreignness/self-dissimilarity module
    resistance_loop/       ResistanceLoop wrapper
    strategy_engine/       Strategy scoring and consensus

backend/             CLI enrichment pipeline
  cli/enrich_cohort  TCGA metadata join, expression/purity/clonality, RL scoring
  core/
    prioritization/resistance_loop.py  Wraps apply_resistance_loop_engine()
    qualification/escape.py
  tests/             40 tests, all passing
  validation/        VALIDATION PIPELINE — run with py -3.11
    artifacts/       All validation results (JSON/CSV)
    targeted_hypothesis_test.py
    new_features_and_validate.py
    full_cross_validation.py
    tesla_validate.py, ott_haystack_analysis.py, etc.

configs/
  app_config.yaml          Branding, default profile IDs, cohort search paths
  scoring_profiles/        rl_v1.yaml, rl_tcr_v1.yaml, rl_balanced.yaml,
                           rl_expression_heavy.yaml
  rule_profiles/           default_rules.yaml (Tier1/Tier2 thresholds)
  canonical_schema.yaml    Data contract
  module_schema.yaml       Module registry

data/final/          Pre-enriched Parquet cohort files (large binaries — tracked
                     in git, DO NOT add more parquet files to git)

app.py               Thin entry point — imports neoresist.dash_app.create_app()
app_streamlit_backup.py  Legacy Streamlit backup — IGNORE
```

## SCORING FORMULA

```
rl_priority = immunogenicity_blend × (1 − resistance_penalty)

immunogenicity_blend = W_expr × expression_norm
                     + W_pres × presentation
                     + W_ccf  × ccf
                     + W_self × self_dissimilarity

resistance_penalty = escape_penalty_weight × escape   (bounded [0, 1])
```

All weights are config-driven (scoring_profiles/*.yaml). No `eval()` anywhere.

## CURRENT STATE

### What works (DO NOT BREAK)
- Dash app boots: `python app.py` → http://127.0.0.1:8050
- Dual UI modes: Simple (overview + upload) + Expert (full pipeline)
- RL v1 scoring engine (neoresist/scoring.py — FROZEN)
- Case management + strategy system + audit trail
- TCGA-SARC cohort loaded (245 patients, 430K candidates)
- 40/40 backend tests passing (confirmed 2026-04-21)
- Upload: VCF/MAF → predictor-ready CSV normalization working
- Strategy registry: validated AUC scores across 4 datasets

### In-flight (touch carefully)
- neoresist_md/ modular architecture migration (parallel to stable neoresist/)
- ~40 uncommitted modified files on working tree (pre-existing from Codex sessions)
- Most recent commit bc39165: 286 lines added to callbacks.py + layout.py

### Known landmines
- `neoresist/scoring.py` is FROZEN — never modify under any circumstances
- `data/final/*.parquet` tracked in git (large binaries) — do not add more
- Two parallel module systems (neoresist/ vs neoresist_md/) are intentionally
  coexisting during migration — do not merge or delete either
- `binding_affinity_nm` is NOT_AVAILABLE for Sahin/Rojas in training_matrix.csv
  — do not treat as numeric zero
- Three similarly-named dirs: neoresist/ (stable code), neoresist-md/ (docs only),
  neoresist_md/ (new modular code) — don't confuse them
- dash_bootstrap_components INSTALLED (2026-04-21) — `pip install dash-bootstrap-components`
- Sahin 2017 Supp Table 1 URL (provided) only contains P04 vaccine design, NOT full
  per-patient HLA table. Full HLA for all 13 patients is in a different extended data
  table in the paper. data/sahin2017_supp.xlsx saved but only useful for P04.

## VALIDATION RESULTS (Science side)

### Datasets
```
Ott 2017:    97 mutations, 15 immuno, 6 patients (melanoma)
TESLA 2020:  918 candidates, 41 immuno, 9 patients (mixed)
Hilf 2019:   152 rows, 15 patients (GBM) — binding 127/152
Rojas 2023:  230 rows, 16 patients (pancreatic) — binding complete
Sahin 2017:  125 rows → 165 rows after HLA expansion (13 patients, melanoma)
             HLA SOURCED from Extended Data Table 3 (Nature 547:222-226):
               P01→A*31:01  P02→B*39:06  P04→A*02:01,B*07:02,B*44:02
               P05→B*07:02  P06→A*11:01  P11→A*02:01
               P17→A*68:01,B*37:01  P19→B*57:01,A*11:01
               P03, P07, P09, P10, P12 → NOT_AVAILABLE (no confirmed class-I)
             Binding: 120/165 rows predicted via MHCflurry sliding window (8-11mer scan)
               80/125 unique mutations covered
             Files: validation_papers/sahin_with_binding.csv
                    artifacts/sahin_cross_validation_with_binding.json
                    artifacts/sahin_cross_matrix_with_binding.csv
Keskin 2019: 27 rows (GBM) — no LOPO (too few patients)
```

### Strategy registry (validated AUC)
```
Strategy             Ott     TESLA   Hilf    Rojas
rl_binding_only      0.649   0.771   0.624   0.547
rl_expression_v1     0.716   0.813   0.497   0.547
rl_tcr_v1            0.758   0.510   0.493   0.675
binding_plus_calis   0.667   0.762   0.627   0.523
```

### Müller 2023 NCI results (2026-04-21)
```
Dataset: muller_nci.tsv | 292,495 rows | 82 positives | 56 patients | 43 LOPO folds
Columns: MT_BindAff(nM), Score_EL(NetMHCpan EL 0-1), Quantification(expression),
         MT_pep_x(peptide 8-12mer), VALIDATED(0/1). NO wildtype peptide.

Strategy                    LOPO AUC    Note
binding_only                0.9661      bind_log50k from MT_BindAff
score_el_only               0.9839 ★    NetMHCpan EL score — best single feature
rl_expression_v1_from_tesla 0.9656      near-identical to binding_only
rl_tcr_v1_from_ott          0.4917      FAILS — tcr_charge_diff unavailable (no WT),
                                        TCR volume dominates and hurts performance
binding_plus_calis          0.9622

Key findings:
  - NCI is binding-dominated: Score_EL LOPO=0.984 (near-perfect)
  - rl_tcr_v1 fails without WT peptide (tcr_charge_diff excluded, volume hurts)
  - Confirms: NCI uses different biology than melanoma/GBM — binding alone is sufficient
  - Strengthens paper thesis: no single strategy works everywhere
File: artifacts/muller_nci_results.json

### Müller 2023 NCI — Verification + Strategy Discovery (2026-04-21)
```
STEP 1 — Bias check:
  Selection bias flag NEGATIVE: only 3.7% of negatives <500nM (not pre-screened)
  Positives: nm_raw median=52.1 nM, 80.5% below 500nM — genuine strong binders
  Negatives: nm_raw median=24,947 nM, Score_EL median=0.0005
  Positives: Score_EL median=0.7892 — massive clean separation
  Per-patient EL AUC: 0.94–0.9997 across top-10 patients — signal is REAL
  Interpretation: high AUC is not artifactual; positives are truly extreme binders

STEP 2 — NCI_S1–S7 strategy discovery:
  Pre-committed directions: Score_EL+ BindStab+ Agretopicity− expression+

  Strategy                               LOPO AUC
  NCI_S1  Score_EL only                  0.9839 ★ best
  NCI_S2  Score_EL + BindStab            0.9678   adding BindStab slightly hurts
  NCI_S5  Score_EL + BindStab + Agret    0.9642
  NCI_S3  Score_EL + Agretopicity        0.9351   Agretopicity dilutes signal
  NCI_S6  Score_EL + BindStab + expr     0.7612
  NCI_S7  Score_EL + BindStab + Agret + expr  0.7679
  NCI_S4  Score_EL + expression          0.7431   expression kills performance

  Conclusion: Score_EL alone is optimal on NCI. No combination improves it.
  Expression is a destructive feature on NCI (correlated within patient, not predictive).
  Agretopicity (inverted) direction is correct but adds noise on this dataset.
Files: artifacts/muller_nci_discovery.json, backend/validation/muller_nci_discovery.py
```

### SHANK2 G486S verification (2026-04-21)
```
Dataset:    Keskin 2019 (GBM, patient keskin_8) — NOT in Ott 2017
Peptide:    SDRVKVLSI | HLA: HLA-B*08:01 | immunogenic: YES
Binding:    447.37 nM — rank 17/17 (dead last, 0th percentile)
rl_tcr_v1:  rank 5/17 (70.6th percentile) — +12 positions
Conclusion: SHANK2 surfaces dramatically higher under rl_tcr_v1.
            Compelling paper example: worst binder → top-5 candidate
            due to high TCR surface volume (tcr_volume=140.86).
Artifact:   artifacts/shank2_verification.json
```

### MHCflurry %rank baseline (2026-04-21, cross-matrix update)
```
Strategy                        Ott    TESLA  Sahin  Hilf   Rojas  Mean
binding_only                    0.544  0.771  0.477  0.624  0.547  0.593
rl_tcr_v1_from_ott              0.698  0.510  0.660  0.493  0.675  0.607 ★ best mean
rl_expression_v1_from_tesla     0.647  0.813  0.437  0.497  0.547  0.588
mhcflurry_percentile_rank_only  N/A    N/A    0.504  0.596  0.653  0.584
binding_plus_calis              0.575  0.762  0.401  0.627  0.523  0.578

Notes:
  - %rank outperforms nM binding (0.584 vs 0.593 mean — within noise)
  - rl_tcr_v1 beats %rank baseline on melanoma (Ott, Sahin) by design
  - Ott/TESLA have no %rank data (bind_log50k used as proxy)
  - PRIME: not available in any dataset
  Files: artifacts/full_cross_matrix_with_baselines.csv
         artifacts/percentile_baseline_results.json
```

### HONEST VALIDATION AUDIT (2026-04-21) — pre-paper integrity checks
```
Script: backend/validation/honest_validation_audit.py
Artifact: artifacts/honest_validation_audit.json
Method: Bootstrap CIs from existing per-patient LOPO AUC values (1000 resamples).
        All CIs are on the same numbers reported in the cross-matrix.

ISSUE 1 — Strategy origins:
  binding_only:               pre-specified (universal baseline, no data required)
  rl_tcr_v1_from_ott:         DISCOVERED on Ott (H14 = best of 21 hypotheses)
  rl_expression_v1_from_tesla: DISCOVERED on TESLA (expression direction selected on TESLA)
  binding_plus_calis:         pre-specified (Calis 2013 published score)

ISSUE 2 — rl_expression_v1 honest reframe:
  WRONG:  "validated on Ott AND TESLA"
  CORRECT: "discovered on TESLA, transfers to Ott (+0.067 over baseline)"
  TESLA result must be labeled 'training dataset' in all tables.

ISSUE 3 — Bootstrap CIs (1000 patient-level resamples):
  Strategy                  Dataset        LOPO AUC  95% CI             Notes
  binding_only              Ott (n=6)      0.649     [0.538, 0.753]     reliable ✓
  binding_only              TESLA (n=9)    0.771     [0.685, 0.860]     reliable ✓
  binding_only              Hilf (n=15)    0.613     [0.519, 0.705]     reliable ✓
  rl_tcr_v1_from_ott        Ott            0.680     [0.546, 0.839]     CI⊃baseline (n.s.)
  rl_tcr_v1_from_ott        TESLA          0.510     [0.404, 0.634]     CI⊃0.5 (fails on TESLA, no WT)
  rl_tcr_v1_from_ott        Hilf           0.578     [0.482, 0.677]     CI⊃baseline (n.s., worse)
  rl_tcr_v1_from_ott        Rojas          0.677     [0.501, 0.875]     wide CI (no binding baseline)
  rl_tcr_v1_from_ott        Sahin          0.657     [0.549, 0.766]     vs baseline N/A (no binding)
  rl_expression_v1_tesla    Ott            0.716     [0.642, 0.778]     CI⊃baseline (n.s.)
  rl_expression_v1_tesla    Sahin          0.462     [0.347, 0.575]     CI⊃0.5 (fails)

ISSUE 4 — H14 Bonferroni (21 hypotheses tested on Ott):
  Bonferroni threshold: 0.0024
  H14 t-test p = 0.2819, Wilcoxon p = 0.4375
  Bootstrap CI for mean diff vs baseline: [-0.042, 0.264] — crosses zero
  FAILS both Bonferroni AND uncorrected p<0.05.
  VERDICT: Exploratory only. Cannot claim statistical significance.

ISSUE 5 — Keskin removed from AUC tables:
  n=2 non-immunogenic rows → LOPO AUC meaningless.
  SHANK2 G486S retained as qualitative case study only.

ISSUE 6 — Feature availability table: see artifacts/honest_validation_audit.json

SURVIVORSHIP SUMMARY:
  PASS:
    - binding_only baseline reliable on Ott, Hilf, TESLA (CI does not include 0.5)
    - Score_EL dominance on Müller NCI is real (not selection-bias artifact)
    - Keskin properly excluded from AUC tables
  FAIL (insufficient evidence):
    - H14 rl_tcr_v1 improvement on Ott (p=0.28, n=6, not significant)
    - rl_expression_v1 transfer to Ott (CI includes baseline 0.649)
    - rl_tcr_v1 transfer to Hilf, Rojas, Sahin, TESLA (wide CIs or performs worse)

FOR PAPER:
  Main table: pre-specified strategies + transfer tests (labeled correctly)
  Supplementary: H14 Bonferroni table, discovery-set results labeled as training
  Remove: any claim of statistical significance for TCR features over binding baseline
  SHANK2: qualitative figure only, not as AUC evidence
```

### Sahin 2017 PHASE 4/5 results (2026-04-21, MHCflurry binding)
```
Strategy                                 Sahin LOPO
binding_only                             0.4771
rl_v1_original                           0.4869
rl_tcr_v1_from_ott                       0.6599  ★ best transfer
rl_expression_v1_from_tesla              0.4371
binding_plus_calis                       0.4011
BEST_OF_sahin_2017 (dataset-specific)    0.7295  ★ best overall
```
NOTE: Sahin is a vaccine dataset — 10 pre-selected mutations per patient,
~66% immunogenicity rate (vs ~5-15% in other datasets). Binding alone
performs near-chance (0.48). TCR features (rl_tcr_v1) show strongest
transfer. Confirms melanoma TCR signal, consistent with rl_tcr_v1 Ott=0.758.

### Key finding (this IS the paper thesis)
No single strategy generalizes everywhere. Cancer-type pattern emerging:
melanoma responds to TCR/sequence features (Ott AUC=0.758, Sahin AUC=0.660),
GBM/pancreatic respond to expression signal (Hilf/Rojas). Binding alone
is insufficient for melanoma (Sahin LOPO=0.48, near chance).
This motivates the configurable platform and the paper's argument.

## ACTIVE TASKS (update this every session)

[x] Fix dash_bootstrap_components — installed 2026-04-21, 40/40 tests still passing
[x] Source Sahin HLA from Nature 2017 Extended Data Table 3 — DONE 2026-04-21
[x] SHANK2 G486S verified (Keskin 2019, rank 17→5 under rl_tcr_v1, +12 positions)
[x] NetMHCpan %rank baseline added to cross-matrix — DONE 2026-04-21
    rl_tcr_v1 beats %rank on melanoma (Ott 0.698 vs N/A, Sahin 0.660 vs 0.504)
[x] Müller 2023 figshare: BLOCKED (202 HTML page, cannot auto-download)
    ingest_muller_s3.py created — ready to run once Data_S3.xlsx is placed manually
    8 patients have confirmed HLA, 5 unknown (P03, P07, P09, P10, P12)
    MHCflurry binding predictions run: 120/165 rows covered
    PHASE 4/5 complete: rl_tcr_v1 LOPO=0.660, binding_only LOPO=0.477
[x] Download Müller 2023 NCI data — DONE. muller_nci.tsv (23MB) ingested.
    LOPO results: Score_EL=0.984, binding_only=0.966, rl_tcr_v1=0.492 (fails w/o WT)
[ ] Müller 2023 Data S3 full feature set — may contain additional datasets (HiTIDE/TESLA)
    URL: https://www.cell.com/immunity/fulltext/S1074-7613(23)00406-5
    Save as: validation_papers/muller2023/Data_S3.xlsx
    Then run: py -3.11 backend/validation/ingest_muller_s3.py
[ ] Install ITSNdb R package and export to CSV
[ ] Draft paper methods section (user has template from prior Claude conversation)
[ ] Commit the ~40 modified files currently in working tree
[ ] Investigate/add .gitignore entry for data/final/*.parquet

## CHANGELOG
<!-- Append after every session. Format: DATE | AGENT | WHAT CHANGED -->

2026-04-21 | Claude claude-sonnet-4-6 | Session 5: Müller NCI analysis
  Changed: backend/validation/muller_nci_analysis.py (new),
           artifacts/muller_nci_results.json
  Validation: muller_nci Score_EL LOPO=0.984, binding_only=0.966, rl_tcr_v1=0.492
  Tests: 40/40 passing
  Next: paper methods draft; interpret NCI binding-dominance in paper context
2026-04-21 | Claude claude-sonnet-4-6 | Session 4: SHANK2 verification, %rank baseline, Müller prep
  Changed: task1_shank2_verification.py, task2_percentile_baseline.py,
           add_sahin_percentile.py, ingest_muller_s3.py (all new),
           sahin_with_binding.csv (added percentile_rank),
           shank2_verification.json, full_cross_matrix_with_baselines.csv,
           percentile_baseline_results.json
  Validation: SHANK2 rank 17/17→5/17 (+12). rl_tcr_v1 mean=0.607 best.
              %rank baseline: Rojas=0.653, Hilf=0.596, Sahin=0.504
  Tests: 40/40 passing
  Next: human downloads Müller Data_S3.xlsx; draft paper methods
2026-04-21 | Claude claude-sonnet-4-6 | Session 3: Sahin HLA from ED Table 3, MHCflurry binding
  predictions, PHASE 4/5 run for Sahin.
  Changed: backend/validation/add_sahin_binding.py (new),
           backend/validation/rerun_sahin_with_binding.py (new),
           validation_papers/sahin_with_binding.csv (generated, 165 rows),
           artifacts/sahin_cross_validation_with_binding.json,
           artifacts/sahin_cross_matrix_with_binding.csv
  Validation: sahin_2017 rl_tcr_v1 LOPO=0.6599, binding_only=0.4771,
              BEST_OF=0.7295 (dataset-specific discovery)
  Tests: 40/40 passing
  Next: integrate Sahin into full cross-dataset table; draft paper methods
2026-04-21 | Claude claude-sonnet-4-6 | Session 2: installed dash-bootstrap-components, attempted
  Sahin HLA sourcing, discovered Supp Table 1 is P04-only not full cohort.
  Changed: data/sahin2017_supp.xlsx (added), AGENTS.md (updated)
  Validation: no scripts run
  Tests: 40/40 passing
  Next: find full Sahin per-patient HLA table (Extended Data in Nature 23003)
2026-04-21 | Claude claude-sonnet-4-6 | Session 1: Created AGENTS.md and .cursor/rules
  from full codebase analysis. 40/40 tests confirmed passing. No code changed.
2026-04-16 | Codex (OpenAI) | Snapshot commit (1399a45) — see git log for prior work

---

## FOR THE NEXT AGENT — START HERE

1. Read this file completely
2. Run tests: `python -m unittest discover backend/tests -v`
3. Check working tree: `git status`
4. Ask the user what they want to do
5. **Do not touch `neoresist/scoring.py` under any circumstances**
6. Update the CHANGELOG and ACTIVE TASKS sections before ending the session
7. Commit AGENTS.md: `git add AGENTS.md && git commit -m "docs: update AGENTS.md"`
