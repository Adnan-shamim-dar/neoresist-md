# AGENTS.md — NeoResist-MD Project State
# Single source of truth for all AI coding agents (Claude, Codex, Cursor, etc.)
# READ THIS FIRST. UPDATE THIS LAST.
# Last updated: 2026-04-21 (session 2) by Claude (claude-sonnet-4-6)

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
Sahin 2017:  125 rows, 13 patients (melanoma) — NO binding
             hla_allele col is EMPTY in training_matrix.csv for all 13 patients
             P04 HLA-I confirmed from supp: A*02:01, B*07:02, B*44:02 | HLA-II: DRB1*15:01
             Remaining 12 patients (P01-P03, P05-P13): HLA still missing
             Need: Extended Data Table from Nature 23003 (NOT Supp Table 1)
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

### Key finding (this IS the paper thesis)
No single strategy generalizes everywhere. Cancer-type pattern emerging:
melanoma responds to expression signal, GBM/pancreatic respond to sequence
features. This motivates the configurable platform and the paper's argument.

## ACTIVE TASKS (update this every session)

[x] Fix dash_bootstrap_components — installed 2026-04-21, 40/40 tests still passing
[~] Source Sahin HLA from Nature 2017 — PARTIAL: P04 done (A*02:01, B*07:02, B*44:02, DRB1*15:01)
    Supp Table 1 only covers P04. Need Extended Data Table for P01-P03, P05-P13.
    data/sahin2017_supp.xlsx saved to repo.
[ ] Download Müller 2023 Data S1-S4 from Cell Immunity paper (manual download needed)
[ ] Install ITSNdb R package and export to CSV
[ ] Draft paper methods section (user has template from prior Claude conversation)
[ ] Commit the ~40 modified files currently in working tree
[ ] Investigate/add .gitignore entry for data/final/*.parquet

## CHANGELOG
<!-- Append after every session. Format: DATE | AGENT | WHAT CHANGED -->

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
