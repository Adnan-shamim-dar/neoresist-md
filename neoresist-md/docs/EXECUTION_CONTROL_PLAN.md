# EXECUTION_CONTROL_PLAN

## Purpose

Persistent execution tracker for implementing the NeoResist-MD plan without losing context across chats.

## Operating Mode

- Perplexity plan = strategic authority.
- This execution layer = implementation authority.
- No core scientific code changes without explicit user confirmation.
- Documentation/scaffolding/schema planning can proceed immediately.

## Current State Snapshot

- Repo skeleton exists for `neoresist-md/`.
- Canonical plan exists in `docs/NEORESIST-MD_MASTER_PLAN.md`.
- Architecture and I/O contracts exist in `docs/ARCHITECTURE.md`.
- Placeholder module folders exist under `backend/core/`.
- Repo-root `app.py` remains the active runtime entrypoint.
- Persistent execution continuity now depends on on-disk case/module state under `local_state/cases/`.

## Phase Plan

### Phase A - Documentation Lock (in progress)

- [x] Canonical master plan file
- [x] Architecture with module contracts
- [x] Roadmap file
- [x] Cursor system prompt file
- [x] Execution control plan file

### Phase B - Schema Lock (approval needed for implementation details)

- [ ] Add JSON schema specs for M0-M4 outputs
- [ ] Add schema versioning policy examples
- [ ] Add module-level TODO files

### Phase C - Core Spine (requires explicit approval before coding)

- [ ] Extract NeoVax core service into `backend/core/neoantigen/`
- [ ] Add QC placeholder module
- [ ] Add CLI pipeline runner
- [ ] Emit schema-valid outputs across modules

### Phase D - Validation and Expansion

- [ ] Add retrospective validation cases
- [ ] Add audit/confidence reporting
- [ ] Add thin web/API integration layer

### Phase E - Hybrid Migration Checkpointing (in progress)

- [x] Add canonical schema config for the shared long-format table contract
- [x] Add runtime helpers to materialize canonical columns with defaults
- [x] Add persistent module checkpoint labels and resume-ready flags
- [x] Recover stale `running` module states safely after interrupted sessions
- [x] Keep the Overview scatterplot visible in Expert mode
- [x] Add tests for canonical schema, UI mode visibility, and stale-run recovery
- [x] Expand Tier 2/Tier 3 module coverage beyond current verified modules
- [x] Expose canonical artifact summaries and confidence badges more deeply in the Expert UI
- [ ] Add regression tests for case detail rendering and diagnostics surfaces

## Decision Gates (must confirm with user)

- Gate 1: start core extraction from existing codebase
- Gate 2: choose schema format/tooling
- Gate 3: choose first validation cohort and metrics
- Gate 4: expose API/FHIR-facing artifacts

## Context Continuity Rules

- Every major change updates:
  - `docs/ROADMAP.md`
  - `docs/ARCHITECTURE.md` (if contracts changed)
  - this file (`docs/EXECUTION_CONTROL_PLAN.md`)
- Keep a short change note for each architecture decision.
- Treat this file as primary implementation memory between chats.

## Latest Completed Checkpoint

Date: 2026-04-14

- Added `configs/canonical_schema.yaml` as the canonical long-format table contract.
- Added `neoresist/canonical_schema.py` to load/fill the canonical schema and build canonical artifacts from supported module outputs.
- Hardened `neoresist/case_store.py` with:
  - `checkpoint_label`
  - `resume_ready`
  - `attempt_count`
- Hardened `neoresist/module_runner.py` to recover stale `running` worker states and resume safely from disk-backed checkpoints.
- Updated `neoresist/case_worker.py` to emit canonical CSV artifacts for supported modules:
  - `neoantigen_generation` -> `canonical_candidates.csv`
  - `expression_join` -> `expression_canonical.csv`
- Added `neoresist/ui_state.py` and routed Dash section visibility through it.
- Corrected the UI behavior so Expert mode keeps the Overview scatterplot instead of hiding it.
- Added/updated tests:
  - `backend/tests/test_canonical_schema.py`
  - `backend/tests/test_ui_state.py`
  - `backend/tests/test_case_platform.py`
- Expanded the module registry with additional prompt-aligned modules:
  - locally supported: `resistance_loop`, `strategy_engine`, `prioritization_tiering`
  - explicit unavailable placeholders: `presentation_netctlpan`, `escape_lohhla`, `recognition_foreignness`
- Hardened `synchronize_manifest()` so newly introduced modules are initialized for older persisted cases.
- Extended `neoresist/case_ui.py` Expert mode with:
  - confidence badges derived from canonical outputs when available
  - an `Artifacts` tab exposing stored artifact paths
  - richer supported-vs-unavailable module messaging
  - previews for resistance, strategy, and prioritization outputs
- Updated worker chaining so module completion automatically calls `resume_case()` and advances downstream modules even outside the browser.
- Added dependency-aware rerun cascades from the UI so upstream evidence changes invalidate downstream scoring/prioritization modules instead of leaving stale outputs.
- Extended the pipeline status panel to summarize persisted case/module state, not only cohort batch artifacts.
- Added unavailable-module adapter-status artifact emission once a placeholder module reaches an unavailable state.
- Verified a real persisted smoke case can run through:
  - `neoantigen_generation`
  - `expression_join`
  - `resistance_loop`
  - `strategy_engine`
  - `prioritization_tiering`
  with rerun/recovery validated on the final module after a real failure fix.

## Verification Completed

- `python -m unittest backend.tests.test_canonical_schema backend.tests.test_case_platform backend.tests.test_ui_state backend.tests.test_strategy_registry backend.tests.test_neoresist_loaders`
- `python -m compileall neoresist backend`
- Expanded regression suite:
  - `backend/tests/test_case_ui.py`
  - `backend/tests/test_ops_status.py`
- Local Dash smoke check against `http://127.0.0.1:8050/_dash-layout`
  - layout served successfully
  - mode toggle present
  - scatter section present
  - cases / advanced strategies / pipeline status sections present
- Real persisted smoke-case execution:
  - created case from uploaded CSV
  - attached RNA sidecar
  - disabled unsupported external modules
  - verified chained completion through prioritization
  - verified rerun of failed prioritization module to successful completion

## Exact Next Continuation Point

Continue from:
- `neoresist/case_ui.py`
- `neoresist/dash_app/layout.py`
- `backend/tests/`
- `docs/architecture.md`

Next task:
- if continuing beyond this build pass, focus on deeper architectural convergence only:
  - optional package-layout convergence toward the prompt’s target tree
  - optional adapter implementations for third-party scientific runtimes

After that:
- keep external runtimes honest unless their tools are actually installed and verifiable

## Active Assumptions

- Repo-root `app.py` remains the runtime entrypoint during migration.
- Existing working cohort views are preserved unless replaced by a verified equivalent.
- Only locally supportable modules are treated as guaranteed end-to-end; unsupported external runtimes remain explicit `unavailable` integrations.

## Recurring Dev Issue - UI changes not visible after code edits

### Symptom

- New Dash UI code is implemented, but the browser still shows the old app state or appears to ignore recent changes.
- Team concludes the app is "not running" or that the new implementation did not land, even though the code is present.

### Root cause

- The root Dash app runs from repo-root `app.py`.
- Historically it was launched with `debug=False`, so hot reload was disabled.
- When an older `python app.py` process is still running, new layout/callback/CSS changes will not appear until that process is restarted.
- Some changes are also gated behind navigation, mode, or selection state:
  - `Advanced Strategies` requires the matching nav view and Expert mode.
  - `Cases` and `Pipeline Status` only appear on those nav routes.
  - patient-specific panels such as evidence expansions require a patient/HLA selection.

### Standard fix

1. Confirm the correct entrypoint is being used: repo-root `python app.py`.
2. Stop any older Dash/python process that is still serving the previous build.
3. Start the app again from repo root.
4. For local development, enable hot reload:
   - Windows PowerShell: `set DASH_DEBUG=1`
   - then run: `python app.py`
5. Hard refresh the browser after restart.
6. If a feature still seems missing, verify whether it is hidden behind:
   - nav section
   - Simple vs Expert mode
   - patient/HLA selection

### Verification checklist

- `app.py` should import `create_app()` from `neoresist.dash_app`.
- The running app should expose current nav items such as `Cases`, `Advanced Strategies`, and `Pipeline Status`.
- The served assets should include the latest `assets/styles.css`.
- The version in `neoresist/version.py` should match the intended release marker.

### Prevention rule

- Whenever we add UI code, assume "stale Dash process / no hot reload / wrong screen state" before assuming the implementation failed.
- Prefer running local UI work with `DASH_DEBUG=1`.
- When reporting "changes not visible," always check process state, active route, active mode, and selection-dependent panels first.
