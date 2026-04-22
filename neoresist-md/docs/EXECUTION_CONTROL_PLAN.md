# EXECUTION_CONTROL_PLAN

## Purpose

Persistent execution tracker for NeoResist-MD implementation and verification continuity.

## Current Project State

- System verified and stabilized through a live-browser + test-suite sprint on 2026-04-17.
- Repo-root `app.py` remains the active runtime entrypoint.
- Current verified startup command: `python app.py`
- Current verified local URL: `http://127.0.0.1:8050`
- Persistent case/module state continues to live under `local_state/cases/`.

## What Was Completed In This Session

- Critical fix (resolved): Simple mode blank-screen routing.
  - Added expert-only nav guard and Simple-mode fallback routing so expert pages no longer resolve to blank content in Simple mode.
  - Added nav tooltips for expert-only items: "Switch to Expert mode to access this section."
- Critical fix (resolved): intermittent `prioritization_tiering` `EmptyDataError`.
  - Added pre-read artifact existence/size guards and empty-data checks in tiering.
  - Tiering now fails with a descriptive message (`No candidates available for tiering — upstream module produced empty output`) and keeps `resume_ready=true`.

- Cleaned stale Dash/process state and re-established a single clean app process.
- Added cache-busting for fonts and local CSS assets:
  - `neoresist/dash_app/constants.py`
  - `neoresist/dash_app/__init__.py`
- Ran full verification gates repeatedly:
  - `python -m unittest discover backend/tests -v`
  - `python -m compileall neoresist backend -q`
- Restored the import gate expected by the sprint prompt using small compatibility shims:
  - `compute_rl_priority`
  - `detect_kras_g12`
  - `CaseStore`
  - `StrategyRegistry`
  - `validate_canonical_table`
- Stabilized callback/layout issues found during live browser testing:
  - seeded hidden initial IDs for drawer close/download callbacks
  - ignored phantom zero-click callback triggers
  - hardened evidence drawer formatting for array-like values
  - restored Simple-mode export callback binding
- Added `tests/fixtures/bad_input.tsv` for broken-upload verification.
- Refreshed verification documentation:
  - `docs/FEATURE_AUDIT.md`
  - `docs/DEMO_VERIFICATION.md`
  - this file

## Verified Now

- App boots cleanly from a single `python app.py`.
- `_dash-layout` returns HTTP 200.
- Inter and JetBrains Mono load correctly in the live app.
- Navigation, mode toggle, Expert workspace, pipeline status page, cases page, and Simple evidence drawer/export all work at browser level.
- Backend regression suite currently passes:
  - 35 tests
- Compile check currently passes.

## Known Issues Remaining

1. Simple mode can still navigate to expert-only routes and show no visible main section.
2. Scatter box-select toggle is visible but did not switch Plotly drag mode during live verification.
3. `tests/fixtures/kras_msi_test.tsv` is not upload-compatible with the documented canonical upload contract, so live KRAS/MSI upload verification is blocked.
4. Repeated TCGA upload/rerun produced one intermittent `prioritization_tiering` failure:
   - `pandas.errors.EmptyDataError: No columns to parse from file`
   - earlier runs of the same file completed, so this looks like a stability/race issue, not a deterministic input failure.
5. Security and deployment baselines are still missing:
   - no auth
   - no encryption at rest
   - no user-attributed audit field
   - no Docker packaging

## Decision Boundary For Next Work

- Safe verification/stabilization edits remain allowed.
- Do not change scoring formula, case store semantics, module runner architecture, strategy semantics, or canonical schema contracts without explicit user direction.
- Prefer documenting risky findings over attempting broad fixes.

## Next Logical Sprint

### Priority 1 — Security / Compliance

- Add authentication foundation.
- Add encryption-at-rest strategy for persisted case data.
- Add user identity fields to audit events.
- Document pre-clinical security limitations explicitly.

### Priority 2 — Deployment / Reproducibility

- Add `Dockerfile`.
- Add `docker-compose.yml`.
- Produce a reproducible local demo environment spec.
- Pin or lock critical runtime dependencies.

### Priority 3 — Demo Stability

- Fix Simple-mode blank-route behavior safely.
- Fix scatter box-selection behavior.
- Harden the `strategy_engine -> prioritization_tiering` handoff against empty-artifact races.
- Re-run the full browser verification checklist after those fixes.

## Continuity Note

If a later session resumes from here, start with:

1. Confirm only one app process is serving port `8050`.
2. Re-run:
   - `python -m unittest discover backend/tests -v`
   - `python -m compileall neoresist backend -q`
3. Read `docs/DEMO_VERIFICATION.md` for the exact PASS/FAIL ledger from 2026-04-17.

## Reusable End-to-End Verification Protocol

Use this as the authoritative handoff checklist for full-system verification. This can run:
- after validation tasks A-D are complete, or
- as a standalone integrity sweep at any time.

### Operating Rules

1. Do not add features.
2. Do not refactor working code.
3. Do not modify scoring formula, case store semantics, module runner architecture, or strategy semantics unless fixing a verified bug.
4. Fix budget:
   - max 15 lines per single fix,
   - max 100 lines total across all fixes in the session.
5. If a fix exceeds budget or risk is high, mark as `KNOWN ISSUE` and continue verification.
6. For each check, record exactly one: `PASS`, `PARTIAL`, or `FAIL`.
7. If a check depends on validation outputs not yet present, keep it in scope and mark `PARTIAL`/`FAIL` with blocker text:
   - `validation artifacts not yet present`
8. Do not remove blocked checks from totals silently.
9. Before starting and after any fix, run:
   - `python -m compileall neoresist backend -q`
   - `python -m unittest discover backend/tests -v`
10. Update both:
   - `docs/FEATURE_AUDIT.md`
   - `neoresist-md/docs/EXECUTION_CONTROL_PLAN.md`

### Canonical Paths (Repo-Truth Normalized)

- Execution control: `neoresist-md/docs/EXECUTION_CONTROL_PLAN.md`
- Feature audit: `docs/FEATURE_AUDIT.md`
- Demo verification ledger: `docs/DEMO_VERIFICATION.md`
- Architecture reference: `docs/architecture.md`
- `rl_v1`: `configs/scoring_profiles/rl_v1.yaml`
- `rl_calibrated`: `configs/profiles/rl_calibrated.yaml`

### Phase 0: Clean Environment (PowerShell)

```powershell
# 1) Find listeners on 8050 and kill all related app python PIDs
$listeners = netstat -ano | Select-String ':8050' | Select-String 'LISTENING'
$pids = @()
foreach ($line in $listeners) {
  $parts = ($line -split '\s+') | Where-Object { $_ -ne '' }
  if ($parts.Count -gt 0) { $pids += $parts[-1] }
}
$pids = $pids | Sort-Object -Unique
foreach ($procId in $pids) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }

# 2) Verify 8050 clear before startup
netstat -ano | Select-String ':8050'

# 3) Clear bytecode caches (repo-local)
$root = (Resolve-Path '.').Path
Get-ChildItem -LiteralPath $root -Recurse -Force -File -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $root -Recurse -Force -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 4) Baseline checks
python -m compileall neoresist backend -q
python -m unittest discover backend/tests -v

# 5) Start app fresh and verify single listener + HTTP 200
$p = Start-Process python -ArgumentList 'app.py' -PassThru
Start-Sleep -Seconds 6
netstat -ano | Select-String ':8050'
(Invoke-WebRequest -Uri 'http://127.0.0.1:8050/_dash-layout' -UseBasicParsing -TimeoutSec 15).StatusCode
```

Required gate before Phase 1:
- `PASS` only if one listener on `8050`, compile/test pass, and HTTP status `200`.

### Phase 1: Import and Module Health

Run and record each as `PASS`/`FAIL`:
- `python -c "from neoresist.dash_app.layout import build_layout; print('layout OK')"`
- `python -c "from neoresist.scoring import compute_rl_priority; print('scoring OK')"`
- `python -c "from neoresist.upload_runtime import validate_uploaded_table; print('upload OK')"`
- `python -c "from neoresist.tumor_features import detect_kras_g12, estimate_msi_status; print('tumor_features OK')"`
- `python -c "from neoresist.case_store import CaseStore; print('case_store OK')"`
- `python -c "from neoresist.module_runner import ModuleRunner; print('module_runner OK')"`
- `python -c "from neoresist.strategy_registry import StrategyRegistry; print('strategy_registry OK')"`
- `python -c "from neoresist.canonical_schema import validate_canonical_table; print('canonical_schema OK')"`
- `python -c "from neoresist.ui_state import resolve_view_state; print('ui_state OK')"`
- `python -c "from neoresist.dash_app.panels import build_evidence_panel; print('panels OK')"`
- `python -c "from neoresist.dash_app.callbacks import *; print('callbacks OK')"`
- `python -c "from neoresist.dash_app.components.simple_mode import build_kpi_hero, build_candidate_cards; print('simple_mode OK')"`
- `python -c "from neoresist.dash_app.components.candidate_card import *; print('candidate_card OK')"`

### Phase 2: Typography and Color System

Record each as `PASS`/`PARTIAL`/`FAIL`:
- Inter and JetBrains Mono are loaded from external stylesheets.
- `assets/custom.css` declares the intended type scale classes.
- `assets/styles.css` has no untracked font-size drift outside approved scale values.
- Uppercase UI labels have letter-spacing (`.kpi-label`, `.nav-section-label`, `.tier-badge`).
- CSS variables exist for background/text/accent/tier palette.
- Hardcoded hex colors in `assets/styles.css` are tagged with `TODO` comments where palette migration is pending.
- Live UI check: dark background, readable contrast, no panel overlap at desktop width.

### Phase 3: Navigation and Mode System

- Sidebar rail has icon+label entries and grouped sections.
- Active nav item has clear visual state.
- Simple/Expert toggle is pill-style and stable.
- Expert-only routes do not render blank in Simple mode.
- Expert-only routes in Simple mode show disabled/tooltip guidance.
- Switch Expert->Simple while on expert-only route lands on a valid Simple route.
- Rapid mode toggle (20x) stays stable.
- Rapid nav cycle (3 loops) stays stable.

### Phase 4: Simple Mode Clinical Surface

- KPI hero renders correctly with major values.
- HLA coverage indicator renders and reflects cohort state.
- Tumor type and MSI badge display properly.
- Candidate cards show gene, peptide, tier, and RL score.
- KRAS G12 and shared MSI flags render when data supports them; otherwise mark logic-only verification.
- Evidence drawer includes binding, expression, clonality, LOH/resistance, confidence, and explanatory text.
- HLA-UNK warning appears only for unknown HLA cases.
- Tumor-board export button downloads usable CSV with ranked candidates/evidence columns.

### Phase 5: Expert Mode Computational Workspace

- Module stack diagram renders with state semantics (complete/running/unavailable/pending/failed).
- Module click opens detail panel with artifacts/confidence/provenance and rerun control.
- Strategy editor sliders + radar render and update.
- `rl_v1` and `rl_calibrated` are available for compare/switch.
- Comparison/consensus views populate.
- Scatter plot renders, recolors by selected dimension, shows tier lines, supports lasso/box.
- Audit trail/log viewer shows timestamped entries.
- Expert-only panels are hidden in Simple mode.

### Phase 6: Upload Flow

- Upload page accepts MAF/TSV/CSV/VCF as documented.
- Tumor type selector exists and persists to case metadata.
- Validation checklist shows format, required columns, HLA, patient ID, mutation count, tumor type.
- Alias auto-mapping is displayed (for example `Gene -> Hugo_Symbol`).
- Missing HLA warning is explicit and non-crashing.
- Module availability section shows available/unavailable with reasons.
- Run confirmation appears before execution.
- Execution feedback shows progress and clear completion/failure messaging.
- Test file A: `neoresist-md/data/tcga_sarc_case1.tsv`.
- Test file B: `tests/fixtures/kras_msi_test.tsv`.
- Test file C: `tests/fixtures/bad_input.tsv`.
- Auto-mapping stress file (alias headers) succeeds end-to-end.

### Phase 7: Case Management

- Case list shows case ID, status, and timestamps.
- Case detail shows manifest fields (including tumor_type) and module progression.
- Resume flow continues from checkpoint when applicable.
- Rerun cascades downstream modules correctly.
- Per-case audit log includes creation/start/complete/fail/rerun.
- Case manifests persist under `local_state/cases/` and survive app restart.

### Phase 8: Strategy System

- `configs/scoring_profiles/rl_v1.yaml` exists and is loadable.
- `configs/profiles/rl_calibrated.yaml` exists and is loadable.
- UI strategy switching changes visible ranking outputs.
- Edited strategies can be saved and persist after reload.
- Comparison and consensus outputs render.
- Strategy audit events are timestamped and coherent.

### Phase 9: Pipeline Status

- Pipeline Status route is accessible.
- Availability table lists registered modules.
- Each row shows module, status, and reason for unavailability.
- Runtime detection is reflected (not hardcoded static labels).

### Phase 10: Scoring Formula Integrity

- `neoresist/scoring.py` aligns with formula in `docs/architecture.md`.
- One-line formula provenance note exists near scoring implementation.
- No conflicting alternative scoring path changes rankings for same inputs.
- BLOSUM62 self-dissimilarity path exists.
- Fallback behavior does not use fake hash values as real signal.
- LOH escape penalty influences final RL priority as expected.
- Expression normalization and cap behavior are consistent with profile-config logic.
- Tier thresholds are configurable and surfaced with provenance note in UI.

### Phase 11: Documentation Check

- Core docs exist and are current: `README.md`, `docs/architecture.md`, `docs/INPUT_FORMAT.md`, `docs/FEATURE_AUDIT.md`, `docs/DEMO_VERIFICATION.md`.
- Config docs exist: `configs/canonical_schema.yaml`, `configs/module_schema.yaml`, profile files.
- Keep repo truth:
  - execution control file is `neoresist-md/docs/EXECUTION_CONTROL_PLAN.md`.
  - do not force use of `docs/EXECUTION_CONTROL_PLAN.md` if absent.

### Phase 12: Stability Stress Tests

- Rapid mode switching.
- Rapid navigation loops.
- Sequential uploads without page reload.
- Strategy switches under load.
- Hard refresh recovery in Expert mode.
- Graceful empty-state startup with no cases.

### Validation Artifact Policy (Always In Scope)

Check and record these artifact paths when present:
- `backend/validation/artifacts/training_matrix.csv`
- `backend/validation/artifacts/experiment1_auc_results.json`
- `backend/validation/artifacts/experiment2_ablation_results.json`
- `backend/validation/artifacts/experiment3_calibration_results.json`
- `backend/validation/artifacts/experiment4_haystack_results.json`
- `backend/validation/artifacts/VALIDATION_SUMMARY.md`
- `backend/validation/figures/` publication figures

If missing in standalone runs, do not skip:
- mark `PARTIAL`/`FAIL`,
- include blocker reason: `validation artifacts not yet present`.

### Reporting Requirements

After verification, update `docs/FEATURE_AUDIT.md` with:
- date, auditor, app version/hash, previous audit reference,
- total checks, PASS/PARTIAL/FAIL counts, completion %,
- phase-by-phase results,
- fixes applied (with line counts),
- known issues (with severity and recommended priority),
- comparison vs previous audit,
- top 5 remaining gaps, top 5 strengths, next actions.

Also append a short session delta in this file with:
- what was verified now,
- what failed/blocked,
- whether regressions were introduced,
- exact restart point for next session.

### Expected Outcome Bar

- Target overall completion: about 90%.
- Below 85% indicates significant regression risk and requires a focused stabilization sprint before publication-facing work.
