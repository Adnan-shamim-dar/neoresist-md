# NeoResist-MD Demo Verification

**Date:** 2026-04-17
**Scope:** Full system verification, stabilization, and cleanup sprint
**App version:** 0.3.0
**Git commit:** `bc39165`
**Start command:** `python app.py`
**Active server PID:** `24052`
**Primary URL:** `http://127.0.0.1:8050`
**Test files:**
- `neoresist-md/data/tcga_sarc_case1.tsv`
- `tests/fixtures/kras_msi_test.tsv`
- `tests/fixtures/bad_input.tsv`

## Phase 0 — Clean Environment

**Result:** PASS

- Cleared stale listeners on port `8050`.
- Rebuilt a clean single-process app run.
- Verified `http://127.0.0.1:8050/_dash-layout` returned HTTP 200.
- Verified only one listener remained on `8050`.
- Added cache-busting versioning for stylesheet/font URLs:
  - Google font URLs now include `?v=2026041701`
  - Dash assets are served from `assets-2026041701`
- Cleared repo-local `__pycache__` / `.pyc` artifacts during stabilization.

**Verification commands used**
- `Get-NetTCPConnection -LocalPort 8050 | Where-Object { $_.State -eq 'Listen' }`
- `Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8050/_dash-layout`

## Phase 1 — Full Test Suite and Compile Check

**Result:** PASS

### Backend tests

- Command: `python -m unittest discover backend/tests -v`
- Result: **35 tests passed**

### Compile check

- Command: `python -m compileall neoresist backend -q`
- Result: PASS

### Import gate

- `from neoresist.dash_app.layout import build_layout` — PASS
- `from neoresist.scoring import compute_rl_priority` — PASS
- `from neoresist.upload_runtime import validate_uploaded_table` — PASS
- `from neoresist.tumor_features import detect_kras_g12, estimate_msi_status` — PASS
- `from neoresist.case_store import CaseStore` — PASS
- `from neoresist.module_runner import ModuleRunner` — PASS
- `from neoresist.strategy_registry import StrategyRegistry` — PASS
- `from neoresist.canonical_schema import validate_canonical_table` — PASS

## Phase 2 — Feature-by-Feature Live Verification

### 2.1 Navigation and Mode Toggle

- App loads without blank screen — PASS
- Navigation rail visible with icon + label pairs — PASS
- Nav grouped into Analysis / Platform sections — PASS
- Clicking nav items switches visible content area in Expert mode — PASS
- Clicking nav items in Simple mode can leave expert-only routes blank — FAIL
- Simple/Expert toggle visible and styled as pill switch — PASS
- Clicking Simple shows the clinical surface — PASS
- Clicking Expert shows the computational workspace — PASS
- Switching modes did not leave a stuck spinner — PASS
- Rapid 5-switch smoke check remained stable — PASS

### 2.2 Overview / Cohort View

- Cohort KPI cards load — PASS
- KPI numbers render prominently — PASS
- HLA coverage indicator visible — PASS
- Tumor type shown (`SARC` in cohort view) — PASS
- Scatter plot renders with data — PASS
- Scatter color selector visible and working — PASS
- Tier boundary reference lines present — PASS
- Lasso mode present as default drag mode — PASS
- Box mode button visible — PASS
- Box mode changed drag mode successfully — FAIL
- Lasso/box toggle visible near scatter — PASS

### 2.3 Simple Mode Clinical Surface

- Hero KPI section visible — PASS
- Ranked candidate cards visible — PASS
- Candidate cards show gene / peptide / tier / score structure — PASS
- Clicking a candidate opens evidence drawer — PASS
- Evidence includes binding block — PASS
- Evidence includes expression block — PASS
- Evidence includes clonality block — PASS
- Evidence includes escape / LOH / resistance block — PASS
- Plain-English explanatory labels present — PASS
- HLA-UNK warning shown in live upload summary — PASS
- Known-HLA no-warning path — NOT VERIFIED
- KRAS G12 badge visible live — NOT VERIFIED
- Shared MSI neoantigen badge visible live — NOT VERIFIED
- MSI status badge visible near patient summary surfaces — PASS
- Tumor type visible near patient summary surfaces — PASS
- “Export for Tumor Board” button visible — PASS
- Export click produced CSV download — PASS
- Downloaded CSV contained ranked/evidence columns — PASS

### 2.4 Expert Mode Computational Workspace

- Pipeline visualization visible — PASS
- Per-module status badges visible — PASS
- Case detail panel exposes artifacts / confidence / provenance — PASS
- Rerun controls visible — PASS
- Strategy editor visible with sliders — PASS
- Weight slider interaction updates live values — NOT VERIFIED
- Radar chart present — PASS
- Strategy comparison workspace visible — PASS
- Comparison table populates — NOT VERIFIED
- Consensus computation surface visible — PASS
- Audit / log viewer present — PASS
- Expert-only panels hidden when returning to Simple — PASS

### 2.5 Upload Flow

#### Test File A — `neoresist-md/data/tcga_sarc_case1.tsv`

- File accepted by upload zone — PASS
- Validation checklist appears — PASS
- Auto-mapped aliases shown — PASS
- Module availability shown — PASS
- Run confirmation step appears — PASS
- Clicking Run starts case execution — PASS
- Progress feedback visible — PASS
- Pipeline completed without errors — FAIL
- Results persisted and visible in Cases view — PASS
- RL scoring/pipeline artifacts generated — PASS
- Switching between Simple and Expert after upload remains stable — PASS

**Observed detail**
- Two TCGA runs completed with terminal states:
  - `complete=5`, `unavailable=2`, `disabled=2`
- A later repeat run of the same fixture failed in `prioritization_tiering` with:
  - `pandas.errors.EmptyDataError: No columns to parse from file`
- This makes the end-to-end upload path **functionally real but not yet fully deterministic**.

#### Test File B — `tests/fixtures/kras_msi_test.tsv`

- File uploads successfully — FAIL
- Clear error shown instead of crash — PASS
- Error text listed missing canonical genomic columns — PASS
- KRAS/MSI live badge verification from this fixture — BLOCKED

**Observed error**
- `TSV/CSV upload is missing critical columns after auto-mapping: alt, chrom, pos, ref. Expected format: TSV/CSV mutation table with genomic columns. Required columns: chrom, pos, ref, alt.`

#### Test File C — `tests/fixtures/bad_input.tsv`

- Upload does not crash app — PASS
- Clear error lists missing required columns — PASS
- Error states expected format — PASS
- App remains functional after failed upload — PASS
- No user-visible Python stack trace — PASS

### 2.6 Strategy System

- `rl_v1` selectable — PASS
- `rl_calibrated` selectable — PASS
- Switching strategies changes displayed RL values live — NOT VERIFIED
- Strategy weights editable via sliders — PASS
- Edited strategy can be saved — NOT VERIFIED
- Saved strategy persists after reload — NOT VERIFIED
- Comparison between `rl_v1` and `rl_calibrated` workspace exists — PASS

### 2.7 Case Management

- Cases page lists existing cases — PASS
- Case list shows status/progress/timestamps context — PASS
- Uploaded case appears in list — PASS
- Case detail shows manifest info including tumor type — PASS
- Case rerun controls visible — PASS
- Rerun cascade buttons visible — PASS
- Case audit log exists on disk and in UI tabs — PASS

### 2.8 Pipeline Status

- Pipeline Status page accessible — PASS
- Module availability table displayed — PASS
- Each module shows name/status/reason — PASS
- Available modules show positive availability state — PASS
- Unavailable modules show explicit reason — PASS

### 2.9 CSS and Visual Consistency

- Inter font visibly loaded — PASS
- JetBrains Mono used for mono surfaces — PASS
- Active nav item has visible indicator — PASS
- Tier badge colors are consistent — PASS
- Background palette stayed consistent during checks — PASS
- No major overlap/clipping at `1440px` width — PASS
- No exhaustive viewport/mobile visual sweep completed — PARTIAL

## Phase 3 — Stability Stress Tests

### 3.1 Rapid Mode Switching

**Result:** PASS

- 20 rapid Simple/Expert toggles
- No stuck `Updating...`
- No blank page
- No browser callback errors captured

### 3.2 Rapid Navigation

**Result:** PASS

- 3 rapid full cycles through:
  - Overview → Patients → Upload → Cases → Strategies → Pipeline Status → About
- Verified in Expert mode
- No blank page or browser error

### 3.3 Upload After Upload

**Result:** PASS (app stability) / FAIL (fixture B compatibility)

- Uploaded Test File A
- Uploaded Test File B without reload
- App remained usable and navigation continued to work
- Second file was rejected cleanly by validation because it is not upload-compatible

### 3.4 Strategy Switch Under Load

**Result:** NOT VERIFIED

- Strategy selector presence verified
- Repeated score-update verification under active patient load was not completed in this session

### 3.5 Browser Refresh Recovery

**Result:** PASS

- Hard reload equivalent performed in browser automation
- App reloaded to a usable state
- No stale CSS observed
- Versioned stylesheet URLs were present after refresh

## Phase 4 — Fixes Applied

### Safe fixes applied in this session

1. Added stylesheet/font cache busting
   - `neoresist/dash_app/constants.py`
   - `neoresist/dash_app/__init__.py`
2. Added compatibility shims required by the import gate
   - `neoresist/scoring.py`
   - `neoresist/tumor_features.py`
   - `neoresist/case_store.py`
   - `neoresist/strategy_registry.py`
   - `neoresist/canonical_schema.py`
3. Seeded hidden initial callback inputs so drawer callbacks do not bind to missing IDs
   - `neoresist/dash_app/layout.py`
4. Guarded zero-click phantom callback fires that were opening the expert strategy modal
   - `neoresist/dash_app/callbacks.py`
5. Hardened evidence-drawer formatting against array-like values
   - `neoresist/dash_app/components/candidate_card.py`
6. Restored Simple-mode export callback binding by seeding hidden `btn-download-patient`
   - `neoresist/dash_app/layout.py`
7. Added broken-input verification fixture
   - `tests/fixtures/bad_input.tsv`

### Full regression gate after code fixes

- `python -m unittest discover backend/tests -v` — PASS
- `python -m compileall neoresist backend -q` — PASS

## Known Issues (Not Safely Fixed in This Sprint)

1. Simple mode can navigate to expert-only pages and render no visible main section.
2. Scatter box-select button is visible but did not switch Plotly `dragmode` from lasso during live testing.
3. `tests/fixtures/kras_msi_test.tsv` is not upload-compatible with the documented canonical upload format, so the planned live KRAS/MSI upload verification is blocked.
4. Repeated upload/rerun of the TCGA fixture produced one intermittent `prioritization_tiering` failure:
   - `pandas.errors.EmptyDataError: No columns to parse from file`
   - prior successful runs of the same file suggest a race/state issue rather than invalid input.

## Session Outcome

- The app now boots cleanly from a single `python app.py`.
- Stale CSS/process issues were stabilized.
- The browser-level modal/callback/export regressions found during this sprint were fixed safely.
- The main upload path is real and persistent, but one downstream module handoff remains intermittently unstable.
- Documentation was refreshed to reflect the verified state rather than the earlier optimistic snapshot.
