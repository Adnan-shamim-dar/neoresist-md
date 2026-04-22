# NeoResist-MD Feature Audit
**Date:** 2026-04-17
**Auditor:** AI coding agent
**App version:** 0.3.0 (`neoresist/version.py`), git `bc39165`
**Audit scope:** Full feature check against Strategic Development Document v1.0
**Audit version:** 2

## Summary
- Total features checked: 114
- ✅ DONE: 85
- 🟡 PARTIAL: 18
- ❌ MISSING: 11
- Completion percentage: 74.6%

## Part I — UI/UX
### Typography
- ✅ Inter font loaded — live browser `font-family` resolved to `Inter, sans-serif`; source remains `neoresist/dash_app/constants.py` + `assets/custom.css`.
- ✅ JetBrains Mono loaded — live browser `font-family` resolved to `"JetBrains Mono", monospace`; source remains `neoresist/dash_app/constants.py` + `assets/custom.css`.
- 🟡 Consistent type scale — a scale exists (`.text-micro`...`.text-kpi` in `assets/custom.css`), but `assets/styles.css` still uses ad-hoc sizes.
- ✅ Uppercase labels use letter-spacing — `.kpi-label`, `.nav-section-label`, `.tier-badge`.

### Color System
- ✅ Defined color palette exists — CSS vars in `assets/custom.css` (`--bg-*`, `--text-*`, `--accent-blue`, tier colors).
- ✅ Background colors consistent — live UI stayed on the defined dark palette during browser verification.
- ✅ Tier badges use distinct colors — `.tier-badge.tier-1/2/3` in `assets/custom.css`.
- ✅ Text color hierarchy exists — `--text-primary/secondary/muted` + usage in CSS.
- 🟡 Accent color consistency — mostly `--accent-blue`, but mixed hardcoded accents remain in `assets/styles.css`.

### Navigation
- ✅ Sidebar nav rail (icon + label) — live left rail rendered correctly from `neoresist/dash_app/layout.py` (`_nav_label`, `.nav-rail`).
- ✅ Nav grouped into sections — “Analysis” and “Platform” headings rendered live.
- ✅ Active nav visual indicator — live DOM included `.nav-item.active`.
- 🟡 Nav controls rendered sections — all nav routes worked in Expert mode, but Simple mode still leaves some expert-only routes blank.

### Simple / Expert Mode Toggle
- ✅ Toggle exists and works — `ui-mode` in layout + `toggle_sections()` callback.
- ✅ Styled as pill switch — `.mode-toggle .toggle-option` in `assets/custom.css`.
- 🟡 Switching mode does not recompute pipeline, but does refresh dashboard dataframe/figures (`update_dashboard` depends on `ui-mode`).
- ✅ Simple mode hides machinery — `resolve_view_state()` disables expert/filter shells.
- ✅ Expert mode shows full pipeline/diagnostics/raw outputs — expert shells + pipeline drawer callbacks.
- ✅ Modes are visually distinct — separate simple/expert hero shells and control visibility.

### Simple Mode — Clinical Surface
- ✅ Hero KPI section — `build_kpi_hero()` (`components/simple_mode.py`); verified live on Overview.
- ✅ Ranked candidate clinical cards — `build_candidate_cards()`; 490 `.candidate-card` rows rendered live in Simple Overview.
- ✅ Evidence drawer/cards on selection — `toggle_evidence_drawer()` + `simple-evidence-cards`; verified live.
- ✅ Evidence includes binding/expression/clonality/LOH/confidence states — verified live in the drawer; source `components/candidate_card.py`.
- ✅ Plain-English evidence explanations — explanatory labels/tooltips present in drawer content.
- ✅ Export button exists — “Export for Tumor Board” in `components/candidate_card.py`; verified live after drawer open.
- ✅ Export works (CSV) — download verified live after restoring callback binding; CSV included ranked/evidence columns.

### Expert Mode — Computational Workspace
- ✅ Module pipeline visualization — `build_pipeline_diagram()` rendered as `expert-pipeline-diagram`; verified live.
- ✅ Per-module status display — pipeline + confidence/status badges rendered live.
- ✅ Per-module detail panel (artifacts/confidence/provenance/rerun) — Cases page showed module tabs, timestamps, artifacts, and rerun controls.
- ✅ Strategy editor with weight controls — sliders rendered live in Expert workspace.
- ✅ Strategy comparison capability — comparison/consensus workspace rendered; `RL calibrated` appeared in live strategy library.
- ✅ Radar chart for weights — `build_strategy_radar_figure()` remains wired into Expert workspace.
- ✅ Scatter plot present and functional — `dcc.Graph(id="scatter")` + `make_scatter_fig()` rendered live.
- ✅ Scatter color dimension selector — `scatter-color-dimension` dropdown rendered live.
- 🟡 Lasso/box selection — lasso mode is the live default; box toggle is visible but did not switch Plotly `dragmode` during verification.
- ✅ Tier boundary reference lines — live scatter `_fullLayout.shapes` count was 5 on Expert Overview.
- ✅ Audit trail/log viewer — strategy audit table + case audit tab remain present; case audit artifacts exist on disk.

### Upload Flow
- 🟡 Upload zone with guidance — exists with MAF/TSV wording, but accepted-format guidance is still minimal despite new input-format docs.
- 🟡 Validation checklist after upload — live checklist shows format, columns, HLA, patient ID, mutation count, tumor type, and alias mapping, but it is still not a richer stepwise wizard.
- ✅ Module availability shown before run — live checklist labels runtime-unavailable modules.
- ✅ Run confirmation step — case summary panel + explicit “Run enabled modules”.
- ✅ Upload triggers real pipeline execution — verified live through `run_case_modules()` -> `ModuleRunner.start_case()`.
- ✅ Explicit progress/success/failure feedback — alerts + global progress bar callbacks verified live.

### Responsive / Mobile
- 🟡 Tablet usability likely (Bootstrap grid), but not fully verified by live viewport test.
- ✅ Sidebar adapts at narrow widths — media query in `assets/custom.css`.
- 🟡 Scatter readability on small screens is only partially addressed (fixed graph heights).

## Part II — Science
### Self-Dissimilarity
- ✅ Real BLOSUM62 computation exists — `neoresist_md/backend/core/recognition/foreignness_module.py`.
- 🟡 Hash proxy still exists as fallback in `neoresist/scoring.py::_self_dissimilarity_fill`.
- ❌ Placeholder limitation is not clearly disclosed in UI when hash fallback is used.

### HLA Binding Prediction
- ✅ MHCflurry integration exists — `mhcflurry_module.py` (`_run_mhcflurry`).
- ✅ `binding_affinity`/`binding_rank`/`binding_tool` populated from real predictor when available.
- 🟡 Raw upload path relies on NeoVax CLI outputs; fallback/stub paths still present.
- ✅ Processing-related scores computed when available (`cleavage_score`, `tap_score`).

### HLA Typing
- ✅ HLA allele validation exists — regex checks in `upload_runtime.py` + generation module.
- ✅ `HLA-UNK` impact is explicitly explained in UI — live upload summary showed “No HLA typing detected — pan-allele estimates will be used.”
- ❌ Optional HLA inference module not found.

### Resistance Model
- ✅ HLA LOH modeled as resistance penalty — `_loh_penalty()` in resistance module.
- ✅ LOH affects final priority — included in resistance composite and final scoring.
- 🟡 Evolutionary stability is proxy-level (volatility/clonality), not true CNV-derived stability.
- ✅ IFN-gamma pathway disruption flags exist — `IFN_GAMMA_GENES` in `lohhla_module.py`.

### Expression Model
- ✅ TPM log normalization used — `expression_norm_from_tpm()` in `neoresist/scoring.py`.
- 🟡 TPM cap configurable by profile, but not clearly tumor-type-specific.
- ❌ Tumor-cell-specific expression flag/future note not found.

### Clonality Model
- ✅ CCF computed/ingested — clonality module + `real_ccf` blending paths.
- ✅ CLONAL/SUBCLONAL/UNKNOWN classes exist.
- ✅ CCF uncertainty communicated — low-confidence proxy messaging in UI.
- ✅ Subclonal penalty present — resistance volatility + tier logic.

### Scoring Model
- ✅ Composite weighted RL scoring exists.
- ✅ Weights configurable via YAML strategy/profile system.
- 🟡 `immunogenicity × (1−resistance)` exists in `neoresist_md` resistance module, but root app still carries a separate scoring path/history that should stay documented clearly.
- ✅ Tier boundaries are applied.
- ✅ Tier thresholds configurable (`rule_profiles` + strategy editor inputs).
- ✅ Tier-threshold provenance note shown in UI (`build_kpi_hero()`).

### Validation
- ✅ Retrospective-style validation harness present on public-trial-labeled dataset key (`ott2017`) with artifacts.
- ✅ AUC-ROC computed — `backend/validation/artifacts/metrics.json`.
- ✅ Calibrated profile file exists — `configs/profiles/rl_calibrated.yaml`; `RL calibrated` also appeared live in the strategy UI.
- ✅ “No validation yet” fallback not needed since validation artifacts exist.

### Tumor Type Specificity
- ✅ Tumor type is captured in case metadata — upload manifest persists `tumor_type`, and Cases page displays it live.
- 🟡 KRAS G12 special handling exists in code/tests (`test_tumor_features.py`, `candidate_card.py`), but full live upload verification was blocked because `tests/fixtures/kras_msi_test.tsv` is not upload-compatible.
- ✅ MSI detection/flag exists — validator populates `msi_status`; live upload/case views displayed MSS status.
- 🟡 Shared neoantigen detection/badge path exists in code, but full live upload verification was blocked by the same fixture incompatibility.

## Part III — Platform
### Case Management
- ✅ Cases created from uploads — `create_case_from_upload()`; verified live.
- ✅ Case manifests persisted to disk (`manifest.yaml`).
- ✅ Module status tracked per case (`status.json`, manifest state map).
- ✅ Cases resumed after interruption (`_reconcile_running_state`).
- ✅ Cases rerunnable with downstream cascades (rerun/reset callbacks).
- ✅ Case audit log exists (`audit/events.jsonl`).

### Strategy System
- ✅ Multiple strategies definable.
- ✅ Strategies persisted (JSON store + config profiles).
- ✅ Active strategy switch without reupload.
- ✅ Strategy comparison implemented.
- ✅ Consensus computation implemented (`build_consensus_table`).
- ✅ Strategy weights editable in UI.

### Module System
- ✅ Explicit I/O contracts — `configs/module_schema.yaml`.
- ✅ Module registry exists.
- ✅ Supported modules run locally end-to-end.
- ✅ Unsupported modules marked unavailable honestly.
- ✅ Module confidence tracked/propagated to UI.

### Pipeline Status
- ✅ Pipeline status section exists.
- ✅ System availability is centralized enough for demo use — live module availability table shows name, status, and reason/setup note for each module.
- ✅ “Requires setup” visibility is explicit in the status page for unavailable runtimes (for example NetCTLpan and LOHHLA).
- ✅ Runtime tool checks executed (`_load_tool_registry_state`, `module_installation`).

### Security / Compliance
- ❌ Encryption at rest not implemented/found.
- ❌ Authentication not implemented/found.
- ❌ Audit log includes timestamp/action/case, but no user identity field.
- ❌ Missing-security pre-clinical requirements not explicitly documented.

### Deployment
- ❌ `Dockerfile` not found.
- ❌ `docker-compose.yml` not found.
- ✅ Single-command startup works (`python app.py`; live `_dash-layout` check returned 200 after clean restart).
- 🟡 Requirements are version-bounded but not strictly pinned (mostly `>=`).

### Documentation
- ✅ README explains product and run flow.
- ✅ Architecture document exists.
- ✅ Assumptions documented (`neoresist-md/docs/EXECUTION_CONTROL_PLAN.md`).
- ✅ Execution control/checkpoint document exists and was refreshed in this session.

## Top 5 Gaps (Ranked by Impact)
1. Missing security baseline (encryption/auth/user-level audit) still blocks any clinical-grade story.
2. Simple mode can still navigate to expert-only routes that render blank content, which is risky in a live demo.
3. Upload-driven case execution showed an intermittent `prioritization_tiering` failure (`EmptyDataError`) on a later rerun of the same TCGA fixture.
4. Tumor-type specificity is only partially demonstrated end to end: KRAS G12 and shared neoantigen surfacing could not be verified live with the provided KRAS/MSI fixture.
5. Deployment packaging remains incomplete (`Dockerfile` / `docker-compose.yml` missing).

## Top 5 Strengths
1. Strong case lifecycle platform: persistent manifests, module states, reruns, resume after interruption.
2. Expert mode is now demo-stable after stale-process/cache cleanup and small callback hardening.
3. Pipeline status is honest and usable, with unavailable runtimes surfaced clearly instead of hidden.
4. Strategy system is rich and investor-legible: saved profiles, `RL calibrated`, comparison, consensus, and audit surfaces are all present.
5. Simple mode cohort review works end to end with candidate cards, evidence drawer, and CSV export.

## Recommended Next Actions
1. Fix demo-visible stability issues before any investor walkthrough: Simple-mode blank routes, box-select behavior, and the intermittent `prioritization_tiering` handoff failure.
2. Implement minimum security/compliance foundation (auth, encryption-at-rest, user-attributed audit events).
3. Add deployment packaging (`Dockerfile`, `docker-compose.yml`) and a reproducible demo environment spec.
