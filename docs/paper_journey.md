# NeoResist-MD Paper Journey Log

> For AI agents continuing this work: read this file first. It records decisions made, what was tried, what failed, and what the current state is. Update it when you change something that affects the paper's claims or structure.

---

## Current State (2026-04-26)

**Active draft:** `docs/paper_draft_v3.md`  
**Previous draft:** `docs/paper_draft_v2.md` (kept for reference)  
**Journal target:** Genome Medicine / NPJ Precision Oncology  
**Autoresearch loop:** iteration 222, phase 3  
**Branch:** `dash-ui-migration`

---

## Validated Claims (safe to include in main text)

| Claim | Numbers | Source |
|---|---|---|
| Melanoma held-out validation | AUC 0.575, delta +0.105, CI [0.511–0.630], p=0.011 | paper_evidence_summary.json → sahin_2017 |
| TCR contact volume dominant feature | −0.086 AUC on removal | ablation on Sahin |
| Borch melanoma: bind+HydroCore beats PRIME | 0.660 vs 0.609 | borch2024_external_validation.json → melanoma cohort |
| Borch bladder: melanoma strategy fails | 0.570 vs PRIME 0.616 | borch2024_external_validation.json → bladder cohort |

## Provisional Claims (supplement / exploratory only)

| Claim | Status | Blocker |
|---|---|---|
| GBM gbm_bind_seq_60_40 transfers to Keskin | provisional | no held-out validation cohort |
| Pancreatic neoguider_v2_melanoma | provisional | no held-out validation cohort |
| TESLA rl_expression_v1 | provisional | no held-out validation cohort |
| Context-aware selector 0.694 vs 0.545 | exploratory | driven by provisional contexts |
| Bladder autoresearch best blend 0.605 | supplement only | same data as Borch bladder; below PRIME |

---

## Key Decisions Made

### Bladder autoresearch (2026-04-26)
- harbst2022_bladder IS the same 6,237 rows as Borch/IMPROVE bladder cohort — same data, different label
- Full bidirectional cross-test attempted: bladder-optimised blend (Foreigness+RankEL) scores 0.547 on melanoma vs PRIME 0.609 — confirms bladder features harm melanoma ✓
- Reverse direction NOT achievable: exhaustive search finds best 0.584 on bladder vs PRIME 0.616 — feature ceiling with available data
- Root cause: Borch bladder has no TCR contact volume, stability, or self-dissimilarity features
- Decision: disclose the gap explicitly as a data limitation, not a biological null result
- harbst2022_bladder removed from main autoresearch loop (was burning compute with no possible gain); standalone harbst2022_bladder_autoresearch.py is the right tool

### Autoresearch loop fix (2026-04-26)
- Bug fixed: `_feature_space()` was including 100%-null columns (binding_stability, self_dissimilarity, calis_simplified in bladder) because it checked `feature in df.columns` but not coverage
- Fix: added `min_coverage=0.5` filter — features where <50% of rows are non-null are excluded from the search space
- File: `backend/strategy_engine/autoresearch_loop.py:386`

### Submission package complete (2026-04-27)
Files created:
- study_audit.txt — provenance log with git-verified dates; Ott discovery before Sahin access confirmed
- registry/melanoma_all_signals.json — pre-specification document with full strategy spec + held-out results
- scripts/generate_figures.py — generates all 4 paper figures (tested, runs clean)
- scripts/borch_significance_test.py — reproduces paired permutation p=0.067
- figures/fig1_framework.png, fig2_melanoma.png, fig3_borch.png, fig4_transfer.png — generated
- data/README_data.txt — dataset attribution with DOIs
- requirements.txt — updated with scipy, matplotlib, seaborn
- README.md — updated with NeoResist-MD description, structure, key results
- docs/cover_letter_draft.md — submission cover letter draft

Reference audit: all 21 refs cited, none dangling. Ref [10] DOI corrected to 10.3389/fimmu.2024.1265584.

IMPORTANT FLAG: The Codex checklist linked to fimmu.2020.576603 (Wells/TESLA 2020) as the Borch reference — this is WRONG. Our ref [10] is Borch 2024 (fimmu.2024.1265584). Do not use the Codex URL.

### Thesis-strengthening additions (2026-04-26)
- NEW Section 2: Feature discriminability varies by cancer context — single-feature AUC across 5 cohorts shows no universal best feature (range 0.14–0.24 AUC units). Binding leads on TESLA, TCR contact volume leads on melanoma, calis leads on GBM. This is Table 2.
- NEW Section 6 (bidirectional cross-cancer transfer): melanoma model on GBM (Keskin) = 0.411, below binding ~0.628. GBM model on Sahin = 0.466, below binding 0.470. Each specialist fails at the other's home context — this is the bidirectional story. Source: cross_cancer_transfer.json. This is Table 6.
- Sections renumbered: old 2→3, old 3→4, old 4→5, old 5 split into 6 (bidirectional) + kept in 6, old 6→7
- Abstract and Conclusions updated to reflect both new findings
- These additions come entirely from existing artifacts — no new data needed

### Draft v3 reviewer feedback fixes (2026-04-26)
- CRITICAL: Borch bind+HydroCore vs PRIME is NOT significant. Computed paired permutation p=0.067, bootstrap CI [-0.028, +0.123] at n=23 patients. All "exceeds PRIME" language changed to "numerically exceeds" with p-value stated throughout.
- Table 3: removed non-standard "14.9% contribution" column — raw delta (−0.086) only
- Methods: added GBM Keskin pre-specification paragraph — Keskin was pre-specified before Hilf run, accessed once
- Supplement S1: removed "harbst2022_bladder" dataset name — renamed to "Borch/IMPROVE bladder cohort with additional derived features"
- Methods Borch section: added explicit statement that HydroCore↔tcr_volume_mean correlation requires ICERFIRE on Borch peptides (not done); Borch result treated as corroborating, not replicating

### Draft v3 final fixes (2026-04-26)
- Title rephrased: "discovery and validation of a melanoma-specific..." (cleaner scope)
- Section 3: added paragraph explaining why 4-feature strategy is reported despite ablation showing expression/presentation redundant — fixed based on Ott, redundancy is a known small-cohort instability
- Permutation test: increased to 10,000 shuffles (resolution ±0.001 at n=13) — previously 1,000 was too coarse
- PRIME version clarified in Methods: Borch used PRIME 2.0 [8] as pre-computed by Borch et al.
- Power statement added to Limitations: 13-patient cohorts, ablation estimates have substantial uncertainty
- LogReg caveat elevated to Results text: trained in-domain, not comparable to fixed external PRIME
- Refs [20]/[21] renumbered: DeepNeo=[20] (Methods, gated_ranking), Müller=[21] (Background)
- Ref [22] (Luksza 2022) removed — was uncited in text
- Grammar: "corroborates on" → "is corroborated on"
- Final word count: ~4,300 words, 21 references

### Draft v2 → v3 (2026-04-26)
- Removed "Temporary caveats for this draft" block — draft artifact, not for submission
- Moved Section 6 (context-aware selector) to supplementary — aggregate result driven by provisional contexts
- Added pre-specification paragraph to Methods — addresses multiple comparison concern
- Tightened Section 4: harbst autoresearch + bidirectional attempt moved to Supplement S4; main text now focused on Borch melanoma validation + bladder specificity only
- Softened pairwise comparison language in Results S2 — overlapping CIs, no pairwise tests done
- Tightened HydroCore proxy language — "corroborating" not "replication"
- Removed TTIF from abbreviations (unused in text)
- Clarified PRIME reference: version 2.0 (Gfeller 2023, ref [8]) used for Borch comparison; ref [24] (Rivière 2022) is the earlier version, removed from results to avoid confusion
- Abstract scope tightened to match evidence

---

## What Still Needs Doing Before Submission

- [ ] Actual figures (Fig 1, 2, 3) don't exist yet — need to be generated
- [ ] Pairwise permutation tests between strategies in Table 2 (or weaken language)
- [ ] HydroCore ↔ tcr_volume_mean correlation on a shared dataset (Ott or Sahin if ICERFIRE re-run)
- [ ] Add second validated context (GBM or pancreatic) — requires new cohort data
- [ ] Author list, affiliations, funding TBD
- [ ] Repository URL TBD for Data Availability
- [ ] Verify ref [22] (Balachandran 2022) is correctly cited in text

---

## Numbers to Never Change Without Re-running the Pipeline

These are crystallised from `paper_evidence_summary.json` and `borch2024_external_validation.json`. Do not update them from memory or estimates:

- Melanoma held-out AUC: **0.575** (not 0.576, not 0.58)
- Binding-only baseline: **0.470**
- CI: **[0.511, 0.630]**
- Borch melanoma bind+HydroCore: **0.660**, CI **[0.609, 0.709]**
- PRIME on Borch melanoma: **0.609**, CI **[0.556, 0.660]**
- PRIME on Borch bladder: **0.616**
- Harbst bladder best blend: **0.605**
