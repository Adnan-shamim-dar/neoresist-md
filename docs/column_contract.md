# Column contract

Canonical names are the **Parquet columns** produced by `backend.cli.qualify_cohort` and `backend.cli.enrich_cohort`. The Dash UI loads cohort tables via `neoresist.loaders.load_cohort_for_dash`, which applies minimal synonym mapping (`neoresist.schema`) then validates required fields.

## `dash_candidate_view` (required after normalization)

| Column | Role |
|--------|------|
| `patient_id` | Cohort key |
| `hla_allele` | HLA context |
| `tier` | Tier 1–3 |
| `rl_priority` | ResistanceLoop score in [0, 1] |
| `exclusion_reasons` | List-like exclusions per row |

**Synonyms → `patient_id`:** `patient`, `caseid`, `case_id`, `barcode`, `Patient`, `PATIENT_ID` (see `neoresist/schema.py`).

**Optional (UI features when present):** `gene`, `mutant_peptide`, `expression_tpm`, `ccf`, `presentation_score`, `self_dissimilarity`, `affinity_nm`, `real_expression_tpm`, `real_ccf`, Phase 5 purity / provenance columns, `scoring_profile`, `scoring_version`, `rule_profile`, `dataset_name`, `app_version`.

## `enrichment_parquet_output`

Enriched rows include TCGA join fields (`rna_tpm_dict`, `purity`, …), real-expression / clonality layers, ResistanceLoop outputs (`rl_priority`, `tier`, `expression_norm`, `evidence_*_source`), Phase 5 purity arbitration and `expression_source` / `ccf_source` / … provenance, plus:

- `scoring_profile`, `scoring_version`, `rule_profile` — from active YAML profiles  
- `dataset_name`, `app_version` — from `neoresist.metadata.stamp_enrichment_metadata` in `enrich_cohort`

## Out of scope (no automatic Dash load)

MAF preview CSVs, supported/rejected mutation tables, and other pipeline diagnostics are **not** normalized into the candidate view without an explicit adapter. Use Parquet outputs from qualify/enrich for the dashboard.
