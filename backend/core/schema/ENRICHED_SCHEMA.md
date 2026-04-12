# Enriched candidates schema (Phase 4 + Phase 5)

`enriched_candidates.parquet` is produced by `python -m backend.cli.enrich_cohort`. Phase 5 adds **optional purity arbitration** and **evidence provenance** columns without removing Phase 4 fields.

## Phase 4 (unchanged semantics)

- TCGA join columns: `rna_tpm_dict`, `purity` (from metadata merge before arbitration), `cnv_segments`, `normal_bam_path`, `tcga_join_status`, `source` (or project-specific source labels).
- Real layers: `real_expression_tpm`, `expression_bin`, `RNA_data_missing`, `real_ccf`, `ccf_naive`, `clonality_data_missing`, etc.
- ResistanceLoop: `rl_priority`, `tier`, `expression_norm`, `evidence_expression_source`, `evidence_ccf_source`, …

## Phase 5 — purity arbitration

Written by `backend.core.qualification.purity_resolution.apply_purity_resolution`.

| Column | Description |
|--------|-------------|
| `purity_tcga_merged` | Copy of merged TCGA `purity` before arbitration (if `purity` existed). |
| `purity` | Canonical tumor purity used downstream; equals `purity_value_used` after arbitration. |
| `purity_value_used` | Chosen purity in `[0, 1]` or `NaN` if unresolved (when stub fallback is disabled). |
| `purity_source_used` | `user_file`, `tcga_metadata`, `computed_estimate`, `stub_fallback`, or `unresolved`. |
| `purity_method_used` | Method label from metadata, user file, or resolver. |
| `purity_confidence_used` | Confidence in `[0, 1]`. |
| `purity_candidates_json` | JSON list of ranked candidates considered. |
| `purity_resolution_reason` | Human-readable deterministic resolution trace. |

**CLI:** `--purity-file`, `--purity-key`, `--purity-source-priority`, `--purity-stub-fallback`.

- Default **`--purity-stub-fallback` is off**: if TCGA purity is absent and no user file applies, purity stays **unresolved** (`NaN`) to match legacy Phase 4 behavior when metadata lacks purity.
- Enable **`--purity-stub-fallback`** to allow the internal stub value (`0.82`) as a last resort.

## Phase 5 — clonality provenance

From `real_clonality.apply_layer`:

| Column | Description |
|--------|-------------|
| `clonality_source_used` | e.g. `vaf_over_purity_naive`, `unresolved`. |
| `clonality_confidence` | Scalar in `[0, 1]`. |
| `clonality_resolution_reason` | Notes on VAF/purity/CNV availability. |
| `cnv_presence_note` | `present` / `absent` placeholder note. |

## Phase 5 — unified evidence provenance

From `apply_evidence_provenance_columns` (after ResistanceLoop):

| Column | Description |
|--------|-------------|
| `expression_source` | Mirrors `evidence_expression_source` when present, else inferred from RNA fields. |
| `ccf_source` | Mirrors `evidence_ccf_source` when present, else inferred from `real_ccf` / clonality flags. |
| `purity_source` | Summary label (typically aligns with `purity_source_used`). |
| `metadata_source` | TCGA `source` / join status. |
| `resolution_status` | `complete`, `partial`, or `minimal_stub`. |
| `evidence_notes` | Pipe-separated summary for auditing. |

## TCGA metadata extensions

Stub and R stub rows may include:

- `purity_method` — e.g. `stub_python_hash_tpm`, `stub_r_deterministic`, or future GDC-derived labels.
- `purity_confidence` — scalar hint for arbitration (default ~0.55 in stubs).

Real R pulls may leave `purity` / `purity_method` / `purity_confidence` as `NA` when not available.
