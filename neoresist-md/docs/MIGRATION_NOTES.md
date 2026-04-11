# MIGRATION_NOTES

## Scope

This note captures the NeoVax extraction from `neovax-builder` into `neoresist-md` Module 1 (`backend/core/neoantigen/`) with parity-first constraints.

## Old -> New file mapping (NeoVax only)

| neovax-builder | neoresist-md | Notes |
| --- | --- | --- |
| `core/predictor.py` | `backend/core/neoantigen/rank.py` | Core candidate generation, ranking, and validation logic preserved; `predict_neoantigens` alias retained for compatibility. |
| `core/run_metadata.py` | `backend/core/neoantigen/serialize.py` | Run metadata and hashing utilities moved into module-local serializer. |
| `input/maf_ingest.py` | `backend/core/neoantigen/parse.py` | MAF reading/parsing and input mode detection consolidated into parser layer. |
| `input/sequence_fetch.py` | `backend/core/neoantigen/filter.py` | Sequence fetch + cache + sequence match validation now in filter layer. |
| `input/maf_pipeline.py` | `backend/core/neoantigen/service.py` (private helper) + `filter.py` | Pipeline orchestration absorbed into `run_neovax()` and `_prepare_maf_predictor_rows()`. |
| (no direct equivalent) | `backend/core/neoantigen/service.py` | New orchestration entrypoint for CSV/MAF/DataFrame inputs and artifact emission. |

## Frozen parity assumptions

- **Deterministic ranking behavior is frozen** for Module 1:
  - peptide lengths `(8, 9, 10, 11)`
  - `priority_score = 0.6 * presentation_component + 0.4 * affinity_component`
  - same triage thresholds and ranking sort order.
- **Validation semantics are frozen** (row-indexed error shape and mutation/reference AA checks).
- **Peptide fan-out semantics are frozen** for candidate generation:
  - one supported mutation row is expanded into mutation-spanning windows in `rank._spanning_mutant_peptides()`
  - for non-edge mutation positions, fan-out is `8 + 9 + 10 + 11 = 38` candidate peptides per supported row
  - edge positions may generate fewer than 38 due to bounded window starts.
- **MAF ingest semantics are frozen** for current scope:
  - only `Missense_Mutation`
  - simple `HGVSp_Short` form (`p.X123Y`)
  - same rejection categories and counts.
- **Metadata contract is frozen** for parity keys (run identity, input hash, formula/predictor fields, candidate counts).
- **Parity guardrail exists in tests** via `backend/tests/test_neoantigen_parity.py` against `../neovax-builder`.

## What remains outside `neoresist-md`

- Legacy NeoVax app/CLI surfaces in `neovax-builder` (for example Streamlit/UI-oriented entrypoints and demo scripts) are **not** migrated as authoritative core logic.
- Legacy project-level wrappers and demos in `neovax-builder` (for example `app.py`, `run_example.py`, `make_demo_report.py`) remain legacy reference surfaces, not core module ownership.
- `neovax-builder` remains the **reference baseline** for parity comparison only during migration hardening.
- Non-NeoVax modules (M0 QC, M2 CloneEscape, M3 ResistanceLoop, M4 TrialMatch, M5 Report/API) are still active architecture targets in `neoresist-md` and are outside this specific NeoVax extraction note.

## Engineering stance

- Keep CLI/API layers thin; orchestration may call `run_neovax()` but should not duplicate scientific logic.
- Keep `backend/core/neoantigen/` deterministic and schema-stable unless explicitly version-bumped with migration docs.
