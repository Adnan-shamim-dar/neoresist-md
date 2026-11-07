# NeoResist-MD

**A systematic framework for cancer-type-specific neoantigen ranking strategy discovery.**

NeoResist-MD searches for interpretable, cancer-type-specific neoantigen ranking strategies in immunogenicity data using an automated loop, scores them against partner cohorts with full provenance logging, and exposes them through a prioritisation interface.

> **Status note (November 2026).** A later internal audit of this codebase found that the cohort described below as "held-out" (`sahin_2017`) was in fact present in the search loop's active dataset list and was read many times during the search, not once at final evaluation. **No cleanly sealed held-out cohort exists in this project at present**, so none of the results here should be read as held-out validation. The melanoma figure is reported below as an exploratory result only. A separate, larger run of the same engine (27,118 evaluations across nine cancer contexts) promoted **nothing**, and that negative result — together with a defect found in the promotion gate's confidence-interval statistic — is the subject of work in preparation. Figures and numbers in this April 2026 snapshot have not been recomputed.

**Methodology reference:** [docs/paper_draft_v3.md](docs/paper_draft_v3.md)

---

## Quick Start

```bash
pip install -r requirements.txt
python scripts/generate_figures.py         # reproduce all 4 paper figures
python scripts/borch_significance_test.py  # reproduce Borch p-value
```

---

## Repository Structure

```
registry/                    # Strategy specifications (JSON)
  melanoma_all_signals.json  # Melanoma candidate strategy (exploratory; not held-out validated)
docs/
  paper_draft_v3.md          # Current manuscript
  paper_journey.md           # Decision log for collaborators / AI agents
  cover_letter_draft.md      # Submission cover letter
data/
  README_data.txt            # Dataset attribution and sources
backend/strategy_engine/
  autoresearch_loop.py       # Main search engine
  paper_evidence.py          # Evaluation pipeline
scripts/
  generate_figures.py        # Reproduce paper figures 1–4
  borch_significance_test.py # Reproduce Borch significance test
study_audit.txt              # Evidence provenance and pre-specification log
```

---

## Key Results

| Result | Value | Status |
|---|---|---|
| Melanoma AUC, `sahin_2017` | 0.575 (delta +0.105 vs binding-only, p=0.011) | **Exploratory** — cohort was read inside the search loop, so this is not held-out validation |
| TCR contact volume ablation | −0.086 AUC on removal | **Exploratory** — same cohort, n = 13 |
| Borch melanoma vs PRIME 2.0 | 0.660 vs 0.609 (p=0.067) | Suggestive, not significant; CI of difference spans zero |
| Bidirectional transfer failure | Melanoma → GBM: 0.411 vs binding ~0.628 | Provisional |

The two sample sizes behind the top rows are 13 patients each. Treat every figure in this table as a direction to test, not a result to rely on.

---

## Strategy Specification

See [`registry/melanoma_all_signals.json`](registry/melanoma_all_signals.json) for the full specification, weights, and provenance. Note that the four weights (0.40 / 0.20 / 0.20 / 0.20) were set from a fixed template and were never fitted.

---

## Evidence Audit

See [`study_audit.txt`](study_audit.txt) for the April 2026 commit timeline. **That file's pre-specification claim does not hold** — see the status note at the top of this README. The timeline shows when the strategy definition was committed; it does not establish that the evaluation cohort was sealed from the search, and the later audit found it was not.

---

## Data Attribution

See [`data/README_data.txt`](data/README_data.txt). All cohorts are from published supplementary materials; no new patient data were collected.

---

*Previously: monorepo for neoantigen tooling (NeoResist-MD Dash UI, NeoVax backend, ResistanceLoop qualification).*

## NeoResist-MD Dash (`app.py`)

Run from the repo root:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8050` (or set `PORT` / see `.env.example`).

For local hot reload while editing the Dash UI:

```bash
set DASH_DEBUG=1
python app.py
```

### Modular layout

- **`neoresist/`** — config loading, cohort loaders, schema synonyms, YAML-driven **scoring** (`configs/scoring_profiles/`) and **rules** (`configs/rule_profiles/`), enrichment metadata stamps.
- **`neoresist/dash_app/`** — Dash layout and callbacks (imported by thin root `app.py`).
- **`configs/app_config.yaml`** — branding subtitle, default dataset/scoring/rule profile ids, **ordered cohort search paths** (Parquet, CSV, XLSX).
- **`docs/architecture.md`** — data flow, how to add profiles. **`docs/column_contract.md`** — required Dash columns.

### Scoring CLI overrides

```bash
python -m backend.cli.enrich_cohort --stub --scoring-profile rl_v1 --rule-profile default_rules --dataset-name tcga_sarc
```

Enriched Parquet includes `scoring_profile`, `scoring_version`, `rule_profile`, `dataset_name`, `app_version`.

### Upgraded for ResistanceLoop v1

The NeoResist-MD console now reads **`qualified_candidates.parquet`** (PyArrow-backed Parquet) from `data/final/` or `neoresist-md/data/final/`, with in-memory caching keyed by file modification time.

- **KPIs**: Tier 1/2 candidate counts, mean RL priority, patients with Tier 1 neoantigens, filtered cohort size, and HLA coverage badges.
- **Filters**: Tier and HLA multi-select, top exclusion-reason tags, RL / TPM / CCF ranges, legacy mutations / fanout / candidate ranges (patient×HLA aggregates), search, and sort.
- **Scatter**: Mutations vs mean RL priority per patient×HLA; marker size = candidates; color = best tier; hypermutator callout.
- **Evidence panel**: Representative row, RL component breakdown, exclusion list, top peptides.
- **Patient detail**: Tier pie, RL histogram, exclusion bar chart, CSV download.
- **Upload (stub)**: MAF upload arms a 3-second timer, then selects the first cohort row as a demo patient×HLA.

Generate the Parquet with:

```bash
python -m backend.cli.qualify_cohort
```

Enrich with TCGA TPM / purity / naive CCF (stub mode needs no GDC access):

```bash
python -m backend.cli.enrich_cohort --stub
```

**Phase 5 (optional purity table)** — CSV/TSV/Parquet/JSON with columns such as `patient_id` or `barcode`, plus `purity`. Matching uses patient id, case barcode (`TCGA-XX-YYYY`), or a 16-character barcode prefix.

```bash
python -m backend.cli.enrich_cohort --stub --purity-file path/to/purity.csv
```

- `--purity-key` — `auto` (default), `patient_id`, or `barcode` (key order for lookups).
- `--purity-source-priority` — comma list: `user_file,tcga_metadata,computed_estimate,stub_fallback`.
- `--purity-stub-fallback` — if no purity can be resolved from file/TCGA/computed, use the built-in stub value. **Omit** this flag to keep legacy behavior when TCGA purity is missing (purity stays unresolved / `NaN`).

See `backend/core/schema/ENRICHED_SCHEMA.md` for enriched output columns.

Full GDC pull (requires R, Bioconductor **TCGAbiolinks**, network, and disk):

```bash
python -m backend.cli.enrich_cohort --pull
```

R script: `backend/scripts/pull_tcga_sarc.R` (use `--stub` for CI). Set `TCGA_STUB=1` to force stub from R.
