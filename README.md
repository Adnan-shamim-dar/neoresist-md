# neovax

Monorepo for neoantigen tooling (NeoResist-MD Dash UI, NeoVax backend, ResistanceLoop qualification).

## NeoResist-MD Dash (`app.py`)

Run from the repo root:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8050`.

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
