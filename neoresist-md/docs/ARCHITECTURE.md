# ARCHITECTURE

## Mission

NeoResist-MD is an end-to-end oncology decision-support platform that converts tumor data into clinically useful outputs: clonal architecture, neoantigen candidates, resistance risks, and trial opportunities.

## Repository Architecture

```text
neoresist-md/
  docs/
    NEORESIST-MD_MASTER_PLAN.md
    ARCHITECTURE.md
    ONCOLOGIST_MANUAL.md
    PHARMA_API_SPEC.md
  data/
    sample_inputs/
      patient_001/
        maf_supported_preview.csv
        maf_rejected_preview.csv
        README.md
    reference/
      depmap_schema.md
      clinicaltrials_schema.md
  backend/
    core/
      qc/
      neoantigen/
      clonality/
      resistance/
      trialmatch/
      report/
    api/
      rest/
      cli/
    tests/
  agents/
    autoresearch/
    evaluation/
  frontend/
    web_app/
    report_templates/
  infra/
    docker/
    configs/
    deployment/
```

## Key Ideas

- **docs/** holds the brain. This is the durable project memory that survives chat expiration.
- **backend/core/** is pure logic: no UI/web assumptions, structured inputs/outputs only.
- **agents/** isolates AutoResearch orchestration from scientific core code.
- **infra/** keeps deployment readiness in place from day one.
- The architecture is mono-repo now, service-ready later.
- **AutoResearch is evidence synthesis only**, never primary scoring logic.

## Stable Module I/O Contracts

To enable large expansion without chaos, each module keeps a stable contract:

- input format is explicit and versioned
- output format is explicit and versioned
- internals can change as long as the contract does not break

### Module 1 - NeoVax

**Input**

- `variants.tsv` (or `maf_supported_preview.csv` style), with canonical columns such as:
  - `sample_id`, `chromosome`, `position`, `ref`, `alt`
  - `gene`, `consequence`, `VAF`
  - optional: transcript/protein consequence fields and HLA context

**Output**

- `neoantigens.json`

```json
{
  "sample_id": "patient_001",
  "neoantigens": [
    {
      "id": "neo_001",
      "gene": "TP53",
      "peptide": "SLMEQSDHF",
      "hla": "HLA-A*02:01",
      "binding_affinity": 50.3,
      "rank": 1
    }
  ]
}
```

### Module 2 - CloneEscape

**Input**

- variants table with VAF
- copy-number context when available
- optional NeoVax candidate linkage

**Output**

- `clones.json`

```json
{
  "sample_id": "patient_001",
  "clones": [
    {
      "clone_id": "clone_A",
      "ccf": 0.72,
      "mutations": [],
      "neoantigens": []
    }
  ]
}
```

### Module 3 - ResistanceLoop

**Input**

- clone-aware mutation context (`clones.json`) and/or variant-level features
- gene-level dependency sources (DepMap/CRISPR-derived features)

**Output**

- `resistance.json`

```json
{
  "sample_id": "patient_001",
  "resistance_drivers": [
    {
      "gene": "PIK3CA",
      "evidence_source": "DepMap",
      "resistance_score": 0.81,
      "rationale": "High dependency and pathway reactivation risk."
    }
  ]
}
```

### Module 4 - TrialMatch

**Input**

- prioritized targets from NeoVax/CloneEscape/ResistanceLoop outputs
- indication and optional patient-level eligibility constraints

**Output**

- `trial_matches.json`

```json
{
  "sample_id": "patient_001",
  "trials": [
    {
      "nct_id": "NCT00000000",
      "title": "Example Precision Trial",
      "eligibility_score": 0.76,
      "matched_targets": ["PIK3CA"],
      "location_count": 12
    }
  ]
}
```

### Module 5 - Report & API

**Input**

- `neoantigens.json`
- `clones.json`
- `resistance.json`
- `trial_matches.json`

**Output**

- human-readable report (`report.pdf`)
- machine-readable payload (`report.json`) for EHR/pharma/CRO systems

### Module 0 - Quality Control (Gatekeeper)

**Input**

- VCF/MAF-style mutation file
- optional purity/coverage metadata

**Output**

- `qc_report.json`
- `qc_flags.json`

**Responsibilities**

- Validate schema and required columns
- Estimate data sufficiency for downstream inference
- Stop pipeline early on low-confidence inputs

## AutoResearch Placement

AutoResearch sits beside deterministic modules and consumes only structured outputs (`neoantigens.json`, `clones.json`, `resistance_profile.json`, `trial_matches.json`).

Allowed responsibilities:

- literature/trial evidence retrieval and synthesis
- human-readable rationale narrative generation
- machine-readable rationale object with references

Not allowed:

- primary scientific scoring
- mutation actionability determination
- clone existence inference
- resistance truth inference

## Contract Stability Rules

- Never remove fields without a version bump.
- Additive fields are allowed if old consumers still work.
- Every module declares `schema_version` in outputs.
- If an internal engine is replaced (for example, PyClone -> alternative), the same input/output contracts must hold.
- New modules (for example, toxicity prediction or combo-therapy suggestion) should consume JSON and emit JSON to stay pipeline-compatible.

## Engineering Rules

- Core scientific logic must remain in `backend/core/` and run without UI dependencies.
- Modules read structured inputs and emit versioned structured outputs.
- JSON schemas are explicit and changed deliberately.
- UI is orchestration, not computation.
- Model/agent upgrades cannot silently break report formats.
- Major inference paths require unit tests plus at least one validation case.
