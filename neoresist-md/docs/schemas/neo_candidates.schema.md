# neo_candidates Schema

## Purpose

Primary deterministic output from NeoVax candidate generation and ranking.

## Version

- `schema_name`: `neo_candidates`
- `schema_version`: `1.0.0`

## Required Top-Level Fields

- `schema_name` (string)
- `schema_version` (string)
- `sample_id` (string)
- `hla_alleles` (array of strings)
- `candidates` (array of candidate objects)
- `generated_at` (string, ISO-8601 UTC)

## Candidate Object Required Fields

- `id` (string)
- `gene` (string)
- `peptide` (string)
- `best_allele` (string)
- `presentation_score` (number)
- `affinity` (number)
- `presentation_component` (number)
- `affinity_component` (number)
- `priority_score` (number)
- `triage_label` (string)
- `ranking_reason` (string)

## Example

```json
{
  "schema_name": "neo_candidates",
  "schema_version": "1.0.0",
  "sample_id": "patient_001",
  "hla_alleles": ["HLA-A*02:01"],
  "candidates": [
    {
      "id": "neo_001",
      "gene": "TP53",
      "peptide": "ETFRDLWKL",
      "best_allele": "HLA-A*02:01",
      "presentation_score": 0.13813,
      "affinity": 1753.59,
      "presentation_component": 0.9823,
      "affinity_component": 0.9823,
      "priority_score": 0.9823,
      "triage_label": "High Priority",
      "ranking_reason": "strong presentation + strong affinity"
    }
  ],
  "generated_at": "2026-04-08T15:30:00Z"
}
```
