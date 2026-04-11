# clones Schema

## Purpose

CloneEscape output representing clonal/subclonal architecture and mapped mutation/neoantigen sets.

## Version

- `schema_name`: `clones`
- `schema_version`: `1.0.0`

## Required Top-Level Fields

- `schema_name` (string)
- `schema_version` (string)
- `sample_id` (string)
- `clones` (array of clone objects)
- `generated_at` (string, ISO-8601 UTC)

## Clone Object Required Fields

- `clone_id` (string)
- `abundance_score` (number)
- `mutation_ids` (array of strings)
- `neoantigen_ids` (array of strings)

## Example

```json
{
  "schema_name": "clones",
  "schema_version": "1.0.0",
  "sample_id": "patient_001",
  "clones": [
    {
      "clone_id": "clone_A",
      "abundance_score": 0.72,
      "mutation_ids": ["mut_001", "mut_003"],
      "neoantigen_ids": ["neo_001", "neo_004"]
    }
  ],
  "generated_at": "2026-04-08T15:30:00Z"
}
```
