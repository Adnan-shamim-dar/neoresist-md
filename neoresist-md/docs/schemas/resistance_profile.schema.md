# resistance_profile Schema

## Purpose

ResistanceLoop output describing resistance-associated drivers and rationale scores.

## Version

- `schema_name`: `resistance_profile`
- `schema_version`: `1.0.0`

## Required Top-Level Fields

- `schema_name` (string)
- `schema_version` (string)
- `sample_id` (string)
- `resistance_drivers` (array of objects)
- `generated_at` (string, ISO-8601 UTC)

## Resistance Driver Required Fields

- `gene` (string)
- `resistance_score` (number)
- `evidence_source` (string)
- `rationale` (string)

## Example

```json
{
  "schema_name": "resistance_profile",
  "schema_version": "1.0.0",
  "sample_id": "patient_001",
  "resistance_drivers": [
    {
      "gene": "PIK3CA",
      "resistance_score": 0.81,
      "evidence_source": "DepMap",
      "rationale": "Pathway-linked resistance signal."
    }
  ],
  "generated_at": "2026-04-08T15:30:00Z"
}
```
