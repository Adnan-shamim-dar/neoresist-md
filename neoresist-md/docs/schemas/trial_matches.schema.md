# trial_matches Schema

## Purpose

TrialMatch output for ranked clinical trial opportunities and rationale.

## Version

- `schema_name`: `trial_matches`
- `schema_version`: `1.0.0`

## Required Top-Level Fields

- `schema_name` (string)
- `schema_version` (string)
- `sample_id` (string)
- `trials` (array of trial objects)
- `generated_at` (string, ISO-8601 UTC)

## Trial Object Required Fields

- `nct_id` (string)
- `title` (string)
- `eligibility_score` (number)
- `matched_targets` (array of strings)
- `status` (string)

## Example

```json
{
  "schema_name": "trial_matches",
  "schema_version": "1.0.0",
  "sample_id": "patient_001",
  "trials": [
    {
      "nct_id": "NCT00000000",
      "title": "Example Precision Trial",
      "eligibility_score": 0.76,
      "matched_targets": ["PIK3CA"],
      "status": "Recruiting"
    }
  ],
  "generated_at": "2026-04-08T15:30:00Z"
}
```
