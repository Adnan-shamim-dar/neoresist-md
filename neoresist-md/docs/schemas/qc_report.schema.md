# qc_report Schema

## Purpose

Quality-control gate output for deciding whether a case can proceed downstream.

## Version

- `schema_name`: `qc_report`
- `schema_version`: `1.0.0`

## Required Fields

- `schema_name` (string)
- `schema_version` (string)
- `sample_id` (string)
- `status` (string; allowed: `pass`, `fail`, `warn`)
- `input_summary` (object)
- `checks` (array of objects)
- `generated_at` (string, ISO-8601 UTC)

## Example

```json
{
  "schema_name": "qc_report",
  "schema_version": "1.0.0",
  "sample_id": "patient_001",
  "status": "pass",
  "input_summary": {
    "input_mode": "maf",
    "total_rows": 100
  },
  "checks": [
    {
      "check_id": "required_columns",
      "result": "pass",
      "message": "Required columns are present."
    }
  ],
  "generated_at": "2026-04-08T15:30:00Z"
}
```
