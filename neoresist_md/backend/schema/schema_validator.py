from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _default_schema_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "config" / "canonical_schema_v1.yaml"


def load_schema(path: str | Path | None = None) -> dict:
    schema_path = Path(path) if path else _default_schema_path()
    return yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}


def all_columns(schema: dict) -> list[str]:
    cols = [str(x) for x in (schema.get("all_columns") or [])]
    if cols:
        return cols
    groups = schema.get("column_groups") or {}
    out: list[str] = []
    for _, group_cols in groups.items():
        out.extend(str(x) for x in (group_cols or []))
    return out


def validate_columns(columns: list[str], schema: dict | None = None) -> tuple[bool, list[str]]:
    schema = schema or load_schema()
    required = [str(x) for x in (schema.get("required_columns") or [])]
    missing = [col for col in required if col not in set(columns)]
    return (len(missing) == 0, missing)


def validate_enum_values(rows: list[dict[str, Any]], schema: dict | None = None) -> tuple[bool, list[str]]:
    schema = schema or load_schema()
    enums = schema.get("enum_values") or {}
    errors: list[str] = []
    for idx, row in enumerate(rows):
        for col, allowed in enums.items():
            val = row.get(col)
            if val is None:
                continue
            if str(val) not in {str(x) for x in allowed}:
                errors.append(f"row {idx} col {col}: {val!r} not in {allowed}")
    return (len(errors) == 0, errors)


def ensure_canonical_row(row: dict[str, Any], schema: dict | None = None) -> dict[str, Any]:
    schema = schema or load_schema()
    out = dict(row)
    defaults = schema.get("defaults") or {}
    for col in all_columns(schema):
        if col not in out:
            out[col] = defaults.get(col)
    return out


def ensure_canonical_table(rows: list[dict[str, Any]], schema: dict | None = None) -> list[dict[str, Any]]:
    schema = schema or load_schema()
    return [ensure_canonical_row(row, schema) for row in rows]


def apply_module_updates(
    rows: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    *,
    module_id: str,
    schema: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Enforces canonical write rules:
    - never delete columns
    - never recompute/overwrite already populated values
    - module can only write columns it owns
    - add missing columns with None/defaults
    """
    schema = schema or load_schema()
    canonical_cols = set(all_columns(schema))
    ownership = schema.get("module_column_ownership") or {}
    writable = set(str(x) for x in (ownership.get(module_id) or []))
    if not writable:
        raise ValueError(f"Unknown module_id for ownership checks: {module_id}")
    if len(rows) != len(updates):
        raise ValueError("rows and updates length mismatch")

    out_rows = ensure_canonical_table(rows, schema)
    for i, (row, patch) in enumerate(zip(out_rows, updates)):
        for col, new_value in patch.items():
            col_s = str(col)
            if col_s not in canonical_cols:
                raise ValueError(f"{module_id} attempted to write non-canonical column: {col_s}")
            if col_s not in writable:
                raise ValueError(f"{module_id} attempted to write column outside ownership: {col_s}")
            if new_value is None:
                continue
            current_value = row.get(col_s)
            if current_value is not None and current_value != new_value:
                raise ValueError(
                    f"{module_id} attempted to overwrite existing value at row {i}, column {col_s}: "
                    f"{current_value!r} -> {new_value!r}"
                )
            row[col_s] = new_value
    return out_rows


def migrate_rows(rows: list[dict[str, Any]], from_version: str, to_version: str) -> list[dict[str, Any]]:
    """
    Schema bump contract:
    - every new schema version must add a migration function below
    - missing migration path raises immediately
    """
    if from_version == to_version:
        return rows
    migrations: dict[tuple[str, str], Any] = {
        # Example for future:
        # ("1.0.0", "1.1.0"): migrate_1_0_0_to_1_1_0,
    }
    fn = migrations.get((str(from_version), str(to_version)))
    if fn is None:
        raise NotImplementedError(
            f"No migration function registered for canonical schema {from_version} -> {to_version}"
        )
    return fn(rows)
