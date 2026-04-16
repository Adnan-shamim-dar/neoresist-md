from __future__ import annotations

from neoresist_md.backend.schema.schema_validator import (
    apply_module_updates,
    ensure_canonical_table,
    load_schema,
    validate_columns,
    validate_enum_values,
)


def test_schema_contains_required_columns():
    schema = load_schema()
    ok, missing = validate_columns(schema.get("all_columns", []), schema)
    assert ok, f"Missing required columns: {missing}"


def test_ensure_canonical_table_fills_missing():
    schema = load_schema()
    rows = [{"patient_id": "P1", "gene": "TP53"}]
    out = ensure_canonical_table(rows, schema)
    assert "expression_tpm" in out[0]
    assert "resistance_composite" in out[0]


def test_apply_updates_blocks_overwrite():
    schema = load_schema()
    rows = ensure_canonical_table([{"patient_id": "P1"}], schema)
    first = apply_module_updates(rows, [{"expression_tpm": 10.0}], module_id="expression", schema=schema)
    try:
        apply_module_updates(first, [{"expression_tpm": 20.0}], module_id="expression", schema=schema)
    except ValueError:
        assert True
        return
    assert False, "Expected overwrite protection error"


def test_enum_validation():
    schema = load_schema()
    rows = ensure_canonical_table([{"expression_flag": "BAD_VALUE"}], schema)
    ok, errors = validate_enum_values(rows, schema)
    assert not ok
    assert errors

