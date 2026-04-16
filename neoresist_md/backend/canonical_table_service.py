from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from neoresist_md.backend.schema.schema_validator import all_columns
from neoresist_md.backend.schema.schema_validator import apply_module_updates
from neoresist_md.backend.schema.schema_validator import ensure_canonical_table
from neoresist_md.backend.schema.schema_validator import load_schema


@dataclass
class CanonicalStore:
    case_id: str
    root_dir: Path

    @property
    def case_dir(self) -> Path:
        return self.root_dir / self.case_id

    @property
    def table_path(self) -> Path:
        return self.case_dir / "canonical_table.csv"

    @property
    def audit_path(self) -> Path:
        return self.case_dir / "canonical_audit.jsonl"

    def ensure_dirs(self) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "intermediate"


def build_canonical(rows: list[dict[str, Any]]) -> pd.DataFrame:
    schema = load_schema()
    out_rows = ensure_canonical_table(rows, schema=schema)
    return pd.DataFrame(out_rows, columns=all_columns(schema))


def persist_canonical(case_id: str, df: pd.DataFrame, root_dir: str | Path | None = None) -> Path:
    store = CanonicalStore(case_id=case_id, root_dir=Path(root_dir) if root_dir else _default_root())
    store.ensure_dirs()
    df.to_csv(store.table_path, index=False)
    return store.table_path


def load_canonical(case_id: str, root_dir: str | Path | None = None) -> pd.DataFrame:
    store = CanonicalStore(case_id=case_id, root_dir=Path(root_dir) if root_dir else _default_root())
    if not store.table_path.is_file():
        return pd.DataFrame(columns=all_columns(load_schema()))
    return pd.read_csv(store.table_path)


def append_audit_event(case_id: str, payload: dict[str, Any], root_dir: str | Path | None = None) -> None:
    store = CanonicalStore(case_id=case_id, root_dir=Path(root_dir) if root_dir else _default_root())
    store.ensure_dirs()
    with store.audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def update_with_module(
    case_id: str,
    module_id: str,
    updates: list[dict[str, Any]],
    root_dir: str | Path | None = None,
) -> pd.DataFrame:
    current = load_canonical(case_id, root_dir=root_dir)
    schema = load_schema()
    rows = current.to_dict("records")
    if not rows and updates:
        rows = ensure_canonical_table([{} for _ in updates], schema=schema)
    updated_rows = apply_module_updates(rows, updates, module_id=module_id, schema=schema)
    out = pd.DataFrame(updated_rows, columns=all_columns(schema))
    persist_canonical(case_id, out, root_dir=root_dir)
    append_audit_event(case_id, {"event": "module_update", "module_id": module_id, "row_count": len(out)}, root_dir=root_dir)
    return out

