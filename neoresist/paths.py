"""Repository root and config-relative paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Project root (parent of ``neoresist`` package)."""
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return repo_root() / "configs"


def local_state_dir() -> Path:
    return repo_root() / "local_state"


def cases_dir() -> Path:
    return local_state_dir() / "cases"


def strategy_store_dir() -> Path:
    return local_state_dir() / "strategies"


def audit_log_path() -> Path:
    return local_state_dir() / "strategy_audit.jsonl"


def upload_runs_dir() -> Path:
    return repo_root() / "outputs" / "upload_runs"


def module_schema_path() -> Path:
    return config_dir() / "module_schema.yaml"
