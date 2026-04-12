"""Repository root and config-relative paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Project root (parent of ``neoresist`` package)."""
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return repo_root() / "configs"
