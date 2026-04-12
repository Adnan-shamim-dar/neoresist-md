from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from neoresist.paths import config_dir, repo_root


@dataclass(frozen=True)
class BrandingConfig:
    title: str
    subtitle: str
    footer_note: str


@dataclass(frozen=True)
class DefaultsConfig:
    dataset_id: str
    scoring_profile_id: str
    rule_profile_id: str


@dataclass(frozen=True)
class AppConfig:
    branding: BrandingConfig
    defaults: DefaultsConfig
    cohort_search_paths: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    raw = _load_yaml(config_dir() / "app_config.yaml")
    b = raw.get("branding") or {}
    d = raw.get("defaults") or {}
    cohort = raw.get("cohort") or {}
    paths = tuple(cohort.get("search_paths") or [])
    return AppConfig(
        branding=BrandingConfig(
            title=str(b.get("title") or "NeoResist-MD"),
            subtitle=str(
                b.get("subtitle")
                or "TCGA-SARC — ResistanceLoop v1 evidence-aware neoantigen qualification"
            ),
            footer_note=str(b.get("footer_note") or "ResistanceLoop v1"),
        ),
        defaults=DefaultsConfig(
            dataset_id=str(d.get("dataset_id") or "tcga_sarc"),
            scoring_profile_id=str(d.get("scoring_profile_id") or "rl_v1"),
            rule_profile_id=str(d.get("rule_profile_id") or "default_rules"),
        ),
        cohort_search_paths=paths,
    )


def resolved_cohort_search_paths() -> list[Path]:
    root = repo_root()
    cfg = get_app_config()
    if cfg.cohort_search_paths:
        return [root / p for p in cfg.cohort_search_paths]
    from neoresist.dash_app.data import qualified_parquet_search_paths

    return qualified_parquet_search_paths()
