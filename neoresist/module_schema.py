from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from neoresist.paths import module_schema_path


@dataclass(frozen=True)
class ModuleArtifactSpec:
    artifact_key: str
    filename: str
    label: str
    display_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModuleTooltipSpec:
    short: str
    long: str


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    display_name: str
    version: str
    heavy: bool
    dependencies: list[str]
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[ModuleArtifactSpec]
    missing_policy: str
    tooltip: ModuleTooltipSpec


def _load_yaml() -> dict[str, Any]:
    path = module_schema_path()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_module_schema() -> dict[str, ModuleSpec]:
    raw = _load_yaml()
    modules = raw.get("modules") or {}
    out: dict[str, ModuleSpec] = {}
    for module_id, payload in modules.items():
        outputs = [
            ModuleArtifactSpec(
                artifact_key=str(item.get("artifact_key") or ""),
                filename=str(item.get("filename") or ""),
                label=str(item.get("label") or item.get("artifact_key") or ""),
                display_columns=[str(x) for x in (item.get("display_columns") or [])],
            )
            for item in (payload.get("outputs") or [])
        ]
        tooltip_payload = payload.get("tooltip") or {}
        out[str(module_id)] = ModuleSpec(
            module_id=str(module_id),
            display_name=str(payload.get("display_name") or module_id),
            version=str(payload.get("version") or "1.0.0"),
            heavy=bool(payload.get("heavy", False)),
            dependencies=[str(x) for x in (payload.get("dependencies") or [])],
            required_inputs=[str(x) for x in (payload.get("required_inputs") or [])],
            optional_inputs=[str(x) for x in (payload.get("optional_inputs") or [])],
            outputs=outputs,
            missing_policy=str(payload.get("missing_policy") or "mark_unavailable"),
            tooltip=ModuleTooltipSpec(
                short=str(tooltip_payload.get("short") or ""),
                long=str(tooltip_payload.get("long") or ""),
            ),
        )
    return out


def module_order() -> list[str]:
    return list(load_module_schema().keys())
