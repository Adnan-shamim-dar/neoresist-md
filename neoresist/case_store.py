from __future__ import annotations

import base64
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from neoresist.module_schema import load_module_schema
from neoresist.paths import cases_dir
from neoresist.upload_runtime import validate_uploaded_table

CASE_SCHEMA_VERSION = "1.0.0"
CASE_AUDIT_VERSION = "1.0.0"
INPUT_ROLE_FILENAMES = {
    "primary_upload": "primary",
    "rna_sidecar": "rna_sidecar",
    "purity_sidecar": "purity_sidecar",
    "cnv_sidecar": "cnv_sidecar",
}


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return clean or "case"


def _decode_upload(contents: str) -> bytes:
    _, encoded = contents.split(",", 1)
    return base64.b64decode(encoded)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def ensure_case_root() -> Path:
    root = cases_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def case_dir(case_id: str) -> Path:
    return ensure_case_root() / case_id


def module_dir(case_id: str, module_id: str) -> Path:
    path = case_dir(case_id) / "modules" / module_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def module_status_path(case_id: str, module_id: str) -> Path:
    return module_dir(case_id, module_id) / "status.json"


def manifest_path(case_id: str) -> Path:
    return case_dir(case_id) / "manifest.yaml"


def case_audit_log_path(case_id: str) -> Path:
    path = case_dir(case_id) / "audit" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _relative_to_case(case_id: str, path: Path) -> str:
    return str(path.relative_to(case_dir(case_id))).replace("\\", "/")


def _absolute_from_case(case_id: str, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    return case_dir(case_id) / rel_path


def append_case_audit_event(case_id: str, event_type: str, payload: dict[str, Any]) -> None:
    event = {
        "timestamp": _now_iso(),
        "event_type": event_type,
        "case_id": case_id,
        "payload": payload,
        "audit_version": CASE_AUDIT_VERSION,
    }
    with case_audit_log_path(case_id).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=_json_default) + "\n")


def read_case_audit(case_id: str, limit: int = 40) -> list[dict[str, Any]]:
    path = case_audit_log_path(case_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def write_case_manifest(manifest: dict[str, Any]) -> None:
    path = manifest_path(str(manifest["case_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=False)
    tmp_path.replace(path)


def read_case_manifest(case_id: str) -> dict[str, Any]:
    path = manifest_path(case_id)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def write_module_status(case_id: str, module_id: str, status: dict[str, Any]) -> None:
    path = module_status_path(case_id, module_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, default=_json_default), encoding="utf-8")


def read_module_status(case_id: str, module_id: str) -> dict[str, Any]:
    path = module_status_path(case_id, module_id)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _initial_module_status(case_id: str, module_id: str, enabled: bool, installed: bool, install_message: str) -> dict[str, Any]:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "module_id": module_id,
        "status": "pending" if enabled else "disabled",
        "enabled": enabled,
        "installed": installed,
        "install_message": install_message,
        "message": "Waiting for run." if enabled else "Disabled for this case.",
        "started_at": None,
        "finished_at": None,
        "updated_at": _now_iso(),
        "artifacts": {},
        "pid": None,
        "error": None,
        "last_checked_at": None,
        "checkpoint_label": "created",
        "resume_ready": False,
        "attempt_count": 0,
    }


def detect_case_id(sample_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_slugify(sample_id)}"


def create_case_from_upload(
    contents: str,
    filename: str,
    *,
    enabled_modules: list[str] | None = None,
    mode_context: str = "Simple",
    install_checks: dict[str, tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    raw = _decode_upload(contents)
    validation = validate_uploaded_table(raw, filename)
    module_specs = load_module_schema()
    enabled = set(enabled_modules or module_specs.keys())
    case_id = detect_case_id(validation.sample_id)
    root = case_dir(case_id)
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "modules").mkdir(parents=True, exist_ok=True)
    (root / "derived").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(parents=True, exist_ok=True)

    effective_raw = validation.normalized_bytes if validation.normalized_bytes is not None else raw
    effective_name = validation.normalized_filename if validation.normalized_filename else filename
    suffix = Path(effective_name).suffix or ".tsv"
    primary_path = root / "input" / f"{INPUT_ROLE_FILENAMES['primary_upload']}{suffix}"
    primary_path.write_bytes(effective_raw)

    manifest = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "created_at": _now_iso(),
        "sample_id": validation.sample_id,
        "input_mode": validation.input_mode,
        "mode_context": mode_context,
        "run_requested": False,
        "input_files": {
            "primary_upload": _relative_to_case(case_id, primary_path),
            "rna_sidecar": None,
            "purity_sidecar": None,
            "cnv_sidecar": None,
        },
        "enabled_modules": {module_id: module_id in enabled for module_id in module_specs},
        "module_status": {},
        "artifact_index": {},
        "audit_version": CASE_AUDIT_VERSION,
        "validation": {
            "row_count": validation.row_count,
            "columns": validation.columns,
            "message": validation.message,
            "germline_contamination_suspected": bool(getattr(validation, "germline_contamination_suspected", False)),
            "ith_entropy": getattr(validation, "ith_entropy", None),
            "hla_format_invalid_count": int(getattr(validation, "hla_format_invalid_count", 0) or 0),
        },
        "germline_contamination_suspected": bool(getattr(validation, "germline_contamination_suspected", False)),
        "ith_entropy": getattr(validation, "ith_entropy", None),
    }
    for module_id in module_specs:
        installed, install_message = (install_checks or {}).get(module_id, (True, ""))
        status = _initial_module_status(case_id, module_id, module_id in enabled, installed, install_message)
        write_module_status(case_id, module_id, status)
        manifest["module_status"][module_id] = status["status"]
    write_case_manifest(manifest)
    append_case_audit_event(
        case_id,
        "case_created",
        {
            "filename": filename,
            "input_mode": validation.input_mode,
            "row_count": validation.row_count,
            "enabled_modules": sorted(enabled),
        },
    )
    return {"case_id": case_id, "validation": validation, "manifest": manifest, "case_dir": str(root)}


def attach_case_input(case_id: str, input_role: str, contents: str, filename: str) -> dict[str, Any]:
    if input_role not in INPUT_ROLE_FILENAMES:
        raise ValueError(f"Unsupported case input role: {input_role}")
    manifest = read_case_manifest(case_id)
    root = case_dir(case_id)
    raw = _decode_upload(contents)
    suffix = Path(filename).suffix or ".txt"
    dest = root / "input" / f"{INPUT_ROLE_FILENAMES[input_role]}{suffix}"
    dest.write_bytes(raw)
    manifest["input_files"][input_role] = _relative_to_case(case_id, dest)
    write_case_manifest(manifest)
    append_case_audit_event(case_id, "sidecar_attached", {"input_role": input_role, "filename": filename})
    return manifest


def set_module_enabled(case_id: str, module_id: str, enabled: bool) -> dict[str, Any]:
    manifest = read_case_manifest(case_id)
    if "case_id" not in manifest:
        manifest["case_id"] = case_id
    if "enabled_modules" not in manifest or not isinstance(manifest.get("enabled_modules"), dict):
        manifest["enabled_modules"] = {}
    if "module_status" not in manifest or not isinstance(manifest.get("module_status"), dict):
        manifest["module_status"] = {}
    manifest["enabled_modules"][module_id] = bool(enabled)
    status = read_module_status(case_id, module_id)
    if not status:
        status = _initial_module_status(case_id, module_id, bool(enabled), True, "")
    status["enabled"] = bool(enabled)
    status["status"] = "pending" if enabled else "disabled"
    status["message"] = "Queued for run." if enabled else "Disabled for this case."
    status["updated_at"] = _now_iso()
    status["error"] = None
    status["resume_ready"] = False
    status["checkpoint_label"] = "enabled" if enabled else "disabled"
    write_module_status(case_id, module_id, status)
    manifest["module_status"][module_id] = status["status"]
    write_case_manifest(manifest)
    append_case_audit_event(case_id, "module_toggle", {"module_id": module_id, "enabled": bool(enabled)})
    return manifest


def mark_run_requested(case_id: str, run_requested: bool) -> dict[str, Any]:
    manifest = read_case_manifest(case_id)
    manifest["run_requested"] = bool(run_requested)
    write_case_manifest(manifest)
    append_case_audit_event(case_id, "run_requested", {"run_requested": bool(run_requested)})
    return manifest


def reset_module_for_rerun(case_id: str, module_id: str) -> dict[str, Any]:
    status = read_module_status(case_id, module_id)
    if not status:
        status = _initial_module_status(case_id, module_id, True, True, "")
    status["status"] = "pending"
    status["message"] = "Queued for rerun."
    status["started_at"] = None
    status["finished_at"] = None
    status["updated_at"] = _now_iso()
    status["error"] = None
    status["pid"] = None
    status["resume_ready"] = False
    status["checkpoint_label"] = "reset_for_rerun"
    write_module_status(case_id, module_id, status)
    manifest = read_case_manifest(case_id)
    if "case_id" not in manifest:
        manifest["case_id"] = case_id
    if "module_status" not in manifest or not isinstance(manifest.get("module_status"), dict):
        manifest["module_status"] = {}
    manifest["module_status"][module_id] = "pending"
    write_case_manifest(manifest)
    append_case_audit_event(case_id, "module_reset", {"module_id": module_id})
    return manifest


def reset_modules_for_rerun(case_id: str, module_ids: list[str]) -> dict[str, Any]:
    manifest = read_case_manifest(case_id)
    if "case_id" not in manifest:
        manifest["case_id"] = case_id
    if "module_status" not in manifest or not isinstance(manifest.get("module_status"), dict):
        manifest["module_status"] = {}
    for module_id in module_ids:
        status = read_module_status(case_id, module_id)
        if not status:
            status = _initial_module_status(case_id, module_id, True, True, "")
        status["status"] = "pending"
        status["message"] = "Queued for rerun."
        status["started_at"] = None
        status["finished_at"] = None
        status["updated_at"] = _now_iso()
        status["error"] = None
        status["pid"] = None
        status["resume_ready"] = False
        status["checkpoint_label"] = "reset_for_rerun"
        write_module_status(case_id, module_id, status)
        manifest["module_status"][module_id] = "pending"
    write_case_manifest(manifest)
    append_case_audit_event(case_id, "module_chain_reset", {"module_ids": module_ids})
    return manifest


def synchronize_manifest(case_id: str) -> dict[str, Any]:
    manifest = read_case_manifest(case_id)
    if not manifest or "case_id" not in manifest:
        return {}
    module_specs = load_module_schema()
    artifact_index: dict[str, dict[str, str]] = {}
    for module_id in module_specs:
        status = read_module_status(case_id, module_id)
        if not status:
            status = _initial_module_status(case_id, module_id, bool(manifest.get("enabled_modules", {}).get(module_id, True)), True, "")
            write_module_status(case_id, module_id, status)
        if "enabled_modules" not in manifest:
            manifest["enabled_modules"] = {}
        manifest["enabled_modules"].setdefault(module_id, bool(status.get("enabled", True)))
        if "module_status" not in manifest:
            manifest["module_status"] = {}
        manifest["module_status"][module_id] = status.get("status", "pending")
        artifact_index[module_id] = status.get("artifacts", {}) or {}
    manifest["artifact_index"] = artifact_index
    write_case_manifest(manifest)
    return manifest


def update_module_checkpoint(
    case_id: str,
    module_id: str,
    *,
    checkpoint_label: str,
    resume_ready: bool,
    message: str | None = None,
) -> dict[str, Any]:
    status = read_module_status(case_id, module_id)
    status["checkpoint_label"] = checkpoint_label
    status["resume_ready"] = bool(resume_ready)
    status["updated_at"] = _now_iso()
    if message is not None:
        status["message"] = message
    write_module_status(case_id, module_id, status)
    return status


def list_cases(limit: int = 24) -> list[dict[str, Any]]:
    root = ensure_case_root()
    manifests: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/manifest.yaml"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        manifests.append(manifest)
        if len(manifests) >= limit:
            break
    return manifests


def delete_case(case_id: str) -> bool:
    root = ensure_case_root().resolve()
    target = case_dir(case_id).resolve()
    if root not in target.parents:
        raise ValueError(f"Refusing to delete case outside case root: {target}")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


def resolve_case_input(case_id: str, input_role: str) -> Path | None:
    manifest = read_case_manifest(case_id)
    return _absolute_from_case(case_id, manifest.get("input_files", {}).get(input_role))


class CaseStore:
    create_case_from_upload = staticmethod(create_case_from_upload)
    read_case_manifest = staticmethod(read_case_manifest)
    write_case_manifest = staticmethod(write_case_manifest)
    list_cases = staticmethod(list_cases)
    delete_case = staticmethod(delete_case)
    resolve_case_input = staticmethod(resolve_case_input)
    attach_case_input = staticmethod(attach_case_input)
