from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime

from neoresist.case_store import (
    append_case_audit_event,
    module_dir,
    mark_run_requested,
    read_case_manifest,
    read_module_status,
    synchronize_manifest,
    write_module_status,
)
from neoresist.module_schema import ModuleSpec, load_module_schema, module_order
from neoresist.paths import repo_root


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _wsl_pyclone_available() -> tuple[bool, str]:
    if not shutil.which("wsl"):
        return False, "WSL is not installed on this machine."
    try:
        proc = subprocess.run(
            ["wsl", "sh", "-lc", "command -v pyclone-vi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:
        return False, f"WSL adapter check failed: {exc}"
    if proc.returncode == 0 and proc.stdout.strip():
        return True, f"PyClone-VI adapter available via WSL at {proc.stdout.strip()}."
    try:
        fallback = subprocess.run(
            ["wsl", "sh", "-lc", "test -x /home/rambe/neoresist-pyclone/bin/pyclone-vi && echo /home/rambe/neoresist-pyclone/bin/pyclone-vi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if fallback.returncode == 0 and fallback.stdout.strip():
            return True, f"PyClone-VI adapter available via WSL at {fallback.stdout.strip()}."
    except Exception:
        pass
    stderr = proc.stderr.replace("\x00", "").strip() or proc.stdout.replace("\x00", "").strip() or "PyClone-VI executable not found in WSL."
    return False, stderr


def module_installation(module_id: str) -> tuple[bool, str]:
    if module_id == "neoantigen_generation":
        exists = (repo_root() / "neoresist-md").is_dir()
        return exists, "Local NeoVax backend detected." if exists else "Missing neoresist-md backend directory."
    if module_id == "expression_join":
        return True, "Expression join is available in the root app."
    if module_id == "clonality_pyclone_vi":
        return _wsl_pyclone_available()
    if module_id in {"resistance_loop", "strategy_engine", "prioritization_tiering"}:
        return True, "Local module available in the root app."
    if module_id == "presentation_netctlpan":
        return False, "NetCTLpan runtime is not installed in this environment."
    if module_id == "escape_lohhla":
        return False, "LOHHLA runtime is not installed in this environment."
    if module_id == "recognition_foreignness":
        has_bio = importlib.util.find_spec("Bio") is not None
        has_module = (repo_root() / "neoresist_md" / "backend" / "core" / "recognition" / "foreignness_module.py").is_file()
        if has_bio and has_module:
            return True, "Foreignness recognition is available via local BLOSUM62 module."
        missing = []
        if not has_bio:
            missing.append("Biopython not installed")
        if not has_module:
            missing.append("foreignness module file missing")
        return False, f"Foreignness recognition unavailable: {', '.join(missing)}."
    return False, "Unknown module."


def _pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


class ModuleRunner:
    def __init__(self) -> None:
        self.modules = load_module_schema()

    def installation_checks(self) -> dict[str, tuple[bool, str]]:
        return {module_id: module_installation(module_id) for module_id in self.modules}

    def start_case(self, case_id: str) -> dict[str, str]:
        mark_run_requested(case_id, True)
        return self.resume_case(case_id)

    def resume_case(self, case_id: str) -> dict[str, str]:
        manifest = synchronize_manifest(case_id)
        decisions: dict[str, str] = {}
        if not bool(manifest.get("run_requested", False)):
            for module_id in module_order():
                decisions[module_id] = str(read_module_status(case_id, module_id).get("status") or "pending")
            return decisions
        for module_id in module_order():
            spec = self.modules[module_id]
            status = read_module_status(case_id, module_id)
            state = str(status.get("status") or "pending")
            enabled = bool(manifest.get("enabled_modules", {}).get(module_id, False))
            if not enabled:
                decisions[module_id] = "disabled"
                continue
            if state == "running":
                if self._reconcile_running_state(case_id, module_id, status):
                    status = read_module_status(case_id, module_id)
                    state = str(status.get("status") or "pending")
                else:
                    decisions[module_id] = "running"
                    continue
            if state == "running":
                decisions[module_id] = "running"
                continue
            if state in {"complete", "failed", "unavailable"}:
                decisions[module_id] = state
                continue
            action, message = self._preflight(case_id, spec, manifest)
            if action == "unavailable":
                self._write_unavailable(case_id, module_id, status, message)
                decisions[module_id] = "unavailable"
                continue
            if action == "blocked":
                status["status"] = "pending"
                status["message"] = message
                status["updated_at"] = _now_iso()
                write_module_status(case_id, module_id, status)
                decisions[module_id] = "pending"
                continue
            self._spawn_worker(case_id, module_id, status, message)
            decisions[module_id] = "running"
        synchronize_manifest(case_id)
        return decisions

    def _reconcile_running_state(self, case_id: str, module_id: str, status: dict[str, object]) -> bool:
        pid = status.get("pid")
        if _pid_is_running(int(pid) if pid is not None else None):
            return False
        status.update(
            {
                "status": "pending",
                "pid": None,
                "resume_ready": True,
                "checkpoint_label": "recovered_stale_run",
                "message": "Recovered stale running state after an interrupted session. Safe to resume from the last disk checkpoint.",
                "updated_at": _now_iso(),
                "last_checked_at": _now_iso(),
            }
        )
        write_module_status(case_id, module_id, status)
        append_case_audit_event(case_id, "module_recovered_for_resume", {"module_id": module_id})
        return True

    def _preflight(self, case_id: str, spec: ModuleSpec, manifest: dict[str, object]) -> tuple[str, str]:
        for dep in spec.dependencies:
            dep_status = read_module_status(case_id, dep).get("status")
            if dep_status != "complete":
                return "blocked", f"Waiting for dependency: {self.modules[dep].display_name}."
        installed, install_message = module_installation(spec.module_id)
        if not installed and spec.module_id != "clonality_pyclone_vi":
            return "unavailable", install_message
        input_files = manifest.get("input_files", {}) or {}
        if spec.module_id == "expression_join" and not input_files.get("rna_sidecar"):
            return "unavailable", "No RNA sidecar attached for this case."
        if spec.module_id == "clonality_pyclone_vi" and not installed:
            return "unavailable", install_message
        if spec.module_id in {"presentation_netctlpan", "escape_lohhla", "recognition_foreignness"} and not installed:
            return "unavailable", install_message
        return "start", install_message or "Starting module."

    def _spawn_worker(self, case_id: str, module_id: str, status: dict[str, object], message: str) -> None:
        mdir = module_dir(case_id, module_id)
        stdout_path = mdir / "stdout.log"
        stderr_path = mdir / "stderr.log"
        stdout_handle = stdout_path.open("a", encoding="utf-8")
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "neoresist.case_worker", "--case-id", case_id, "--module", module_id],
            cwd=str(repo_root()),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            creationflags=creationflags,
        )
        stdout_handle.close()
        stderr_handle.close()
        status.update(
            {
                "status": "running",
                "message": message,
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "finished_at": None,
                "pid": proc.pid,
                "error": None,
                "last_checked_at": _now_iso(),
                "checkpoint_label": "worker_started",
                "resume_ready": False,
                "attempt_count": int(status.get("attempt_count") or 0) + 1,
            }
        )
        write_module_status(case_id, module_id, status)
        append_case_audit_event(case_id, "module_started", {"module_id": module_id, "pid": proc.pid})

    def _write_unavailable(self, case_id: str, module_id: str, status: dict[str, object], message: str) -> None:
        artifacts = dict(status.get("artifacts") or {})
        for output in self.modules[module_id].outputs:
            if output.artifact_key == "adapter_status_json":
                artifact_dir = module_dir(case_id, module_id) / "artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                adapter_status_path = artifact_dir / output.filename
                adapter_status_path.write_text(
                    json.dumps(
                        {
                            "available": False,
                            "module_id": module_id,
                            "message": message,
                            "checkpoint_label": "unavailable",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                artifacts[output.artifact_key] = str(adapter_status_path)
        status.update(
            {
                "status": "unavailable",
                "message": message,
                "updated_at": _now_iso(),
                "finished_at": _now_iso(),
                "pid": None,
                "error": message,
                "artifacts": artifacts,
                "resume_ready": True,
                "checkpoint_label": "unavailable",
            }
        )
        write_module_status(case_id, module_id, status)
        append_case_audit_event(case_id, "module_unavailable", {"module_id": module_id, "message": message})
