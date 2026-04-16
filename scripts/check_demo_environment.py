from __future__ import annotations

import importlib.metadata
import shlex
import subprocess
from pathlib import Path

import yaml


def _parse_requirement_name(line: str) -> str:
    for sep in (">=", "==", "<=", "~=", ">", "<"):
        if sep in line:
            return line.split(sep, 1)[0].strip()
    return line.strip()


def check_python_packages(requirements_path: Path) -> list[str]:
    lines = [ln.strip() for ln in requirements_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    missing: list[str] = []
    for req in lines:
        name = _parse_requirement_name(req)
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(req)
    return missing


def check_tools(tool_registry_path: Path) -> dict[str, str]:
    raw = yaml.safe_load(tool_registry_path.read_text(encoding="utf-8")) or {}
    status: dict[str, str] = {}
    for group_name, group in raw.items():
        if not isinstance(group, dict):
            continue
        for tool_id, tool in group.items():
            if not isinstance(tool, dict):
                continue
            cmd = tool.get("check_command")
            key = f"{group_name}.{tool_id}"
            if not cmd:
                status[key] = "no_check_command"
                continue
            try:
                proc = subprocess.run(shlex.split(str(cmd)), capture_output=True, text=True, timeout=4)
                status[key] = "available" if proc.returncode in {0, 1, 2} else "unavailable"
            except Exception:
                status[key] = "unavailable"
    return status


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    req_path = root / "requirements.txt"
    tool_path = root / "neoresist_md" / "config" / "tool_registry.yaml"
    missing = check_python_packages(req_path)
    tool_status = check_tools(tool_path)
    print("=== Python package check ===")
    if missing:
        print("Missing packages:")
        for req in missing:
            print(f"- {req}")
        print("\nInstall command:")
        print("pip install " + " ".join(f'"{req}"' for req in missing))
    else:
        print("All required Python packages are installed.")
    print("\n=== External tool check ===")
    for key in sorted(tool_status):
        print(f"- {key}: {tool_status[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

