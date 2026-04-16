from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from neoresist.case_store import list_cases, read_module_status
from neoresist.module_schema import module_order
from neoresist.paths import repo_root, upload_runs_dir


def collect_batch_status() -> dict[str, Any]:
    root = repo_root()
    final_summary_path = root / "data" / "final" / "cohort_bundle_summary.json"
    patient_metrics_path = root / "data" / "final" / "patient_metrics.csv"
    proof_table_path = root / "data" / "final" / "proof_table.csv"
    run_root = root / "neoresist-md" / "data" / "neovax_runs"

    summary: dict[str, Any] = {
        "active_job": False,
        "reference_hla": "HLA-A0201",
        "discovered_run_dirs": 0,
        "runs_with_case_report": 0,
        "runs_with_candidates": 0,
        "failed_or_incomplete_runs": 0,
        "latest_runs": [],
        "proof_rows": 0,
        "patient_metric_rows": 0,
        "cohort_bundle": None,
        "recent_upload_runs": [],
        "persisted_cases": 0,
        "module_state_counts": {},
        "recent_case_ids": [],
    }
    if run_root.is_dir():
        run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
        summary["discovered_run_dirs"] = len(run_dirs)
        latest = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:6]
        summary["latest_runs"] = [p.name for p in latest]
        for p in run_dirs:
            has_case_report = (p / "case_report.md").is_file()
            has_candidates = (p / "candidates.csv").is_file()
            summary["runs_with_case_report"] += int(has_case_report)
            summary["runs_with_candidates"] += int(has_candidates)
            if not has_case_report or not has_candidates:
                summary["failed_or_incomplete_runs"] += 1
    if final_summary_path.is_file():
        try:
            summary["cohort_bundle"] = json.loads(final_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary["cohort_bundle"] = None
    if patient_metrics_path.is_file():
        try:
            summary["patient_metric_rows"] = int(len(pd.read_csv(patient_metrics_path)))
        except Exception:
            summary["patient_metric_rows"] = 0
    if proof_table_path.is_file():
        try:
            summary["proof_rows"] = int(len(pd.read_csv(proof_table_path)))
        except Exception:
            summary["proof_rows"] = 0
    if upload_runs_dir().is_dir():
        recent_uploads = sorted(upload_runs_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
        summary["recent_upload_runs"] = [p.name for p in recent_uploads if p.is_dir()]
    cases = list_cases(limit=12)
    summary["persisted_cases"] = len(cases)
    summary["recent_case_ids"] = [str(item.get("case_id")) for item in cases if item.get("case_id")]
    counts: Counter[str] = Counter()
    for manifest in cases:
        case_id = str(manifest.get("case_id") or "")
        if not case_id:
            continue
        for module_id in module_order():
            state = str(read_module_status(case_id, module_id).get("status") or "pending")
            counts[state] += 1
    summary["module_state_counts"] = dict(counts)
    return summary
