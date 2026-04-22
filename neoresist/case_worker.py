from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backend.core.qualification.real_clonality import write_pyclone_stub_tsv
from backend.core.qualification.real_expression import tpm_expression_bin
from neoresist.canonical_schema import build_canonical_from_candidates, ensure_canonical_columns
from neoresist.case_store import (
    append_case_audit_event,
    module_dir,
    read_case_manifest,
    read_module_status,
    resolve_case_input,
    update_module_checkpoint,
    write_module_status,
)
from neoresist.module_runner import module_installation
from neoresist.module_runner import ModuleRunner
from neoresist.paths import repo_root
from neoresist.profiles import load_scoring_profile
from neoresist.strategy_registry import build_consensus_table, list_strategies, score_candidates_for_strategy
from neoresist.scoring import apply_resistance_loop_engine
from neoresist.upload_runtime import FIXED_HLA
from neoresist_md.backend.core.recognition.foreignness_module import ForeignnessModule


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _update_status(case_id: str, module_id: str, **fields) -> None:
    status = read_module_status(case_id, module_id)
    status.update(fields)
    status["updated_at"] = _now_iso()
    write_module_status(case_id, module_id, status)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _first_existing_artifact(case_id: str, artifact_keys: list[tuple[str, str]]) -> Path | None:
    for module_id, artifact_key in artifact_keys:
        status = read_module_status(case_id, module_id)
        path_str = (status.get("artifacts") or {}).get(artifact_key)
        if path_str:
            path = Path(str(path_str))
            if path.is_file():
                return path
    return None


def _infer_separator(path):
    if path.suffix.lower() in {".maf", ".tsv", ".txt"}:
        return "\t"
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.splitlines()[0] if text.splitlines() else ""
    return "\t" if first.count("\t") > first.count(",") else ","


def _load_table(path):
    return pd.read_csv(path, sep=_infer_separator(path), comment="#" if path.suffix.lower() in {".maf", ".tsv", ".txt"} else None)


def _detect_gene_column(columns):
    for candidate in ("gene", "gene_name", "Hugo_Symbol", "symbol"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not detect gene column in sidecar: {columns[:12]}")


def _detect_tpm_column(columns):
    for candidate in ("TPM", "tpm", "expression_tpm", "expr_tpm"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not detect TPM column in sidecar: {columns[:12]}")


def _run_neoantigen(case_id: str) -> dict[str, object]:
    manifest = read_case_manifest(case_id)
    input_path = resolve_case_input(case_id, "primary_upload")
    if input_path is None or not input_path.is_file():
        raise ValueError("Primary upload file is missing.")
    output_dir = module_dir(case_id, "neoantigen_generation") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "backend.api.cli.neovax_cli",
        "--input",
        str(input_path),
        "--input-mode",
        str(manifest.get("input_mode") or "auto"),
        "--hla",
        FIXED_HLA,
        "--sample-id",
        str(manifest.get("sample_id") or "patient_case"),
        "--output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root() / "neoresist-md"),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"NeoVax backend failed with code {proc.returncode}.")
    artifacts = {}
    for name in ("candidates.csv", "neo_candidates.json", "run_metadata.json", "case_report.md", "supported.csv", "rejected.csv"):
        path = output_dir / name
        if path.is_file():
            artifacts[name.replace(".", "_")] = str(path)
    preview = []
    candidates_csv = output_dir / "candidates.csv"
    if candidates_csv.is_file():
        candidates_df = pd.read_csv(candidates_csv)
        preview = candidates_df.head(12).to_dict("records")
        canonical = build_canonical_from_candidates(
            candidates_df,
            source_module="neoantigen_generation",
            run_id=case_id,
            sample_barcode=str(manifest.get("sample_id") or case_id),
        )
        canonical_path = output_dir / "canonical_candidates.csv"
        canonical.to_csv(canonical_path, index=False)
        artifacts["canonical_candidates_csv"] = str(canonical_path)
        summary_path = output_dir / "canonical_summary.json"
        _write_json(
            summary_path,
            {
                "schema_version": str(canonical.get("schema_version").iloc[0]) if not canonical.empty else "1.0.0",
                "rows": int(len(canonical)),
                "columns": list(canonical.columns),
                "source_module": "neoantigen_generation",
            },
        )
        artifacts["canonical_summary_json"] = str(summary_path)
    return {"artifacts": artifacts, "preview": preview, "message": "Neoantigen generation completed."}


def _run_expression_join(case_id: str) -> dict[str, object]:
    neo_dir = module_dir(case_id, "neoantigen_generation") / "artifacts"
    candidates_path = neo_dir / "candidates.csv"
    if not candidates_path.is_file():
        raise ValueError("Neoantigen candidates are missing for expression join.")
    sidecar_path = resolve_case_input(case_id, "rna_sidecar")
    if sidecar_path is None or not sidecar_path.is_file():
        raise ValueError("RNA sidecar is missing.")
    candidates = pd.read_csv(candidates_path)
    sidecar = _load_table(sidecar_path)
    gene_col = _detect_gene_column([str(c) for c in sidecar.columns])
    tpm_col = _detect_tpm_column([str(c) for c in sidecar.columns])
    tpm_map = {
        str(row[gene_col]).strip().upper(): float(row[tpm_col])
        for _, row in sidecar.iterrows()
        if pd.notna(row.get(gene_col)) and pd.notna(row.get(tpm_col))
    }
    real_tpm = []
    expr_bins = []
    missing = []
    for _, row in candidates.iterrows():
        gene = str(row.get("gene_name") or row.get("gene") or "").strip().upper()
        value = tpm_map.get(gene)
        real_tpm.append(value)
        expr_bins.append(tpm_expression_bin(value))
        missing.append(value is None)
    out = candidates.copy()
    out["real_expression_tpm"] = real_tpm
    out["expression_bin"] = expr_bins
    out["RNA_data_missing"] = missing
    output_dir = module_dir(case_id, "expression_join") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "expression_candidates.csv"
    out.to_csv(csv_path, index=False)
    canonical = build_canonical_from_candidates(
        out,
        source_module="expression_join",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    )
    canonical["expression_tool"] = "rna_sidecar_join"
    canonical["expression_flag"] = out["expression_bin"].map(
        {
            "high": "EXPRESSED",
            "medium": "EXPRESSED",
            "low": "LOW",
            "unexpressed": "ABSENT",
            "absent": "ABSENT",
        }
    ).fillna("UNKNOWN")
    canonical["expression_confidence"] = out["RNA_data_missing"].map(lambda is_missing: "UNAVAILABLE" if bool(is_missing) else "HIGH")
    canonical_path = output_dir / "expression_canonical.csv"
    ensure_canonical_columns(
        canonical,
        source_module="expression_join",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    ).to_csv(canonical_path, index=False)
    summary = {
        "rows": int(len(out)),
        "resolved_rows": int((~pd.Series(missing)).sum()),
        "missing_rows": int(pd.Series(missing).sum()),
    }
    summary_path = output_dir / "expression_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "artifacts": {
            "expression_candidates_csv": str(csv_path),
            "expression_summary_json": str(summary_path),
            "expression_canonical_csv": str(canonical_path),
        },
        "preview": out.head(12).to_dict("records"),
        "message": "Expression join completed.",
    }


def _run_clonality(case_id: str) -> dict[str, object]:
    available, message = module_installation("clonality_pyclone_vi")
    output_dir = module_dir(case_id, "clonality_pyclone_vi") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_status_path = output_dir / "adapter_status.json"
    primary_input = resolve_case_input(case_id, "primary_upload")
    if primary_input is None or not primary_input.is_file():
        raise ValueError("Primary upload is missing for clonality.")
    input_df = _load_table(primary_input)
    if not available:
        adapter_status = {"available": False, "message": message}
        adapter_status_path.write_text(json.dumps(adapter_status, indent=2), encoding="utf-8")
        raise RuntimeError(message)
    if "vaf" not in {str(c).lower() for c in input_df.columns} and "VAF" not in input_df.columns:
        adapter_status = {"available": True, "message": "PyClone-VI adapter detected, but the input file lacks a usable VAF column."}
        adapter_status_path.write_text(json.dumps(adapter_status, indent=2), encoding="utf-8")
        raise RuntimeError(adapter_status["message"])
    work_df = input_df.copy()
    lowered = {str(c).lower(): c for c in work_df.columns}
    if "vaf" in lowered and lowered["vaf"] != "vaf":
        work_df["vaf"] = pd.to_numeric(work_df[lowered["vaf"]], errors="coerce")
    if "patient_id" not in work_df.columns:
        manifest = read_case_manifest(case_id)
        work_df["patient_id"] = str(manifest.get("sample_id") or case_id)
    if "gene_name" not in work_df.columns and "Hugo_Symbol" in work_df.columns:
        work_df["gene_name"] = work_df["Hugo_Symbol"]
    input_tsv = write_pyclone_stub_tsv(output_dir, str(work_df["patient_id"].iloc[0]), work_df)
    adapter_status = {
        "available": True,
        "message": "PyClone-VI adapter detected. Prepared canonical input, but execution is not yet configured automatically in this environment.",
        "prepared_input": str(input_tsv),
    }
    adapter_status_path.write_text(json.dumps(adapter_status, indent=2), encoding="utf-8")
    raise RuntimeError(adapter_status["message"])


def _run_recognition(case_id: str) -> dict[str, object]:
    source_path = _first_existing_artifact(
        case_id,
        [
            ("expression_join", "expression_candidates_csv"),
            ("neoantigen_generation", "candidates_csv"),
        ],
    )
    if source_path is None:
        raise ValueError("No candidate artifact exists for foreignness recognition.")
    df = pd.read_csv(source_path)
    if "mutant_peptide" not in df.columns and "peptide" in df.columns:
        df["mutant_peptide"] = df["peptide"]
    if "wildtype_peptide" not in df.columns:
        df["wildtype_peptide"] = None
    if "gene" not in df.columns and "gene_name" in df.columns:
        df["gene"] = df["gene_name"]
    if "patient_id" not in df.columns:
        manifest = read_case_manifest(case_id)
        df["patient_id"] = str(manifest.get("sample_id") or case_id)
    df["run_id"] = case_id

    scored = ForeignnessModule().safe_run(df)
    output_dir = module_dir(case_id, "recognition_foreignness") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_path = output_dir / "recognition_scored.csv"
    scored.to_csv(scored_path, index=False)

    canonical = build_canonical_from_candidates(
        scored,
        source_module="recognition_foreignness",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    )
    canonical["self_dissimilarity"] = pd.to_numeric(scored.get("self_dissimilarity"), errors="coerce")
    canonical["mutant_wt_distance"] = pd.to_numeric(scored.get("mutant_wt_distance"), errors="coerce")
    canonical["recognition_score"] = pd.to_numeric(scored.get("recognition_score"), errors="coerce")
    canonical["recognition_tool"] = scored.get("recognition_tool", pd.Series(["blosum62_alignment"] * len(scored)))
    canonical["recognition_confidence"] = scored.get("recognition_confidence", pd.Series(["UNAVAILABLE"] * len(scored)))
    canonical_path = output_dir / "recognition_canonical.csv"
    ensure_canonical_columns(
        canonical,
        source_module="recognition_foreignness",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    ).to_csv(canonical_path, index=False)

    summary_path = output_dir / "recognition_summary.json"
    _write_json(
        summary_path,
        {
            "rows": int(len(scored)),
            "recognition_score_min": float(pd.to_numeric(scored.get("recognition_score"), errors="coerce").dropna().min())
            if "recognition_score" in scored.columns and not pd.to_numeric(scored.get("recognition_score"), errors="coerce").dropna().empty
            else 0.0,
            "recognition_score_max": float(pd.to_numeric(scored.get("recognition_score"), errors="coerce").dropna().max())
            if "recognition_score" in scored.columns and not pd.to_numeric(scored.get("recognition_score"), errors="coerce").dropna().empty
            else 0.0,
            "recognition_tool": str(scored.get("recognition_tool", pd.Series(["blosum62_alignment"])).iloc[0]),
        },
    )
    return {
        "artifacts": {
            "recognition_scored_csv": str(scored_path),
            "recognition_canonical_csv": str(canonical_path),
            "recognition_summary_json": str(summary_path),
        },
        "preview": scored.head(12).to_dict("records"),
        "message": "Foreignness recognition completed.",
    }


def _run_resistance_loop(case_id: str) -> dict[str, object]:
    source_path = _first_existing_artifact(
        case_id,
        [
            ("expression_join", "expression_candidates_csv"),
            ("neoantigen_generation", "candidates_csv"),
        ],
    )
    if source_path is None:
        raise ValueError("No candidate artifact exists for ResistanceLoop scoring.")
    df = pd.read_csv(source_path)
    if "presentation_score" not in df.columns and "priority_score" in df.columns:
        df["presentation_score"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0.0)
    if "gene" not in df.columns and "gene_name" in df.columns:
        df["gene"] = df["gene_name"]
    if "mutant_peptide" not in df.columns and "peptide" in df.columns:
        df["mutant_peptide"] = df["peptide"]
    if "escape_penalty" not in df.columns:
        loh_series = df.get("hla_loh_status")
        if loh_series is not None:
            df["escape_penalty"] = loh_series.astype(str).str.lower().map({"lost": 0.75, "loh_detected": 0.75}).fillna(0.0)
        else:
            df["escape_penalty"] = 0.0
    scored = apply_resistance_loop_engine(
        df,
        prefer_real_evidence=True,
        profile_id="rl_v1",
        rule_profile_id="default_rules",
    )
    output_dir = module_dir(case_id, "resistance_loop") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "resistance_scored.csv"
    scored.to_csv(scored_path, index=False)
    canonical = build_canonical_from_candidates(
        scored,
        source_module="resistance_loop",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    )
    resistance_weight = abs(float(load_scoring_profile("rl_v1").escape_penalty_weight))
    canonical["resistance_composite"] = pd.to_numeric(scored.get("escape_penalty"), errors="coerce").fillna(0.0) * resistance_weight
    canonical["resistance_confidence"] = scored["evidence_expression_source"].map(lambda src: "HIGH" if src != "stub" else "MEDIUM")
    canonical["strategy_source"] = "rl_v1"
    canonical["weight_vector_json"] = json.dumps({"profile_id": "rl_v1", "rule_profile_id": "default_rules"})
    canonical["composite_priority"] = scored["rl_priority"]
    canonical_path = output_dir / "resistance_canonical.csv"
    ensure_canonical_columns(
        canonical,
        source_module="resistance_loop",
        run_id=case_id,
        sample_barcode=str(read_case_manifest(case_id).get("sample_id") or case_id),
    ).to_csv(canonical_path, index=False)
    summary = {
        "rows": int(len(scored)),
        "tier_1_rows": int((scored["tier"] == 1).sum()) if "tier" in scored.columns else 0,
        "mean_rl_priority": float(pd.to_numeric(scored["rl_priority"], errors="coerce").fillna(0.0).mean()),
        "used_real_expression_rows": int((scored["evidence_expression_source"] != "stub").sum()) if "evidence_expression_source" in scored.columns else 0,
        "used_real_ccf_rows": int((scored["evidence_ccf_source"] != "stub").sum()) if "evidence_ccf_source" in scored.columns else 0,
    }
    summary_path = output_dir / "resistance_summary.json"
    _write_json(summary_path, summary)
    return {
        "artifacts": {
            "resistance_scored_csv": str(scored_path),
            "resistance_summary_json": str(summary_path),
            "resistance_canonical_csv": str(canonical_path),
        },
        "preview": scored.head(12).to_dict("records"),
        "message": "ResistanceLoop scoring completed.",
    }


def _run_strategy_engine(case_id: str) -> dict[str, object]:
    source_path = _first_existing_artifact(case_id, [("resistance_loop", "resistance_scored_csv")])
    if source_path is None:
        raise ValueError("ResistanceLoop output is missing for strategy scoring.")
    df = pd.read_csv(source_path)
    strategies = list_strategies()[:3]
    if not strategies:
        raise ValueError("No strategies are available for strategy scoring.")
    frames: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        frames[strategy.strategy_id] = score_candidates_for_strategy(df, strategy)
    combined = pd.concat(frames.values(), ignore_index=True)
    consensus = build_consensus_table(frames, top_n=50)
    output_dir = module_dir(case_id, "strategy_engine") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "strategy_scores.csv"
    combined.to_csv(scores_path, index=False)
    consensus_path = output_dir / "strategy_consensus.csv"
    consensus.to_csv(consensus_path, index=False)
    summary_path = output_dir / "strategy_summary.json"
    _write_json(
        summary_path,
        {
            "strategies": [strategy.strategy_id for strategy in strategies],
            "rows": int(len(combined)),
            "consensus_rows": int(len(consensus)),
            "top_consensus_score": float(pd.to_numeric(consensus.get("consensus_score"), errors="coerce").fillna(0.0).max()) if not consensus.empty else 0.0,
        },
    )
    return {
        "artifacts": {
            "strategy_scores_csv": str(scores_path),
            "strategy_consensus_csv": str(consensus_path),
            "strategy_summary_json": str(summary_path),
        },
        "preview": consensus.head(12).to_dict("records"),
        "message": "Strategy engine completed.",
    }


def _run_prioritization_tiering(case_id: str) -> dict[str, object]:
    strategy_scores_path = _first_existing_artifact(case_id, [("strategy_engine", "strategy_scores_csv")])
    consensus_path = _first_existing_artifact(case_id, [("strategy_engine", "strategy_consensus_csv")])
    if strategy_scores_path is None or consensus_path is None:
        raise ValueError("Strategy engine artifacts are missing for prioritization.")
    if not strategy_scores_path.exists() or strategy_scores_path.stat().st_size == 0:
        raise ValueError("No candidates available for tiering — upstream module produced empty output")
    if not consensus_path.exists() or consensus_path.stat().st_size == 0:
        raise ValueError("No candidates available for tiering — upstream module produced empty output")
    try:
        scores = pd.read_csv(strategy_scores_path)
        consensus = pd.read_csv(consensus_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("No candidates available for tiering — upstream module produced empty output") from exc
    if scores is None or scores.empty:
        raise ValueError("No candidates available for tiering — upstream module produced empty output")
    if consensus is None or consensus.empty:
        raise ValueError("No candidates available for tiering — upstream module produced empty output")
    manifest = read_case_manifest(case_id)
    if "patient_id" not in scores.columns:
        scores["patient_id"] = str(manifest.get("sample_id") or case_id)
    if "hla_allele" not in scores.columns:
        scores["hla_allele"] = FIXED_HLA
    if "gene" not in scores.columns and "gene_name" in scores.columns:
        scores["gene"] = scores["gene_name"]
    if "mutant_peptide" not in scores.columns and "peptide" in scores.columns:
        scores["mutant_peptide"] = scores["peptide"]
    if "protein_change" not in scores.columns:
        scores["protein_change"] = ""
    sort_cols = [col for col in ["coherence_score", "rl_priority"] if col in scores.columns]
    ascending = [False] * len(sort_cols)
    base = scores.sort_values(sort_cols or list(scores.columns[:1]), ascending=ascending or [True])
    dedupe_cols = [col for col in ["patient_id", "hla_allele", "gene", "mutant_peptide", "protein_change"] if col in base.columns]
    if dedupe_cols:
        base = base.drop_duplicates(subset=dedupe_cols)
    if "consensus_score" in consensus.columns:
        key_cols = [c for c in ["patient_id", "hla_allele", "gene", "mutant_peptide", "protein_change"] if c in consensus.columns and c in base.columns]
        if key_cols:
            for col in key_cols:
                base[col] = base[col].fillna("").astype(str)
                consensus[col] = consensus[col].fillna("").astype(str)
            base = base.merge(consensus[key_cols + [c for c in ["consensus_score", "consensus_agreement_count"] if c in consensus.columns]], on=key_cols, how="left")
    base["composite_priority"] = pd.to_numeric(base.get("consensus_score", base.get("rl_priority")), errors="coerce").fillna(pd.to_numeric(base.get("rl_priority"), errors="coerce").fillna(0.0))
    prioritized = base.sort_values(["tier", "composite_priority"], ascending=[True, False]).copy()
    prioritized["report_summary"] = prioritized.apply(
        lambda row: f"{row.get('gene', 'candidate')} {row.get('mutant_peptide', '')} tier {row.get('tier', '—')} priority {float(row.get('composite_priority', 0.0)):.3f}",
        axis=1,
    )
    output_dir = module_dir(case_id, "prioritization_tiering") / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    prioritized_path = output_dir / "prioritized_candidates.csv"
    prioritized.to_csv(prioritized_path, index=False)
    report_path = output_dir / "patient_report.json"
    _write_json(
        report_path,
        {
            "case_id": case_id,
            "rows": int(len(prioritized)),
            "top_tier": int(pd.to_numeric(prioritized["tier"], errors="coerce").fillna(3).min()) if not prioritized.empty else 3,
            "top_candidates": prioritized[["gene", "mutant_peptide", "tier", "composite_priority"]].head(10).to_dict("records"),
        },
    )
    return {
        "artifacts": {
            "prioritized_candidates_csv": str(prioritized_path),
            "patient_report_json": str(report_path),
        },
        "preview": prioritized.head(12).to_dict("records"),
        "message": "Prioritization and tiering completed.",
    }


def run_module(case_id: str, module_id: str) -> None:
    _update_status(case_id, module_id, status="running", last_checked_at=_now_iso())
    update_module_checkpoint(
        case_id,
        module_id,
        checkpoint_label="worker_executing",
        resume_ready=False,
        message="Worker is executing and writing module artifacts to disk.",
    )
    try:
        if module_id == "neoantigen_generation":
            result = _run_neoantigen(case_id)
        elif module_id == "expression_join":
            result = _run_expression_join(case_id)
        elif module_id == "clonality_pyclone_vi":
            result = _run_clonality(case_id)
        elif module_id == "recognition_foreignness":
            result = _run_recognition(case_id)
        elif module_id == "resistance_loop":
            result = _run_resistance_loop(case_id)
        elif module_id == "strategy_engine":
            result = _run_strategy_engine(case_id)
        elif module_id == "prioritization_tiering":
            result = _run_prioritization_tiering(case_id)
        else:
            raise ValueError(f"Unknown module id: {module_id}")
        _update_status(
            case_id,
            module_id,
            status="complete",
            finished_at=_now_iso(),
            message=str(result.get("message") or "Completed."),
            artifacts=result.get("artifacts") or {},
            preview=result.get("preview") or [],
            error=None,
            pid=None,
            resume_ready=True,
            checkpoint_label="complete",
        )
        append_case_audit_event(case_id, "module_completed", {"module_id": module_id})
        try:
            ModuleRunner().resume_case(case_id)
        except Exception:
            pass
    except Exception as exc:
        current = read_module_status(case_id, module_id)
        state = "unavailable" if module_id in {"expression_join", "clonality_pyclone_vi"} and not current.get("artifacts") else "failed"
        _update_status(
            case_id,
            module_id,
            status=state,
            finished_at=_now_iso(),
            message=str(exc),
            error=str(exc),
            pid=None,
            resume_ready=True,
            checkpoint_label="failed" if state == "failed" else "unavailable",
        )
        append_case_audit_event(case_id, "module_failed", {"module_id": module_id, "error": str(exc)})
        try:
            ModuleRunner().resume_case(case_id)
        except Exception:
            pass
        if module_id == "prioritization_tiering" and "No candidates available for tiering" in str(exc):
            return
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one NeoResist case module.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args(argv)
    run_module(args.case_id, args.module)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
