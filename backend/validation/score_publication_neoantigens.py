from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neoresist.scoring import apply_resistance_loop_engine
from neoresist_md.backend.core.recognition.foreignness_module import ForeignnessModule

NA_TOKENS = {"", "NA", "N/A", "NONE", "NULL", "NOT_AVAILABLE", "NOT AVAILABLE"}
NUMERIC_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
VALID_HLA_PATTERN = re.compile(r"^HLA-[ABC]\*\d{2}:\d{2}$")
HLA_COLUMNS = ("hla_a1", "hla_a2", "hla_b1", "hla_b2", "hla_c1", "hla_c2")


@dataclass(frozen=True)
class ScoreConfig:
    labels_csv: Path
    hla_csv: Path
    output_dir: Path


def _clean_token(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in NA_TOKENS:
        return None
    return text


def _numeric(value: object) -> float | None:
    text = _clean_token(value)
    if text is None:
        return None
    if not NUMERIC_PATTERN.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_tpm_value(tpm_value: object) -> float | None:
    if tpm_value is None:
        return None
    tpm_str = str(tpm_value).strip()
    if tpm_str in ("NOT_AVAILABLE", "NA", "N/A", "nan", "NaN", "", "None"):
        return None
    tpm_str = tpm_str.split()[0]
    tpm_str = tpm_str.replace(",", "")
    try:
        return float(tpm_str)
    except (ValueError, TypeError):
        print(f"  WARNING: Cannot parse TPM value: '{tpm_value}'")
        return None


def score_expression(tpm_value: object) -> float:
    """Log-normalize TPM to 0-1 score."""
    tpm = _parse_tpm_value(tpm_value)
    if tpm is None:
        return float("nan")
    if tpm <= 0:
        return 0.0
    return float(min(np.log10(tpm + 1) / np.log10(101), 1.0))


def _usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    usable = pd.to_numeric(df["usable_for_training"], errors="coerce").fillna(0).astype(int).eq(1)
    out = df.loc[usable].copy()
    if out.empty:
        raise RuntimeError("No usable_for_training=1 rows were found.")
    return out


def _load_patient_alleles(hla_df: pd.DataFrame) -> dict[str, list[str]]:
    patient_map: dict[str, list[str]] = {}
    for _, row in hla_df.iterrows():
        patient_id = _clean_token(row.get("patient_id"))
        if patient_id is None:
            continue
        alleles: list[str] = []
        for col in HLA_COLUMNS:
            token = _clean_token(row.get(col))
            if token is None:
                continue
            allele = token.upper()
            if VALID_HLA_PATTERN.match(allele):
                alleles.append(allele)
        if alleles:
            patient_map[patient_id] = sorted(set(alleles))
    return patient_map


def _normalize_hla(value: object) -> str | None:
    token = _clean_token(value)
    if token is None:
        return None
    allele = token.upper()
    if not VALID_HLA_PATTERN.match(allele):
        return None
    return allele


def _presentation_from_affinity(affinity_nm: float | None, cap_nm: float = 50000.0) -> float:
    if affinity_nm is None or not np.isfinite(affinity_nm) or affinity_nm <= 0:
        return 0.0
    ratio = np.log1p(min(float(affinity_nm), cap_nm)) / np.log1p(cap_nm)
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def _affinity_baseline_score(affinity_nm: float | None) -> float:
    return _presentation_from_affinity(affinity_nm)


def _predict_missing_affinity(df: pd.DataFrame, patient_alleles: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    out["predicted_hla_allele"] = pd.NA
    out["predicted_affinity_nm"] = pd.NA
    out["predicted_presentation_rank"] = pd.NA
    out["binding_source"] = "reported"

    needs = out["binding_affinity_nm_numeric"].isna() & out["mutant_peptide"].astype(str).str.len().between(8, 11)
    if not bool(needs.any()):
        return out

    try:
        from mhcflurry import Class1PresentationPredictor

        predictor = Class1PresentationPredictor.load()
    except Exception:
        out.loc[needs, "binding_source"] = "missing"
        return out

    for idx, row in out.loc[needs].iterrows():
        peptide = str(row.get("mutant_peptide") or "").strip().upper()
        if not peptide:
            out.at[idx, "binding_source"] = "missing"
            continue
        alleles: list[str] = []
        row_hla = _normalize_hla(row.get("hla_allele"))
        if row_hla:
            alleles.append(row_hla)
        else:
            patient_id = _clean_token(row.get("patient_id"))
            if patient_id:
                alleles.extend(patient_alleles.get(patient_id, []))
        alleles = [a for a in dict.fromkeys(alleles) if VALID_HLA_PATTERN.match(a)]
        if not alleles:
            out.at[idx, "binding_source"] = "missing"
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pred = predictor.predict(peptides=[peptide], alleles=alleles)
        except Exception:
            out.at[idx, "binding_source"] = "missing"
            continue
        if pred is None or pred.empty:
            out.at[idx, "binding_source"] = "missing"
            continue
        ranked = pred.sort_values("affinity", ascending=True)
        best = ranked.iloc[0]
        out.at[idx, "predicted_hla_allele"] = str(best.get("allele"))
        out.at[idx, "predicted_affinity_nm"] = float(best.get("affinity"))
        out.at[idx, "predicted_presentation_rank"] = float(best.get("presentation_percentile"))
        out.at[idx, "binding_source"] = "predicted"
    return out


def _build_scoring_frame(labels_df: pd.DataFrame, hla_df: pd.DataFrame) -> pd.DataFrame:
    df = _usable_rows(labels_df)
    patient_alleles = _load_patient_alleles(hla_df)

    df["mutant_peptide"] = df["mutant_peptide"].map(lambda v: (_clean_token(v) or "").upper())
    df["wildtype_peptide"] = df["wildtype_peptide"].map(lambda v: (_clean_token(v) or "").upper())
    df["hla_allele"] = df["hla_allele"].map(_normalize_hla)
    df["binding_affinity_nm_numeric"] = df["binding_affinity_nm"].map(_numeric)
    df["expression_tpm_numeric"] = df["expression_tpm"].map(_parse_tpm_value)
    df["expression_score"] = df["expression_tpm"].map(score_expression)
    vaf = df["vaf"].map(_numeric)
    df["ccf"] = (vaf.fillna(0.0) * 2.0).clip(0.0, 1.0)
    df["immunogenic_label"] = pd.to_numeric(df["immunogenic"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df["escape_penalty"] = 0.0

    df = _predict_missing_affinity(df, patient_alleles)

    final_affinity = df["binding_affinity_nm_numeric"].copy()
    pred_aff = pd.to_numeric(df["predicted_affinity_nm"], errors="coerce")
    final_affinity = final_affinity.fillna(pred_aff)
    df["final_binding_affinity_nm"] = final_affinity

    row_hla = df["hla_allele"].copy()
    pred_hla = df["predicted_hla_allele"].astype("string")
    df["final_hla_allele"] = row_hla.fillna(pred_hla)

    df["presentation_score"] = df["final_binding_affinity_nm"].map(_presentation_from_affinity)
    df["binding_affinity_baseline_score"] = df["final_binding_affinity_nm"].map(_affinity_baseline_score)
    df["gene"] = df["gene"].map(lambda v: _clean_token(v) or "")
    df["protein_change"] = df["protein_change"].map(lambda v: _clean_token(v) or "")

    rec_input = df[["mutant_peptide", "wildtype_peptide", "gene"]].copy()
    rec_scored = ForeignnessModule().safe_run(rec_input)
    df["self_dissimilarity"] = pd.to_numeric(rec_scored.get("self_dissimilarity"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    scoring_input = pd.DataFrame(
        {
            "mutant_peptide": df["mutant_peptide"],
            "gene": df["gene"],
            "expression_tpm": pd.to_numeric(df["expression_tpm_numeric"], errors="coerce").fillna(0.0).clip(lower=0.0),
            "presentation_score": df["presentation_score"],
            "ccf": df["ccf"],
            "self_dissimilarity": df["self_dissimilarity"],
            "escape_penalty": df["escape_penalty"],
            "hla_allele": df["final_hla_allele"],
        }
    )
    rl_scored = apply_resistance_loop_engine(
        scoring_input,
        prefer_real_evidence=False,
        profile_id="rl_v1",
        rule_profile_id="default_rules",
    )
    df["rl_priority"] = pd.to_numeric(rl_scored.get("rl_priority"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["rl_tier"] = pd.to_numeric(rl_scored.get("tier"), errors="coerce")
    return df


def _compute_metrics(scored: pd.DataFrame) -> dict[str, object]:
    valid = scored["immunogenic_label"].isin([0, 1])
    y = scored.loc[valid, "immunogenic_label"].to_numpy()
    rl = scored.loc[valid, "rl_priority"].to_numpy()
    base = scored.loc[valid, "binding_affinity_baseline_score"].to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Immunogenic labels collapsed to one class.")

    rl_auc = float(roc_auc_score(y, rl))
    base_auc = float(roc_auc_score(y, base))
    counts_by_source = scored["paper_source"].value_counts().to_dict()
    return {
        "n_rows_scored": int(len(scored)),
        "n_labels": int(valid.sum()),
        "n_positive": int((scored["immunogenic_label"] == 1).sum()),
        "n_negative": int((scored["immunogenic_label"] == 0).sum()),
        "auc_roc_rl_priority": rl_auc,
        "auc_roc_binding_affinity_only": base_auc,
        "auc_delta_rl_minus_binding": rl_auc - base_auc,
        "binding_source_counts": scored["binding_source"].value_counts().to_dict(),
        "rows_by_paper_source": counts_by_source,
        "expression_score_summary": {
            "nan_count": int(scored["expression_score"].isna().sum()),
            "zero_count": int((scored["expression_score"] == 0).sum()),
            "nonzero_count": int(((scored["expression_score"] > 0) & scored["expression_score"].notna()).sum()),
            "mean": float(pd.to_numeric(scored["expression_score"], errors="coerce").mean(skipna=True)),
        },
    }


def _write_outputs(scored: pd.DataFrame, metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cols = [
        "paper_source",
        "patient_id",
        "cancer_type",
        "gene",
        "protein_change",
        "mutant_peptide",
        "wildtype_peptide",
        "peptide_length",
        "hla_allele",
        "final_hla_allele",
        "binding_source",
        "binding_affinity_nm",
        "binding_affinity_nm_numeric",
        "predicted_affinity_nm",
        "final_binding_affinity_nm",
        "binding_affinity_baseline_score",
        "expression_tpm",
        "expression_tpm_numeric",
        "expression_score",
        "vaf",
        "ccf",
        "self_dissimilarity",
        "presentation_score",
        "escape_penalty",
        "rl_priority",
        "rl_tier",
        "immunogenic",
        "immunogenic_label",
        "response_type",
        "detection_method",
        "notes",
    ]
    matrix_path = output_dir / "training_matrix.csv"
    scored.loc[:, [c for c in training_cols if c in scored.columns]].to_csv(matrix_path, index=False)

    scored_path = output_dir / "publication_scored_neoantigens.csv"
    scored.to_csv(scored_path, index=False)

    results = pd.DataFrame(
        [
            {"ranker": "ResistanceLoop (rl_priority)", "auc_roc": metrics["auc_roc_rl_priority"], "n": metrics["n_labels"]},
            {
                "ranker": "Binding-affinity-only",
                "auc_roc": metrics["auc_roc_binding_affinity_only"],
                "n": metrics["n_labels"],
            },
        ]
    )
    results_path = output_dir / "results_table.csv"
    results.to_csv(results_path, index=False)

    metrics_payload = dict(metrics)
    metrics_payload.update(
        {
            "training_matrix_csv": str(matrix_path),
            "scored_rows_csv": str(scored_path),
            "results_table_csv": str(results_path),
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def run(config: ScoreConfig) -> dict[str, object]:
    labels_df = pd.read_csv(config.labels_csv)
    hla_df = pd.read_csv(config.hla_csv)
    scored = _build_scoring_frame(labels_df, hla_df)
    metrics = _compute_metrics(scored)
    _write_outputs(scored, metrics, config.output_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"This scorer must run on Python 3.11 for MHCflurry compatibility. Current: {sys.version.split()[0]}. "
            "Run with: py -3.11 backend/validation/score_publication_neoantigens.py"
        )
    parser = argparse.ArgumentParser(description="Score publication neoantigens with ResistanceLoop.")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "validation_papers" / "merged_immunogenicity_labels.csv",
    )
    parser.add_argument(
        "--hla-csv",
        type=Path,
        default=ROOT / "validation_papers" / "patient_hla_alleles.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "backend" / "validation" / "artifacts",
    )
    args = parser.parse_args(argv)

    metrics = run(
        ScoreConfig(
            labels_csv=args.labels_csv,
            hla_csv=args.hla_csv,
            output_dir=args.output_dir,
        )
    )
    print("Publication scoring complete.")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
