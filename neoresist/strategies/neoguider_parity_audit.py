from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.strategy_engine.metrics.tfa_mean import compute_tfa_mean_tesla_exact
from neoresist.paths import repo_root
from neoresist.strategy_registry import get_strategy, score_candidates_for_strategy

CANONICAL_FEATURES = (
    "presentation_score_el",
    "binding_nm",
    "binding_stability",
    "expression_log2",
    "dai_agretopicity",
)
RAW_FEATURES = ("Score_EL", "MT_BindAff", "BindStab", "Quantification", "Agretopicity")


def _load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if "patient_id" not in df.columns and "PatientID" in df.columns:
        df["patient_id"] = df["PatientID"].astype(str)
    if "immunogenic" not in df.columns and "VALIDATED" in df.columns:
        df["immunogenic"] = pd.to_numeric(df["VALIDATED"], errors="coerce")
    return df


def _coverage(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            report[col] = {
                "present": True,
                "non_na": int(series.notna().sum()),
                "coverage_pct": float(100.0 * series.notna().sum() / len(df)) if len(df) else 0.0,
            }
        else:
            report[col] = {"present": False, "non_na": 0, "coverage_pct": 0.0}
    return report


def _metric_for_scores(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    rows = df.dropna(subset=["immunogenic", score_col]).copy()
    groups = list(rows.groupby("patient_id", sort=True))
    y_true = [pd.to_numeric(g["immunogenic"], errors="coerce").to_numpy(dtype=float) for _, g in groups]
    scores = [pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float) for _, g in groups]
    tfa = compute_tfa_mean_tesla_exact(y_true, scores)
    return {
        "score_col": score_col,
        "n_rows": int(len(rows)),
        "n_patients": int(rows["patient_id"].nunique()),
        "TFA_mean": float(tfa["mean"]["TFA_mean"]),
        "TTIF": float(tfa["mean"]["TTIF"]),
        "FR": float(tfa["mean"]["FR"]),
        "AUPRC": float(tfa["mean"]["AUPRC"]),
    }


def run_audit(data_file: Path, strategy_id: str, published_tfa: float = 19.6) -> dict[str, Any]:
    df = _load_table(data_file)
    if "patient_id" not in df.columns or "immunogenic" not in df.columns:
        raise ValueError("data_file must contain patient_id and immunogenic (or aliases PatientID/VALIDATED)")
    strategy = get_strategy(strategy_id)
    scored = score_candidates_for_strategy(df, strategy)
    scored["binding_baseline"] = pd.to_numeric(scored.get("binding_nm"), errors="coerce")
    if "binding_baseline" in scored.columns:
        ranked = scored["binding_baseline"].rank(method="average", ascending=True, na_option="bottom")
        max_rank = float(ranked.max()) if len(ranked) else 1.0
        scored["binding_rank_score"] = 1.0 - ((ranked - 1.0) / max(max_rank - 1.0, 1.0))
    else:
        scored["binding_rank_score"] = 0.0

    metrics = {
        "neoguider_v1": _metric_for_scores(scored, "rl_priority"),
        "binding_rank_score": _metric_for_scores(scored, "binding_rank_score"),
    }
    if "PredictedProbability" in scored.columns:
        metrics["supplementary_predicted_probability"] = _metric_for_scores(scored, "PredictedProbability")

    faults: list[dict[str, Any]] = []
    canonical_cov = _coverage(scored, CANONICAL_FEATURES)
    raw_cov = _coverage(scored, RAW_FEATURES)
    missing_canonical = [k for k, v in canonical_cov.items() if not v["present"] or v["coverage_pct"] < 99.0]
    if missing_canonical:
        faults.append(
            {
                "severity": "high",
                "code": "FEATURE_PARITY_MISSING",
                "detail": f"Missing/incomplete canonical NeoGuider F-features: {missing_canonical}",
            }
        )

    if "source_sup3_file" not in scored.columns:
        faults.append(
            {
                "severity": "medium",
                "code": "SOURCE_PROVENANCE_WEAK",
                "detail": "Input table lacks source_sup3_file provenance; parity traceability is weaker.",
            }
        )
    else:
        if scored["source_sup3_file"].astype(str).str.contains("top1000", case=False, na=False).any():
            faults.append(
                {
                    "severity": "medium",
                    "code": "TOP1000_FILTERED_INPUT",
                    "detail": "Source files are top1000 outputs; this may differ from full-candidate evaluation used in some comparisons.",
                }
            )

    model_tfa = metrics["neoguider_v1"]["TFA_mean"]
    delta = float(model_tfa - published_tfa)
    if abs(delta) > 5.0:
        faults.append(
            {
                "severity": "medium",
                "code": "PUBLISHED_GAP_LARGE",
                "detail": f"Observed TFA gap vs published 19.6 is {delta:+.3f}.",
            }
        )

    faults.append(
        {
            "severity": "medium",
            "code": "METRIC_IMPLEMENTATION_LIMIT",
            "detail": (
                "AUPRC is computed via sklearn average_precision_score, while TESLA's original R uses PRROC over rank partitions; "
                "small numeric drift is expected."
            ),
        }
    )

    full_universe_available = False
    conclusion_payload = {
        "conclusion": "BENCHMARK_NOT_REPRODUCIBLE_WITHOUT_PIPELINE",
        "reason": (
            "NeoGuider published TFA-mean=19.6 was computed on full candidate universe "
            "from raw FASTQ via netMHCpan 4.1. Supplementary Data 3 contains only "
            "top-1000 pre-ranked output per patient. Direct numerical comparison to 19.6 "
            "requires running the full NeoGuider pipeline locally "
            "(BWA-MEM + UVC + netMHCpan + Kallisto). This is not reproduced here."
        ),
        "what_our_numbers_mean": (
            "TFA-mean values in neoguider_variant_screen.json are valid for RELATIVE "
            "comparison of variants against each other on consistent inputs. They are not "
            "valid for claiming numerical superiority over published NeoGuider 19.6."
        ),
        "promotion_unblocked": True,
        "promotion_criteria": (
            "Promote variants that (1) beat neoguider_v1 on internal screen, "
            "(2) pass transfer test (train on NCI, test on Ott or vice versa), "
            "(3) have written biological rationale in YAML."
        ),
        "paper_framing": (
            "Report TFA-mean on NeoResist's own full-universe datasets (Ott haystack, "
            "NCI neo-pep). Claim structural advantages: frameshift coverage, cancer-type "
            "specificity. Do not claim numerical superiority over NeoGuider 19.6 without "
            "full pipeline parity."
        ),
    }

    return {
        "data_file": str(data_file),
        "strategy_id": strategy_id,
        "n_rows": int(len(scored)),
        "n_patients": int(scored["patient_id"].nunique()),
        "n_immunogenic": int(pd.to_numeric(scored["immunogenic"], errors="coerce").fillna(0).sum()),
        "patient_ids": sorted(scored["patient_id"].astype(str).unique().tolist()),
        "canonical_feature_coverage": canonical_cov,
        "raw_feature_coverage": raw_cov,
        "metrics": metrics,
        "published_reference": {"tesla_tfa_mean": published_tfa},
        "full_universe_available": full_universe_available,
        "decision": conclusion_payload,
        "faults": faults,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NeoGuider TESLA parity and surface benchmark faults.")
    parser.add_argument(
        "--data-file",
        default="validation_papers/tesla/tesla_neoguider_fig3_sup3.csv",
        help="CSV/TSV path for parity evaluation.",
    )
    parser.add_argument("--strategy-id", default="neoguider_v1")
    parser.add_argument("--published-tfa", type=float, default=19.6)
    parser.add_argument(
        "--output",
        default="backend/strategy_engine/artifacts/neoguider_parity_audit_tesla.json",
        help="Audit artifact path.",
    )
    args = parser.parse_args()
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = repo_root() / data_file
    result = run_audit(data_file=data_file, strategy_id=args.strategy_id, published_tfa=args.published_tfa)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
