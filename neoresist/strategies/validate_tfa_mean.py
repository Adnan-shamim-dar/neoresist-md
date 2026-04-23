from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.strategy_engine.metrics.tfa_mean import compute_tfa_mean_tesla_exact
from neoresist.paths import repo_root
from neoresist.strategies.neoguider import DEFAULT_FEATURE_MAP, DEFAULT_FEATURE_ORDER
from neoresist.strategy_registry import get_strategy, score_candidates_for_strategy

NEOGUIDER_PUBLISHED = {
    "tesla": {"TFA_mean": 19.6, "source": "Zhao et al. Figure 3, Genome Medicine 2026"},
    "nci_test": {"top20_true": 15.4, "source": "Zhao et al. Table 4, NeoRanking framework"},
    "hitide": {"top20_true": 18.1, "source": "Zhao et al. Table 4, NeoRanking framework"},
}

def _find_dataset_file(dataset: str, data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV/TSV files found under {data_dir}")
    dataset_lower = dataset.lower()
    preferred = [p for p in candidates if dataset_lower in p.name.lower()]
    strong = [
        p for p in preferred
        if "features" in p.name.lower() and "full_features" not in p.name.lower()
    ]
    exact = [p for p in strong if f"{dataset_lower}_2020_features" in p.name.lower()]
    if exact:
        return sorted(exact)[0]
    if strong:
        return sorted(strong)[0]
    if preferred:
        return sorted(preferred)[0]
    return sorted(candidates)[0]


def _load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns and "PatientID" in out.columns:
        out["patient_id"] = out["PatientID"].astype(str)
    if "immunogenic" not in out.columns and "VALIDATED" in out.columns:
        out["immunogenic"] = pd.to_numeric(out["VALIDATED"], errors="coerce")
    if "presentation_score_el" not in out.columns and "Score_EL" in out.columns:
        out["presentation_score_el"] = pd.to_numeric(out["Score_EL"], errors="coerce")
    if "binding_nm" not in out.columns and "MT_BindAff" in out.columns:
        out["binding_nm"] = pd.to_numeric(out["MT_BindAff"], errors="coerce")
    if "binding_stability" not in out.columns and "BindStab" in out.columns:
        out["binding_stability"] = pd.to_numeric(out["BindStab"], errors="coerce")
    if "expression_log2" not in out.columns and "Quantification" in out.columns:
        quant = pd.to_numeric(out["Quantification"], errors="coerce").clip(lower=0.0)
        out["expression_log2"] = np.log2(1.0 + quant)
    if "dai_agretopicity" not in out.columns and "Agretopicity" in out.columns:
        out["dai_agretopicity"] = pd.to_numeric(out["Agretopicity"], errors="coerce")
    if "patient_id" not in out.columns:
        raise ValueError("Validation dataset must include patient_id")
    if "immunogenic" not in out.columns:
        raise ValueError("Validation dataset must include immunogenic")
    return out


def _apply_patient_filter(df: pd.DataFrame, patient_ids: list[str] | None) -> pd.DataFrame:
    if not patient_ids:
        return df
    keep = {str(x).strip() for x in patient_ids if str(x).strip()}
    return df[df["patient_id"].astype(str).isin(keep)].copy()


def _feature_coverage_report(df: pd.DataFrame, strategy_id: str) -> dict[str, dict[str, float | int | bool]]:
    feature_order = DEFAULT_FEATURE_ORDER
    feature_map = DEFAULT_FEATURE_MAP
    report: dict[str, dict[str, float | int | bool]] = {}
    if strategy_id != "neoguider_v1":
        return report
    for feature in feature_order:
        source = feature_map.get(feature, feature)
        if source in df.columns:
            series = pd.to_numeric(df[source], errors="coerce")
            non_na = int(series.notna().sum())
            report[feature] = {
                "source_column": source,
                "present_in_table": True,
                "non_na_rows": non_na,
                "coverage_pct": float(100.0 * non_na / len(df)) if len(df) else 0.0,
            }
        else:
            report[feature] = {
                "source_column": source,
                "present_in_table": False,
                "non_na_rows": 0,
                "coverage_pct": 0.0,
            }
    return report


def _print_comparison(dataset: str, result: dict) -> None:
    print("Metric      | NeoGuider Published | NeoResist | Delta")
    published = NEOGUIDER_PUBLISHED.get(dataset, {})
    if "TFA_mean" in published:
        theirs = float(published["TFA_mean"])
        ours = float(result["mean"]["TFA_mean"])
        print(f"TFA-mean    | {theirs:18.4f} | {ours:9.4f} | {ours - theirs:+.4f}")
    elif "top20_true" in published:
        theirs = float(published["top20_true"])
        ours = float(result["mean"]["TTIF"]) * 20.0
        print(f"top20#T     | {theirs:18.4f} | {ours:9.4f} | {ours - theirs:+.4f}")
    else:
        ours = float(result['mean']['TFA_mean'])
        print(f"TFA-mean    | {'N/A':>18} | {ours:9.4f} | {'N/A'}")


def _print_frameshift_report(scored: pd.DataFrame) -> None:
    if "is_frameshift" not in scored.columns:
        return
    fs = scored[scored["is_frameshift"].fillna(False).astype(bool)].copy()
    if fs.empty:
        return
    fs = fs.sort_values("rl_priority", ascending=False).reset_index(drop=True)
    print("Frameshift report:")
    for idx, row in fs.iterrows():
        gene = row.get("gene_name", row.get("gene", ""))
        binding = row.get("binding_nm", np.nan)
        would_rank_last = "yes" if pd.isna(binding) else "no"
        print(f"{gene} | {binding} | {idx + 1} | {would_rank_last}")


def run_validation(
    dataset,
    data_dir,
    strategy_id,
    patient_ids: list[str] | None = None,
    data_file: str | None = None,
):
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root() / data_dir
    if data_file:
        chosen_file = Path(data_file)
        if not chosen_file.is_absolute():
            chosen_file = repo_root() / chosen_file
    else:
        chosen_file = _find_dataset_file(dataset, data_dir)
    df = _apply_patient_filter(_ensure_columns(_load_table(chosen_file)), patient_ids)
    strategy = get_strategy(strategy_id)
    scored = score_candidates_for_strategy(df, strategy)
    grouped = list(scored.groupby("patient_id", sort=True))
    y_true_per_patient = [grp["immunogenic"].to_numpy(dtype=float) for _, grp in grouped]
    scores_per_patient = [grp["rl_priority"].to_numpy(dtype=float) for _, grp in grouped]
    result = compute_tfa_mean_tesla_exact(y_true_per_patient, scores_per_patient)
    result["dataset"] = dataset
    result["strategy_id"] = strategy_id
    result["source_file"] = str(chosen_file)
    result["patient_ids"] = sorted(df["patient_id"].astype(str).unique().tolist())
    result["n_rows"] = int(len(df))
    result["n_patients"] = int(df["patient_id"].nunique())
    result["n_immunogenic"] = int(pd.to_numeric(df["immunogenic"], errors="coerce").fillna(0).sum())
    result["feature_coverage"] = _feature_coverage_report(df, strategy_id)
    artifact = repo_root() / f"backend/strategy_engine/artifacts/tfa_mean_vs_neoguider_{dataset}.json"
    artifact.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_comparison(dataset, result)
    if result["feature_coverage"]:
        print("NeoGuider F-feature coverage:")
        for feature, meta in result["feature_coverage"].items():
            print(
                f"{feature:12s} | source={meta['source_column']} | "
                f"present={meta['present_in_table']} | "
                f"non_na={meta['non_na_rows']} | "
                f"coverage={meta['coverage_pct']:.1f}%"
            )
    _print_frameshift_report(scored)
    print(f"Saved {artifact}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare NeoResist TFA-mean with NeoGuider published numbers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_file", default="", help="Optional explicit CSV/TSV path.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--patient-ids", default="", help="Optional comma-separated patient_id subset.")
    args = parser.parse_args()
    patient_ids = [x.strip() for x in args.patient_ids.split(",") if x.strip()]
    run_validation(
        args.dataset,
        args.data_dir,
        args.strategy,
        patient_ids=patient_ids or None,
        data_file=args.data_file or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
