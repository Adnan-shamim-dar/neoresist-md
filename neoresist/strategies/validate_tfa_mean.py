from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from neoresist.paths import repo_root
from neoresist.strategy_registry import get_strategy, score_candidates_for_strategy

NEOGUIDER_PUBLISHED = {
    "tesla": {"TFA_mean": 19.6, "source": "Zhao et al. Figure 3, Genome Medicine 2026"},
    "nci_test": {"top20_true": 15.4, "source": "Zhao et al. Table 4, NeoRanking framework"},
    "hitide": {"top20_true": 18.1, "source": "Zhao et al. Table 4, NeoRanking framework"},
}


def compute_tfa_mean(y_true_per_patient, scores_per_patient, top_k=20):
    results = []
    for y, s in zip(y_true_per_patient, scores_per_patient, strict=False):
        y, s = np.array(y), np.array(s)
        order = np.argsort(s)[::-1]
        y_sorted = y[order]
        n_immuno = y.sum()
        top_k_arr = y_sorted[:top_k]
        ttif = top_k_arr.sum() / top_k if len(top_k_arr) == top_k else top_k_arr.sum() / len(top_k_arr)
        threshold = max(len(y) // 2, top_k)
        fr = y_sorted[:threshold].sum() / n_immuno if n_immuno > 0 else 0.0
        auprc = average_precision_score(y, s) if n_immuno > 0 else 0.0
        results.append({"TTIF": ttif, "FR": fr, "AUPRC": auprc, "TFA_mean": float(np.mean([ttif, fr, auprc]))})
    mean_r = {k: float(np.mean([r[k] for r in results])) for k in ["TTIF", "FR", "AUPRC", "TFA_mean"]}
    return {"per_patient": results, "mean": mean_r}


def _find_dataset_file(dataset: str, data_dir: Path) -> Path:
    candidates = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV/TSV files found under {data_dir}")
    preferred = [p for p in candidates if dataset.lower() in p.name.lower()]
    return sorted(preferred or candidates)[0]


def _load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "patient_id" not in out.columns:
        raise ValueError("Validation dataset must include patient_id")
    if "immunogenic" not in out.columns:
        raise ValueError("Validation dataset must include immunogenic")
    return out


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


def run_validation(dataset, data_dir, strategy_id):
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = repo_root() / data_dir
    data_file = _find_dataset_file(dataset, data_dir)
    df = _ensure_columns(_load_table(data_file))
    strategy = get_strategy(strategy_id)
    scored = score_candidates_for_strategy(df, strategy)
    grouped = list(scored.groupby("patient_id", sort=True))
    y_true_per_patient = [grp["immunogenic"].to_numpy(dtype=float) for _, grp in grouped]
    scores_per_patient = [grp["rl_priority"].to_numpy(dtype=float) for _, grp in grouped]
    result = compute_tfa_mean(y_true_per_patient, scores_per_patient)
    result["dataset"] = dataset
    result["strategy_id"] = strategy_id
    result["source_file"] = str(data_file)
    artifact = repo_root() / f"backend/strategy_engine/artifacts/tfa_mean_vs_neoguider_{dataset}.json"
    artifact.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_comparison(dataset, result)
    _print_frameshift_report(scored)
    print(f"Saved {artifact}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare NeoResist TFA-mean with NeoGuider published numbers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()
    run_validation(args.dataset, args.data_dir, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
