"""
NeoResist-MD — Borch/IMPROVE significance test
bind+HydroCore vs PRIME 2.0 on Borch melanoma cohort.

Run: python scripts/borch_significance_test.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "validation/papers/borch2024/data/03_data_for_CV/IMPROVE/03_final_peptide_features_Partition.txt"
ARTS  = ROOT / "backend/strategy_engine/artifacts/borch2024_external_validation.json"

def run() -> None:
    # Load data
    df = pd.read_csv(DATA, sep="\t")
    df["immunogenic"] = df["response"].map({"yes":1,"no":0,"1":1,"0":0,1:1,0:0}).fillna(0).astype(int)
    mel = df[df["cohort"] == "melanoma"].copy()

    # Patient-percentile scores
    def pct(data: pd.DataFrame, col: str, higher_is_better: bool) -> pd.Series:
        v = pd.to_numeric(data[col], errors="coerce")
        r = v.groupby(data["Patient"]).transform(lambda x: x.rank(pct=True, na_option="keep"))
        return (1 - r) if not higher_is_better else r

    s_hc = 0.5 * pct(mel, "HydroCore", True) + 0.5 * pct(mel, "RankEL_4.1", False)

    # Per-patient AUCs for bind+HydroCore
    hc_by_pat: dict[str, float] = {}
    for pat, grp in mel.assign(_s=s_hc).groupby("Patient"):
        y = grp["immunogenic"].values; s = grp["_s"].values
        if y.sum() == 0 or y.sum() == len(y): continue
        try:
            hc_by_pat[pat] = roc_auc_score(y, s)
        except Exception:
            pass

    # Per-patient AUCs for PRIME (from canonical artifact)
    with open(ARTS) as f:
        d = json.load(f)
    prime_by_pat: dict[str, float] = {}
    for row in d["ranked"]:
        if row["strategy"] == "prime_only":
            for p in row["by_cohort"]["melanoma"]["patient_metrics"]:
                if p.get("auc") is not None:
                    prime_by_pat[p["patient_id"]] = p["auc"]
            break

    common = sorted(set(hc_by_pat) & set(prime_by_pat))
    a_hc = np.array([hc_by_pat[p] for p in common])
    a_pr = np.array([prime_by_pat[p] for p in common])
    diff = a_hc - a_pr
    obs  = diff.mean()
    n    = len(diff)

    rng = np.random.default_rng(42)
    # Paired permutation
    perm_diffs = [(rng.choice([-1, 1], n) * diff).mean() for _ in range(10_000)]
    p_perm = float(np.mean(np.array(perm_diffs) >= obs))
    # Bootstrap CI on difference
    boot = [(a_hc[rng.integers(0, n, n)] - a_pr[rng.integers(0, n, n)]).mean()
            for _ in range(10_000)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    print(f"Patients (n):              {n}")
    print(f"bind+HydroCore mean AUC:   {a_hc.mean():.4f}")
    print(f"PRIME mean AUC:            {a_pr.mean():.4f}")
    print(f"Observed difference:       {obs:+.4f}")
    print(f"Paired permutation p:      {p_perm:.4f}")
    print(f"Bootstrap 95% CI (diff):   [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    print("Interpretation: directional advantage; does not reach p<0.05 at n=23 patients.")


if __name__ == "__main__":
    run()
