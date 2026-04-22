from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ARTIFACTS = Path("backend/validation/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    m = np.isfinite(y_true) & np.isfinite(y_score)
    if m.sum() < 2:
        return None
    y = y_true[m]
    s = y_score[m]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def tcr_positions(pep_len: int) -> list[int]:
    if pep_len == 8:
        return [2, 3, 4, 5, 6]
    if pep_len == 9:
        return [3, 4, 5, 6, 7]
    if pep_len == 10:
        return [3, 4, 5, 6, 7, 8]
    if pep_len == 11:
        return [3, 4, 5, 6, 7, 8, 9]
    return list(range(2, max(3, pep_len - 1)))


def anchor_positions(pep_len: int) -> list[int]:
    return [1, pep_len - 1]


AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
    "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
    "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
AA_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_CHARGE = {"R": 1, "K": 1, "H": 0.5, "D": -1, "E": -1}
AA_AROMATIC = {"F": 1, "W": 1, "Y": 1, "H": 0.5}
AA_POLAR = {"S": 1, "T": 1, "N": 1, "Q": 1, "Y": 0.5, "C": 0.5}
AA_SMALL = {"G": 1, "A": 1, "S": 0.5, "T": 0.5, "P": 0.5}
AA_ALIPHATIC = {"A": 1, "V": 1, "L": 1, "I": 1, "M": 0.5}
AA_MW = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115, "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}
AA_BETA = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87,
    "I": 1.60, "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37,
    "Y": 1.47, "V": 1.70,
}
AA_HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00,
    "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08,
    "Y": 0.69, "V": 1.06,
}
CALIS_SCORES = {
    "A": -0.02, "C": 0.08, "D": -0.19, "E": -0.19, "F": 0.12, "G": -0.04, "H": 0.10, "I": 0.05,
    "K": -0.20, "L": 0.06, "M": 0.05, "N": -0.10, "P": -0.06, "Q": -0.10, "R": -0.12, "S": -0.04,
    "T": -0.03, "V": 0.02, "W": 0.13, "Y": 0.09,
}


def compute_aa_property(peptide: object, prop_dict: dict[str, float], positions: str = "all") -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    if len(pep) < 8:
        return np.nan
    if positions == "all":
        pos_list = range(len(pep))
    elif positions == "tcr":
        pos_list = tcr_positions(len(pep))
    elif positions == "anchor":
        pos_list = anchor_positions(len(pep))
    elif positions == "n_term":
        pos_list = range(min(3, len(pep)))
    elif positions == "c_term":
        pos_list = range(max(0, len(pep) - 3), len(pep))
    else:
        mid = len(pep) // 2
        pos_list = range(max(0, mid - 1), min(len(pep), mid + 2))
    vals = [prop_dict[pep[p]] for p in pos_list if p < len(pep) and pep[p] in prop_dict]
    return float(np.mean(vals)) if vals else np.nan


def compute_aa_composition(peptide: object, aa_set: set[str], positions: str = "all") -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    if len(pep) < 8:
        return np.nan
    if positions == "all":
        pos_list = list(range(len(pep)))
    elif positions == "tcr":
        pos_list = tcr_positions(len(pep))
    else:
        pos_list = anchor_positions(len(pep))
    valid = [p for p in pos_list if p < len(pep)]
    if not valid:
        return np.nan
    count = sum(1 for p in valid if pep[p] in aa_set)
    return float(count / len(valid))


def peptide_entropy(peptide: object) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    if len(pep) < 8:
        return np.nan
    counts = Counter(pep)
    total = len(pep)
    return float(-sum((c / total) * np.log2(c / total) for c in counts.values()))


def calis_immunogenicity(peptide: object) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    if len(pep) < 8:
        return np.nan
    vals = [CALIS_SCORES[pep[p]] for p in tcr_positions(len(pep)) if p < len(pep) and pep[p] in CALIS_SCORES]
    return float(np.mean(vals)) if vals else np.nan


def tcr_property_std(peptide: object, prop_dict: dict[str, float]) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    vals = [prop_dict.get(pep[p], np.nan) for p in tcr_positions(len(pep)) if p < len(pep)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else np.nan


def tcr_property_max(peptide: object, prop_dict: dict[str, float]) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    vals = [prop_dict.get(pep[p], np.nan) for p in tcr_positions(len(pep)) if p < len(pep)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


def tcr_property_min(peptide: object, prop_dict: dict[str, float]) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    vals = [prop_dict.get(pep[p], np.nan) for p in tcr_positions(len(pep)) if p < len(pep)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else np.nan


def score_hypothesis(df_subset: pd.DataFrame, features: list[tuple[str, float, bool]]) -> np.ndarray:
    scores = np.full(len(df_subset), np.nan)
    for i in range(len(df_subset)):
        row = df_subset.iloc[i]
        vals = []
        ws = []
        for feat, w, inv in features:
            if feat not in df_subset.columns:
                continue
            v = row[feat]
            if pd.isna(v) or not np.isfinite(v):
                continue
            vals.append(-v if inv else v)
            ws.append(w)
        if not vals:
            scores[i] = 0.0
            continue
        s = sum(ws)
        scores[i] = 0.0 if s <= 0 else sum(v * (w / s) for v, w in zip(vals, ws))
    return scores


def main() -> int:
    tesla = pd.read_csv("validation_papers/tesla_prepared.csv")
    tesla = tesla[tesla["immunogenic"].isin([0, 1])].copy()
    y = tesla["immunogenic"].values.astype(float)

    print("=" * 70)
    print("TESLA STRATEGY DISCOVERY")
    print("=" * 70)
    print(f"Rows: {len(tesla)}")
    print(f"Immunogenic: {int(tesla['immunogenic'].sum())}")
    print(f"Patients: {tesla['patient_id'].nunique()}")

    nm = pd.to_numeric(tesla.get("binding_affinity_nm"), errors="coerce")
    if nm.notna().sum() < 50:
        for alt in ["binding_affinity_nm_numeric", "predicted_affinity_nm", "final_binding_affinity_nm", "affinity_nm"]:
            if alt in tesla.columns:
                nm = pd.to_numeric(tesla[alt], errors="coerce")
                if nm.notna().sum() > 50:
                    break
    tesla["nm"] = nm
    tesla["bind_log50k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
    tesla["bind_log5k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(5000)).clip(0, 1)
    tesla["bind_log500"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(500)).clip(0, 1)
    tesla["bind_sigmoid500"] = 1.0 / (1.0 + nm / 500)
    tesla["bind_sigmoid50"] = 1.0 / (1.0 + nm / 50)
    tesla["bind_exp500"] = np.exp(-nm / 500)
    tesla["bind_exp150"] = np.exp(-nm / 150)
    tesla["bind_exp50"] = np.exp(-nm / 50)
    tesla["bind_neglog"] = -np.log10(nm.clip(lower=0.1))
    tesla["bind_linear500"] = (1 - nm / 500).clip(0, 1)
    tesla["bind_linear50k"] = (1 - nm / 50000).clip(0, 1)
    tesla["bind_binary500"] = (nm < 500).astype(float)
    tesla["bind_binary50"] = (nm < 50).astype(float)
    tesla["bind_rank_global"] = 1 - nm.rank(pct=True)
    tesla["bind_rank_patient"] = 1 - tesla.groupby("patient_id")["nm"].rank(pct=True)
    tesla["bind_tiered"] = np.select([nm < 50, nm < 500], [1.0, 0.5], default=0.0)
    tesla["bind_ultra"] = (nm < 10).astype(float)

    pep_col = next((c for c in ["mutant_peptide", "peptide", "mutant_seq", "mt_peptide"] if c in tesla.columns and tesla[c].notna().sum() > 50), "mutant_peptide")
    pep = tesla[pep_col]
    t0 = time.time()
    for pname, pdict in [
        ("volume", AA_VOLUME), ("hydro", AA_HYDRO), ("charge", AA_CHARGE), ("aromatic", AA_AROMATIC),
        ("polar", AA_POLAR), ("small", AA_SMALL), ("aliphatic", AA_ALIPHATIC), ("mw", AA_MW), ("beta", AA_BETA), ("helix", AA_HELIX),
    ]:
        tesla[f"tcr_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "tcr"))
        tesla[f"anchor_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "anchor"))
        tesla[f"full_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "all"))
        tesla[f"nterm_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "n_term"))
        tesla[f"cterm_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "c_term"))
        tesla[f"center_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, "center"))

    for name, aa_set in [
        ("aromatic_frac", set("FWY")), ("charged_frac", set("RKHDE")), ("pos_charge_frac", set("RKH")), ("neg_charge_frac", set("DE")),
        ("hydrophobic_frac", set("AILMFVW")), ("polar_frac", set("STNQY")), ("small_frac", set("GAST")), ("large_frac", set("FWYRH")),
    ]:
        tesla[f"tcr_{name}"] = pep.apply(lambda p: compute_aa_composition(p, aa_set, "tcr"))
        tesla[f"full_{name}"] = pep.apply(lambda p: compute_aa_composition(p, aa_set, "all"))

    tesla["calis_immuno"] = pep.apply(calis_immunogenicity)
    tesla["pep_entropy"] = pep.apply(peptide_entropy)
    tesla["pep_length"] = pep.apply(lambda p: len(str(p).strip()) if pd.notna(p) else np.nan)
    tesla["pep_is_9mer"] = (tesla["pep_length"] == 9).astype(float)
    tesla["pep_is_10mer"] = (tesla["pep_length"] == 10).astype(float)
    tesla["tcr_hydro_std"] = pep.apply(lambda p: tcr_property_std(p, AA_HYDRO))
    tesla["tcr_volume_std"] = pep.apply(lambda p: tcr_property_std(p, AA_VOLUME))
    tesla["tcr_charge_std"] = pep.apply(lambda p: tcr_property_std(p, AA_CHARGE))
    tesla["tcr_hydro_max"] = pep.apply(lambda p: tcr_property_max(p, AA_HYDRO))
    tesla["tcr_hydro_min"] = pep.apply(lambda p: tcr_property_min(p, AA_HYDRO))
    tesla["tcr_hydro_range"] = tesla["tcr_hydro_max"] - tesla["tcr_hydro_min"]
    tesla["tcr_volume_max"] = pep.apply(lambda p: tcr_property_max(p, AA_VOLUME))
    tesla["tcr_volume_min"] = pep.apply(lambda p: tcr_property_min(p, AA_VOLUME))
    tesla["tcr_volume_range"] = tesla["tcr_volume_max"] - tesla["tcr_volume_min"]

    for aa in "ACDEFGHIKLMNPQRSTVWY":
        tesla[f"cterm_is_{aa}"] = pep.apply(lambda p: 1.0 if pd.notna(p) and str(p).strip().upper().endswith(aa) else 0.0)
        tesla[f"pos2_is_{aa}"] = pep.apply(lambda p: 1.0 if pd.notna(p) and len(str(p).strip()) > 1 and str(p).strip().upper()[1] == aa else 0.0)
    print(f"Feature computation time: {time.time() - t0:.1f}s")

    hla_col = next((c for c in ["hla_allele", "final_hla_allele", "hla", "mhc_allele", "allele"] if c in tesla.columns and tesla[c].notna().sum() > 50), None)
    if hla_col:
        def hla_supertype(hla: object) -> str:
            if pd.isna(hla):
                return "unknown"
            h = str(hla).replace("HLA-", "").replace("*", "")
            if h.startswith("A02"):
                return "A02"
            if h.startswith("A01"):
                return "A01"
            if h.startswith("A03") or h.startswith("A11"):
                return "A03"
            if h.startswith("A24"):
                return "A24"
            if h.startswith("B07"):
                return "B07"
            if h.startswith("B08"):
                return "B08"
            if h.startswith("B15"):
                return "B15"
            if h.startswith("B27"):
                return "B27"
            if h.startswith("B44"):
                return "B44"
            return "other"
        tesla["hla_supertype"] = tesla[hla_col].apply(hla_supertype)
        for st in tesla["hla_supertype"].dropna().unique():
            tesla[f"hla_is_{st}"] = (tesla["hla_supertype"] == st).astype(float)
        tesla["hla_candidate_count"] = tesla.groupby(hla_col)["immunogenic"].transform("count")
        tesla["patient_candidate_count"] = tesla.groupby("patient_id")["immunogenic"].transform("count")

    tesla["bind_x_calis"] = tesla["bind_log50k"] * tesla["calis_immuno"]
    tesla["bind_x_tcr_hydro"] = tesla["bind_log50k"] * tesla["tcr_hydro"]
    tesla["bind_x_tcr_volume"] = tesla["bind_log50k"] * tesla["tcr_volume"]
    tesla["bind_x_tcr_aromatic"] = tesla["bind_log50k"] * tesla["tcr_aromatic"]
    tesla["bind_x_entropy"] = tesla["bind_log50k"] * tesla["pep_entropy"]
    tesla["strong_bind_x_calis"] = tesla["bind_binary500"] * tesla["calis_immuno"]

    for feat in ["bind_log50k", "calis_immuno", "tcr_hydro", "tcr_volume", "tcr_aromatic", "tcr_polar", "tcr_beta", "tcr_hydro_std", "pep_entropy"]:
        if feat in tesla.columns:
            tesla[f"{feat}_prank"] = tesla.groupby("patient_id")[feat].rank(pct=True)

    drop_cols = {
        "immunogenic", "patient_id", "paper_source", "cancer_type", "gene", "protein_change", "mutant_peptide",
        "wildtype_peptide", "hla_allele", "final_hla_allele", "hla_supertype", "response_type", "detection_method", "notes", "nm", pep_col,
    }
    feat_cols = [c for c in tesla.columns if c not in drop_cols and tesla[c].dtype.kind in {"f", "i"}]

    single_results: list[dict[str, object]] = []
    for feat in feat_cols:
        vals = pd.to_numeric(tesla[feat], errors="coerce")
        m = vals.notna() & np.isfinite(vals)
        if m.sum() < 50:
            continue
        ys = y[m.values]
        xs = vals[m].values
        if len(np.unique(ys)) < 2 or np.std(xs) < 1e-10:
            continue
        auc = roc_auc_score(ys, xs)
        auc_inv = roc_auc_score(ys, -xs)
        best = max(auc, auc_inv)
        direction = "pos" if auc >= auc_inv else "neg"
        try:
            _, pval = mannwhitneyu(xs[ys == 1], xs[ys == 0], alternative="two-sided")
        except Exception:
            pval = 1.0
        single_results.append(
            {
                "feature": feat,
                "auc": float(auc),
                "auc_inv": float(auc_inv),
                "best_auc": float(best),
                "direction": direction,
                "p_value": float(pval),
                "n": int(m.sum()),
                "immuno_mean": float(np.mean(xs[ys == 1])),
                "non_immuno_mean": float(np.mean(xs[ys == 0])),
            }
        )
    single_results.sort(key=lambda r: r["best_auc"], reverse=True)

    significant = [r for r in single_results if r["p_value"] < 0.05 and r["feature"] != "bind_log50k" and not str(r["feature"]).startswith("bind_")]
    hypotheses: list[dict[str, object]] = [{"name": "H0_binding_only", "features": [("bind_log50k", 1.0, False)], "rationale": "Baseline"}]
    for sf in significant[:15]:
        feat = str(sf["feature"])
        inv = sf["direction"] == "neg"
        hypotheses.extend(
            [
                {"name": f"H_{feat}_60_40", "features": [("bind_log50k", 0.6, False), (feat, 0.4, inv)], "rationale": f"Binding + {feat}"},
                {"name": f"H_{feat}_80_20", "features": [("bind_log50k", 0.8, False), (feat, 0.2, inv)], "rationale": f"Binding heavy + {feat}"},
                {"name": f"H_{feat}_50_50", "features": [("bind_log50k", 0.5, False), (feat, 0.5, inv)], "rationale": f"Equal binding + {feat}"},
            ]
        )
    if len(significant) >= 2:
        f1, f2 = significant[:2]
        hypotheses.append(
            {
                "name": "H_top2_with_bind",
                "features": [("bind_log50k", 0.5, False), (str(f1["feature"]), 0.25, f1["direction"] == "neg"), (str(f2["feature"]), 0.25, f2["direction"] == "neg")],
                "rationale": "Binding + top2",
            }
        )
    if len(significant) >= 3:
        f1, f2, f3 = significant[:3]
        hypotheses.append(
            {
                "name": "H_top3_with_bind",
                "features": [("bind_log50k", 0.4, False), (str(f1["feature"]), 0.2, f1["direction"] == "neg"), (str(f2["feature"]), 0.2, f2["direction"] == "neg"), (str(f3["feature"]), 0.2, f3["direction"] == "neg")],
                "rationale": "Binding + top3",
            }
        )
    hypotheses.append({"name": "H_calis_60_40", "features": [("bind_log50k", 0.6, False), ("calis_immuno", 0.4, False)], "rationale": "Binding + Calis"})
    for b in ["bind_exp500", "bind_sigmoid500", "bind_exp150", "bind_neglog"]:
        hypotheses.append({"name": f"H_{b}_only", "features": [(b, 1.0, False)], "rationale": f"Alternative binding: {b}"})
    if "bind_log50k_prank" in tesla.columns:
        hypotheses.append({"name": "H_prank_binding", "features": [("bind_log50k_prank", 1.0, False)], "rationale": "Patient normalized binding rank"})
        if "calis_immuno_prank" in tesla.columns:
            hypotheses.append({"name": "H_prank_bind_calis", "features": [("bind_log50k_prank", 0.6, False), ("calis_immuno_prank", 0.4, False)], "rationale": "Patient normalized bind+calis"})

    patients = tesla["patient_id"].astype(str).unique()
    cv_results: dict[str, dict[str, object]] = {}
    for hyp in hypotheses:
        name = str(hyp["name"])
        feats = hyp["features"]  # type: ignore[assignment]
        per: dict[str, dict[str, object]] = {}
        aucs = []
        for p in patients:
            m = tesla["patient_id"].astype(str) == p
            test = tesla[m]
            yt = test["immunogenic"].values.astype(float)
            if len(np.unique(yt)) < 2:
                per[p] = {"auc": None, "status": "skipped", "n": int(len(test)), "n_pos": int(yt.sum())}
                continue
            s = score_hypothesis(test, feats)  # type: ignore[arg-type]
            auc = safe_auc(yt, s)
            per[p] = {"auc": auc, "status": "computed" if auc is not None else "insufficient", "n": int(len(test)), "n_pos": int(yt.sum())}
            if auc is not None:
                aucs.append(auc)
        s_all = score_hypothesis(tesla, feats)  # type: ignore[arg-type]
        cv_results[name] = {
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "median_auc": float(np.median(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
            "in_sample_auc": safe_auc(y, s_all),
            "n_computable": int(len(aucs)),
            "per_patient": per,
            "rationale": str(hyp["rationale"]),
        }

    lr_candidates = [
        str(r["feature"])
        for r in single_results[:30]
        if not str(r["feature"]).startswith(("bind_", "hla_is_", "cterm_is_", "pos2_is_")) and int(r["n"]) > 500
    ][:10]
    lr_features = ["bind_log50k"] + lr_candidates
    lr_per: dict[str, dict[str, object]] = {}
    lr_aucs = []
    for p in patients:
        tr = tesla["patient_id"].astype(str) != p
        te = tesla["patient_id"].astype(str) == p
        train = tesla[tr]
        test = tesla[te]
        ytr = train["immunogenic"].values.astype(float)
        yte = test["immunogenic"].values.astype(float)
        if len(np.unique(yte)) < 2:
            lr_per[p] = {"auc": None, "status": "skipped"}
            continue
        xtr = train[lr_features].fillna(0.0).values
        xte = test[lr_features].fillna(0.0).values
        try:
            sc = StandardScaler()
            xtr = sc.fit_transform(xtr)
            xte = sc.transform(xte)
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(xtr, ytr)
            pr = lr.predict_proba(xte)[:, 1]
            auc = safe_auc(yte, pr)
            lr_per[p] = {"auc": auc, "status": "computed", "coefs": dict(zip(lr_features, lr.coef_[0].tolist()))}
            if auc is not None:
                lr_aucs.append(auc)
        except Exception as e:
            lr_per[p] = {"auc": None, "status": f"error: {e}"}
    cv_results["TRAIN_FOLD_LR"] = {
        "mean_auc": float(np.mean(lr_aucs)) if lr_aucs else None,
        "median_auc": float(np.median(lr_aucs)) if lr_aucs else None,
        "std_auc": float(np.std(lr_aucs)) if lr_aucs else None,
        "in_sample_auc": None,
        "n_computable": int(len(lr_aucs)),
        "per_patient": lr_per,
        "rationale": f"LR trained on folds with {lr_features}",
    }

    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]["mean_auc"] if x[1]["mean_auc"] is not None else -1, reverse=True)
    baseline_mean = cv_results.get("H0_binding_only", {}).get("mean_auc", 0.0) or 0.0
    winners = [(n, r, (r["mean_auc"] - baseline_mean)) for n, r in sorted_results if r["mean_auc"] is not None and (r["mean_auc"] - baseline_mean) > 0.01]

    repertoire: dict[str, object] = {
        "rl_binding_only": {
            "description": "MHC binding affinity only",
            "features": {"bind_log50k": 1.0},
            "validated_on": {"tesla_2020": baseline_mean},
            "best_for": "Universal baseline, feature-poor data",
        }
    }
    for n, r, _ in winners:
        hyp = next((h for h in hypotheses if h["name"] == n), None)
        if hyp is None:
            continue
        repertoire[f"rl_tesla_{str(n).lower()}"] = {
            "description": r["rationale"],
            "features": {f: w for f, w, _ in hyp["features"]},  # type: ignore[index]
            "validated_on": {"tesla_2020": r["mean_auc"]},
            "best_for": "TESLA-like data (mutant-only peptide data)",
        }

    output = {
        "dataset": {
            "name": "TESLA 2020 (Wells et al.)",
            "n_candidates": int(len(tesla)),
            "n_immunogenic": int(tesla["immunogenic"].sum()),
            "n_patients": int(tesla["patient_id"].nunique()),
            "patients": tesla["patient_id"].astype(str).unique().tolist(),
            "wildtype_available": False,
        },
        "binding_baseline_lopo": baseline_mean,
        "single_feature_results": single_results[:50],
        "n_hypotheses_tested": len(hypotheses),
        "cv_results": {
            n: {
                "mean_auc": float(r["mean_auc"]) if r["mean_auc"] is not None else None,
                "in_sample_auc": float(r["in_sample_auc"]) if r.get("in_sample_auc") is not None else None,
                "rationale": r["rationale"],
                "n_computable": int(r["n_computable"]),
            }
            for n, r in cv_results.items()
        },
        "winners": [{"name": n, "auc": float(r["mean_auc"]), "delta": float(d)} for n, r, d in winners],
        "repertoire": repertoire,
        "ranked_lopo": [{"name": n, "mean_auc": r["mean_auc"], "delta_vs_binding": (r["mean_auc"] - baseline_mean) if r["mean_auc"] is not None else None} for n, r in sorted_results],
    }

    (ARTIFACTS / "tesla_strategy_discovery.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    tesla.to_csv(ARTIFACTS / "tesla_full_features.csv", index=False)
    pd.DataFrame(single_results).to_csv(ARTIFACTS / "tesla_single_feature_ranking.csv", index=False)

    print(f"Baseline LOPO (binding): {baseline_mean:.4f}")
    if winners:
        print(f"Best winner: {winners[0][0]} LOPO={winners[0][1]['mean_auc']:.4f} delta={winners[0][2]:+.4f}")
    else:
        print("No winner above +0.01 vs binding baseline.")
    print(f"Saved: {ARTIFACTS / 'tesla_strategy_discovery.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

