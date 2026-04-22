from __future__ import annotations

import json
import warnings
from collections import OrderedDict, Counter
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


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    m = np.isfinite(y_true) & np.isfinite(y_score)
    if m.sum() < 2:
        return None
    yy = y_true[m]
    ss = y_score[m]
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, ss))


def is_leaky_feature_name(name: str) -> bool:
    n = name.lower()
    banned = [
        "immunogenic",
        "label",
        "response",
        "detection",
        "validated",
        "usable_for_training",
        "tesla_tcr_flow",
        "tcr_flow",
        "nanoparticle",
        "notes",
    ]
    return any(k in n for k in banned)


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
    elif positions == "center":
        mid = len(pep) // 2
        pos_list = range(max(0, mid - 1), min(len(pep), mid + 2))
    else:
        pos_list = []
    vals = [prop_dict[pep[p]] for p in pos_list if p < len(pep) and pep[p] in prop_dict]
    return float(np.mean(vals)) if vals else np.nan


def compute_aa_fraction(peptide: object, aa_set: set[str], positions: str = "all") -> float:
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
    else:
        pos_list = []
    valid = [p for p in pos_list if p < len(pep)]
    if not valid:
        return np.nan
    count = sum(1 for p in valid if pep[p] in aa_set)
    return float(count / len(valid))


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
    if len(pep) < 8:
        return np.nan
    vals = [prop_dict.get(pep[p], np.nan) for p in tcr_positions(len(pep)) if p < len(pep)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else np.nan


def hamming_prop(mut: object, wt: object) -> float:
    if pd.isna(mut) or pd.isna(wt):
        return np.nan
    ms, ws = str(mut).strip(), str(wt).strip()
    if not ms or not ws:
        return np.nan
    n = min(len(ms), len(ws))
    if n == 0:
        return np.nan
    return float(sum(a != b for a, b in zip(ms[:n], ws[:n])) / n)


def peptide_entropy(peptide: object) -> float:
    if pd.isna(peptide) or str(peptide).strip() == "":
        return np.nan
    pep = str(peptide).strip().upper()
    if len(pep) < 8:
        return np.nan
    counts = Counter(pep)
    tot = len(pep)
    return float(-sum((c / tot) * np.log2(c / tot) for c in counts.values()))


def score_rows(df: pd.DataFrame, feature_list: list[tuple[str, float, bool]]) -> np.ndarray:
    scores = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.iloc[i]
        av, aw = [], []
        for feat, w, inv in feature_list:
            if feat not in df.columns:
                continue
            val = row[feat]
            if pd.isna(val) or not np.isfinite(val):
                continue
            av.append(-val if inv else val)
            aw.append(w)
        if not av:
            scores[i] = 0.0
            continue
        tw = float(sum(aw))
        scores[i] = float(sum(v * (w / tw) for v, w in zip(av, aw))) if tw > 0 else 0.0
    return scores


def resolve_auto_directions(
    train_df: pd.DataFrame, feature_list: list[tuple[str, float, bool | str]]
) -> list[tuple[str, float, bool]]:
    out: list[tuple[str, float, bool]] = []
    y = train_df["immunogenic"].values.astype(float)
    for feat, w, inv in feature_list:
        if inv != "AUTO":
            out.append((feat, w, bool(inv)))
            continue
        if feat not in train_df.columns:
            out.append((feat, w, False))
            continue
        vals = pd.to_numeric(train_df[feat], errors="coerce")
        m = vals.notna() & np.isfinite(vals)
        if m.sum() < 5 or len(np.unique(y[m.values])) < 2:
            out.append((feat, w, False))
            continue
        v = vals[m].values
        yy = y[m.values]
        imm = np.mean(v[yy == 1]) if np.sum(yy == 1) > 0 else np.nan
        non = np.mean(v[yy == 0]) if np.sum(yy == 0) > 0 else np.nan
        out.append((feat, w, False if imm >= non else True))
    return out


def lopo_auc_with_train_fold_directions(
    df: pd.DataFrame, feature_list: list[tuple[str, float, bool | str]]
) -> tuple[float | None, dict[str, float | None]]:
    aucs: dict[str, float | None] = {}
    for tp in df["patient_id"].astype(str).unique():
        tr = df[df["patient_id"].astype(str) != tp]
        te = df[df["patient_id"].astype(str) == tp]
        yte = te["immunogenic"].values.astype(float)
        if len(np.unique(yte)) < 2:
            aucs[tp] = None
            continue
        resolved = resolve_auto_directions(tr, feature_list)
        s = score_rows(te, resolved)
        aucs[tp] = safe_auc(yte, s)
    comp = [v for v in aucs.values() if v is not None]
    return (float(np.mean(comp)) if comp else None), aucs


def lopo_auc(df: pd.DataFrame, feature_list: list[tuple[str, float, bool]]) -> tuple[float | None, dict[str, float | None]]:
    aucs: dict[str, float | None] = {}
    for tp in df["patient_id"].astype(str).unique():
        test = df[df["patient_id"].astype(str) == tp]
        y = test["immunogenic"].values.astype(float)
        if len(np.unique(y)) < 2:
            aucs[tp] = None
            continue
        s = score_rows(test, feature_list)
        aucs[tp] = safe_auc(y, s)
    comp = [v for v in aucs.values() if v is not None]
    return (float(np.mean(comp)) if comp else None), aucs


def insample_auc(df: pd.DataFrame, feature_list: list[tuple[str, float, bool]]) -> float | None:
    y = df["immunogenic"].values.astype(float)
    s = score_rows(df, feature_list)
    return safe_auc(y, s)


def compute_all_features(df: pd.DataFrame, paper_name: str) -> pd.DataFrame:
    out = df.copy()
    print(f"\n--- Computing features for {paper_name} ---")
    nm = None
    for col in ["binding_affinity_nm_numeric", "binding_affinity_nm", "final_binding_affinity_nm", "predicted_affinity_nm", "affinity_nm", "ic50_nm"]:
        if col in out.columns:
            c = pd.to_numeric(out[col], errors="coerce")
            if c.notna().sum() > 5:
                nm = c
                break
    if nm is not None:
        out["nm_raw"] = nm
        out["bind_log50k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
        out["bind_log5k"] = (1 - np.log10(nm.clip(lower=0.1)) / np.log10(5000)).clip(0, 1)
        out["bind_sigmoid500"] = 1.0 / (1.0 + nm / 500)
        out["bind_exp500"] = np.exp(-nm / 500)
        out["bind_exp150"] = np.exp(-nm / 150)
        out["bind_neglog"] = -np.log10(nm.clip(lower=0.1))
        out["bind_binary500"] = (nm < 500).astype(float)
        out["bind_rank_patient"] = 1 - out.groupby("patient_id")["nm_raw"].rank(pct=True)
    else:
        out["nm_raw"] = np.nan
        out["bind_log50k"] = np.nan

    expr = None
    for col in ["expression_tpm_numeric", "expression_tpm", "tpm", "expression_score"]:
        if col in out.columns:
            c = pd.to_numeric(out[col], errors="coerce")
            if c.notna().sum() > 5:
                expr = c
                break
    if expr is not None:
        out["expr_raw"] = expr
        out["expr_log2"] = np.log2(expr.clip(lower=0) + 1)
        out["expr_sqrt"] = np.sqrt(expr.clip(lower=0))
        out["expr_sigmoid10"] = expr / (expr + 10)
        out["expr_rank"] = expr.rank(pct=True)
    else:
        out["expr_raw"] = np.nan
    if "expression_score" in out.columns:
        out["expr_existing"] = pd.to_numeric(out["expression_score"], errors="coerce")
    if "presentation_score" in out.columns:
        out["pres_existing"] = pd.to_numeric(out["presentation_score"], errors="coerce")
    if "self_dissimilarity" in out.columns:
        out["dissim_existing"] = pd.to_numeric(out["self_dissimilarity"], errors="coerce")

    pep_col = next((c for c in ["mutant_peptide", "peptide", "mutant_seq", "mt_peptide", "neoantigen_sequence"] if c in out.columns and out[c].notna().sum() > 5), None)
    wt_col = next((c for c in ["wildtype_peptide", "wt_peptide", "wildtype_seq"] if c in out.columns and out[c].notna().sum() > 5), None)

    if pep_col is not None:
        pep = out[pep_col]
        for pname, pdict in [
            ("volume", AA_VOLUME), ("hydro", AA_HYDRO), ("charge", AA_CHARGE), ("aromatic", AA_AROMATIC),
            ("polar", AA_POLAR), ("small", AA_SMALL), ("aliphatic", AA_ALIPHATIC), ("beta", AA_BETA), ("helix", AA_HELIX),
        ]:
            for pos in ["tcr", "anchor", "all", "n_term", "c_term", "center"]:
                out[f"{pos}_{pname}"] = pep.apply(lambda p: compute_aa_property(p, pdict, pos))
        for name, aset in [
            ("aromatic_frac", set("FWY")), ("charged_frac", set("RKHDE")), ("pos_charge_frac", set("RKH")),
            ("neg_charge_frac", set("DE")), ("hydrophobic_frac", set("AILMFVW")), ("polar_frac", set("STNQY")),
            ("small_frac", set("GAST")), ("large_frac", set("FWYRH")),
        ]:
            for pos in ["tcr", "all", "anchor"]:
                out[f"{pos}_{name}"] = pep.apply(lambda p: compute_aa_fraction(p, aset, pos))
        out["calis_immuno"] = pep.apply(calis_immunogenicity)
        out["pep_entropy"] = pep.apply(peptide_entropy)
        for pname, pdict in [("hydro", AA_HYDRO), ("volume", AA_VOLUME), ("charge", AA_CHARGE)]:
            out[f"tcr_{pname}_std"] = pep.apply(lambda p: tcr_property_std(p, pdict))
        out["pep_length"] = pep.apply(lambda p: len(str(p).strip()) if pd.notna(p) else np.nan)
        out["pep_is_9"] = (out["pep_length"] == 9).astype(float)
        out["pep_is_10"] = (out["pep_length"] == 10).astype(float)

        if wt_col is not None:
            out["hamming"] = out.apply(lambda r: hamming_prop(r[pep_col], r[wt_col]), axis=1)
            for pname, pdict in [("volume", AA_VOLUME), ("hydro", AA_HYDRO), ("charge", AA_CHARGE)]:
                m = pep.apply(lambda p: compute_aa_property(p, pdict, "tcr"))
                w = out[wt_col].apply(lambda p: compute_aa_property(p, pdict, "tcr"))
                out[f"tcr_{pname}_diff"] = m - w
                out[f"tcr_{pname}_abs_diff"] = (m - w).abs()
            out["tcr_surface_change"] = (
                out.get("tcr_hydro_abs_diff", pd.Series(0, index=out.index)).fillna(0) / 9.0
                + out.get("tcr_volume_abs_diff", pd.Series(0, index=out.index)).fillna(0) / 167.0
                + out.get("tcr_charge_abs_diff", pd.Series(0, index=out.index)).fillna(0) / 2.0
            )
        if nm is not None:
            for sf in ["calis_immuno", "tcr_hydro", "tcr_volume", "pep_entropy", "tcr_polar", "tcr_beta"]:
                if sf in out.columns:
                    out[f"bind_x_{sf}"] = out["bind_log50k"] * out[sf]
        for f in ["bind_log50k", "calis_immuno", "tcr_hydro", "tcr_volume", "pep_entropy"]:
            if f in out.columns and out[f].notna().sum() > 10:
                out[f"{f}_prank"] = out.groupby("patient_id")[f].rank(pct=True)
    return out


def make_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    return obj


def main() -> int:
    print("=" * 70)
    print("FULL CROSS-DATASET STRATEGY DISCOVERY")
    print("=" * 70)

    tm = pd.read_csv(ARTIFACTS / "training_matrix.csv")
    tm = tm[tm["immunogenic"].isin([0, 1])].copy()
    papers: dict[str, pd.DataFrame] = {}
    for pn in sorted(tm["paper_source"].unique()):
        if pn == "ott_2017":
            continue
        papers[pn] = tm[tm["paper_source"] == pn].copy()
        d = papers[pn]
        print(f"  {pn}: {len(d)} rows, {int(d['immunogenic'].sum())} immunogenic, {d['patient_id'].nunique()} patients")

    tesla_path = Path("validation_papers/tesla_prepared.csv")
    if tesla_path.exists():
        tesla = pd.read_csv(tesla_path)
        tesla = tesla[tesla["immunogenic"].isin([0, 1])].copy()
        papers["tesla_2020"] = tesla
        print(f"  tesla_2020: {len(tesla)} rows, {int(tesla['immunogenic'].sum())} immunogenic, {tesla['patient_id'].nunique()} patients")

    ott = pd.read_csv(ARTIFACTS / "ott_mutation_level_with_new_features.csv")
    ott = ott[ott["immunogenic"].isin([0, 1])].copy()
    papers["ott_2017_mutation"] = ott
    print(f"  ott_2017_mutation: {len(ott)} rows, {int(ott['immunogenic'].sum())} immunogenic, {ott['patient_id'].nunique()} patients")

    featured = {k: compute_all_features(v, k) for k, v in papers.items()}

    TRANSFER_STRATEGIES: OrderedDict[str, dict[str, object]] = OrderedDict()
    TRANSFER_STRATEGIES["binding_only"] = {"features": [("bind_log50k", 1.0, False)], "description": "MHC binding log50k"}
    TRANSFER_STRATEGIES["rl_v1_original"] = {"features": [("pres_existing", 0.50, False), ("expr_existing", 0.33, False), ("dissim_existing", 0.17, False)], "description": "Original rl_v1"}
    TRANSFER_STRATEGIES["rl_tcr_v1_from_ott"] = {"features": [("bind_log50k", 0.4, False), ("tcr_volume", 0.2, False), ("tcr_charge_diff", 0.2, False), ("tcr_hydro", 0.2, False)], "description": "Ott H14 locked directions"}
    TRANSFER_STRATEGIES["rl_expression_v1_from_tesla"] = {"features": [("bind_log50k", 0.50, False), ("nm_raw", 0.25, True), ("expr_raw", 0.25, False)], "description": "TESLA expression strategy"}
    TRANSFER_STRATEGIES["binding_plus_calis"] = {"features": [("bind_log50k", 0.6, False), ("calis_immuno", 0.4, False)], "description": "Bind + Calis"}
    TRANSFER_STRATEGIES["bind_exp500"] = {"features": [("bind_exp500", 1.0, False)], "description": "Binding exp500"}
    TRANSFER_STRATEGIES["bind_sigmoid500"] = {"features": [("bind_sigmoid500", 1.0, False)], "description": "Binding sigmoid500"}

    print("\n" + "=" * 70)
    print("TRANSFER STRATEGY TESTING")
    print("=" * 70)
    transfer_results: dict[str, object] = {}
    for pn, df in featured.items():
        n_pat = int(df["patient_id"].nunique())
        can_lopo = n_pat >= 3
        print(f"\n--- {pn}: rows={len(df)} pos={int(df['immunogenic'].sum())} patients={n_pat} LOPO={'yes' if can_lopo else 'NO'}")
        pres = {}
        for sn, sd in TRANSFER_STRATEGIES.items():
            fl = sd["features"]  # type: ignore[index]
            available, missing = [], []
            for feat, _, _ in fl:  # type: ignore[misc]
                if feat in df.columns and pd.to_numeric(df[feat], errors="coerce").notna().sum() > 5:
                    available.append(feat)
                else:
                    missing.append(feat)
            if not available:
                pres[sn] = {"lopo_auc": None, "insample_auc": None, "status": f"no features available (missing: {missing})"}
                print(f"  {sn:<35} LOPO={'N/A':>8} InSamp={'N/A':>8} missing={missing}")
                continue
            is_auc = insample_auc(df, fl)  # type: ignore[arg-type]
            if can_lopo:
                lopo, per = lopo_auc(df, fl)  # type: ignore[arg-type]
            else:
                lopo, per = None, {}
            pres[sn] = {"lopo_auc": lopo, "insample_auc": is_auc, "per_patient": per, "features_available": available, "features_missing": missing, "status": "computed"}
            print(f"  {sn:<35} LOPO={(f'{lopo:.4f}' if lopo is not None else 'N/A'):>8} InSamp={(f'{is_auc:.4f}' if is_auc is not None else 'N/A'):>8} missing={missing if missing else 'none'}")
        transfer_results[pn] = {"n_rows": int(len(df)), "n_immunogenic": int(df["immunogenic"].sum()), "n_patients": n_pat, "can_lopo": can_lopo, "strategies": pres}

    print("\n" + "=" * 70)
    print("PAPER-SPECIFIC DISCOVERY")
    print("=" * 70)
    discovery_results: dict[str, object] = {}
    for pn, df in featured.items():
        y = df["immunogenic"].values.astype(float)
        n_rows, n_pos, n_pat = len(df), int(df["immunogenic"].sum()), int(df["patient_id"].nunique())
        print(f"\n=== {pn}: rows={n_rows} pos={n_pos} patients={n_pat}")
        if n_pos < 5:
            discovery_results[pn] = {"status": "skipped_too_few_positives", "n_pos": n_pos}
            print("  SKIP: too few positives")
            continue
        feat_cols = [
            c
            for c in df.columns
            if c != "immunogenic"
            and not is_leaky_feature_name(c)
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().sum() > max(20, int(n_rows * 0.3))
            and pd.to_numeric(df[c], errors="coerce").std() > 1e-10
        ]
        single = []
        for f in feat_cols:
            vals = pd.to_numeric(df[f], errors="coerce")
            m = vals.notna() & np.isfinite(vals)
            if m.sum() < max(20, int(n_rows * 0.3)):
                continue
            yy, vv = y[m.values], vals[m].values
            if len(np.unique(yy)) < 2:
                continue
            auc = roc_auc_score(yy, vv)
            auc_inv = roc_auc_score(yy, -vv)
            best = max(auc, auc_inv)
            direction = "pos" if auc >= auc_inv else "neg"
            try:
                _, pval = mannwhitneyu(vv[yy == 1], vv[yy == 0], alternative="two-sided")
            except Exception:
                pval = 1.0
            single.append({"feature": f, "best_auc": float(best), "direction": direction, "p_value": float(pval), "n": int(m.sum())})
        single.sort(key=lambda x: x["best_auc"], reverse=True)
        sig = [
            s
            for s in single
            if s["p_value"] < 0.05
            and not str(s["feature"]).startswith(("bind_", "pres_"))
            and s["feature"] != "nm_raw"
            and not is_leaky_feature_name(str(s["feature"]))
        ]

        hyps = [{"name": f"{pn}_binding_only", "features": [("bind_log50k", 1.0, False)]}]
        for sf in sig[:8]:
            feat = sf["feature"]
            for bw, fw in [(0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]:
                hyps.append({"name": f"{pn}_{feat}_{int(bw*100)}_{int(fw*100)}", "features": [("bind_log50k", bw, False), (feat, fw, "AUTO")]})
        if len(sig) >= 2:
            f1, f2 = sig[:2]
            hyps.append({"name": f"{pn}_top2_bind", "features": [("bind_log50k", 0.5, False), (f1["feature"], 0.25, "AUTO"), (f2["feature"], 0.25, "AUTO")]})
        if len(sig) >= 3:
            f1, f2, f3 = sig[:3]
            hyps.append({"name": f"{pn}_top3_bind", "features": [("bind_log50k", 0.4, False), (f1["feature"], 0.2, "AUTO"), (f2["feature"], 0.2, "AUTO"), (f3["feature"], 0.2, "AUTO")]})
        if "expr_raw" in df.columns and pd.to_numeric(df["expr_raw"], errors="coerce").notna().sum() > 10:
            hyps.append({"name": f"{pn}_bind_expr_60_40", "features": [("bind_log50k", 0.6, False), ("expr_raw", 0.4, False)]})
            hyps.append({"name": f"{pn}_bind_expr_80_20", "features": [("bind_log50k", 0.8, False), ("expr_raw", 0.2, False)]})
        if "calis_immuno" in df.columns:
            hyps.append({"name": f"{pn}_bind_calis", "features": [("bind_log50k", 0.6, False), ("calis_immuno", 0.4, False)]})

        can_lopo = n_pat >= 3
        hyp_res = {}
        for h in hyps:
            resolved_full = resolve_auto_directions(df, h["features"])
            is_auc = insample_auc(df, resolved_full)
            lopo, per = lopo_auc_with_train_fold_directions(df, h["features"]) if can_lopo else (None, {})
            hyp_res[h["name"]] = {"lopo_auc": lopo, "insample_auc": is_auc, "features": h["features"], "resolved_full_features": resolved_full, "per_patient": per}

        if can_lopo and len(sig) >= 1:
            lr_feats = [
                "bind_log50k"
            ] + [
                s["feature"]
                for s in sig[:5]
                if s["feature"] in df.columns and not is_leaky_feature_name(str(s["feature"]))
            ]
            lr_feats = [f for f in lr_feats if f in df.columns]
            if len(lr_feats) >= 2:
                lr_pp = {}
                for tp in df["patient_id"].astype(str).unique():
                    tr = df[df["patient_id"].astype(str) != tp]
                    te = df[df["patient_id"].astype(str) == tp]
                    ytr = tr["immunogenic"].values.astype(float)
                    yte = te["immunogenic"].values.astype(float)
                    if len(np.unique(yte)) < 2:
                        lr_pp[tp] = None
                        continue
                    xtr = tr[lr_feats].fillna(0).values
                    xte = te[lr_feats].fillna(0).values
                    try:
                        sc = StandardScaler()
                        xtr = sc.fit_transform(xtr)
                        xte = sc.transform(xte)
                        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
                        lr.fit(xtr, ytr)
                        pr = lr.predict_proba(xte)[:, 1]
                        lr_pp[tp] = float(roc_auc_score(yte, pr))
                    except Exception:
                        lr_pp[tp] = None
                comp = [v for v in lr_pp.values() if v is not None]
                hyp_res[f"{pn}_train_fold_lr"] = {"lopo_auc": (float(np.mean(comp)) if comp else None), "insample_auc": None, "features": lr_feats, "per_patient": lr_pp}

        sorted_h = sorted(hyp_res.items(), key=lambda x: (x[1]["lopo_auc"] if x[1]["lopo_auc"] is not None else x[1]["insample_auc"] if x[1]["insample_auc"] is not None else -1), reverse=True)
        baseline = hyp_res.get(f"{pn}_binding_only", {}).get("lopo_auc")
        if baseline is None:
            baseline = hyp_res.get(f"{pn}_binding_only", {}).get("insample_auc", 0)
        best_name = sorted_h[0][0] if sorted_h else None
        best_auc = (sorted_h[0][1]["lopo_auc"] if sorted_h and sorted_h[0][1]["lopo_auc"] is not None else sorted_h[0][1]["insample_auc"] if sorted_h else None)
        best_delta = (best_auc - baseline) if (best_auc is not None and baseline is not None) else None
        print(f"  Best: {best_name} AUC={best_auc:.4f} delta={best_delta:+.4f}" if best_auc is not None else "  Best: N/A")

        discovery_results[pn] = {
            "n_rows": n_rows,
            "n_immunogenic": n_pos,
            "n_patients": n_pat,
            "can_lopo": can_lopo,
            "eval_type": "LOPO" if can_lopo else "IN-SAMPLE (no LOPO)",
            "binding_baseline": baseline,
            "single_feature_top10": single[:10],
            "significant_non_binding": [{"feature": s["feature"], "auc": s["best_auc"], "p": s["p_value"], "direction": s["direction"]} for s in sig[:10]],
            "hypotheses": {k: {"lopo_auc": v["lopo_auc"], "insample_auc": v["insample_auc"], "features": v.get("features")} for k, v in hyp_res.items()},
            "best_strategy": best_name,
            "best_auc": best_auc,
            "best_delta": best_delta,
        }

    print("\n" + "=" * 70)
    print("CROSS MATRIX")
    print("=" * 70)
    all_papers = list(transfer_results.keys())
    header = f"{'Strategy':<35}" + "".join([f" {p[:12]:>12}" for p in all_papers]) + f" {'Mean':>8}"
    print(header)
    print("-" * len(header))
    for sn in TRANSFER_STRATEGIES.keys():
        row = f"{sn:<35}"
        aucs = []
        for pn in all_papers:
            r = transfer_results[pn]["strategies"].get(sn, {})
            a = r.get("lopo_auc") if r else None
            if a is None:
                a = r.get("insample_auc") if r else None
            if a is None:
                row += f" {'N/A':>12}"
            else:
                aucs.append(a)
                row += f" {a:>12.4f}"
        row += f" {(f'{np.mean(aucs):.4f}' if aucs else 'N/A'):>8}"
        print(row)

    repertoire = {
        "rl_binding_only": {
            "description": "MHC binding affinity only",
            "type": "universal",
            "features": {"bind_log50k": 1.0},
            "performance": {},
        },
        "rl_expression_v1": {
            "description": "Binding (dual) + Expression",
            "type": "cross-validated",
            "features": {"bind_log50k": 0.50, "nm_raw_inv": 0.25, "expression": 0.25},
            "performance": {},
        },
    }
    for pn in all_papers:
        rb = transfer_results[pn]["strategies"].get("binding_only", {})
        a = rb.get("lopo_auc") if rb else None
        if a is None:
            a = rb.get("insample_auc") if rb else None
        if a is not None:
            repertoire["rl_binding_only"]["performance"][pn] = a
        re = transfer_results[pn]["strategies"].get("rl_expression_v1_from_tesla", {})
        e = re.get("lopo_auc") if re else None
        if e is None:
            e = re.get("insample_auc") if re else None
        if e is not None:
            repertoire["rl_expression_v1"]["performance"][pn] = e

    for pn, d in discovery_results.items():
        if d.get("best_strategy") and d.get("best_delta") is not None and d["best_delta"] > 0.01:
            bs = d["best_strategy"]
            bh = d["hypotheses"].get(bs, {})
            repertoire[f"rl_{pn}_v1"] = {
                "description": f"Best strategy for {pn}",
                "type": "paper-specific",
                "features": bh.get("features", "see details"),
                "performance": {pn: d["best_auc"]},
                "delta_over_binding": d["best_delta"],
            }

    output = {
        "summary": {
            "papers_tested": all_papers,
            "transfer_strategies_tested": len(TRANSFER_STRATEGIES),
            "total_hypotheses_tested": int(sum(len(d.get("hypotheses", {})) for d in discovery_results.values() if isinstance(d, dict))),
            "evaluation_method": "LOPO where possible, in-sample otherwise",
        },
        "transfer_results": make_serializable(transfer_results),
        "discovery_results": make_serializable(discovery_results),
        "repertoire": make_serializable(repertoire),
    }
    (ARTIFACTS / "full_cross_validation_results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    rows = []
    for sn in TRANSFER_STRATEGIES.keys():
        r = {"strategy": sn}
        for pn in all_papers:
            rs = transfer_results[pn]["strategies"].get(sn, {})
            r[f"{pn}_lopo"] = rs.get("lopo_auc") if rs else None
            r[f"{pn}_insample"] = rs.get("insample_auc") if rs else None
        rows.append(r)
    for pn, d in discovery_results.items():
        if d.get("best_strategy"):
            r = {"strategy": f"BEST_OF_{pn}"}
            if d.get("can_lopo"):
                r[f"{pn}_lopo"] = d.get("best_auc")
            else:
                r[f"{pn}_insample"] = d.get("best_auc")
            rows.append(r)
    pd.DataFrame(rows).to_csv(ARTIFACTS / "full_cross_matrix.csv", index=False)
    (ARTIFACTS / "strategy_repertoire.json").write_text(json.dumps(make_serializable(repertoire), indent=2, default=str), encoding="utf-8")

    print("\nSaved:")
    print(f"  {ARTIFACTS / 'full_cross_validation_results.json'}")
    print(f"  {ARTIFACTS / 'full_cross_matrix.csv'}")
    print(f"  {ARTIFACTS / 'strategy_repertoire.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
