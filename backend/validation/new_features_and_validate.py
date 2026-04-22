from __future__ import annotations

import json
import warnings
from itertools import combinations
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


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < 10:
        return None
    yy = y[m]
    ss = s[m]
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, ss))


def tcr_positions(pep_len: int) -> list[int]:
    if pep_len == 8:
        return [2, 3, 4, 5, 6]
    if pep_len == 9:
        return [3, 4, 5, 6, 7]
    if pep_len == 10:
        return [3, 4, 5, 6, 7, 8]
    if pep_len == 11:
        return [3, 4, 5, 6, 7, 8, 9]
    start = 2
    end = max(start + 1, pep_len - 1)
    return list(range(start, end))


def hamming_prop(mut, wt):
    if pd.isna(mut) or pd.isna(wt):
        return np.nan
    mut = str(mut).strip().upper()
    wt = str(wt).strip().upper()
    if not mut or not wt:
        return np.nan
    n = min(len(mut), len(wt))
    if n == 0:
        return np.nan
    return sum(a != b for a, b in zip(mut[:n], wt[:n])) / n


def bind_log50k(nm: pd.Series) -> pd.Series:
    n = pd.to_numeric(nm, errors="coerce")
    return (1 - np.log10(n.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)


def bind_exp500(nm: pd.Series) -> pd.Series:
    n = pd.to_numeric(nm, errors="coerce")
    return np.exp(-n / 500)


def dynamic_blend(df: pd.DataFrame, features: list[str], weights: list[float]) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    vals = [pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float) for f in features]
    w = np.array(weights, dtype=float)
    for i in range(len(df)):
        num = 0.0
        den = 0.0
        for arr, ww in zip(vals, w):
            v = arr[i]
            if ww <= 0 or not np.isfinite(v):
                continue
            if v == 0:
                continue
            num += ww * v
            den += ww
        out[i] = (num / den) if den > 0 else 0.0
    return out


def normalize_hla(h):
    if pd.isna(h):
        return None
    s = str(h).strip().replace(" ", "")
    if not s or "UNK" in s.upper() or "UNKNOWN" in s.upper():
        return None
    if not s.startswith("HLA-"):
        s = "HLA-" + s
    return s


def main() -> int:
    print("=" * 70)
    print("NEW FEATURES + FULL REVALIDATION")
    print("=" * 70)

    tm_path = ARTIFACTS / "training_matrix.csv"
    if not tm_path.exists():
        raise FileNotFoundError(f"Missing: {tm_path}")

    tm = pd.read_csv(tm_path)
    ott_all = tm[tm["paper_source"] == "ott_2017"].copy()
    ott = ott_all[ott_all["immunogenic"].isin([0, 1])].copy()

    print(f"Peptide-level Ott rows: {len(ott)}")
    print(f"Immunogenic peptide rows: {int(pd.to_numeric(ott['immunogenic'], errors='coerce').fillna(0).sum())}")
    print(f"Patients: {sorted(ott['patient_id'].astype(str).unique().tolist())}")

    mutation_level = (
        ott.groupby(["patient_id", "gene", "protein_change"], dropna=False)
        .agg(
            {
                "immunogenic": "max",
                "expression_tpm_numeric": "first",
                "expression_score": "first",
                "presentation_score": "max",
                "binding_affinity_nm_numeric": "min",
                "final_binding_affinity_nm": "min",
                "predicted_affinity_nm": "min",
                "self_dissimilarity": "mean",
                "mutant_peptide": "first",
                "wildtype_peptide": "first",
                "hla_allele": "first",
                "final_hla_allele": "first",
                "peptide_length": "first",
                "cancer_type": "first",
            }
        )
        .reset_index()
    )
    mutation_level = mutation_level[mutation_level["immunogenic"].isin([0, 1])].copy()

    num_cols = [
        "immunogenic",
        "expression_tpm_numeric",
        "expression_score",
        "presentation_score",
        "binding_affinity_nm_numeric",
        "final_binding_affinity_nm",
        "predicted_affinity_nm",
        "self_dissimilarity",
    ]
    for c in num_cols:
        if c in mutation_level.columns:
            mutation_level[c] = pd.to_numeric(mutation_level[c], errors="coerce")
    peptide_df = ott.copy()
    for c in ["binding_affinity_nm_numeric", "expression_tpm_numeric", "expression_score", "presentation_score", "self_dissimilarity"]:
        if c in peptide_df.columns:
            peptide_df[c] = pd.to_numeric(peptide_df[c], errors="coerce")

    print(f"Mutation-level rows: {len(mutation_level)} | positives: {int(mutation_level['immunogenic'].sum())}")

    feature_status: dict[str, bool] = {}

    print("\n" + "=" * 70)
    print("PHASE 1: FEATURE CONSTRUCTION")
    print("=" * 70)

    # Feature 1: DAI (mut vs wt affinity)
    print("\n[1] DAI")
    peptide_df["dai"] = np.nan
    try:
        from mhcflurry import Class1AffinityPredictor  # type: ignore

        predictor = Class1AffinityPredictor.load()
        cache: dict[tuple[str, str], float] = {}

        def predict_nm(pep: str, hla: str) -> float:
            key = (pep, hla)
            if key in cache:
                return cache[key]
            v = float(predictor.predict(peptides=[pep], alleles=[hla])[0])
            cache[key] = v
            return v

        dai_vals = []
        ok = 0
        for _, r in peptide_df.iterrows():
            mut = str(r.get("mutant_peptide", "")).strip().upper()
            wt = str(r.get("wildtype_peptide", "")).strip().upper()
            hla = normalize_hla(r.get("final_hla_allele", r.get("hla_allele")))
            if not mut or not wt or hla is None:
                dai_vals.append(np.nan)
                continue
            try:
                mut_nm = predict_nm(mut, hla)
                wt_nm = predict_nm(wt, hla)
                if mut_nm > 0 and wt_nm > 0:
                    dai_vals.append(float(np.log10(wt_nm) - np.log10(mut_nm)))
                    ok += 1
                else:
                    dai_vals.append(np.nan)
            except Exception:
                dai_vals.append(np.nan)
        peptide_df["dai"] = dai_vals
        feature_status["dai"] = ok > 0
        print(f"Computed DAI rows: {ok}/{len(peptide_df)}")
    except Exception as e:
        feature_status["dai"] = False
        print(f"Skipped DAI (mhcflurry unavailable/failed): {e}")

    # Feature 2: IEDB-like immunogenicity
    print("\n[2] IEDB-like TCR recognition score")
    calis = {
        "A": -0.02,
        "C": 0.08,
        "D": -0.19,
        "E": -0.19,
        "F": 0.12,
        "G": -0.04,
        "H": 0.10,
        "I": 0.05,
        "K": -0.20,
        "L": 0.06,
        "M": 0.05,
        "N": -0.10,
        "P": -0.06,
        "Q": -0.10,
        "R": -0.12,
        "S": -0.04,
        "T": -0.03,
        "V": 0.02,
        "W": 0.13,
        "Y": 0.09,
    }

    def iedb_like(pep):
        if pd.isna(pep):
            return np.nan
        s = str(pep).strip().upper()
        if not s:
            return np.nan
        pos = tcr_positions(len(s))
        vals = [calis[a] for i, a in enumerate(s) if i in pos and a in calis]
        if not vals:
            return np.nan
        return float(np.mean(vals))

    peptide_df["iedb_immuno"] = peptide_df["mutant_peptide"].apply(iedb_like)
    peptide_df["iedb_immuno_wt"] = peptide_df["wildtype_peptide"].apply(iedb_like)
    peptide_df["iedb_immuno_diff"] = peptide_df["iedb_immuno"] - peptide_df["iedb_immuno_wt"]
    feature_status["iedb_immuno"] = peptide_df["iedb_immuno"].notna().any()
    print(f"Computed IEDB-like rows: {int(peptide_df['iedb_immuno'].notna().sum())}/{len(peptide_df)}")

    # Feature 3: Proteome foreignness (best-effort)
    print("\n[3] Self-proteome foreignness")
    peptide_df["proteome_foreignness"] = np.nan
    peptide_df["proteome_foreignness_wt"] = np.nan
    peptide_df["proteome_foreignness_diff"] = np.nan

    # first: local self library from wildtype peptides
    self_lib = {
        str(x).strip().upper()
        for x in peptide_df["wildtype_peptide"].dropna().astype(str).tolist()
        if str(x).strip()
    }

    def nearest_hamming_foreignness(pep: str, library: list[str], early_stop: int = 1) -> float:
        if not pep:
            return np.nan
        p = pep.upper()
        cands = [x for x in library if len(x) == len(p)]
        if not cands:
            return np.nan
        if p in cands:
            return 0.0
        best = len(p)
        for s in cands:
            h = sum(a != b for a, b in zip(p, s))
            if h < best:
                best = h
                if best <= early_stop:
                    break
        return best / len(p)

    self_lib_list = list(self_lib)
    peptide_df["proteome_foreignness"] = peptide_df["mutant_peptide"].apply(
        lambda x: nearest_hamming_foreignness(str(x).strip().upper() if pd.notna(x) else "", self_lib_list)
    )
    peptide_df["proteome_foreignness_wt"] = peptide_df["wildtype_peptide"].apply(
        lambda x: nearest_hamming_foreignness(str(x).strip().upper() if pd.notna(x) else "", self_lib_list)
    )
    peptide_df["proteome_foreignness_diff"] = (
        peptide_df["proteome_foreignness"] - peptide_df["proteome_foreignness_wt"]
    )
    feature_status["proteome_foreignness"] = peptide_df["proteome_foreignness"].notna().any()
    print(f"Computed foreignness rows: {int(peptide_df['proteome_foreignness'].notna().sum())}/{len(peptide_df)}")

    # Feature 4: TCR-facing residue properties
    print("\n[4] TCR-facing physicochemical properties")
    aa_hydro = {
        "A": 1.8,
        "R": -4.5,
        "N": -3.5,
        "D": -3.5,
        "C": 2.5,
        "Q": -3.5,
        "E": -3.5,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "L": 3.8,
        "K": -3.9,
        "M": 1.9,
        "F": 2.8,
        "P": -1.6,
        "S": -0.8,
        "T": -0.7,
        "W": -0.9,
        "Y": -1.3,
        "V": 4.2,
    }
    aa_vol = {
        "A": 88.6,
        "R": 173.4,
        "N": 114.1,
        "D": 111.1,
        "C": 108.5,
        "Q": 143.8,
        "E": 138.4,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "M": 162.9,
        "F": 189.9,
        "P": 112.7,
        "S": 89.0,
        "T": 116.1,
        "W": 227.8,
        "Y": 193.6,
        "V": 140.0,
    }
    aa_arom = {"F": 1.0, "W": 1.0, "Y": 1.0, "H": 0.5}

    def tcr_props(pep):
        if pd.isna(pep):
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        s = str(pep).strip().upper()
        if not s:
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        pos = tcr_positions(len(s))
        hs, vs, ars, cs = [], [], [], []
        for i in pos:
            if i >= len(s):
                continue
            aa = s[i]
            hs.append(aa_hydro.get(aa, 0.0))
            vs.append(aa_vol.get(aa, 100.0))
            ars.append(aa_arom.get(aa, 0.0))
            c = 0.0
            if aa in ("R", "K"):
                c = 1.0
            elif aa == "H":
                c = 0.5
            elif aa in ("D", "E"):
                c = -1.0
            cs.append(c)
        if not hs:
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        return {
            "tcr_hydrophobicity": float(np.mean(hs)),
            "tcr_volume": float(np.mean(vs)),
            "tcr_aromaticity": float(np.mean(ars)),
            "tcr_charge": float(np.mean(cs)),
        }

    mut_props = pd.DataFrame(peptide_df["mutant_peptide"].apply(tcr_props).tolist(), index=peptide_df.index)
    wt_props = pd.DataFrame(peptide_df["wildtype_peptide"].apply(tcr_props).tolist(), index=peptide_df.index)
    for c in ["tcr_hydrophobicity", "tcr_volume", "tcr_aromaticity", "tcr_charge"]:
        peptide_df[f"{c}_mut"] = mut_props[c]
        peptide_df[f"{c}_wt"] = wt_props[c]
        peptide_df[f"{c}_diff"] = peptide_df[f"{c}_mut"] - peptide_df[f"{c}_wt"]
        peptide_df[f"{c}_abs_diff"] = peptide_df[f"{c}_diff"].abs()

    peptide_df["tcr_surface_change"] = (
        peptide_df["tcr_hydrophobicity_abs_diff"].fillna(0) / 9.0
        + peptide_df["tcr_volume_abs_diff"].fillna(0) / 167.0
        + peptide_df["tcr_aromaticity_abs_diff"].fillna(0)
        + peptide_df["tcr_charge_abs_diff"].fillna(0) / 2.0
    )
    feature_status["tcr_properties"] = True
    print("Computed TCR property features")

    # Feature 5: anchor quality
    print("\n[5] Anchor residue quality")
    anchor_prefs = {
        "A*02": {1: {"L": 1.0, "M": 0.9, "V": 0.8, "I": 0.8}, -1: {"V": 1.0, "L": 0.9, "I": 0.8}},
        "A*24": {1: {"Y": 1.0, "F": 0.9, "W": 0.8}, -1: {"F": 1.0, "L": 0.9, "I": 0.8}},
        "B*07": {1: {"P": 1.0}, -1: {"L": 1.0, "M": 0.8}},
    }

    def anchor_quality(pep, hla):
        if pd.isna(pep) or pd.isna(hla):
            return np.nan
        p = str(pep).strip().upper()
        h = str(hla).upper().replace("HLA-", "")
        if len(p) < 8:
            return np.nan
        pref = None
        for k, v in anchor_prefs.items():
            if k in h:
                pref = v
                break
        if pref is None:
            hyd = set("LIVMFWY")
            return float((p[1] in hyd) * 0.5 + (p[-1] in hyd) * 0.5)
        s = 0.0
        n = 0
        for pos, d in pref.items():
            aa = p[pos]
            s += d.get(aa, 0.1)
            n += 1
        return s / n if n > 0 else np.nan

    peptide_df["anchor_quality"] = peptide_df.apply(
        lambda r: anchor_quality(r.get("mutant_peptide"), r.get("final_hla_allele", r.get("hla_allele"))), axis=1
    )
    peptide_df["anchor_quality_wt"] = peptide_df.apply(
        lambda r: anchor_quality(r.get("wildtype_peptide"), r.get("final_hla_allele", r.get("hla_allele"))), axis=1
    )
    peptide_df["anchor_quality_diff"] = peptide_df["anchor_quality"] - peptide_df["anchor_quality_wt"]
    feature_status["anchor_quality"] = True
    print(f"Computed anchor quality rows: {int(peptide_df['anchor_quality'].notna().sum())}/{len(peptide_df)}")

    # Feature 6: mutation position descriptors
    print("\n[6] Mutation position features")

    def mutation_pos_features(mut, wt):
        if pd.isna(mut) or pd.isna(wt):
            return {"mut_at_tcr_contact": np.nan, "mut_at_anchor": np.nan, "mut_position_centrality": np.nan, "n_mutations": np.nan}
        m = str(mut).strip().upper()
        w = str(wt).strip().upper()
        if not m or not w:
            return {"mut_at_tcr_contact": np.nan, "mut_at_anchor": np.nan, "mut_position_centrality": np.nan, "n_mutations": np.nan}
        n = min(len(m), len(w))
        if n < 8:
            return {"mut_at_tcr_contact": np.nan, "mut_at_anchor": np.nan, "mut_position_centrality": np.nan, "n_mutations": np.nan}
        muts = [i for i in range(n) if m[i] != w[i]]
        if not muts:
            return {"mut_at_tcr_contact": 0.0, "mut_at_anchor": 0.0, "mut_position_centrality": 0.0, "n_mutations": 0.0}
        tcr = set(tcr_positions(n))
        anchors = {1, n - 1}
        center = (n - 1) / 2.0
        at_tcr = sum(1 for i in muts if i in tcr) / len(muts)
        at_anchor = sum(1 for i in muts if i in anchors) / len(muts)
        centrality = []
        for i in muts:
            if i in tcr:
                dist = abs(i - center) / center if center > 0 else 0.0
                centrality.append(1.0 - dist)
        cval = float(np.mean(centrality)) if centrality else 0.0
        return {
            "mut_at_tcr_contact": float(at_tcr),
            "mut_at_anchor": float(at_anchor),
            "mut_position_centrality": cval,
            "n_mutations": float(len(muts)),
        }

    mpos = pd.DataFrame(peptide_df.apply(lambda r: mutation_pos_features(r.get("mutant_peptide"), r.get("wildtype_peptide")), axis=1).tolist(), index=peptide_df.index)
    for c in mpos.columns:
        peptide_df[c] = mpos[c]
    feature_status["mutation_position"] = True
    print("Computed mutation position features")

    # Feature 7: hamming dissimilarity
    print("\n[7] Hamming dissimilarity")
    peptide_df["hamming_distance"] = peptide_df.apply(lambda r: hamming_prop(r.get("mutant_peptide"), r.get("wildtype_peptide")), axis=1)
    feature_status["hamming_distance"] = True
    print(f"Computed hamming rows: {int(peptide_df['hamming_distance'].notna().sum())}/{len(peptide_df)}")

    # Merge peptide-aggregated features to mutation-level
    agg_cols_mean = [
        "dai",
        "iedb_immuno",
        "iedb_immuno_wt",
        "iedb_immuno_diff",
        "proteome_foreignness",
        "proteome_foreignness_wt",
        "proteome_foreignness_diff",
        "tcr_hydrophobicity_mut",
        "tcr_hydrophobicity_diff",
        "tcr_hydrophobicity_abs_diff",
        "tcr_volume_mut",
        "tcr_volume_diff",
        "tcr_volume_abs_diff",
        "tcr_aromaticity_mut",
        "tcr_aromaticity_diff",
        "tcr_charge_mut",
        "tcr_charge_diff",
        "tcr_charge_abs_diff",
        "tcr_surface_change",
        "mut_at_tcr_contact",
        "mut_at_anchor",
        "mut_position_centrality",
        "n_mutations",
        "hamming_distance",
    ]
    agg_cols_max = ["anchor_quality", "anchor_quality_wt", "anchor_quality_diff"]

    by = ["patient_id", "gene", "protein_change"]
    avail_mean = [c for c in agg_cols_mean if c in peptide_df.columns]
    avail_max = [c for c in agg_cols_max if c in peptide_df.columns]

    if avail_mean:
        gmean = peptide_df.groupby(by, dropna=False)[avail_mean].mean().reset_index()
        mutation_level = mutation_level.merge(gmean, on=by, how="left")
    if avail_max:
        gmax = peptide_df.groupby(by, dropna=False)[avail_max].max().reset_index()
        mutation_level = mutation_level.merge(gmax, on=by, how="left")

    out_aug = ARTIFACTS / "ott_mutation_level_with_new_features.csv"
    mutation_level.to_csv(out_aug, index=False)
    print(f"\nSaved augmented mutation table: {out_aug}")

    print("\n" + "=" * 70)
    print("PHASE 2: SINGLE-FEATURE AUC")
    print("=" * 70)

    y = mutation_level["immunogenic"].to_numpy(dtype=float)

    all_features = [
        "presentation_score",
        "expression_score",
        "self_dissimilarity",
        "dai",
        "iedb_immuno",
        "iedb_immuno_diff",
        "proteome_foreignness",
        "proteome_foreignness_diff",
        "tcr_hydrophobicity_mut",
        "tcr_hydrophobicity_diff",
        "tcr_hydrophobicity_abs_diff",
        "tcr_volume_mut",
        "tcr_volume_diff",
        "tcr_volume_abs_diff",
        "tcr_aromaticity_mut",
        "tcr_aromaticity_diff",
        "tcr_charge_mut",
        "tcr_charge_diff",
        "tcr_charge_abs_diff",
        "tcr_surface_change",
        "anchor_quality",
        "anchor_quality_diff",
        "mut_at_tcr_contact",
        "mut_position_centrality",
        "hamming_distance",
    ]

    feature_results = []
    print(f"{'Feature':<32} {'AUC':>7} {'N':>4} {'P':>10} {'Dir':>9}")
    print("-" * 70)
    for f in all_features:
        if f not in mutation_level.columns:
            continue
        v = pd.to_numeric(mutation_level[f], errors="coerce")
        m = np.isfinite(v.to_numpy(dtype=float))
        if m.sum() < 15:
            continue
        yy = y[m]
        vv = v.to_numpy(dtype=float)[m]
        if len(np.unique(yy)) < 2:
            continue
        auc = roc_auc_score(yy, vv)
        auc_inv = roc_auc_score(yy, -vv)
        best = float(max(auc, auc_inv))
        direction = "positive" if auc >= auc_inv else "inverted"
        iv = vv[yy == 1]
        nv = vv[yy == 0]
        p = 1.0
        if len(iv) > 0 and len(nv) > 0:
            p = float(mannwhitneyu(iv, nv, alternative="two-sided").pvalue)
        feature_results.append(
            {
                "feature": f,
                "auc": best,
                "auc_raw": float(auc),
                "auc_inverted": float(auc_inv),
                "n": int(m.sum()),
                "p_value": p,
                "direction": direction,
                "mean_immunogenic": float(np.mean(iv)) if len(iv) else np.nan,
                "mean_non_immunogenic": float(np.mean(nv)) if len(nv) else np.nan,
            }
        )
        print(f"{f:<32} {best:>7.4f} {int(m.sum()):>4} {p:>10.3e} {direction:>9}")

    feature_results.sort(key=lambda x: x["auc"], reverse=True)

    print("\n" + "=" * 70)
    print("PHASE 3: STRATEGY SWEEP")
    print("=" * 70)

    mutation_level["bind_log50k"] = bind_log50k(mutation_level["binding_affinity_nm_numeric"])
    mutation_level["bind_exp500"] = bind_exp500(mutation_level["binding_affinity_nm_numeric"])

    strategies = []

    baseline_auc = safe_auc(y, mutation_level["bind_log50k"].to_numpy(dtype=float))
    strategies.append({"name": "binding_only_log50k", "auc": baseline_auc, "kind": "baseline", "config": {"features": ["bind_log50k"], "weights": [1.0]}})

    # RL-v1 style reference
    rl_features = ["bind_log50k", "expression_score", "self_dissimilarity"]
    rl_weights = [0.50, 0.33, 0.17]
    rl_scores = dynamic_blend(mutation_level, rl_features, rl_weights)
    strategies.append({"name": "rl_v1_ref", "auc": safe_auc(y, rl_scores), "kind": "fixed", "config": {"features": rl_features, "weights": rl_weights}})

    # Binding + one extra feature (weight scan)
    candidates = [
        "expression_score",
        "hamming_distance",
        "dai",
        "iedb_immuno",
        "iedb_immuno_diff",
        "proteome_foreignness",
        "proteome_foreignness_diff",
        "tcr_surface_change",
        "anchor_quality",
        "anchor_quality_diff",
        "mut_at_tcr_contact",
        "mut_position_centrality",
        "self_dissimilarity",
    ]

    for bind_col in ["bind_log50k", "bind_exp500"]:
        for feat in candidates:
            if feat not in mutation_level.columns:
                continue
            best_auc = -1.0
            best_w = None
            for w in np.arange(0.0, 1.0001, 0.05):
                ws = [float(w), float(1.0 - w)]
                s = dynamic_blend(mutation_level, [bind_col, feat], ws)
                a = safe_auc(y, s)
                if a is None:
                    continue
                if a > best_auc:
                    best_auc = a
                    best_w = ws
            if best_w is not None:
                strategies.append(
                    {
                        "name": f"{bind_col}+{feat}",
                        "auc": float(best_auc),
                        "kind": "pair_scan",
                        "config": {"features": [bind_col, feat], "weights": best_w},
                    }
                )

    # Logistic models on top non-binding features + binding
    top_non_binding = [x["feature"] for x in feature_results if x["feature"] != "presentation_score"][:8]
    lr_pool = ["bind_log50k"] + [f for f in top_non_binding if f in mutation_level.columns]
    for r in [1, 2, 3]:
        for combo in combinations(lr_pool, r):
            X = np.column_stack([pd.to_numeric(mutation_level[c], errors="coerce").fillna(0).to_numpy(dtype=float) for c in combo])
            mask = (X != 0).any(axis=1)
            if mask.sum() < 20:
                continue
            yy = y[mask]
            if len(np.unique(yy)) < 2:
                continue
            XX = X[mask]
            try:
                sc = StandardScaler()
                XXs = sc.fit_transform(XX)
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
                lr.fit(XXs, yy)
                prob = lr.predict_proba(XXs)[:, 1]
                a = float(roc_auc_score(yy, prob))
                strategies.append(
                    {
                        "name": "lr:" + "+".join(combo),
                        "auc": a,
                        "kind": "lr_in_sample",
                        "config": {"features": list(combo), "coef": [float(x) for x in lr.coef_[0].tolist()]},
                    }
                )
            except Exception:
                pass

    strategies = [s for s in strategies if s.get("auc") is not None and np.isfinite(s.get("auc"))]
    strategies.sort(key=lambda x: x["auc"], reverse=True)

    print("Top 10 in-sample strategies:")
    for i, s in enumerate(strategies[:10], start=1):
        d = (s["auc"] - baseline_auc) if baseline_auc is not None else np.nan
        print(f"  {i:>2}. {s['name']:<45} AUC={s['auc']:.4f} delta={d:+.4f}")

    print("\n" + "=" * 70)
    print("PHASE 4: NESTED LOPO CV (HONEST)")
    print("=" * 70)

    patients = sorted(mutation_level["patient_id"].astype(str).unique().tolist())

    # fixed strategies for fair comparison
    fixed_configs = {
        "binding_only": {"features": ["bind_log50k"], "weights": [1.0]},
        "rl_v1_ref": {"features": ["bind_log50k", "expression_score", "self_dissimilarity"], "weights": [0.50, 0.33, 0.17]},
        "equal3_ref": {"features": ["bind_log50k", "expression_score", "hamming_distance"], "weights": [1 / 3, 1 / 3, 1 / 3]},
    }

    sweep_file = ARTIFACTS / "feature_sweep_results.json"
    sweep_w = [0.77, 0.15, 0.08]
    if sweep_file.exists():
        try:
            sweep = json.loads(sweep_file.read_text(encoding="utf-8"))
            bw = sweep.get("best_combo", {}).get("weights")
            if isinstance(bw, list) and len(bw) == 3:
                sweep_w = [float(bw[0]), float(bw[1]), float(bw[2])]
        except Exception:
            pass
    fixed_configs["sweep_winner_ref"] = {
        "features": ["bind_exp500", "expression_score", "hamming_distance"],
        "weights": sweep_w,
    }

    cv_rows = []
    nested_selected = []

    # candidate features for fold-internal optimization
    fold_candidate_feats = [
        "expression_score",
        "hamming_distance",
        "iedb_immuno",
        "iedb_immuno_diff",
        "tcr_surface_change",
        "anchor_quality",
        "mut_position_centrality",
        "proteome_foreignness",
        "dai",
        "self_dissimilarity",
    ]
    fold_candidate_feats = [f for f in fold_candidate_feats if f in mutation_level.columns]

    for i, p in enumerate(patients, start=1):
        print(f"Fold {i}/{len(patients)}: test patient {p}")
        test_mask = mutation_level["patient_id"].astype(str) == p
        tr = mutation_level[~test_mask].copy()
        te = mutation_level[test_mask].copy()

        y_tr = tr["immunogenic"].to_numpy(dtype=float)
        y_te = te["immunogenic"].to_numpy(dtype=float)

        row = {
            "patient": p,
            "n": int(len(te)),
            "n_pos": int(y_te.sum()),
            "status": "computed",
        }

        if len(np.unique(y_te)) < 2:
            row["status"] = "skipped_no_both_classes"
            cv_rows.append(row)
            continue

        # fixed configs eval
        for name, cfg in fixed_configs.items():
            s_te = dynamic_blend(te, cfg["features"], cfg["weights"])
            row[f"auc_{name}"] = safe_auc(y_te, s_te)

        # inner optimize: bind + one feature with scanned weight on training only
        best_inner = {"auc": -1.0, "bind": None, "feat": None, "w": None}
        for bcol in ["bind_log50k", "bind_exp500"]:
            for f in fold_candidate_feats:
                for w in np.arange(0.0, 1.0001, 0.1):
                    ws = [float(w), float(1.0 - w)]
                    s_tr = dynamic_blend(tr, [bcol, f], ws)
                    a = safe_auc(y_tr, s_tr)
                    if a is None:
                        continue
                    if a > best_inner["auc"]:
                        best_inner = {"auc": float(a), "bind": bcol, "feat": f, "w": ws}

        if best_inner["bind"] is not None:
            s_te = dynamic_blend(te, [best_inner["bind"], best_inner["feat"]], best_inner["w"])
            row["auc_nested_ws"] = safe_auc(y_te, s_te)
            row["inner_train_auc_ws"] = best_inner["auc"]
            row["inner_choice"] = {
                "bind": best_inner["bind"],
                "feat": best_inner["feat"],
                "weights": best_inner["w"],
            }
            nested_selected.append(row["inner_choice"])
        else:
            row["auc_nested_ws"] = None
            row["inner_train_auc_ws"] = None
            row["inner_choice"] = None

        # nested lr: fit LR on top 3 train features by train single-feature auc
        train_auc_feat = []
        for f in fold_candidate_feats + ["bind_log50k", "bind_exp500"]:
            a = safe_auc(y_tr, pd.to_numeric(tr[f], errors="coerce").to_numpy(dtype=float))
            if a is not None:
                train_auc_feat.append((f, a))
        train_auc_feat.sort(key=lambda x: x[1], reverse=True)
        top_lr_feats = [x[0] for x in train_auc_feat[:3]]

        row["auc_nested_lr"] = None
        row["inner_train_auc_lr"] = None
        if len(top_lr_feats) >= 1:
            Xtr = np.column_stack([pd.to_numeric(tr[c], errors="coerce").fillna(0).to_numpy(dtype=float) for c in top_lr_feats])
            Xte = np.column_stack([pd.to_numeric(te[c], errors="coerce").fillna(0).to_numpy(dtype=float) for c in top_lr_feats])
            try:
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr)
                Xte_s = sc.transform(Xte)
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
                lr.fit(Xtr_s, y_tr)
                pr_tr = lr.predict_proba(Xtr_s)[:, 1]
                pr_te = lr.predict_proba(Xte_s)[:, 1]
                row["inner_train_auc_lr"] = safe_auc(y_tr, pr_tr)
                row["auc_nested_lr"] = safe_auc(y_te, pr_te)
                row["inner_lr_features"] = top_lr_feats
            except Exception:
                pass

        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)

    def mean_col(col: str) -> float | None:
        if col not in cv_df.columns:
            return None
        v = pd.to_numeric(cv_df[col], errors="coerce")
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return None
        return float(v.mean())

    means = {
        "binding_only": mean_col("auc_binding_only"),
        "rl_v1_ref": mean_col("auc_rl_v1_ref"),
        "equal3_ref": mean_col("auc_equal3_ref"),
        "sweep_winner_ref": mean_col("auc_sweep_winner_ref"),
        "nested_ws": mean_col("auc_nested_ws"),
        "nested_lr": mean_col("auc_nested_lr"),
    }

    baseline_mean = means["binding_only"]

    print("\nPer-patient nested CV summary:")
    cols_show = [
        "patient",
        "n",
        "n_pos",
        "auc_binding_only",
        "auc_rl_v1_ref",
        "auc_equal3_ref",
        "auc_sweep_winner_ref",
        "auc_nested_ws",
        "auc_nested_lr",
        "status",
    ]
    have_cols = [c for c in cols_show if c in cv_df.columns]
    print(cv_df[have_cols].to_string(index=False))

    print("\nMean AUCs:")
    for k, v in means.items():
        if v is None:
            print(f"  {k:<17} N/A")
        else:
            delta = (v - baseline_mean) if baseline_mean is not None else np.nan
            print(f"  {k:<17} {v:.4f}  delta={delta:+.4f}")

    # top5 strategy means
    strat_mean_map = {
        "binding_only": means["binding_only"],
        "rl_v1_ref": means["rl_v1_ref"],
        "equal3_ref": means["equal3_ref"],
        "sweep_winner_ref": means["sweep_winner_ref"],
        "nested_ws": means["nested_ws"],
        "nested_lr": means["nested_lr"],
    }
    top5 = sorted(
        [{"strategy": k, "mean_auc": v} for k, v in strat_mean_map.items() if v is not None],
        key=lambda x: x["mean_auc"],
        reverse=True,
    )[:5]

    print("\nTop 5 strategy scores (mean CV AUC):")
    for i, r in enumerate(top5, start=1):
        delta = (r["mean_auc"] - baseline_mean) if baseline_mean is not None else np.nan
        print(f"  {i}. {r['strategy']:<16} {r['mean_auc']:.4f}  delta={delta:+.4f}")

    # train-test gap for nested strategies
    train_ws = pd.to_numeric(cv_df.get("inner_train_auc_ws", pd.Series(dtype=float)), errors="coerce")
    test_ws = pd.to_numeric(cv_df.get("auc_nested_ws", pd.Series(dtype=float)), errors="coerce")
    ws_gap = None
    if np.isfinite(train_ws).any() and np.isfinite(test_ws).any():
        ws_gap = float(np.nanmean(train_ws) - np.nanmean(test_ws))

    train_lr = pd.to_numeric(cv_df.get("inner_train_auc_lr", pd.Series(dtype=float)), errors="coerce")
    test_lr = pd.to_numeric(cv_df.get("auc_nested_lr", pd.Series(dtype=float)), errors="coerce")
    lr_gap = None
    if np.isfinite(train_lr).any() and np.isfinite(test_lr).any():
        lr_gap = float(np.nanmean(train_lr) - np.nanmean(test_lr))

    results = {
        "dataset": {
            "peptide_rows": int(len(ott)),
            "mutation_rows": int(len(mutation_level)),
            "positives": int(mutation_level["immunogenic"].sum()),
            "patients": sorted(mutation_level["patient_id"].astype(str).unique().tolist()),
        },
        "feature_status": feature_status,
        "single_feature_results": feature_results,
        "top_single_features": feature_results[:15],
        "strategy_sweep_top20": strategies[:20],
        "nested_cv_per_patient": cv_rows,
        "nested_cv_means": means,
        "top5_strategy_scores": top5,
        "nested_train_test_gap": {
            "weighted_sum": ws_gap,
            "logistic": lr_gap,
        },
        "inner_selected_configs": nested_selected,
    }

    out_json = ARTIFACTS / "new_features_validation_results.json"
    out_cv = ARTIFACTS / "new_features_nested_cv_per_patient.csv"
    out_top = ARTIFACTS / "new_features_top_strategies.csv"

    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    cv_df.to_csv(out_cv, index=False)
    pd.DataFrame(strategies).head(100).to_csv(out_top, index=False)

    print("\nSaved artifacts:")
    print(f"  {out_json}")
    print(f"  {out_cv}")
    print(f"  {out_top}")
    print(f"  {out_aug}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
