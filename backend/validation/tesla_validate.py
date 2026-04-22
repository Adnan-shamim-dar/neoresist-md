from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "validation" / "artifacts"
RAW_DIR = ARTIFACTS / "tesla_raw"
TESLA_OUT = ROOT / "validation_papers" / "tesla_prepared.csv"
TESLA_RESULTS = ARTIFACTS / "tesla_validation_results.json"
CROSS_RESULTS = ARTIFACTS / "cross_dataset_validation.json"
ALL_PAPERS_RESULTS = ARTIFACTS / "all_papers_h14_results.json"
TARGETED_RESULTS = ARTIFACTS / "targeted_hypothesis_results.json"
TRAINING_MATRIX = ARTIFACTS / "training_matrix.csv"

TESLA_FILES = [f"1-s2.0-S0092867420311569-mmc{i}.xlsx" for i in range(1, 8)]
TESLA_BASE = "https://authors.library.caltech.edu/records/y57ap-h8p92/files"


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
    start = 2
    end = max(start + 1, pep_len - 1)
    return list(range(start, end))


def hamming_prop(mut: object, wt: object) -> float:
    if pd.isna(mut) or pd.isna(wt):
        return np.nan
    m = str(mut).strip().upper()
    w = str(wt).strip().upper()
    if not m or not w:
        return np.nan
    n = min(len(m), len(w))
    if n == 0:
        return np.nan
    return float(sum(a != b for a, b in zip(m[:n], w[:n])) / n)


def normalize_hla(h: object) -> str | None:
    if pd.isna(h):
        return None
    s = str(h).strip().replace(" ", "")
    if not s:
        return None
    if not s.startswith("HLA-"):
        s = f"HLA-{s}"
    return s


def parse_hla_from_pmhc(pmhc: object) -> str | None:
    if pd.isna(pmhc):
        return None
    s = str(pmhc)
    if "_" not in s:
        return None
    return normalize_hla(s.split("_", 1)[0])


def bind_log50k(nm: pd.Series) -> pd.Series:
    n = pd.to_numeric(nm, errors="coerce")
    return (1 - np.log10(n.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)


def score_hypothesis(df: pd.DataFrame, features: list[list[object]]) -> np.ndarray:
    scores = np.full(len(df), np.nan)
    for i in range(len(df)):
        row = df.iloc[i]
        vals: list[float] = []
        weights: list[float] = []
        for feat, weight, invert in features:
            if feat not in df.columns:
                continue
            v = row[feat]
            if pd.isna(v) or not np.isfinite(v):
                continue
            vv = -float(v) if bool(invert) else float(v)
            vals.append(vv)
            weights.append(float(weight))
        if not vals:
            scores[i] = 0.0
            continue
        total_w = float(sum(weights))
        scores[i] = 0.0 if total_w <= 0 else sum(v * (w / total_w) for v, w in zip(vals, weights))
    return scores


def ensure_tesla_files() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    import requests

    for fn in TESLA_FILES:
        out = RAW_DIR / fn
        if out.exists() and out.stat().st_size > 1024:
            continue
        url = f"{TESLA_BASE}/{fn}?download=1"
        r = requests.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < 1024:
            raise RuntimeError(f"Failed to download TESLA supplement: {url} (status={r.status_code})")
        out.write_bytes(r.content)


def load_tesla_tables() -> tuple[pd.DataFrame, dict[int, str]]:
    mmc3 = pd.read_excel(RAW_DIR / TESLA_FILES[2], sheet_name="PBMCs and TILs")
    patient_cancer = {
        int(r["Subject ID"]): str(r["Tumor type"]).strip().lower()
        for _, r in mmc3.iterrows()
        if pd.notna(r.get("Subject ID")) and pd.notna(r.get("Tumor type"))
    }

    m4 = pd.read_excel(RAW_DIR / TESLA_FILES[3], sheet_name="master-bindings-selected").copy()
    m4["source_table"] = "mmc4_master-bindings-selected"
    m4["PRED_BINDING"] = pd.to_numeric(m4["NETMHC_PAN_BINDING_AFFINITY"], errors="coerce")
    m4["HLA_RAW"] = m4["MHC"].astype(str)

    m7 = pd.read_excel(RAW_DIR / TESLA_FILES[6], sheet_name="SM_Table_S7_PEPTIDE_VALIDATION_").copy()
    m7["source_table"] = "mmc7_SM_Table_S7_PEPTIDE_VALIDATION_"
    m7["PRED_BINDING"] = pd.to_numeric(m7["PREDICTED_BINDING_AFFINITY"], errors="coerce")
    m7["HLA_RAW"] = m7["PMHC"].apply(parse_hla_from_pmhc)
    for col in ["TCR_FLOW_I", "TCR_FLOW_I_QUANT", "TCR_NANOPARTICLE", "NUMBER_PREDICTING", "MEASURED_BINDING_AFFINITY"]:
        if col not in m7.columns:
            m7[col] = np.nan

    common = sorted(set(m4.columns).union(set(m7.columns)))
    for c in common:
        if c not in m4.columns:
            m4[c] = np.nan
        if c not in m7.columns:
            m7[c] = np.nan

    tesla = pd.concat([m4[common], m7[common]], ignore_index=True)
    tesla["PATIENT_ID"] = pd.to_numeric(tesla["PATIENT_ID"], errors="coerce").astype("Int64")
    return tesla, patient_cancer


def build_tesla_prepared(tesla: pd.DataFrame, patient_cancer: dict[int, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["paper_source"] = "tesla_2020"
    out["patient_id"] = tesla["PATIENT_ID"].apply(lambda x: f"TESLA_{int(x)}" if pd.notna(x) else "TESLA_UNKNOWN")
    out["cancer_type"] = tesla["PATIENT_ID"].apply(
        lambda x: patient_cancer.get(int(x), "unknown") if pd.notna(x) else "unknown"
    )
    out["gene"] = "NOT_AVAILABLE"
    out["protein_change"] = "NOT_AVAILABLE"
    out["mutant_peptide"] = tesla["ALT_EPI_SEQ"].astype(str).str.strip().str.upper()
    out["wildtype_peptide"] = "NOT_AVAILABLE"
    out["peptide_length"] = pd.to_numeric(tesla["PEP_LEN"], errors="coerce")
    out["hla_allele"] = tesla["HLA_RAW"].apply(normalize_hla)
    out["binding_affinity_nm"] = pd.to_numeric(tesla["PRED_BINDING"], errors="coerce")
    out["expression_tpm"] = pd.to_numeric(tesla["TUMOR_ABUNDANCE"], errors="coerce")
    out["vaf"] = "NOT_AVAILABLE"
    out["immunogenic"] = (
        tesla["VALIDATED"]
        .map(lambda v: 1 if str(v).strip().lower() in {"true", "1"} or v is True else 0)
        .astype(int)
    )
    out["response_type"] = np.where(out["immunogenic"] == 1, "validated_positive", "validated_negative")
    out["detection_method"] = "TESLA tetramer validation (public supplementary)"
    out["usable_for_training"] = 1
    out["notes"] = tesla.apply(
        lambda r: f"{r.get('source_table','unknown')}; tissue={r.get('TISSUE_TYPE','NA')}; PMHC={r.get('PMHC','NA')}",
        axis=1,
    )

    # Keep extra TESLA assay metadata for reporting.
    out["tesla_tcr_flow_i"] = pd.to_numeric(tesla.get("TCR_FLOW_I"), errors="coerce")
    out["tesla_tcr_flow_i_quant"] = pd.to_numeric(tesla.get("TCR_FLOW_I_QUANT"), errors="coerce")
    out["tesla_tcr_nanoparticle"] = pd.to_numeric(tesla.get("TCR_NANOPARTICLE"), errors="coerce")
    out["tesla_tcr_flow_ii"] = pd.to_numeric(tesla.get("TCR_FLOW_II"), errors="coerce")
    out["tesla_tcr_flow_ii_quant"] = pd.to_numeric(tesla.get("TCR_FLOW_II_QUANT"), errors="coerce")
    return out


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    out["mutant_peptide"] = out["mutant_peptide"].astype(str).str.strip().str.upper()
    out["wildtype_peptide"] = (
        out["wildtype_peptide"]
        .map(lambda x: np.nan if pd.isna(x) or str(x).strip().upper() in {"", "NOT_AVAILABLE", "NA"} else str(x).strip().upper())
    )
    out["binding_affinity_nm_numeric"] = pd.to_numeric(out["binding_affinity_nm"], errors="coerce")
    out["expression_tpm_numeric"] = pd.to_numeric(out["expression_tpm"], errors="coerce")
    out["bind_log50k"] = bind_log50k(out["binding_affinity_nm_numeric"])

    # DAI (best-effort).
    out["dai"] = np.nan
    try:
        from mhcflurry import Class1AffinityPredictor  # type: ignore

        predictor = Class1AffinityPredictor.load()
        cache: dict[tuple[str, str], float] = {}

        def pred_nm(pep: str, hla: str) -> float:
            key = (pep, hla)
            if key in cache:
                return cache[key]
            val = float(predictor.predict(peptides=[pep], alleles=[hla])[0])
            cache[key] = val
            return val

        vals = []
        for _, r in out.iterrows():
            mut = str(r.get("mutant_peptide", "")).strip().upper()
            wt = str(r.get("wildtype_peptide", "")).strip().upper() if pd.notna(r.get("wildtype_peptide")) else ""
            hla = normalize_hla(r.get("hla_allele"))
            if not mut or not wt or hla is None:
                vals.append(np.nan)
                continue
            try:
                mut_nm = pred_nm(mut, hla)
                wt_nm = pred_nm(wt, hla)
                vals.append(float(np.log10(wt_nm) - np.log10(mut_nm)) if mut_nm > 0 and wt_nm > 0 else np.nan)
            except Exception:
                vals.append(np.nan)
        out["dai"] = vals
    except Exception:
        pass

    calis = {
        "A": -0.02, "C": 0.08, "D": -0.19, "E": -0.19, "F": 0.12, "G": -0.04, "H": 0.10,
        "I": 0.05, "K": -0.20, "L": 0.06, "M": 0.05, "N": -0.10, "P": -0.06, "Q": -0.10,
        "R": -0.12, "S": -0.04, "T": -0.03, "V": 0.02, "W": 0.13, "Y": 0.09,
    }

    def iedb_like(pep: object) -> float:
        if pd.isna(pep):
            return np.nan
        s = str(pep).strip().upper()
        if not s:
            return np.nan
        vals = [calis[a] for i, a in enumerate(s) if i in tcr_positions(len(s)) and a in calis]
        return float(np.mean(vals)) if vals else np.nan

    out["iedb_immuno"] = out["mutant_peptide"].apply(iedb_like)
    out["iedb_immuno_wt"] = out["wildtype_peptide"].apply(iedb_like)
    out["iedb_immuno_diff"] = out["iedb_immuno"] - out["iedb_immuno_wt"]

    # Foreignness: nearest hamming distance against observed wt peptide library.
    self_lib = [x for x in out["wildtype_peptide"].dropna().astype(str).tolist() if str(x).strip()]

    def nearest_hamming_foreignness(pep: object) -> float:
        if pd.isna(pep):
            return np.nan
        p = str(pep).strip().upper()
        cands = [x for x in self_lib if len(x) == len(p)]
        if not p or not cands:
            return np.nan
        if p in cands:
            return 0.0
        best = len(p)
        for s in cands:
            h = sum(a != b for a, b in zip(p, s))
            best = min(best, h)
        return float(best / len(p))

    out["proteome_foreignness"] = out["mutant_peptide"].apply(nearest_hamming_foreignness)
    out["proteome_foreignness_wt"] = out["wildtype_peptide"].apply(nearest_hamming_foreignness)
    out["proteome_foreignness_diff"] = out["proteome_foreignness"] - out["proteome_foreignness_wt"]

    aa_hydro = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
        "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
        "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    aa_vol = {
        "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
        "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
        "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
    }
    aa_arom = {"F": 1.0, "W": 1.0, "Y": 1.0, "H": 0.5}

    def tcr_props(pep: object) -> dict[str, float]:
        if pd.isna(pep):
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        s = str(pep).strip().upper()
        if not s:
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        hs: list[float] = []
        vs: list[float] = []
        ars: list[float] = []
        cs: list[float] = []
        for i in tcr_positions(len(s)):
            if i >= len(s):
                continue
            aa = s[i]
            hs.append(float(aa_hydro.get(aa, 0.0)))
            vs.append(float(aa_vol.get(aa, 100.0)))
            ars.append(float(aa_arom.get(aa, 0.0)))
            charge = 1.0 if aa in ("R", "K") else 0.5 if aa == "H" else -1.0 if aa in ("D", "E") else 0.0
            cs.append(charge)
        if not hs:
            return {"tcr_hydrophobicity": np.nan, "tcr_volume": np.nan, "tcr_aromaticity": np.nan, "tcr_charge": np.nan}
        return {
            "tcr_hydrophobicity": float(np.mean(hs)),
            "tcr_volume": float(np.mean(vs)),
            "tcr_aromaticity": float(np.mean(ars)),
            "tcr_charge": float(np.mean(cs)),
        }

    mut_props = pd.DataFrame(out["mutant_peptide"].apply(tcr_props).tolist(), index=out.index)
    wt_props = pd.DataFrame(out["wildtype_peptide"].apply(tcr_props).tolist(), index=out.index)
    for c in ["tcr_hydrophobicity", "tcr_volume", "tcr_aromaticity", "tcr_charge"]:
        out[f"{c}_mut"] = mut_props[c]
        out[f"{c}_wt"] = wt_props[c]
        out[f"{c}_diff"] = out[f"{c}_mut"] - out[f"{c}_wt"]
        out[f"{c}_abs_diff"] = out[f"{c}_diff"].abs()

    out["tcr_surface_change"] = (
        out["tcr_hydrophobicity_abs_diff"].fillna(0) / 9.0
        + out["tcr_volume_abs_diff"].fillna(0) / 167.0
        + out["tcr_aromaticity_abs_diff"].fillna(0)
        + out["tcr_charge_abs_diff"].fillna(0) / 2.0
    )

    anchor_prefs = {
        "A*02": {1: {"L": 1.0, "M": 0.9, "V": 0.8, "I": 0.8}, -1: {"V": 1.0, "L": 0.9, "I": 0.8}},
        "A*24": {1: {"Y": 1.0, "F": 0.9, "W": 0.8}, -1: {"F": 1.0, "L": 0.9, "I": 0.8}},
        "B*07": {1: {"P": 1.0}, -1: {"L": 1.0, "M": 0.8}},
    }

    def anchor_quality(pep: object, hla: object) -> float:
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
        vals = [pref[pos].get(p[pos], 0.1) for pos in pref]
        return float(np.mean(vals)) if vals else np.nan

    out["anchor_quality"] = out.apply(lambda r: anchor_quality(r.get("mutant_peptide"), r.get("hla_allele")), axis=1)
    out["anchor_quality_wt"] = out.apply(lambda r: anchor_quality(r.get("wildtype_peptide"), r.get("hla_allele")), axis=1)
    out["anchor_quality_diff"] = out["anchor_quality"] - out["anchor_quality_wt"]

    def mutation_pos_features(mut: object, wt: object) -> dict[str, float]:
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
        centrality = [1.0 - (abs(i - center) / center if center > 0 else 0.0) for i in muts if i in tcr]
        cval = float(np.mean(centrality)) if centrality else 0.0
        return {
            "mut_at_tcr_contact": float(at_tcr),
            "mut_at_anchor": float(at_anchor),
            "mut_position_centrality": cval,
            "n_mutations": float(len(muts)),
        }

    mpos = pd.DataFrame(out.apply(lambda r: mutation_pos_features(r.get("mutant_peptide"), r.get("wildtype_peptide")), axis=1).tolist(), index=out.index)
    for c in mpos.columns:
        out[c] = mpos[c]
    out["hamming_distance"] = out.apply(lambda r: hamming_prop(r.get("mutant_peptide"), r.get("wildtype_peptide")), axis=1)

    availability = {
        "binding_affinity_nm": int(out["binding_affinity_nm_numeric"].notna().sum()),
        "expression_tpm": int(out["expression_tpm_numeric"].notna().sum()),
        "mutant_peptide": int(out["mutant_peptide"].notna().sum()),
        "wildtype_peptide": int(out["wildtype_peptide"].notna().sum()),
        "hla_allele": int(out["hla_allele"].notna().sum()),
        "both_mut_and_wt": int((out["mutant_peptide"].notna() & out["wildtype_peptide"].notna()).sum()),
    }
    return out, availability


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float | None, list[float | None]]:
    auc = safe_auc(y, s)
    if auc is None:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = safe_auc(y[idx], s[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return auc, [None, None]
    return auc, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate_hypotheses(df: pd.DataFrame, hypothesis_features: dict[str, list[list[object]]]) -> dict[str, object]:
    y = pd.to_numeric(df["immunogenic"], errors="coerce").fillna(0).astype(float).values
    patients = sorted(df["patient_id"].astype(str).unique().tolist())

    all_scores: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, object]] = {}
    for name, feats in hypothesis_features.items():
        s = score_hypothesis(df, feats)
        all_scores[name] = s
        per_patient: dict[str, dict[str, object]] = {}
        patient_aucs: list[float] = []
        for pat in patients:
            sub = df["patient_id"].astype(str) == pat
            yy = y[sub.values]
            ss = s[sub.values]
            auc = safe_auc(yy, ss)
            per_patient[pat] = {
                "auc": auc,
                "n": int(sub.sum()),
                "n_pos": int(np.sum(yy == 1)),
            }
            if auc is not None:
                patient_aucs.append(auc)
        results[name] = {
            "in_sample_auc": safe_auc(y, s),
            "per_patient_auc": per_patient,
            "mean_per_patient_auc": float(np.mean(patient_aucs)) if patient_aucs else None,
        }

    h0 = all_scores["H0_binding_only"]
    h14 = all_scores["H14_bind_plus_3tcr"]
    h0_auc, h0_ci = bootstrap_auc(y, h0)
    h14_auc, h14_ci = bootstrap_auc(y, h14)

    rng = np.random.default_rng(123)
    deltas: list[float] = []
    n = len(y)
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        a0 = safe_auc(y[idx], h0[idx])
        a14 = safe_auc(y[idx], h14[idx])
        if a0 is not None and a14 is not None:
            deltas.append(a14 - a0)
    delta = (h14_auc - h0_auc) if (h14_auc is not None and h0_auc is not None) else None
    delta_ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))] if deltas else [None, None]

    lopo_summary = None
    if len(patients) > 2:
        pats = results["H14_bind_plus_3tcr"]["per_patient_auc"]
        aucs = [v["auc"] for v in pats.values() if v["auc"] is not None]
        lopo_summary = {
            "n_patients": len(patients),
            "n_computable": len(aucs),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "median_auc": float(np.median(aucs)) if aucs else None,
        }

    ranked = sorted(
        [{"hypothesis": k, "in_sample_auc": v["in_sample_auc"]} for k, v in results.items()],
        key=lambda x: x["in_sample_auc"] if x["in_sample_auc"] is not None else -1,
        reverse=True,
    )

    return {
        "n_rows": int(len(df)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "patients": patients,
        "hypotheses": results,
        "ranked_in_sample": ranked,
        "h14_summary": {
            "in_sample_auc": h14_auc,
            "in_sample_auc_ci95": h14_ci,
            "binding_only_auc": h0_auc,
            "binding_only_auc_ci95": h0_ci,
            "delta_h14_minus_binding": delta,
            "delta_ci95": delta_ci,
        },
        "lopo_tesla": lopo_summary,
    }


def get_hypothesis_features_from_ott() -> dict[str, list[list[object]]]:
    data = json.loads(TARGETED_RESULTS.read_text(encoding="utf-8"))
    out: dict[str, list[list[object]]] = {}
    for name, info in data["cv_results"].items():
        out[name] = info["features"]
    return out


def cross_dataset_h14(ott_df: pd.DataFrame, tesla_df: pd.DataFrame, h14_feats: list[list[object]]) -> dict[str, object]:
    ott_y = pd.to_numeric(ott_df["immunogenic"], errors="coerce").fillna(0).astype(float).values
    tesla_y = pd.to_numeric(tesla_df["immunogenic"], errors="coerce").fillna(0).astype(float).values
    ott_s = score_hypothesis(ott_df, h14_feats)
    tesla_s = score_hypothesis(tesla_df, h14_feats)
    return {
        "setup": "train_on_ott_fixed_h14_then_test_on_tesla",
        "h14_features": h14_feats,
        "ott_in_sample_auc": safe_auc(ott_y, ott_s),
        "tesla_test_auc": safe_auc(tesla_y, tesla_s),
        "ott_n": int(len(ott_df)),
        "tesla_n": int(len(tesla_df)),
    }


def eval_all_papers_h14(training: pd.DataFrame, h14_feats: list[list[object]]) -> dict[str, object]:
    papers = ["sahin_2017", "keskin_2019", "hilf_2019", "rojas_2023"]
    out: dict[str, object] = {"h14_features": h14_feats, "papers": {}}
    for p in papers:
        sub = training[(training["paper_source"] == p) & (training["immunogenic"].isin([0, 1]))].copy()
        if sub.empty:
            out["papers"][p] = {"n_rows": 0, "auc": None}
            continue
        feat_sub, _ = add_features(sub)
        y = pd.to_numeric(feat_sub["immunogenic"], errors="coerce").fillna(0).astype(float).values
        s = score_hypothesis(feat_sub, h14_feats)
        per_pat = {}
        for pat in sorted(feat_sub["patient_id"].astype(str).unique()):
            m = feat_sub["patient_id"].astype(str) == pat
            per_pat[pat] = safe_auc(y[m.values], s[m.values])
        out["papers"][p] = {
            "n_rows": int(len(feat_sub)),
            "n_pos": int(np.sum(y == 1)),
            "auc": safe_auc(y, s),
            "per_patient_auc": per_pat,
        }
    return out


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    TESLA_OUT.parent.mkdir(parents=True, exist_ok=True)

    ensure_tesla_files()
    tesla_raw, patient_cancer = load_tesla_tables()
    tesla_prepared = build_tesla_prepared(tesla_raw, patient_cancer)
    tesla_prepared.to_csv(TESLA_OUT, index=False)

    tesla_feat, tesla_avail = add_features(tesla_prepared)
    hypothesis_features = get_hypothesis_features_from_ott()
    tesla_eval = evaluate_hypotheses(tesla_feat, hypothesis_features)

    per_patient = (
        tesla_feat.groupby("patient_id", dropna=False)["immunogenic"]
        .agg(total="count", immunogenic="sum")
        .reset_index()
        .to_dict(orient="records")
    )

    tesla_payload = {
        "source": {
            "paper": "Wells et al., Cell 2020 (10.1016/j.cell.2020.09.015)",
            "retrieval": "CaltechAUTHORS DOI mirror supplementary xlsx files",
            "files": TESLA_FILES,
        },
        "prepared_csv": str(TESLA_OUT),
        "summary": {
            "total_candidates": int(len(tesla_feat)),
            "total_immunogenic": int(tesla_feat["immunogenic"].sum()),
            "total_non_immunogenic": int((tesla_feat["immunogenic"] == 0).sum()),
            "per_patient_breakdown": per_patient,
            "feature_availability_counts": tesla_avail,
        },
        "hypothesis_results": tesla_eval,
    }
    TESLA_RESULTS.write_text(json.dumps(tesla_payload, indent=2), encoding="utf-8")

    training = pd.read_csv(TRAINING_MATRIX)
    ott = training[(training["paper_source"] == "ott_2017") & (training["immunogenic"].isin([0, 1]))].copy()
    ott_feat, _ = add_features(ott)
    h14_feats = hypothesis_features["H14_bind_plus_3tcr"]

    cross = cross_dataset_h14(ott_feat, tesla_feat, h14_feats)
    CROSS_RESULTS.write_text(json.dumps(cross, indent=2), encoding="utf-8")

    all_papers = eval_all_papers_h14(training, h14_feats)
    ALL_PAPERS_RESULTS.write_text(json.dumps(all_papers, indent=2), encoding="utf-8")

    print(f"Saved: {TESLA_OUT}")
    print(f"Saved: {TESLA_RESULTS}")
    print(f"Saved: {CROSS_RESULTS}")
    print(f"Saved: {ALL_PAPERS_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
