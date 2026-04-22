import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import product as iprod
from scipy.stats import mannwhitneyu
import json
import warnings

warnings.filterwarnings("ignore")

# Load training matrix
tm = pd.read_csv("backend/validation/artifacts/training_matrix.csv")
ott = tm[tm["paper_source"] == "ott_2017"].copy()

print("=" * 70)
print("SYSTEMATIC FEATURE ENGINEERING SWEEP")
print("=" * 70)

# -----------------------------------------------------------
# LEVEL 1: Peptide-HLA level (167 rows, 18 immunogenic)
# -----------------------------------------------------------
peptide_level = ott[ott["immunogenic"].isin([0, 1])].copy()
print(
    f"\nPeptide level: {len(peptide_level)} rows, "
    f"{peptide_level['immunogenic'].sum()} immunogenic"
)

# -----------------------------------------------------------
# LEVEL 2: Mutation level (collapsed)
# -----------------------------------------------------------
mutation_level = (
    ott.groupby(["patient_id", "gene", "protein_change"])
    .agg(
        {
            "immunogenic": "max",
            "expression_tpm": "first",
            "expression_tpm_numeric": "first",
            "expression_score": "first",
            "presentation_score": "max",  # best binding across HLAs
            "binding_affinity_nm_numeric": "min",  # best (lowest) nM
            "predicted_affinity_nm": "min",
            "final_binding_affinity_nm": "min",
            "self_dissimilarity": "mean",
            "mutant_peptide": "first",
            "wildtype_peptide": "first",
            "ccf": "first",
            "hla_allele": "first",
            "cancer_type": "first",
        }
    )
    .reset_index()
)
mutation_level = mutation_level[mutation_level["immunogenic"].isin([0, 1])].copy()
print(
    f"Mutation level: {len(mutation_level)} rows, "
    f"{mutation_level['immunogenic'].sum()} immunogenic"
)

# Print raw data for inspection
print("\n--- RAW VALUES (mutation level, first 15) ---")
print(
    mutation_level[
        [
            "gene",
            "protein_change",
            "immunogenic",
            "binding_affinity_nm_numeric",
            "expression_tpm_numeric",
            "expression_score",
            "presentation_score",
            "self_dissimilarity",
        ]
    ]
    .head(15)
    .to_string()
)

# Also print immunogenic rows specifically
print("\n--- IMMUNOGENIC MUTATIONS ---")
imm = mutation_level[mutation_level["immunogenic"] == 1]
print(
    imm[
        [
            "gene",
            "protein_change",
            "binding_affinity_nm_numeric",
            "expression_tpm_numeric",
            "presentation_score",
            "self_dissimilarity",
        ]
    ].to_string()
)

# =============================================================
# THE SWEEP: Try every combination
# =============================================================

# We'll store all results in a leaderboard
leaderboard = []


def score_auc(y, scores, name, level, details=""):
    """Compute AUC and store in leaderboard."""
    valid_mask = np.isfinite(scores) & np.isfinite(y)
    y_clean = y[valid_mask]
    s_clean = scores[valid_mask]
    if len(np.unique(y_clean)) < 2 or len(y_clean) < 10:
        return None
    auc = roc_auc_score(y_clean, s_clean)
    n = len(y_clean)
    n_pos = int(y_clean.sum())
    leaderboard.append(
        {
            "name": name,
            "level": level,
            "auc": auc,
            "n": n,
            "n_pos": n_pos,
            "details": details,
        }
    )
    return auc


# =============================================================
# CATEGORY 1: BINDING TRANSFORMS
# Try every way to convert binding affinity to a score
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 1: BINDING TRANSFORMS")
print("=" * 70)

for level_name, df in [("peptide", peptide_level), ("mutation", mutation_level)]:
    y = df["immunogenic"].values.astype(float)

    # Get raw binding nM (use best available)
    nm_raw = df["binding_affinity_nm_numeric"].copy()
    if nm_raw.isna().all():
        nm_raw = df["final_binding_affinity_nm"].copy()

    has_nm = nm_raw.notna() & (nm_raw > 0)

    if has_nm.sum() < 10:
        print(f"  {level_name}: insufficient binding data")
        continue

    # Transform 1: 1 - log10(nM)/log10(50000)
    t1 = (1 - np.log10(nm_raw.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
    score_auc(y, t1.values, "bind_log50k", level_name, "1 - log10(nM)/log10(50000)")

    # Transform 2: 1 - log10(nM)/log10(5000)
    t2 = (1 - np.log10(nm_raw.clip(lower=0.1)) / np.log10(5000)).clip(0, 1)
    score_auc(y, t2.values, "bind_log5k", level_name, "1 - log10(nM)/log10(5000)")

    # Transform 3: 1 - log10(nM)/log10(500)
    t3 = (1 - np.log10(nm_raw.clip(lower=0.1)) / np.log10(500)).clip(0, 1)
    score_auc(y, t3.values, "bind_log500", level_name, "1 - log10(nM)/log10(500)")

    # Transform 4: Linear 1 - nM/50000
    t4 = (1 - nm_raw / 50000).clip(0, 1)
    score_auc(y, t4.values, "bind_linear50k", level_name, "1 - nM/50000")

    # Transform 5: Linear 1 - nM/500
    t5 = (1 - nm_raw / 500).clip(0, 1)
    score_auc(y, t5.values, "bind_linear500", level_name, "1 - nM/500, clipped")

    # Transform 6: Sigmoid 1/(1 + nM/500)
    t6 = 1.0 / (1.0 + nm_raw / 500)
    score_auc(y, t6.values, "bind_sigmoid500", level_name, "1/(1 + nM/500)")

    # Transform 7: Sigmoid 1/(1 + nM/50)
    t7 = 1.0 / (1.0 + nm_raw / 50)
    score_auc(y, t7.values, "bind_sigmoid50", level_name, "1/(1 + nM/50)")

    # Transform 8: Exponential exp(-nM/500)
    t8 = np.exp(-nm_raw / 500)
    score_auc(y, t8.values, "bind_exp500", level_name, "exp(-nM/500)")

    # Transform 9: Exponential exp(-nM/150)
    t9 = np.exp(-nm_raw / 150)
    score_auc(y, t9.values, "bind_exp150", level_name, "exp(-nM/150)")

    # Transform 10: Raw inverse 1/nM
    t10 = 1.0 / nm_raw.clip(lower=0.1)
    score_auc(y, t10.values, "bind_inverse", level_name, "1/nM")

    # Transform 11: Within-patient percentile rank
    t11_v2 = (
        df.groupby("patient_id")["binding_affinity_nm_numeric"]
        .rank(ascending=False, pct=True)
        .fillna(0.5)
    )
    score_auc(
        y,
        t11_v2.values,
        "bind_patient_rank",
        level_name,
        "within-patient percentile rank",
    )

    # Transform 12: Global percentile rank (inverted)
    t12 = 1 - nm_raw.rank(ascending=True, pct=True)
    score_auc(
        y,
        t12.values,
        "bind_global_rank",
        level_name,
        "global percentile rank (inverted)",
    )

    # Transform 13: Binary strong binder (<500nM)
    t13 = (nm_raw < 500).astype(float)
    score_auc(y, t13.values, "bind_binary500", level_name, "binary: nM < 500")

    # Transform 14: Binary strong binder (<50nM)
    t14 = (nm_raw < 50).astype(float)
    score_auc(y, t14.values, "bind_binary50", level_name, "binary: nM < 50")

    # Transform 15: Tiered
    t15 = pd.Series(0.0, index=df.index)
    t15[nm_raw < 500] = 0.5
    t15[nm_raw < 50] = 1.0
    score_auc(
        y,
        t15.values,
        "bind_tiered",
        level_name,
        "<50nM=1.0, <500nM=0.5, else=0",
    )

    # Transform 16: -log10(nM)
    t16 = -np.log10(nm_raw.clip(lower=0.1))
    score_auc(y, t16.values, "bind_neglog", level_name, "-log10(nM)")


# =============================================================
# CATEGORY 2: EXPRESSION TRANSFORMS
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 2: EXPRESSION TRANSFORMS")
print("=" * 70)

for level_name, df in [("peptide", peptide_level), ("mutation", mutation_level)]:
    y = df["immunogenic"].values.astype(float)
    tpm = df["expression_tpm_numeric"].copy()
    has_expr = tpm.notna() & (tpm > 0)

    if has_expr.sum() < 10:
        print(f"  {level_name}: insufficient expression data")
        continue

    score_auc(
        y,
        df["expression_score"].values,
        "expr_current",
        level_name,
        "current expression_score",
    )

    y_expr = y[has_expr]
    tpm_valid = tpm[has_expr]

    e1 = np.log2(tpm_valid + 1)
    e1_norm = e1 / e1.max() if e1.max() > 0 else e1
    score_auc(y_expr, e1_norm.values, "expr_log2_maxnorm", level_name, "log2(TPM+1)/max")
    score_auc(y_expr, e1.values, "expr_log2_raw", level_name, "log2(TPM+1)")

    e3 = np.log10(tpm_valid + 1)
    score_auc(y_expr, e3.values, "expr_log10", level_name, "log10(TPM+1)")

    e4 = np.sqrt(tpm_valid)
    score_auc(y_expr, e4.values, "expr_sqrt", level_name, "sqrt(TPM)")

    score_auc(y_expr, tpm_valid.values, "expr_raw_tpm", level_name, "raw TPM")

    e6 = (tpm_valid > 1).astype(float)
    if e6.nunique() > 1:
        score_auc(y_expr, e6.values, "expr_binary_1", level_name, "binary: TPM > 1")

    e7 = (tpm_valid > 10).astype(float)
    if e7.nunique() > 1:
        score_auc(y_expr, e7.values, "expr_binary_10", level_name, "binary: TPM > 10")

    e8 = df.loc[has_expr].groupby("patient_id")["expression_tpm_numeric"].rank(pct=True, ascending=True)
    score_auc(y_expr, e8.values, "expr_patient_rank", level_name, "within-patient percentile")

    from scipy.stats import rankdata

    e9 = rankdata(tpm_valid) / len(tpm_valid)
    score_auc(y_expr, e9, "expr_quantile", level_name, "global quantile")

    e10 = tpm_valid / (tpm_valid + 10)
    score_auc(y_expr, e10.values, "expr_sigmoid10", level_name, "TPM/(TPM+10)")

    e11 = tpm_valid / (tpm_valid + 100)
    score_auc(y_expr, e11.values, "expr_sigmoid100", level_name, "TPM/(TPM+100)")


# =============================================================
# CATEGORY 3: SELF-DISSIMILARITY ALTERNATIVES
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 3: SELF-DISSIMILARITY TRANSFORMS")
print("=" * 70)


def hamming_prop(mut, wt):
    if len(mut) != len(wt):
        minlen = min(len(mut), len(wt))
        mut = mut[:minlen]
        wt = wt[:minlen]
    if len(mut) == 0:
        return np.nan
    return sum(a != b for a, b in zip(mut, wt)) / len(mut)


def hamming_count(mut, wt):
    if len(mut) != len(wt):
        minlen = min(len(mut), len(wt))
        mut = mut[:minlen]
        wt = wt[:minlen]
    return sum(a != b for a, b in zip(mut, wt))


for level_name, df in [("peptide", peptide_level), ("mutation", mutation_level)]:
    y = df["immunogenic"].values.astype(float)
    score_auc(y, df["self_dissimilarity"].values, "dissim_current", level_name, "current BLOSUM62")

    has_both = (
        df["mutant_peptide"].notna()
        & df["wildtype_peptide"].notna()
        & (df["mutant_peptide"] != "")
        & (df["wildtype_peptide"] != "")
    )

    if has_both.sum() < 10:
        print(f"  {level_name}: insufficient peptide pairs")
        continue

    df_pairs = df[has_both].copy()
    y_pairs = df_pairs["immunogenic"].values.astype(float)

    hamming = df_pairs.apply(
        lambda r: hamming_prop(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
        axis=1,
    )
    score_auc(y_pairs, hamming.values, "dissim_hamming", level_name, "Hamming distance proportion")

    hamming_n = df_pairs.apply(
        lambda r: hamming_count(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
        axis=1,
    )
    score_auc(y_pairs, hamming_n.values, "dissim_hamming_n", level_name, "Hamming distance count")

    try:
        from Bio.Align import substitution_matrices

        blosum = substitution_matrices.load("BLOSUM62")

        def blosum_diff(mut, wt):
            if len(mut) != len(wt):
                minlen = min(len(mut), len(wt))
                mut = mut[:minlen]
                wt = wt[:minlen]
            score = 0
            for m, w in zip(mut, wt):
                if m != w:
                    try:
                        wt_self = blosum[w][w]
                        mut_cross = blosum[w][m]
                        score += wt_self - mut_cross
                    except (KeyError, IndexError):
                        pass
            return score

        blosum_d = df_pairs.apply(
            lambda r: blosum_diff(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
            axis=1,
        )
        score_auc(
            y_pairs,
            blosum_d.values,
            "dissim_blosum_diff",
            level_name,
            "BLOSUM62 self-score minus cross-score",
        )

        def blosum_max_diff(mut, wt):
            if len(mut) != len(wt):
                minlen = min(len(mut), len(wt))
                mut = mut[:minlen]
                wt = wt[:minlen]
            max_d = 0
            for m, w in zip(mut, wt):
                if m != w:
                    try:
                        d = blosum[w][w] - blosum[w][m]
                        max_d = max(max_d, d)
                    except (KeyError, IndexError):
                        pass
            return max_d

        blosum_mx = df_pairs.apply(
            lambda r: blosum_max_diff(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
            axis=1,
        )
        score_auc(
            y_pairs,
            blosum_mx.values,
            "dissim_blosum_max",
            level_name,
            "max single-position BLOSUM diff",
        )
    except ImportError:
        print("  Biopython not available for BLOSUM alternatives")

    kd = {
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

    def hydro_change(mut, wt):
        if len(mut) != len(wt):
            minlen = min(len(mut), len(wt))
            mut = mut[:minlen]
            wt = wt[:minlen]
        total = 0
        for m, w in zip(mut, wt):
            if m != w:
                total += abs(kd.get(m, 0) - kd.get(w, 0))
        return total

    hydro = df_pairs.apply(
        lambda r: hydro_change(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
        axis=1,
    )
    score_auc(
        y_pairs,
        hydro.values,
        "dissim_hydro_change",
        level_name,
        "Kyte-Doolittle hydrophobicity change",
    )

    charge = {"R": 1, "K": 1, "H": 0.5, "D": -1, "E": -1}

    def charge_change(mut, wt):
        if len(mut) != len(wt):
            minlen = min(len(mut), len(wt))
            mut = mut[:minlen]
            wt = wt[:minlen]
        total = 0
        for m, w in zip(mut, wt):
            if m != w:
                total += abs(charge.get(m, 0) - charge.get(w, 0))
        return total

    chg = df_pairs.apply(
        lambda r: charge_change(str(r["mutant_peptide"]), str(r["wildtype_peptide"])),
        axis=1,
    )
    if chg.nunique() > 1:
        score_auc(y_pairs, chg.values, "dissim_charge_change", level_name, "charge change magnitude")

    inv_dissim = 1 - df["self_dissimilarity"]
    score_auc(y, inv_dissim.values, "dissim_inverted", level_name, "1 - self_dissimilarity")


# =============================================================
# CATEGORY 4: INTERACTION FEATURES
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 4: INTERACTION / DERIVED FEATURES")
print("=" * 70)

for level_name, df in [("peptide", peptide_level), ("mutation", mutation_level)]:
    y = df["immunogenic"].values.astype(float)
    pres = df["presentation_score"].values
    expr = df["expression_score"].values
    dissim = df["self_dissimilarity"].values

    i1 = pres * expr
    score_auc(y, i1, "interact_bind_x_expr", level_name, "presentation × expression")

    i2 = pres * dissim
    score_auc(y, i2, "interact_bind_x_dissim", level_name, "presentation × dissimilarity")

    i3 = expr * dissim
    score_auc(y, i3, "interact_expr_x_dissim", level_name, "expression × dissimilarity")

    i4 = pres * expr * dissim
    score_auc(y, i4, "interact_all_three", level_name, "pres × expr × dissim")

    def geomean_available(*vals):
        available = [v for v in vals if v > 0]
        if not available:
            return 0
        return np.prod(available) ** (1 / len(available))

    i5 = np.array([geomean_available(p, e, d) for p, e, d in zip(pres, expr, dissim)])
    score_auc(
        y,
        i5,
        "interact_geomean",
        level_name,
        "geometric mean of nonzero features",
    )

    def harmean_available(*vals):
        available = [v for v in vals if v > 0]
        if not available:
            return 0
        return len(available) / sum(1 / v for v in available)

    i6 = np.array([harmean_available(p, e, d) for p, e, d in zip(pres, expr, dissim)])
    score_auc(y, i6, "interact_harmonic", level_name, "harmonic mean of nonzero features")

    i7 = np.maximum(np.maximum(pres, expr), dissim)
    score_auc(y, i7, "interact_max", level_name, "max(pres, expr, dissim)")

    def min_nonzero(*vals):
        available = [v for v in vals if v > 0]
        if not available:
            return 0
        return min(available)

    i8 = np.array([min_nonzero(p, e, d) for p, e, d in zip(pres, expr, dissim)])
    score_auc(
        y,
        i8,
        "interact_min_nonzero",
        level_name,
        "min of nonzero features (bottleneck)",
    )

    if "binding_affinity_nm_numeric" in df.columns:
        bind_rank = (
            df.groupby("patient_id")["binding_affinity_nm_numeric"]
            .rank(ascending=True, pct=True)
            .fillna(0.5)
        )
        bind_rank = 1 - bind_rank
        expr_rank = (
            df.groupby("patient_id")["expression_score"]
            .rank(ascending=True, pct=True)
            .fillna(0.5)
        )
        d1 = bind_rank * expr_rank
        score_auc(y, d1.values, "derived_rank_product", level_name, "binding_rank × expression_rank")


# =============================================================
# CATEGORY 5: MISSING DATA STRATEGIES
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 5: MISSING DATA HANDLING")
print("=" * 70)

weights_v1 = {"expression_score": 0.20, "presentation_score": 0.30, "self_dissimilarity": 0.10}

for level_name, df in [("mutation", mutation_level)]:
    y = df["immunogenic"].values.astype(float)
    features = ["expression_score", "presentation_score", "self_dissimilarity"]

    X1 = df[features].fillna(0).values
    w1 = np.array([0.20, 0.30, 0.10])
    w1 = w1 / w1.sum()
    s1 = (X1 * w1).sum(axis=1)
    score_auc(y, s1, "impute_zero_v1weights", level_name, "zero-fill, rl_v1 weights")

    X2 = df[features].copy()
    for col in features:
        X2[col] = X2[col].fillna(X2[col].mean())
    X2 = X2.values
    s2 = (X2 * w1).sum(axis=1)
    score_auc(y, s2, "impute_mean_v1weights", level_name, "mean-fill, rl_v1 weights")

    X3 = df[features].copy()
    for col in features:
        X3[col] = X3[col].fillna(X3[col].median())
    X3 = X3.values
    s3 = (X3 * w1).sum(axis=1)
    score_auc(y, s3, "impute_median_v1weights", level_name, "median-fill, rl_v1 weights")

    def dynamic_score(row, weights_dict):
        available = {}
        for feat, w in weights_dict.items():
            val = row[feat]
            if pd.notna(val) and val > 0:
                available[feat] = (val, w)
        if not available:
            return np.nan
        tw = sum(w for _, w in available.values())
        return sum(v * (w / tw) for v, w in available.values())

    s4 = df.apply(lambda r: dynamic_score(r, weights_v1), axis=1).values
    score_auc(y, s4, "impute_dynamic_v1weights", level_name, "dynamic renorm, rl_v1 weights")

    has_all = (
        df["expression_score"].notna()
        & (df["expression_score"] > 0)
        & (df["presentation_score"] > 0)
        & (df["self_dissimilarity"] > 0)
        & (df["self_dissimilarity"] != 0.5)
    )
    if has_all.sum() >= 10:
        y_all = df.loc[has_all, "immunogenic"].values.astype(float)
        X_all = df.loc[has_all, features].values
        s5 = (X_all * w1).sum(axis=1)
        n_pos = int(y_all.sum())
        if n_pos > 0 and n_pos < len(y_all):
            score_auc(
                y_all,
                s5,
                "impute_complete_only",
                level_name,
                f"complete cases only (n={has_all.sum()})",
            )


# =============================================================
# CATEGORY 6: COMBINED — BEST TRANSFORMS + GRID SEARCH
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 6: BEST TRANSFORM COMBINATIONS (grid search)")
print("=" * 70)

df = mutation_level.copy()
y = df["immunogenic"].values.astype(float)
nm = df["binding_affinity_nm_numeric"].copy()

bind_transforms = {}
if nm.notna().any():
    bind_transforms["bind_log50k"] = (
        1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)
    ).clip(0, 1)
    bind_transforms["bind_sigmoid500"] = 1.0 / (1.0 + nm / 500)
    bind_transforms["bind_exp500"] = np.exp(-nm / 500)
    bind_transforms["bind_neglog"] = -np.log10(nm.clip(lower=0.1))
    bind_transforms["bind_exp150"] = np.exp(-nm / 150)

tpm = df["expression_tpm_numeric"].copy()
expr_transforms = {}
if tpm.notna().any():
    expr_transforms["expr_current"] = df["expression_score"]
    expr_transforms["expr_log2"] = np.log2(tpm.clip(lower=0) + 1)
    expr_transforms["expr_sqrt"] = np.sqrt(tpm.clip(lower=0))
    expr_transforms["expr_sigmoid10"] = tpm / (tpm + 10)
    expr_transforms["expr_quantile"] = tpm.rank(pct=True)

dissim_transforms = {}
dissim_transforms["dissim_current"] = df["self_dissimilarity"]
dissim_transforms["dissim_inverted"] = 1 - df["self_dissimilarity"]
if (df["mutant_peptide"].notna() & df["wildtype_peptide"].notna()).sum() > 10:
    dissim_transforms["dissim_hamming"] = df.apply(
        lambda r: hamming_prop(str(r["mutant_peptide"]), str(r["wildtype_peptide"]))
        if pd.notna(r["mutant_peptide"]) and pd.notna(r["wildtype_peptide"])
        else np.nan,
        axis=1,
    )

print(f"Binding transforms: {len(bind_transforms)}")
print(f"Expression transforms: {len(expr_transforms)}")
print(f"Dissimilarity transforms: {len(dissim_transforms)}")

best_overall = {"auc": 0, "name": "", "weights": {}}
combo_results = []

weight_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for bt_name, bt in bind_transforms.items():
    for et_name, et in expr_transforms.items():
        for dt_name, dt in dissim_transforms.items():
            best_combo_auc = 0
            best_combo_w = None

            for w0, w1, w2 in iprod(weight_steps, weight_steps, weight_steps):
                t = w0 + w1 + w2
                if t == 0:
                    continue
                ws = np.array([w0, w1, w2]) / t

                scores = []
                for i in range(len(df)):
                    vals = [
                        bt.iloc[i] if pd.notna(bt.iloc[i]) and bt.iloc[i] > 0 else None,
                        et.iloc[i] if pd.notna(et.iloc[i]) and et.iloc[i] > 0 else None,
                        dt.iloc[i] if pd.notna(dt.iloc[i]) and dt.iloc[i] > 0 else None,
                    ]
                    wts = [ws[0], ws[1], ws[2]]

                    avail_v = []
                    avail_w = []
                    for v, w in zip(vals, wts):
                        if v is not None and w > 0:
                            avail_v.append(v)
                            avail_w.append(w)

                    if not avail_v:
                        scores.append(0)
                        continue
                    tw = sum(avail_w)
                    scores.append(sum(v * (w / tw) for v, w in zip(avail_v, avail_w)))

                auc = roc_auc_score(y, scores)
                if auc > best_combo_auc:
                    best_combo_auc = auc
                    best_combo_w = ws

            combo_name = f"{bt_name}+{et_name}+{dt_name}"
            combo_results.append(
                {
                    "combo": combo_name,
                    "auc": best_combo_auc,
                    "weights": best_combo_w.tolist() if best_combo_w is not None else None,
                    "bind": bt_name,
                    "expr": et_name,
                    "dissim": dt_name,
                }
            )

            if best_combo_auc > best_overall["auc"]:
                best_overall = {
                    "auc": best_combo_auc,
                    "name": combo_name,
                    "weights": best_combo_w.tolist() if best_combo_w is not None else None,
                    "bind": bt_name,
                    "expr": et_name,
                    "dissim": dt_name,
                }

combo_results.sort(key=lambda x: x["auc"], reverse=True)
print("\nTop 20 transform combinations:")
print(f"{'Rank':>4} {'AUC':>6} {'Bind Weight':>10} {'Expr Weight':>10} {'Dissim Weight':>12} {'Combo'}")
for i, cr in enumerate(combo_results[:20]):
    w = cr["weights"] if cr["weights"] else [0, 0, 0]
    print(f"{i+1:>4} {cr['auc']:.4f} {w[0]:>10.2f} {w[1]:>10.2f} {w[2]:>12.2f} {cr['combo']}")


# =============================================================
# CATEGORY 7: LOGISTIC REGRESSION WITH BEST TRANSFORMS
# =============================================================

print("\n" + "=" * 70)
print("CATEGORY 7: LOGISTIC REGRESSION WITH FEATURE SELECTION")
print("=" * 70)

all_transforms = {}
all_transforms.update(bind_transforms)
all_transforms.update(expr_transforms)
all_transforms.update(dissim_transforms)

for bt_name, bt in bind_transforms.items():
    for et_name, et in expr_transforms.items():
        all_transforms[f"{bt_name}_x_{et_name}"] = bt * et

lr_results = []
feat_names = list(all_transforms.keys())

from itertools import combinations

for r in [1, 2, 3]:
    for combo in combinations(range(len(feat_names)), r):
        names = [feat_names[i] for i in combo]
        X = np.column_stack([all_transforms[n].fillna(0).values for n in names])
        mask = (X != 0).any(axis=1)
        if mask.sum() < 20:
            continue

        X_sub = X[mask]
        y_sub = y[mask]
        if len(np.unique(y_sub)) < 2:
            continue

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sub)
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(X_scaled, y_sub)
            probs = lr.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y_sub, probs)

            lr_results.append(
                {
                    "features": "+".join(names),
                    "auc": auc,
                    "n": int(mask.sum()),
                    "coefs": dict(zip(names, lr.coef_[0].tolist())),
                }
            )
        except Exception:
            pass

lr_results.sort(key=lambda x: x["auc"], reverse=True)
print("\nTop 20 logistic regression feature sets:")
print(f"{'Rank':>4} {'AUC':>6} {'N':>5} {'Features'}")
for i, lr_r in enumerate(lr_results[:20]):
    print(f"{i+1:>4} {lr_r['auc']:.4f} {lr_r['n']:>5} {lr_r['features']}")
    if i < 5:
        for feat, coef in lr_r["coefs"].items():
            direction = "+" if coef > 0 else "NEG"
            print(f"       {feat}: {coef:+.4f} [{direction}]")


# =============================================================
# FINAL LEADERBOARD
# =============================================================

print("\n" + "=" * 70)
print("FINAL LEADERBOARD — ALL APPROACHES")
print("=" * 70)

leaderboard.append(
    {
        "name": f"BEST_COMBO: {best_overall['name']}",
        "level": "mutation",
        "auc": best_overall["auc"],
        "n": len(mutation_level),
        "n_pos": int(mutation_level["immunogenic"].sum()),
        "details": f"weights: {best_overall['weights']}",
    }
)

if lr_results:
    leaderboard.append(
        {
            "name": f"BEST_LR: {lr_results[0]['features']}",
            "level": "mutation",
            "auc": lr_results[0]["auc"],
            "n": lr_results[0]["n"],
            "n_pos": -1,
            "details": "logistic regression",
        }
    )

leaderboard.sort(key=lambda x: x["auc"], reverse=True)

baseline_auc = None
for entry in leaderboard:
    if "bind_log50k" in entry["name"] and entry["level"] == "mutation":
        baseline_auc = entry["auc"]
        break

print(f"\n{'Rank':>4} {'AUC':>7} {'Delta':>7} {'Level':>8} {'N':>5} {'Name'}")
print("-" * 80)
for i, entry in enumerate(leaderboard[:40]):
    delta = entry["auc"] - baseline_auc if baseline_auc else 0
    delta_str = f"{delta:+.4f}" if baseline_auc else "  ---"
    print(
        f"{i+1:>4} {entry['auc']:.4f} {delta_str} {entry['level']:>8} {entry['n']:>5} {entry['name']}"
    )

print(
    f"\nBaseline (bind_log50k mutation-level): {baseline_auc:.4f}"
    if baseline_auc
    else "not found"
)
print(f"Best overall: {leaderboard[0]['name']} = {leaderboard[0]['auc']:.4f}")
if baseline_auc:
    print(f"Best delta over baseline: {leaderboard[0]['auc'] - baseline_auc:+.4f}")


# =============================================================
# SAVE RESULTS
# =============================================================

results = {
    "leaderboard": leaderboard[:50],
    "best_combo": best_overall,
    "top_lr": lr_results[:10] if lr_results else [],
    "combo_results_top20": combo_results[:20],
    "baseline_auc": baseline_auc,
}

with open("backend/validation/artifacts/feature_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved to feature_sweep_results.json")
print("\n" + "=" * 70)
print("SWEEP COMPLETE")
print("=" * 70)
