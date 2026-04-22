from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def main() -> int:
    tesla = pd.read_csv("validation_papers/tesla_prepared.csv")

    print("=" * 70)
    print("TESLA COLUMN AUDIT — IDENTIFY LEAKAGE")
    print("=" * 70)

    print(f"\nTotal columns: {len(tesla.columns)}")
    print(f"\n{'Column':<40} {'Type':<10} {'NonNull':>7} {'Sample Values'}")
    print("-" * 90)

    for col in tesla.columns:
        dtype = str(tesla[col].dtype)
        non_null = int(tesla[col].notna().sum())
        samples = tesla[col].dropna().head(3).tolist()
        sample_str = str(samples)[:40]
        print(f"{col:<40} {dtype:<10} {non_null:>7} {sample_str}")

    print("\n" + "=" * 70)
    print("LEAKAGE CLASSIFICATION")
    print("=" * 70)

    LEAKAGE_KEYWORDS = [
        "tcr_flow",
        "flow_",
        "elispot",
        "tetramer",
        "reactiv",
        "response",
        "validated",
        "confirmed",
        "assay",
        "ifn",
        "cytokine",
        "killing",
        "recognition",
        "tcr_result",
        "immune_response",
        "multimer",
        "dextramer",
        "pentamer",
        "cd8_response",
        "cd4_response",
        "t_cell",
    ]

    SAFE_KEYWORDS = [
        "peptide",
        "hla",
        "allele",
        "binding",
        "affinity",
        "expression",
        "tpm",
        "gene",
        "protein",
        "mutation",
        "patient",
        "cancer",
        "paper",
        "length",
        "sequence",
        "vaf",
        "ccf",
        "clonal",
    ]

    leaky_cols: list[str] = []
    safe_cols: list[str] = []
    uncertain_cols: list[str] = []

    for col in tesla.columns:
        col_lower = col.lower()
        is_leaky = any(kw in col_lower for kw in LEAKAGE_KEYWORDS)
        is_safe = any(kw in col_lower for kw in SAFE_KEYWORDS)
        if col == "immunogenic":
            print(f"  {col}: LABEL (target variable)")
        elif is_leaky:
            leaky_cols.append(col)
            print(f"  {col}: LEAKAGE — assay/outcome data")
        elif is_safe:
            safe_cols.append(col)
            print(f"  {col}: SAFE — predictive feature")
        else:
            uncertain_cols.append(col)
            print(f"  {col}: UNCERTAIN — needs manual check")

    print(f"\nSafe: {len(safe_cols)}")
    print(f"Leaky: {len(leaky_cols)}")
    print(f"Uncertain: {len(uncertain_cols)}")

    print("\n--- Uncertain column correlation with immunogenic ---")
    for col in uncertain_cols:
        if pd.api.types.is_numeric_dtype(tesla[col]):
            corr = tesla[col].corr(tesla["immunogenic"])
            flag = " <- SUSPICIOUS" if pd.notna(corr) and abs(float(corr)) > 0.3 else ""
            print(f"  {col}: r={float(corr) if pd.notna(corr) else np.nan:.3f}{flag}")

    tesla_feat = pd.read_csv("backend/validation/artifacts/tesla_full_features.csv")

    ALWAYS_DROP = ["immunogenic", "response_type", "detection_method", "notes"]
    drop_set = set(leaky_cols + ALWAYS_DROP + uncertain_cols)

    feature_cols_clean = [
        c
        for c in tesla_feat.columns
        if c not in drop_set
        and c
        not in [
            "patient_id",
            "paper_source",
            "cancer_type",
            "gene",
            "protein_change",
            "mutant_peptide",
            "wildtype_peptide",
            "hla_allele",
            "final_hla_allele",
            "hla_supertype",
            "binding_source",
            "nm",
        ]
        and pd.api.types.is_numeric_dtype(tesla_feat[c])
        and tesla_feat[c].notna().sum() > 50
        and pd.to_numeric(tesla_feat[c], errors="coerce").std() > 1e-10
    ]

    print(f"\n{'=' * 70}")
    print(f"CLEAN FEATURE SET: {len(feature_cols_clean)} features")
    print(f"{'=' * 70}")
    for fc in sorted(feature_cols_clean):
        print(f"  {fc}")

    y = tesla_feat["immunogenic"].values.astype(float)
    clean_single_results: list[dict[str, object]] = []

    print(f"\n{'=' * 70}")
    print("CLEAN SINGLE-FEATURE AUC")
    print(f"{'=' * 70}")

    for feat in feature_cols_clean:
        vals = pd.to_numeric(tesla_feat[feat], errors="coerce")
        mask = vals.notna() & np.isfinite(vals)
        if int(mask.sum()) < 50:
            continue
        y_sub = y[mask.values]
        v_sub = vals[mask].values
        if len(np.unique(y_sub)) < 2 or np.std(v_sub) < 1e-10:
            continue
        auc = float(roc_auc_score(y_sub, v_sub))
        auc_inv = float(roc_auc_score(y_sub, -v_sub))
        best = max(auc, auc_inv)
        direction = "pos" if auc >= auc_inv else "neg"
        try:
            _, pval = mannwhitneyu(v_sub[y_sub == 1], v_sub[y_sub == 0], alternative="two-sided")
            pval = float(pval)
        except Exception:
            pval = 1.0
        clean_single_results.append(
            {
                "feature": feat,
                "auc": auc,
                "best_auc": best,
                "direction": direction,
                "p_value": pval,
                "n": int(mask.sum()),
            }
        )

    clean_single_results.sort(key=lambda x: float(x["best_auc"]), reverse=True)

    print(f"\n{'Rank':>4} {'Feature':<45} {'AUC':>6} {'Dir':>4} {'p':>10}")
    print("-" * 80)
    for i, sr in enumerate(clean_single_results[:30], 1):
        p = float(sr["p_value"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{i:>4} {str(sr['feature']):<45} {float(sr['best_auc']):.4f} {str(sr['direction']):>4} {p:>10.4e} {sig}")

    patients = tesla_feat["patient_id"].unique()

    bind_auc_clean = None
    for sr in clean_single_results:
        if sr["feature"] == "bind_log50k":
            bind_auc_clean = float(sr["best_auc"])
            break

    sig_clean = [
        sr
        for sr in clean_single_results
        if float(sr["p_value"]) < 0.05 and not str(sr["feature"]).startswith("bind_") and int(sr["n"]) > 500
    ]

    print(f"\nBinding baseline (in-sample): {bind_auc_clean if bind_auc_clean is not None else np.nan:.4f}")
    print(f"Significant clean non-binding features: {len(sig_clean)}")

    CLEAN_HYPOTHESES: list[dict[str, object]] = [
        {"name": "CLEAN_binding_only", "features": [("bind_log50k", 1.0, False)], "rationale": "Baseline"}
    ]

    for sf in sig_clean[:10]:
        feat = str(sf["feature"])
        inv = str(sf["direction"]) == "neg"
        for bw, fw in [(0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]:
            CLEAN_HYPOTHESES.append(
                {
                    "name": f"CLEAN_{feat}_{int(bw*100)}_{int(fw*100)}",
                    "features": [("bind_log50k", bw, False), (feat, fw, inv)],
                    "rationale": f"Binding({bw}) + {feat}({fw})",
                }
            )

    if len(sig_clean) >= 2:
        f1, f2 = sig_clean[:2]
        CLEAN_HYPOTHESES.append(
            {
                "name": "CLEAN_top2_bind",
                "features": [
                    ("bind_log50k", 0.5, False),
                    (str(f1["feature"]), 0.25, str(f1["direction"]) == "neg"),
                    (str(f2["feature"]), 0.25, str(f2["direction"]) == "neg"),
                ],
                "rationale": f"Bind + {f1['feature']} + {f2['feature']}",
            }
        )

    if len(sig_clean) >= 3:
        f1, f2, f3 = sig_clean[:3]
        CLEAN_HYPOTHESES.append(
            {
                "name": "CLEAN_top3_bind",
                "features": [
                    ("bind_log50k", 0.4, False),
                    (str(f1["feature"]), 0.2, str(f1["direction"]) == "neg"),
                    (str(f2["feature"]), 0.2, str(f2["direction"]) == "neg"),
                    (str(f3["feature"]), 0.2, str(f3["direction"]) == "neg"),
                ],
                "rationale": "Bind + top 3 clean features",
            }
        )

    CLEAN_HYPOTHESES.append(
        {
            "name": "CLEAN_calis_60_40",
            "features": [("bind_log50k", 0.6, False), ("calis_immuno", 0.4, False)],
            "rationale": "Binding + Calis (biological prior)",
        }
    )

    print(f"\nClean hypotheses: {len(CLEAN_HYPOTHESES)}")

    def score_hypothesis(df_sub: pd.DataFrame, hyp: dict[str, object]) -> np.ndarray:
        scores = np.full(len(df_sub), np.nan)
        for i in range(len(df_sub)):
            row = df_sub.iloc[i]
            av, aw = [], []
            for feat, w, inv in hyp["features"]:  # type: ignore[index]
                if feat not in df_sub.columns:
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

    clean_cv: dict[str, dict[str, object]] = {}
    for hyp in CLEAN_HYPOTHESES:
        name = str(hyp["name"])
        patient_aucs: dict[str, dict[str, float | None]] = {}
        for tp in patients:
            test_df = tesla_feat[tesla_feat["patient_id"] == tp]
            y_test = test_df["immunogenic"].values.astype(float)
            if len(np.unique(y_test)) < 2:
                patient_aucs[str(tp)] = {"auc": None}
                continue
            scores = score_hypothesis(test_df, hyp)
            valid = np.isfinite(scores)
            if int(valid.sum()) < 5 or len(np.unique(y_test[valid])) < 2:
                patient_aucs[str(tp)] = {"auc": None}
                continue
            patient_aucs[str(tp)] = {"auc": float(roc_auc_score(y_test[valid], scores[valid]))}

        comp = [v["auc"] for v in patient_aucs.values() if v["auc"] is not None]
        scores_all = score_hypothesis(tesla_feat, hyp)
        va = np.isfinite(scores_all)
        is_auc = (
            float(roc_auc_score(y[va], scores_all[va]))
            if int(va.sum()) > 20 and len(np.unique(y[va])) == 2
            else None
        )
        clean_cv[name] = {
            "mean_auc": float(np.mean(comp)) if comp else None,
            "in_sample": is_auc,
            "n_comp": int(len(comp)),
            "per_patient": patient_aucs,
            "rationale": str(hyp["rationale"]),
        }

    lr_feats_clean = ["bind_log50k"] + [str(sr["feature"]) for sr in sig_clean[:10]]
    lr_feats_clean = [f for f in lr_feats_clean if f in tesla_feat.columns]

    lr_patient_aucs_clean: dict[str, dict[str, object]] = {}
    for tp in patients:
        train = tesla_feat[tesla_feat["patient_id"] != tp]
        test = tesla_feat[tesla_feat["patient_id"] == tp]
        y_tr = train["immunogenic"].values.astype(float)
        y_te = test["immunogenic"].values.astype(float)
        if len(np.unique(y_te)) < 2:
            lr_patient_aucs_clean[str(tp)] = {"auc": None}
            continue
        X_tr = train[lr_feats_clean].fillna(0).values
        X_te = test[lr_feats_clean].fillna(0).values
        try:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(X_tr_s, y_tr)
            probs = lr.predict_proba(X_te_s)[:, 1]
            lr_patient_aucs_clean[str(tp)] = {
                "auc": float(roc_auc_score(y_te, probs)),
                "coefs": dict(zip(lr_feats_clean, lr.coef_[0].tolist())),
            }
        except Exception:
            lr_patient_aucs_clean[str(tp)] = {"auc": None}

    lr_comp_clean = [v["auc"] for v in lr_patient_aucs_clean.values() if v["auc"] is not None]
    clean_cv["CLEAN_TRAIN_FOLD_LR"] = {
        "mean_auc": float(np.mean(lr_comp_clean)) if lr_comp_clean else None,
        "in_sample": None,
        "n_comp": int(len(lr_comp_clean)),
        "per_patient": lr_patient_aucs_clean,
        "rationale": f"LR on fold: {lr_feats_clean}",
    }

    sorted_clean = sorted(clean_cv.items(), key=lambda x: x[1]["mean_auc"] if x[1]["mean_auc"] is not None else 0, reverse=True)
    baseline_clean = clean_cv.get("CLEAN_binding_only", {}).get("mean_auc", 0) or 0

    print(f"\n{'=' * 70}")
    print("CLEAN LOPO RESULTS")
    print(f"{'=' * 70}")
    print(f"\n{'Rank':>4} {'Strategy':<50} {'LOPO':>6} {'InSamp':>6} {'Delta':>7}")
    print("-" * 80)
    for rank, (name, res) in enumerate(sorted_clean, 1):
        if res["mean_auc"] is None:
            continue
        is_str = f"{res['in_sample']:.4f}" if res.get("in_sample") else "N/A"
        delta = float(res["mean_auc"]) - float(baseline_clean)
        marker = " **" if delta > 0.03 else " *" if delta > 0.01 else ""
        print(f"{rank:>4} {name:<50} {float(res['mean_auc']):.4f} {is_str:>6} {delta:+.4f}{marker}")

    print(f"\nBaseline (binding only) LOPO: {float(baseline_clean):.4f}")
    if sorted_clean:
        best_name, best_res = sorted_clean[0]
        if best_res["mean_auc"] is not None:
            print(f"Best: {best_name} LOPO: {float(best_res['mean_auc']):.4f}")
            print(f"Delta: {float(best_res['mean_auc']) - float(baseline_clean):+.4f}")

    output_clean = {
        "leaky_columns_removed": leaky_cols,
        "uncertain_columns_removed": uncertain_cols,
        "n_clean_features": len(feature_cols_clean),
        "binding_baseline_lopo": baseline_clean,
        "results": {
            name: {
                "mean_auc": res["mean_auc"],
                "in_sample": res.get("in_sample"),
                "delta": (float(res["mean_auc"]) - float(baseline_clean)) if res["mean_auc"] is not None else None,
                "rationale": res["rationale"],
            }
            for name, res in sorted_clean
        },
        "significant_clean_features": [
            {
                "feature": sr["feature"],
                "auc": sr["best_auc"],
                "p": sr["p_value"],
                "direction": sr["direction"],
            }
            for sr in sig_clean
        ],
    }

    with open("backend/validation/artifacts/tesla_clean_results.json", "w", encoding="utf-8") as f:
        json.dump(output_clean, f, indent=2, default=str)

    print("\nSaved: tesla_clean_results.json")
    print(f"\n{'=' * 70}")
    print("CLEAN RERUN COMPLETE")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

