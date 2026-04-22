from __future__ import annotations

import json
from itertools import combinations, product as iprod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < 2:
        return None
    yy = y[m]
    ss = s[m]
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, ss))


def hamming_prop(mut, wt):
    if pd.isna(mut) or pd.isna(wt) or mut == "" or wt == "":
        return np.nan
    mut, wt = str(mut), str(wt)
    minlen = min(len(mut), len(wt))
    if minlen == 0:
        return np.nan
    return sum(a != b for a, b in zip(mut[:minlen], wt[:minlen])) / minlen


def get_transform(df: pd.DataFrame, name: str) -> pd.Series:
    nm = pd.to_numeric(df["binding_affinity_nm_numeric"], errors="coerce").copy()
    tpm = pd.to_numeric(df["expression_tpm_numeric"], errors="coerce").copy()
    sd = pd.to_numeric(df["self_dissimilarity"], errors="coerce").copy()
    if name == "bind_log50k":
        return (1 - np.log10(nm.clip(lower=0.1)) / np.log10(50000)).clip(0, 1)
    elif name == "bind_log5k":
        return (1 - np.log10(nm.clip(lower=0.1)) / np.log10(5000)).clip(0, 1)
    elif name == "bind_sigmoid500":
        return 1.0 / (1.0 + nm / 500)
    elif name == "bind_sigmoid50":
        return 1.0 / (1.0 + nm / 50)
    elif name == "bind_exp500":
        return np.exp(-nm / 500)
    elif name == "bind_exp150":
        return np.exp(-nm / 150)
    elif name == "bind_neglog":
        return -np.log10(nm.clip(lower=0.1))
    elif name == "expr_current":
        return pd.to_numeric(df["expression_score"], errors="coerce").copy()
    elif name == "expr_log2":
        return np.log2(tpm.clip(lower=0) + 1)
    elif name == "expr_sqrt":
        return np.sqrt(tpm.clip(lower=0))
    elif name == "expr_sigmoid10":
        return tpm / (tpm + 10)
    elif name == "expr_quantile":
        return tpm.rank(pct=True)
    elif name == "dissim_current":
        return sd
    elif name == "dissim_inverted":
        return 1 - sd
    elif name == "dissim_hamming":
        return df.apply(lambda r: hamming_prop(r["mutant_peptide"], r["wildtype_peptide"]), axis=1)
    else:
        raise ValueError(f"Unknown transform: {name}")


def dynamic_score_array(bind_s: pd.Series, expr_s: pd.Series, dissim_s: pd.Series, weights: np.ndarray) -> np.ndarray:
    b = pd.to_numeric(bind_s, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(expr_s, errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(dissim_s, errors="coerce").to_numpy(dtype=float)
    w0, w1, w2 = float(weights[0]), float(weights[1]), float(weights[2])

    mb = np.isfinite(b) & (b > 0) & (w0 > 0)
    me = np.isfinite(e) & (e > 0) & (w1 > 0)
    md = np.isfinite(d) & (d > 0) & (w2 > 0)

    num = np.zeros_like(b, dtype=float)
    den = np.zeros_like(b, dtype=float)
    num += np.where(mb, b * w0, 0.0)
    den += np.where(mb, w0, 0.0)
    num += np.where(me, e * w1, 0.0)
    den += np.where(me, w1, 0.0)
    num += np.where(md, d * w2, 0.0)
    den += np.where(md, w2, 0.0)

    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def weight_grid(step: float = 0.1) -> list[np.ndarray]:
    k = int(round(1.0 / step))
    uniq: set[tuple[float, float, float]] = set()
    for a, b, c in iprod(range(k + 1), range(k + 1), range(k + 1)):
        s = a + b + c
        if s == 0:
            continue
        uniq.add((round(a / s, 6), round(b / s, 6), round(c / s, 6)))
    return [np.array(w, dtype=float) for w in sorted(uniq)]


def score_config(df: pd.DataFrame, config: dict) -> np.ndarray:
    b = get_transform(df, config["bind"]) if config.get("bind") else pd.Series(np.nan, index=df.index)
    e = get_transform(df, config["expr"]) if config.get("expr") else pd.Series(np.nan, index=df.index)
    d = get_transform(df, config["dissim"]) if config.get("dissim") else pd.Series(np.nan, index=df.index)
    w = np.array(config["weights"], dtype=float)
    return dynamic_score_array(b, e, d, w)


def mean_nonnull(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def main() -> int:
    art = Path("backend/validation/artifacts")
    tm = pd.read_csv(art / "training_matrix.csv")
    ott = tm[tm["paper_source"] == "ott_2017"].copy()

    # mutation-level collapse
    mutation_level = (
        ott.groupby(["patient_id", "gene", "protein_change"])
        .agg(
            {
                "immunogenic": "max",
                "expression_tpm_numeric": "first",
                "expression_score": "first",
                "presentation_score": "max",
                "binding_affinity_nm_numeric": "min",
                "self_dissimilarity": "mean",
                "mutant_peptide": "first",
                "wildtype_peptide": "first",
                "cancer_type": "first",
            }
        )
        .reset_index()
    )
    mutation_level = mutation_level[mutation_level["immunogenic"].isin([0, 1])].copy()
    for c in ["immunogenic", "expression_tpm_numeric", "expression_score", "presentation_score", "binding_affinity_nm_numeric", "self_dissimilarity"]:
        mutation_level[c] = pd.to_numeric(mutation_level[c], errors="coerce")

    bind_transform_names = ["bind_log50k", "bind_sigmoid500", "bind_exp500", "bind_exp150", "bind_neglog"]
    expr_transform_names = ["expr_current", "expr_log2", "expr_sqrt", "expr_sigmoid10", "expr_quantile"]
    dissim_transform_names = ["dissim_current", "dissim_inverted", "dissim_hamming"]
    w_grid = weight_grid(step=0.1)

    # load sweep winner if available
    sweep_file = art / "feature_sweep_results.json"
    sweep_weights = [0.77, 0.15, 0.08]
    if sweep_file.exists():
        try:
            sweep = json.loads(sweep_file.read_text(encoding="utf-8"))
            bw = sweep.get("best_combo", {}).get("weights")
            if isinstance(bw, list) and len(bw) == 3:
                sweep_weights = [float(bw[0]), float(bw[1]), float(bw[2])]
        except Exception:
            pass

    config_binding_only = {"bind": "bind_log50k", "expr": None, "dissim": None, "weights": [1.0, 0.0, 0.0]}
    config_rl_v1 = {"bind": "bind_log50k", "expr": "expr_current", "dissim": "dissim_current", "weights": [0.50, 0.33, 0.17]}
    config_equal = {"bind": "bind_log50k", "expr": "expr_current", "dissim": "dissim_current", "weights": [0.333, 0.333, 0.333]}
    config_sweep_winner = {"bind": "bind_exp500", "expr": "expr_current", "dissim": "dissim_hamming", "weights": sweep_weights}

    patients = sorted(mutation_level["patient_id"].astype(str).unique().tolist())
    results_by_patient: dict[str, dict] = {}

    print("=" * 70)
    print("NESTED CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(f"Rows: {len(mutation_level)}, Positives: {int(mutation_level['immunogenic'].sum())}, Patients: {len(patients)}")

    # outer loop
    for idx, test_patient in enumerate(patients, start=1):
        print(f"Fold {idx}/{len(patients)}: testing patient {test_patient}...")
        test_mask = mutation_level["patient_id"].astype(str) == test_patient
        train = mutation_level[~test_mask].copy()
        test = mutation_level[test_mask].copy()

        y_train = train["immunogenic"].to_numpy(dtype=int)
        y_test = test["immunogenic"].to_numpy(dtype=int)
        if len(np.unique(y_test)) < 2:
            results_by_patient[test_patient] = {"status": "skipped_no_both_classes", "n": len(test), "n_pos": int(y_test.sum())}
            continue

        best_inner_auc = -1.0
        best_inner_config = None

        # precompute transforms for speed
        all_t = bind_transform_names + expr_transform_names + dissim_transform_names
        train_t = {name: get_transform(train, name) for name in all_t}
        test_t = {name: get_transform(test, name) for name in all_t}

        # inner grid
        for bt_name in bind_transform_names:
            bt_train = train_t[bt_name]
            for et_name in expr_transform_names:
                et_train = train_t[et_name]
                for dt_name in dissim_transform_names:
                    dt_train = train_t[dt_name]
                    for ws in w_grid:
                        scores_train = dynamic_score_array(bt_train, et_train, dt_train, ws)
                        auc = _safe_auc(y_train.astype(float), scores_train.astype(float))
                        if auc is None:
                            continue
                        if auc > best_inner_auc:
                            best_inner_auc = auc
                            best_inner_config = {
                                "method": "weighted_sum",
                                "bind": bt_name,
                                "expr": et_name,
                                "dissim": dt_name,
                                "weights": ws.tolist(),
                                "train_auc": float(auc),
                            }

        lr = None
        scaler = None
        lr_train_auc = None
        if best_inner_config:
            bt_name = best_inner_config["bind"]
            et_name = best_inner_config["expr"]
            dt_name = best_inner_config["dissim"]
            bt_train = train_t[bt_name]
            et_train = train_t[et_name]
            dt_train = train_t[dt_name]
            X_train = np.column_stack([bt_train.fillna(0).values, et_train.fillna(0).values, dt_train.fillna(0).values])
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
                lr.fit(X_scaled, y_train)
                lr_train_auc = float(roc_auc_score(y_train, lr.predict_proba(X_scaled)[:, 1]))
            except Exception:
                lr = None
                scaler = None
                lr_train_auc = None

        auc_test_ws = None
        auc_test_lr = None
        if best_inner_config:
            bt_test = test_t[best_inner_config["bind"]]
            et_test = test_t[best_inner_config["expr"]]
            dt_test = test_t[best_inner_config["dissim"]]
            ws = np.array(best_inner_config["weights"], dtype=float)
            scores_test_ws = dynamic_score_array(bt_test, et_test, dt_test, ws)
            auc_test_ws = _safe_auc(y_test.astype(float), scores_test_ws.astype(float))

        if lr is not None and scaler is not None and best_inner_config:
            bt_test = test_t[best_inner_config["bind"]]
            et_test = test_t[best_inner_config["expr"]]
            dt_test = test_t[best_inner_config["dissim"]]
            X_test = np.column_stack([bt_test.fillna(0).values, et_test.fillna(0).values, dt_test.fillna(0).values])
            try:
                X_test_scaled = scaler.transform(X_test)
                probs_test = lr.predict_proba(X_test_scaled)[:, 1]
                auc_test_lr = _safe_auc(y_test.astype(float), probs_test.astype(float))
            except Exception:
                auc_test_lr = None

        # fixed configs
        fixed_scores = {}
        for label, cfg in [
            ("baseline", config_binding_only),
            ("rl_v1", config_rl_v1),
            ("equal", config_equal),
            ("sweep_winner", config_sweep_winner),
        ]:
            s = score_config(test, cfg)
            fixed_scores[f"{label}_auc"] = _safe_auc(y_test.astype(float), s.astype(float))

        results_by_patient[test_patient] = {
            "status": "computed",
            "n": int(len(test)),
            "n_pos": int(y_test.sum()),
            **fixed_scores,
            "weighted_sum_auc": auc_test_ws,
            "lr_auc": auc_test_lr,
            "inner_config": best_inner_config,
            "inner_train_auc": float(best_inner_auc) if np.isfinite(best_inner_auc) else None,
            "lr_train_auc": lr_train_auc,
        }

    # summary tables
    rows = []
    for p in patients:
        r = results_by_patient.get(p, {})
        rows.append(
            {
                "patient": p,
                "N": r.get("n"),
                "N_pos": r.get("n_pos"),
                "Baseline": r.get("baseline_auc"),
                "rl_v1": r.get("rl_v1_auc"),
                "Equal": r.get("equal_auc"),
                "SweepWin": r.get("sweep_winner_auc"),
                "NestedCV_WS": r.get("weighted_sum_auc"),
                "NestedCV_LR": r.get("lr_auc"),
            }
        )
    table = pd.DataFrame(rows)

    means = {
        "baseline": mean_nonnull([results_by_patient[p].get("baseline_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
        "rl_v1": mean_nonnull([results_by_patient[p].get("rl_v1_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
        "equal": mean_nonnull([results_by_patient[p].get("equal_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
        "sweep_winner": mean_nonnull([results_by_patient[p].get("sweep_winner_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
        "nested_ws": mean_nonnull([results_by_patient[p].get("weighted_sum_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
        "nested_lr": mean_nonnull([results_by_patient[p].get("lr_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"]),
    }

    baseline = means["baseline"]
    deltas = {k: (None if v is None or baseline is None else float(v - baseline)) for k, v in means.items() if k != "baseline"}

    inner_configs = [
        {
            "test_patient": p,
            "bind": results_by_patient[p]["inner_config"]["bind"],
            "expr": results_by_patient[p]["inner_config"]["expr"],
            "dissim": results_by_patient[p]["inner_config"]["dissim"],
            "weights": results_by_patient[p]["inner_config"]["weights"],
        }
        for p in patients
        if results_by_patient.get(p, {}).get("status") == "computed" and results_by_patient[p].get("inner_config") is not None
    ]
    config_signatures = [f"{x['bind']}|{x['expr']}|{x['dissim']}|{','.join(f'{w:.2f}' for w in x['weights'])}" for x in inner_configs]
    unique_config_count = len(set(config_signatures))

    inner_train_mean = mean_nonnull([results_by_patient[p].get("inner_train_auc") for p in patients if results_by_patient.get(p, {}).get("status") == "computed"])
    outer_test_mean = means["nested_ws"]
    train_test_gap = None if inner_train_mean is None or outer_test_mean is None else float(inner_train_mean - outer_test_mean)

    # top 5 strategy scores (by mean AUC)
    strategy_mean = {
        "Binding only (pre-specified)": means["baseline"],
        "rl_v1 (pre-specified)": means["rl_v1"],
        "Equal weights (pre-specified)": means["equal"],
        "Sweep winner (pre-specified)": means["sweep_winner"],
        "Nested CV weighted sum": means["nested_ws"],
        "Nested CV logistic regression": means["nested_lr"],
    }
    top5 = sorted(
        [(k, v) for k, v in strategy_mean.items() if v is not None and np.isfinite(v)],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    print("\nPER-PATIENT RESULTS:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))

    print("\nMEAN AUC ACROSS PATIENTS:")
    for k, label in [
        ("baseline", "Binding only (pre-specified)"),
        ("rl_v1", "rl_v1 weights (pre-specified)"),
        ("equal", "Equal weights (pre-specified)"),
        ("sweep_winner", "Sweep winner (pre-specified)"),
        ("nested_ws", "Nested CV weighted sum"),
        ("nested_lr", "Nested CV logistic regression"),
    ]:
        v = means[k]
        print(f"  {label:<35} {v:.4f}" if v is not None else f"  {label:<35} N/A")

    print("\nDELTA OVER BASELINE:")
    for k, label in [
        ("rl_v1", "rl_v1"),
        ("equal", "Equal"),
        ("sweep_winner", "Sweep winner"),
        ("nested_ws", "Nested CV WS"),
        ("nested_lr", "Nested CV LR"),
    ]:
        d = deltas.get(k)
        print(f"  {label:<15} {d:+.4f}" if d is not None else f"  {label:<15} N/A")

    print("\nINNER LOOP SELECTED CONFIGS:")
    for i, c in enumerate(inner_configs, start=1):
        ws = ",".join(f"{w:.2f}" for w in c["weights"])
        print(f"  Fold {i} (test={c['test_patient']}): bind={c['bind']} expr={c['expr']} dissim={c['dissim']} w=[{ws}]")

    print("\nCONSISTENCY CHECK:")
    print(f"  Unique selected configs across folds: {unique_config_count}/{len(inner_configs)}")
    if unique_config_count <= 2:
        print("  Signal appears stable across folds.")
    elif unique_config_count <= 4:
        print("  Moderate config drift across folds.")
    else:
        print("  High config instability across folds (likely noise/overfit).")

    print("\nTRAIN vs TEST GAP:")
    print(f"  Mean inner train AUC: {inner_train_mean:.4f}" if inner_train_mean is not None else "  Mean inner train AUC: N/A")
    print(f"  Mean outer test AUC:  {outer_test_mean:.4f}" if outer_test_mean is not None else "  Mean outer test AUC: N/A")
    if train_test_gap is not None:
        print(f"  Gap:                  {train_test_gap:.4f}")
        if train_test_gap > 0.15:
            print("  Severe overfitting warning.")

    print("\nTOP 5 STRATEGY SCORES (mean AUC):")
    for i, (name, auc) in enumerate(top5, start=1):
        print(f"  {i}. {name}: {auc:.4f}")

    payload = {
        "dataset": {
            "rows": int(len(mutation_level)),
            "positives": int(mutation_level["immunogenic"].sum()),
            "patients": patients,
        },
        "results_by_patient": results_by_patient,
        "means": means,
        "deltas_over_baseline": deltas,
        "inner_configs": inner_configs,
        "consistency": {
            "unique_config_count": unique_config_count,
            "total_computed_folds": len(inner_configs),
        },
        "train_test_gap": {
            "mean_inner_train_auc": inner_train_mean,
            "mean_outer_test_auc": outer_test_mean,
            "gap": train_test_gap,
        },
        "top5_strategies": [{"name": n, "mean_auc": float(a)} for n, a in top5],
        "fixed_configs": {
            "binding_only": config_binding_only,
            "rl_v1": config_rl_v1,
            "equal": config_equal,
            "sweep_winner": config_sweep_winner,
        },
    }
    out = Path("backend/validation/artifacts/nested_cv_results.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
