from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.validation.full_cross_validation import (
    ARTIFACTS,
    compute_all_features,
    insample_auc,
    is_leaky_feature_name,
    lopo_auc,
    lopo_auc_with_train_fold_directions,
    make_serializable,
    resolve_auto_directions,
)


def build_transfer_strategies() -> OrderedDict[str, dict[str, object]]:
    transfer_strategies: OrderedDict[str, dict[str, object]] = OrderedDict()
    transfer_strategies["binding_only"] = {
        "features": [("bind_log50k", 1.0, False)],
        "description": "MHC binding log50k",
    }
    transfer_strategies["rl_v1_original"] = {
        "features": [("pres_existing", 0.50, False), ("expr_existing", 0.33, False), ("dissim_existing", 0.17, False)],
        "description": "Original rl_v1",
    }
    transfer_strategies["rl_tcr_v1_from_ott"] = {
        "features": [("bind_log50k", 0.4, False), ("tcr_volume", 0.2, False), ("tcr_charge_diff", 0.2, False), ("tcr_hydro", 0.2, False)],
        "description": "Ott H14 locked directions",
    }
    transfer_strategies["rl_expression_v1_from_tesla"] = {
        "features": [("bind_log50k", 0.50, False), ("nm_raw", 0.25, True), ("expr_raw", 0.25, False)],
        "description": "TESLA expression strategy",
    }
    transfer_strategies["binding_plus_calis"] = {
        "features": [("bind_log50k", 0.6, False), ("calis_immuno", 0.4, False)],
        "description": "Bind + Calis",
    }
    transfer_strategies["bind_exp500"] = {
        "features": [("bind_exp500", 1.0, False)],
        "description": "Binding exp500",
    }
    transfer_strategies["bind_sigmoid500"] = {
        "features": [("bind_sigmoid500", 1.0, False)],
        "description": "Binding sigmoid500",
    }
    return transfer_strategies


def run_discovery(pn: str, df: pd.DataFrame) -> dict[str, object]:
    y = df["immunogenic"].values.astype(float)
    n_rows, n_pos, n_pat = len(df), int(df["immunogenic"].sum()), int(df["patient_id"].nunique())
    if n_pos < 5:
        return {"status": "skipped_too_few_positives", "n_pos": n_pos}

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
    hyp_res: dict[str, dict[str, object]] = {}
    for h in hyps:
        resolved_full = resolve_auto_directions(df, h["features"])
        is_auc = insample_auc(df, resolved_full)
        lopo, per = lopo_auc_with_train_fold_directions(df, h["features"]) if can_lopo else (None, {})
        hyp_res[h["name"]] = {"lopo_auc": lopo, "insample_auc": is_auc, "features": h["features"], "resolved_full_features": resolved_full, "per_patient": per}

    if can_lopo and len(sig) >= 1:
        lr_feats = ["bind_log50k"] + [s["feature"] for s in sig[:5] if s["feature"] in df.columns and not is_leaky_feature_name(str(s["feature"]))]
        lr_feats = [f for f in lr_feats if f in df.columns]
        if len(lr_feats) >= 2:
            lr_pp: dict[str, float | None] = {}
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

    sorted_h = sorted(
        hyp_res.items(),
        key=lambda x: (
            x[1]["lopo_auc"] if x[1]["lopo_auc"] is not None else x[1]["insample_auc"] if x[1]["insample_auc"] is not None else -1
        ),
        reverse=True,
    )
    baseline = hyp_res.get(f"{pn}_binding_only", {}).get("lopo_auc")
    if baseline is None:
        baseline = hyp_res.get(f"{pn}_binding_only", {}).get("insample_auc", 0)
    best_name = sorted_h[0][0] if sorted_h else None
    best_auc = (
        sorted_h[0][1]["lopo_auc"]
        if sorted_h and sorted_h[0][1]["lopo_auc"] is not None
        else sorted_h[0][1]["insample_auc"]
        if sorted_h
        else None
    )
    best_delta = (best_auc - baseline) if (best_auc is not None and baseline is not None) else None

    return {
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


def main() -> int:
    rojas_path = Path("validation_papers/rojas_with_binding.csv")
    hilf_path = Path("validation_papers/hilf_with_binding.csv")
    if not rojas_path.exists() or not hilf_path.exists():
        raise FileNotFoundError("Required files missing. Run Task 1 first: validation_papers/rojas_with_binding.csv and validation_papers/hilf_with_binding.csv")

    rojas = pd.read_csv(rojas_path)
    hilf = pd.read_csv(hilf_path)
    rojas = rojas[rojas["immunogenic"].isin([0, 1])].copy()
    hilf = hilf[hilf["immunogenic"].isin([0, 1])].copy()

    featured = {
        "rojas_2023": compute_all_features(rojas, "rojas_2023"),
        "hilf_2019": compute_all_features(hilf, "hilf_2019"),
    }
    transfer_strategies = build_transfer_strategies()

    transfer_results: dict[str, object] = {}
    for pn, df in featured.items():
        n_pat = int(df["patient_id"].nunique())
        can_lopo = n_pat >= 3
        pres = {}
        for sn, sd in transfer_strategies.items():
            fl = sd["features"]  # type: ignore[index]
            available, missing = [], []
            for feat, _, _ in fl:  # type: ignore[misc]
                if feat in df.columns and pd.to_numeric(df[feat], errors="coerce").notna().sum() > 5:
                    available.append(feat)
                else:
                    missing.append(feat)
            if not available:
                pres[sn] = {"lopo_auc": None, "insample_auc": None, "status": f"no features available (missing: {missing})"}
                continue
            is_auc = insample_auc(df, fl)  # type: ignore[arg-type]
            lopo, per = lopo_auc(df, fl) if can_lopo else (None, {})
            pres[sn] = {
                "lopo_auc": lopo,
                "insample_auc": is_auc,
                "per_patient": per,
                "features_available": available,
                "features_missing": missing,
                "status": "computed",
            }
        transfer_results[pn] = {
            "n_rows": int(len(df)),
            "n_immunogenic": int(df["immunogenic"].sum()),
            "n_patients": n_pat,
            "can_lopo": can_lopo,
            "strategies": pres,
        }

    discovery_results = {pn: run_discovery(pn, df) for pn, df in featured.items()}

    rows = []
    for sn in transfer_strategies.keys():
        row = {"strategy": sn}
        for pn in ["rojas_2023", "hilf_2019"]:
            rs = transfer_results[pn]["strategies"].get(sn, {})  # type: ignore[index]
            row[f"{pn}_lopo"] = rs.get("lopo_auc") if rs else None
            row[f"{pn}_insample"] = rs.get("insample_auc") if rs else None
        rows.append(row)
    for pn, d in discovery_results.items():
        if d.get("best_strategy"):
            row = {"strategy": f"BEST_OF_{pn}"}
            if d.get("can_lopo"):
                row[f"{pn}_lopo"] = d.get("best_auc")
            else:
                row[f"{pn}_insample"] = d.get("best_auc")
            rows.append(row)
    cross_matrix = pd.DataFrame(rows)

    old = json.loads((ARTIFACTS / "full_cross_validation_results.json").read_text(encoding="utf-8"))
    old_hilf_binding = old["transfer_results"]["hilf_2019"]["strategies"]["binding_only"].get("lopo_auc")
    old_hilf_expr = old["transfer_results"]["hilf_2019"]["strategies"]["rl_expression_v1_from_tesla"].get("lopo_auc")
    new_hilf_binding = transfer_results["hilf_2019"]["strategies"]["binding_only"].get("lopo_auc")  # type: ignore[index]
    new_hilf_expr = transfer_results["hilf_2019"]["strategies"]["rl_expression_v1_from_tesla"].get("lopo_auc")  # type: ignore[index]

    output = {
        "summary": {
            "papers_tested": ["rojas_2023", "hilf_2019"],
            "note": "PHASE 4/5 rerun using rojas_with_binding.csv and hilf_with_binding.csv",
        },
        "transfer_results": make_serializable(transfer_results),
        "discovery_results": make_serializable(discovery_results),
        "hilf_binding_vs_expression_lopo": {
            "old_binding_only": old_hilf_binding,
            "old_rl_expression_v1_from_tesla": old_hilf_expr,
            "new_binding_only": new_hilf_binding,
            "new_rl_expression_v1_from_tesla": new_hilf_expr,
        },
    }

    out_json = ARTIFACTS / "rojas_hilf_cross_validation_with_binding.json"
    out_csv = ARTIFACTS / "rojas_hilf_cross_matrix_with_binding.csv"
    out_json.write_text(json.dumps(make_serializable(output), indent=2, default=str), encoding="utf-8")
    cross_matrix.to_csv(out_csv, index=False)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(json.dumps(output["hilf_binding_vs_expression_lopo"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
