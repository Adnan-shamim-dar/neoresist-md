"""
PHASE 10 — Hilf 2019 + Keskin 2019 GBM LOPO Validation.

Dataset: backend/strategy_engine/artifacts/hilf_2019_features.csv  (152 rows, 15 patients)
         backend/strategy_engine/artifacts/keskin_2019_features.csv  (27 rows, 2 patients)
  Pooled: 179 rows, 17 patients.  Peptide-level (no dedup: protein_change is NaN for
  many Hilf rows where only the peptide sequence was reported, not the amino-acid change).

Scorers:
  1. binding_only       (-binding_log)
  2. rl_tcr_v1          (weighted composite; uses presentation_score + TCR features)
  3. gbm_ml_v1          (LOPO XGBoost, same hyperparams as melanoma_ml_v1)

Cross-transfer:
  4. melanoma_ml_v1.pkl → predict GBM pooled data → AUC
  5. gbm_ml_v1 (trained on all GBM) → predict Ott+Sahin pooled → AUC

If gbm_ml_v1 LOPO AUC > binding_only + 0.05:
  Update strategies/gbm_ml_v1.yaml  validated: true
  Else: validated: false  (corrects Phase-5 over-optimistic flag)

Run: py -3.11 backend/strategy_engine/phase10_gbm_validation.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS  = Path("backend/strategy_engine/artifacts")
STRATEGIES = Path("backend/strategy_engine/strategies")

XGB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                  eval_metric="logloss", random_state=42, verbosity=0)

CANDIDATE_FEATURES = [
    "tcr_hydro_mean", "tcr_hydro_max", "tcr_volume_mean", "tcr_charge_sum",
    "tcr_aromatic_count", "tcr_hydro_diff", "tcr_volume_diff", "tcr_charge_diff",
    "self_dissimilarity", "hamming_distance", "blosum62_score", "blosum62_at_mutation",
    "calis_simplified", "pep_length", "mutation_position",
    "hydro_full_mean", "hydro_full_max", "aliphatic_index",
    "expression_log2", "expression_zscore",
    "binding_log", "binding_sigmoid", "presentation_score",
]


def make_xgb(scale_pos_weight: float = 1.0):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y, s = np.array(y), np.array(s)
    m = np.isfinite(y) & np.isfinite(s) & np.isin(y, [0, 1])
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), s[m]))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n: int = 1000,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = safe_auc(y[idx], s[idx])
        if a is not None:
            aucs.append(a)
    if not aucs:
        return (0.5, 0.5)
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    idx = np.argsort(-np.array(scores))[:k]
    return float(np.array(y)[idx].sum() / max(np.array(y).sum(), 1))


def select_features(df: pd.DataFrame, candidates: list[str],
                    min_coverage: float = 0.50) -> list[str]:
    selected = []
    for f in candidates:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if col.notna().mean() >= min_coverage and col.std() > 1e-9:
            selected.append(f)
    return selected


def rl_tcr_v1_score(df: pd.DataFrame) -> pd.Series:
    def _n(col: str) -> pd.Series:
        return (pd.to_numeric(df[col], errors="coerce")
                if col in df.columns
                else pd.Series(np.nan, index=df.index))

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=s.index)

    bind_nm = _n("binding_nm").where(_n("binding_nm").notna(),
                                     _n("final_binding_affinity_nm"))
    bind_log = _n("binding_log")
    bind_for_present = bind_nm.where(bind_nm.notna(),
                                     10 ** bind_log.where(bind_log.notna(), np.nan))
    present = _norm(-np.log10(bind_for_present.replace(0, np.nan)))

    if "presentation_score" in df.columns:
        ps = pd.to_numeric(df["presentation_score"], errors="coerce")
        if ps.notna().mean() > 0.5 and ps.std() > 1e-6:
            present = _norm(ps)

    tcr_vol = _n("tcr_volume_diff")
    if tcr_vol.isna().mean() > 0.5:
        tcr_vol = _n("tcr_volume_mean")
    tcr_chg = _n("tcr_charge_diff")
    tcr_hyd = _n("tcr_hydro_diff")
    if tcr_hyd.isna().mean() > 0.5:
        tcr_hyd = _n("tcr_hydro_mean")

    terms = [
        (present, 0.40),
        (_norm(tcr_vol.fillna(0)), 0.20),
        (_norm(tcr_chg.fillna(0)) if tcr_chg.notna().sum() > 5 else pd.Series(0.0, index=df.index), 0.20),
        (_norm(tcr_hyd.fillna(0)) if tcr_hyd.notna().sum() > 5 else pd.Series(0.0, index=df.index), 0.20),
    ]
    return sum(v * w for v, w in terms)


def build_X(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cols = []
    for f in feats:
        col = pd.to_numeric(df[f], errors="coerce").values if f in df.columns else np.zeros(len(df))
        med = np.nanmedian(col)
        col = np.where(np.isnan(col), med if np.isfinite(med) else 0.0, col)
        cols.append(col)
    return np.column_stack(cols) if cols else np.zeros((len(df), 1))


def lopo(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof_y, oof_s = np.zeros(len(df)), np.full(len(df), np.nan)
    for pt, idx in df.groupby("patient_id").groups.items():
        mask_train = df.index.difference(idx)
        if len(mask_train) == 0:
            continue
        tr, te = df.loc[mask_train], df.loc[idx]
        y_tr = tr["immunogenic"].values.astype(int)
        y_te = te["immunogenic"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = neg / max(pos, 1)
        clf = make_xgb(spw)
        clf.fit(build_X(tr, feats), y_tr)
        oof_y[idx - df.index[0]] = y_te
        oof_s[idx - df.index[0]] = clf.predict_proba(build_X(te, feats))[:, 1]
    return oof_y, oof_s


def _lopo_by_position(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """LOPO returning arrays aligned to df.iloc positions."""
    df = df.reset_index(drop=True)
    oof_y = np.zeros(len(df))
    oof_s = np.full(len(df), np.nan)
    for pt, grp in df.groupby("patient_id"):
        te_idx = grp.index.tolist()
        tr_idx = df.index.difference(te_idx).tolist()
        if not tr_idx:
            continue
        tr, te = df.loc[tr_idx], df.loc[te_idx]
        y_tr = tr["immunogenic"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        pos = y_tr.sum()
        spw = (len(y_tr) - pos) / max(pos, 1)
        clf = make_xgb(spw)
        clf.fit(build_X(tr, feats), y_tr)
        oof_y[te_idx] = te["immunogenic"].values.astype(int)
        oof_s[te_idx] = clf.predict_proba(build_X(te, feats))[:, 1]
    return oof_y, oof_s


def main() -> int:
    print("=" * 70)
    print("PHASE 10 — GBM Validation (Hilf 2019 + Keskin 2019)")
    print("=" * 70)

    # ── load and pool ──────────────────────────────────────────────────────────
    hilf   = pd.read_csv(ARTIFACTS / "hilf_2019_features.csv")
    keskin = pd.read_csv(ARTIFACTS / "keskin_2019_features.csv")
    df = pd.concat([hilf, keskin], ignore_index=True)
    df["immunogenic"] = pd.to_numeric(df["immunogenic"], errors="coerce")
    df = df[df["immunogenic"].isin([0, 1])].reset_index(drop=True)

    print(f"\nPooled: {len(df)} rows | {df['patient_id'].nunique()} patients "
          f"| {int((df['immunogenic']==1).sum())} positives")

    # ── feature selection ──────────────────────────────────────────────────────
    feats = select_features(df, CANDIDATE_FEATURES)
    print(f"\nSelected features ({len(feats)}): {feats}")

    # ── scorer 1: binding_only ─────────────────────────────────────────────────
    bind_score = -pd.to_numeric(df["binding_log"], errors="coerce").values
    y_all = df["immunogenic"].values.astype(int)

    bind_auc = safe_auc(y_all, bind_score)
    bind_ci  = bootstrap_ci(y_all, bind_score)

    # ── scorer 2: rl_tcr_v1 ───────────────────────────────────────────────────
    rl_score = rl_tcr_v1_score(df).values
    rl_auc   = safe_auc(y_all, rl_score)
    rl_ci    = bootstrap_ci(y_all, rl_score)

    # ── scorer 3: gbm_ml_v1 LOPO ──────────────────────────────────────────────
    print("\nRunning LOPO …")
    ml_y, ml_s = _lopo_by_position(df, feats)
    valid_mask = np.isfinite(ml_s)
    ml_auc = safe_auc(ml_y[valid_mask], ml_s[valid_mask])
    ml_ci  = bootstrap_ci(ml_y[valid_mask], ml_s[valid_mask])

    r10_bind = recall_at_k(y_all, bind_score, 10)
    r10_rl   = recall_at_k(y_all, rl_score, 10)
    r10_ml   = recall_at_k(ml_y[valid_mask], ml_s[valid_mask], 10)
    r20_bind = recall_at_k(y_all, bind_score, 20)
    r20_rl   = recall_at_k(y_all, rl_score, 20)
    r20_ml   = recall_at_k(ml_y[valid_mask], ml_s[valid_mask], 20)

    # ── cross-transfer 1: melanoma_ml_v1 → GBM ────────────────────────────────
    mel_obj   = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    mel_model = mel_obj["model"]
    mel_feats = mel_obj["features"]
    mel_X     = build_X(df, mel_feats)
    mel_s_gbm = mel_model.predict_proba(mel_X)[:, 1]
    mel_to_gbm_auc = safe_auc(y_all, mel_s_gbm)

    # ── cross-transfer 2: gbm_ml_v1 → Ott+Sahin ──────────────────────────────
    ott   = pd.read_csv(ARTIFACTS / "ott_2017_features.csv")
    sahin = pd.read_csv(ARTIFACTS / "sahin_2017_features.csv")
    mel_df = pd.concat([ott, sahin], ignore_index=True)
    mel_df["immunogenic"] = pd.to_numeric(mel_df["immunogenic"], errors="coerce")
    mel_df = mel_df[mel_df["immunogenic"].isin([0, 1])].reset_index(drop=True)

    gbm_full_clf = make_xgb((y_all == 0).sum() / max((y_all == 1).sum(), 1))
    gbm_full_clf.fit(build_X(df, feats), y_all)
    mel_y_true = mel_df["immunogenic"].values.astype(int)
    gbm_to_mel_s   = gbm_full_clf.predict_proba(build_X(mel_df, feats))[:, 1]
    gbm_to_mel_auc = safe_auc(mel_y_true, gbm_to_mel_s)

    # Save full model for cross-transfer reuse
    joblib.dump({"model": gbm_full_clf, "features": feats},
                ARTIFACTS / "stage2_gbm_model_v2.pkl")

    # ── print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Scorer':<28} {'AUC':>8}   {'95% CI':>18}   {'R@10':>6}  {'R@20':>6}")
    print("-" * 70)
    print(f"{'binding_only':<28} {bind_auc:>8.4f}   "
          f"[{bind_ci[0]:.4f},{bind_ci[1]:.4f}]   {r10_bind:>6.3f}  {r20_bind:>6.3f}")
    print(f"{'rl_tcr_v1':<28} {rl_auc:>8.4f}   "
          f"[{rl_ci[0]:.4f},{rl_ci[1]:.4f}]   {r10_rl:>6.3f}  {r20_rl:>6.3f}")
    print(f"{'gbm_ml_v1 (LOPO)':<28} {ml_auc:>8.4f}   "
          f"[{ml_ci[0]:.4f},{ml_ci[1]:.4f}]   {r10_ml:>6.3f}  {r20_ml:>6.3f}")
    print("-" * 70)
    print(f"{'melanoma→GBM (transfer)':<28} {mel_to_gbm_auc:>8.4f}")
    print(f"{'gbm→melanoma (transfer)':<28} {gbm_to_mel_auc:>8.4f}")
    print("=" * 70)

    threshold = (bind_auc or 0) + 0.05
    validated = bool(ml_auc is not None and ml_auc > threshold)
    print(f"\nThreshold (binding + 0.05): {threshold:.4f}")
    print(f"gbm_ml_v1 AUC {ml_auc:.4f} → validated={'True' if validated else 'False'}")

    # ── update gbm_ml_v1.yaml ─────────────────────────────────────────────────
    yaml_path = STRATEGIES / "gbm_ml_v1.yaml"
    with open(yaml_path) as fh:
        strat = yaml.safe_load(fh)

    strat["validated"]        = validated
    strat["lopo_auc"]         = round(ml_auc or 0, 4)
    strat["lopo_auc_ci"]      = [round(ml_ci[0], 4), round(ml_ci[1], 4)]
    strat["binding_only_baseline"] = round(bind_auc or 0, 4)
    strat["n_patients"]       = int(df["patient_id"].nunique())
    strat["n_rows"]           = len(df)
    strat["features"]         = feats
    strat["cross_transfer"]   = {
        "melanoma_to_gbm": round(mel_to_gbm_auc or 0, 4),
        "gbm_to_melanoma": round(gbm_to_mel_auc or 0, 4),
    }

    if not validated:
        strat["notes"] = (
            f"Phase 10 re-validation: LOPO AUC {ml_auc:.4f} is below threshold "
            f"({threshold:.4f} = binding_only + 0.05). binding_only baseline "
            f"({bind_auc:.4f}) outperforms ML. Recommend binding_only for GBM."
        )
        recommended = "binding_only"
    else:
        strat["notes"] = (
            f"Phase 10 re-validation: LOPO AUC {ml_auc:.4f} exceeds threshold "
            f"({threshold:.4f}). ML adds lift over binding_only ({bind_auc:.4f})."
        )
        recommended = "gbm_ml_v1"

    with open(yaml_path, "w") as fh:
        yaml.dump(strat, fh, default_flow_style=False, sort_keys=False)
    print(f"\nUpdated {yaml_path}")

    # ── update registry.yaml ──────────────────────────────────────────────────
    reg_path = STRATEGIES / "registry.yaml"
    with open(reg_path) as fh:
        reg = yaml.safe_load(fh)

    for s in reg["strategies"]:
        if s["strategy_id"] == "gbm_ml_v1":
            s["validated"]  = validated
            s["lopo_auc"]   = round(ml_auc or 0, 4)

    reg["recommended_by_cancer_type"]["gbm"] = recommended
    import datetime
    reg["updated"] = str(datetime.date.today())

    with open(reg_path, "w") as fh:
        yaml.dump(reg, fh, default_flow_style=False, sort_keys=False)
    print(f"Updated {reg_path}")

    # ── save JSON results ─────────────────────────────────────────────────────
    out = {
        "dataset": "gbm_hilf_keskin_phase10",
        "n_rows": len(df),
        "n_patients": int(df["patient_id"].nunique()),
        "n_positive": int(y_all.sum()),
        "features": feats,
        "lopo_auc": {
            "binding_only": {"auc": round(bind_auc or 0, 4),
                             "ci": [round(bind_ci[0], 4), round(bind_ci[1], 4)],
                             "recall_at_10": round(r10_bind, 3),
                             "recall_at_20": round(r20_bind, 3)},
            "rl_tcr_v1":    {"auc": round(rl_auc or 0, 4),
                             "ci": [round(rl_ci[0], 4), round(rl_ci[1], 4)],
                             "recall_at_10": round(r10_rl, 3),
                             "recall_at_20": round(r20_rl, 3)},
            "gbm_ml_v1":    {"auc": round(ml_auc or 0, 4),
                             "ci": [round(ml_ci[0], 4), round(ml_ci[1], 4)],
                             "recall_at_10": round(r10_ml, 3),
                             "recall_at_20": round(r20_ml, 3)},
        },
        "cross_transfer": {
            "melanoma_to_gbm": round(mel_to_gbm_auc or 0, 4),
            "gbm_to_melanoma": round(gbm_to_mel_auc or 0, 4),
        },
        "validated": validated,
        "recommended_scorer": recommended,
    }
    out_path = ARTIFACTS / "phase10_gbm_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")
    print("\nPhase 10 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
