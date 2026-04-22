"""
PHASE 9 — Rojas 2023 Pancreatic LOPO Validation.

Dataset: backend/strategy_engine/artifacts/rojas_2023_features.csv
  230 rows (peptide-level) → deduplicated to mutation level (keep min binding_log per mutation)
  16 patients, 30 positives

Scorers:
  1. binding_only       (-binding_log)
  2. rl_tcr_v1          (from configs/scoring_profiles/rl_tcr_v1.yaml — falls back to
                         binding when TCR features absent)
  3. pancreatic_ml_v1   (LOPO XGBoost, same hyperparams as melanoma_ml_v1.yaml)

Cross-transfer:
  4. melanoma_ml_v1.pkl → predict Rojas → AUC
  5. pancreatic_ml_v1   → predict Ott+Sahin merged → AUC

If pancreatic LOPO AUC > binding_only + 0.05:
  Save strategies/pancreatic_ml_v1.yaml, update registry.yaml

Run: py -3.11 backend/strategy_engine/phase9_rojas_pancreatic.py
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

ARTIFACTS   = Path("backend/strategy_engine/artifacts")
STRATEGIES  = Path("backend/strategy_engine/strategies")

# ── hyperparameters matching melanoma_ml_v1 ───────────────────────────────────
XGB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                  eval_metric="logloss", random_state=42, verbosity=0)


def make_xgb(scale_pos_weight: float = 1.0):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)


# ── metrics ───────────────────────────────────────────────────────────────────

def safe_auc(y, s) -> float | None:
    y, s = np.array(y, dtype=float), np.array(s, dtype=float)
    m = np.isfinite(y) & np.isfinite(s) & np.isin(y, [0.0, 1.0])
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m].astype(int), s[m]))


def recall_at_k(y, s, k: int) -> float:
    idx = np.argsort(-np.array(s))[:k]
    return float(np.array(y)[idx].sum() / max(np.array(y).sum(), 1))


def bootstrap_ci(y, s, n: int = 1000, seed: int = 42) -> list[float | None]:
    rng = np.random.RandomState(seed)
    y, s = np.array(y, dtype=int), np.array(s, dtype=float)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        yt, sp = y[idx], s[idx]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, sp))
    if len(aucs) < 10:
        return [None, None]
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


# ── rl_tcr_v1 (graceful fallback when TCR absent) ────────────────────────────

def rl_tcr_v1_score(df: pd.DataFrame) -> pd.Series:
    def _n(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else \
               pd.Series(np.nan, index=df.index)

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9) if hi > lo else pd.Series(0.0, index=s.index)

    present = _norm(-_n("binding_log"))
    if "presentation_score" in df.columns:
        ps = pd.to_numeric(df["presentation_score"], errors="coerce")
        if ps.notna().mean() > 0.5 and ps.std() > 1e-6:
            present = _norm(ps)

    tcr_vol = _n("tcr_volume_diff")
    if tcr_vol.isna().mean() > 0.5:
        tcr_vol = _n("tcr_volume_mean")
    tcr_chg = _n("tcr_charge_diff")
    tcr_hyd = _n("tcr_hydrophobicity_diff")
    if tcr_hyd.isna().mean() > 0.5:
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


# ── feature preparation ───────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, candidate_feats: list[str],
                    min_coverage: float = 0.50) -> list[str]:
    return [f for f in candidate_feats
            if f in df.columns and df[f].notna().mean() >= min_coverage]


def prep_X(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = np.column_stack([
        pd.to_numeric(df[f], errors="coerce").values if f in df.columns
        else np.zeros(len(df))
        for f in feats
    ])
    for i in range(X.shape[1]):
        col = X[:, i]
        med = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        X[~np.isfinite(col), i] = med
    return X


# ── LOPO ──────────────────────────────────────────────────────────────────────

def lopo(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    scores = np.full(len(df), np.nan)
    for pt in df["patient_id"].unique():
        tr_mask = df["patient_id"] != pt
        te_mask = df["patient_id"] == pt
        y_tr = df.loc[tr_mask, "immunogenic"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = make_xgb(spw)
        X_tr = prep_X(df[tr_mask].reset_index(drop=True), feats)
        X_te = prep_X(df[te_mask].reset_index(drop=True), feats)
        m.fit(X_tr, y_tr)
        te_idx = df.index[te_mask] - df.index[0]
        scores[te_idx] = m.predict_proba(X_te)[:, 1]
    return scores


# ── cross-transfer helper ─────────────────────────────────────────────────────

def cross_transfer_auc(model, feats: list[str], df: pd.DataFrame,
                       label: str) -> float | None:
    y = df["immunogenic"].values.astype(int)
    X = prep_X(df, feats)
    try:
        s = model.predict_proba(X)[:, 1]
    except Exception as exc:
        print(f"  {label} prediction failed: {exc}")
        return None
    return safe_auc(y, s)


# ── load merged Ott+Sahin for reverse transfer ────────────────────────────────

def load_ott_sahin() -> pd.DataFrame:
    ott  = pd.read_csv(ARTIFACTS / "ott_2017_features.csv",   low_memory=False)
    sah  = pd.read_csv(ARTIFACTS / "sahin_2017_features.csv", low_memory=False)
    df   = pd.concat([ott, sah], ignore_index=True)
    return df[df["immunogenic"].isin([0, 1])].reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 65)
    print("PHASE 9 — ROJAS 2023 PANCREATIC LOPO VALIDATION")
    print("=" * 65)

    # ── load + deduplicate ────────────────────────────────────────────────────
    raw = pd.read_csv(ARTIFACTS / "rojas_2023_features.csv", low_memory=False)
    raw = raw[raw["immunogenic"].isin([0, 1])].copy()

    dup_before = len(raw)
    df = (raw
          .sort_values("binding_log")                         # ascending: lowest nm first
          .groupby(["patient_id", "gene", "protein_change"],
                   as_index=False)
          .first())
    print(f"\nRows: {dup_before} → {len(df)} after deduplication "
          f"({dup_before - len(df)} duplicates removed)")
    print(f"Patients: {df['patient_id'].nunique()}, "
          f"Positive: {int(df['immunogenic'].sum())}, "
          f"Negative: {int((df['immunogenic'] == 0).sum())}")

    # ── select features ───────────────────────────────────────────────────────
    CANDIDATE_FEATS = [
        "tcr_hydro_mean", "tcr_volume_mean", "tcr_charge_sum", "tcr_aromatic_count",
        "self_dissimilarity", "hamming_distance", "blosum62_score", "blosum62_at_mutation",
        "calis_simplified", "pep_length", "hydro_full_mean", "aliphatic_index",
        "expression_log2", "expression_zscore", "binding_log", "binding_sigmoid",
        "presentation_probability",
    ]
    feats = select_features(df, CANDIDATE_FEATS)
    dropped = [f for f in CANDIDATE_FEATS if f not in feats]
    print(f"\nFeatures used ({len(feats)}): {feats}")
    print(f"Dropped (<50% coverage): {dropped}")

    # ── score 1: binding_only ─────────────────────────────────────────────────
    df["_bind"] = -pd.to_numeric(df["binding_log"], errors="coerce")

    # ── score 2: rl_tcr_v1 ───────────────────────────────────────────────────
    df["_rl"] = rl_tcr_v1_score(df).values

    # ── score 3: pancreatic_ml_v1 LOPO ───────────────────────────────────────
    print("\nRunning LOPO (pancreatic_ml_v1)...")
    df["_ml"] = lopo(df, feats)

    y = df["immunogenic"].values.astype(int)

    bind_auc = safe_auc(y, df["_bind"].values)
    rl_auc   = safe_auc(y, df["_rl"].values)
    ml_auc   = safe_auc(y, df["_ml"].values)

    bind_ci = bootstrap_ci(y, df["_bind"].fillna(0).values)
    rl_ci   = bootstrap_ci(y, df["_rl"].values)
    ml_ci   = bootstrap_ci(y, df["_ml"].fillna(0).values)

    # ── cross-transfer ────────────────────────────────────────────────────────
    mel_obj    = joblib.load(ARTIFACTS / "stage2_melanoma_model.pkl")
    mel_model  = mel_obj["model"]
    mel_feats  = mel_obj["features"]

    mel_on_rojas = cross_transfer_auc(mel_model, mel_feats, df,
                                      "melanoma_ml_v1 → Rojas")

    # Train full pancreatic model on all Rojas data for reverse transfer
    spw_all = (y == 0).sum() / max((y == 1).sum(), 1)
    panc_full = make_xgb(spw_all)
    panc_full.fit(prep_X(df, feats), y)

    ott_sahin = load_ott_sahin()
    panc_on_mel = cross_transfer_auc(panc_full, feats, ott_sahin,
                                     "pancreatic_ml_v1 → Ott+Sahin")

    # ── results table ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"{'Scorer':<24} {'AUC':>7}  {'95% CI':>18}  {'R@10':>6}  {'R@20':>6}")
    print("-" * 65)
    for name, auc, ci, col in [
        ("binding_only",       bind_auc, bind_ci, "_bind"),
        ("rl_tcr_v1",          rl_auc,   rl_ci,   "_rl"),
        ("pancreatic_ml_v1",   ml_auc,   ml_ci,   "_ml"),
    ]:
        r10 = recall_at_k(y, df[col].fillna(-np.inf).values, 10)
        r20 = recall_at_k(y, df[col].fillna(-np.inf).values, 20)
        auc_s = f"{auc:.4f}" if auc else "  N/A"
        ci_s  = f"[{ci[0]},{ci[1]}]" if ci[0] else "       N/A      "
        print(f"{name:<24} {auc_s:>7}  {ci_s:>18}  {r10:>6.3f}  {r20:>6.3f}")
    print("-" * 65)
    print(f"{'melanoma→Rojas (transfer)':<24} "
          f"{'N/A' if mel_on_rojas is None else f'{mel_on_rojas:.4f}':>7}")
    print(f"{'pancreatic→Ott+Sahin (transfer)':<24} "
          f"{'N/A' if panc_on_mel is None else f'{panc_on_mel:.4f}':>7}")
    print("=" * 65)

    # ── save YAML + update registry iff significant improvement ──────────────
    threshold = (bind_auc or 0.0) + 0.05
    validated = ml_auc is not None and ml_auc > threshold

    print(f"\nThreshold (binding + 0.05): {threshold:.4f}")
    print(f"pancreatic_ml_v1 AUC {ml_auc:.4f} → validated={validated}")

    strategy_data = dict(
        strategy_id="pancreatic_ml_v1",
        cancer_type="pancreatic",
        display_name="Pancreatic ML v1 (XGBoost, binding+sequence)",
        model_type="xgboost",
        model_path="backend/strategy_engine/artifacts/stage2_pancreatic_model.pkl",
        validated=validated,
        validation_method="leave-one-patient-out (LOPO)",
        lopo_auc=round(ml_auc, 4) if ml_auc else None,
        lopo_auc_ci=ml_ci,
        lopo_aupr=None,
        binding_only_baseline=round(bind_auc, 4) if bind_auc else None,
        training_datasets=["rojas_2023"],
        n_patients=int(df["patient_id"].nunique()),
        n_rows=len(df),
        features=feats,
        cross_transfer=dict(
            melanoma_to_pancreatic=round(mel_on_rojas, 4) if mel_on_rojas else None,
            pancreatic_to_melanoma=round(panc_on_mel, 4) if panc_on_mel else None,
        ),
        notes=(
            "TCR contact features unavailable (Rojas 2023 has no WT peptide). "
            "Model relies on binding + sequence features only. "
            f"{'LOPO AUC exceeds binding baseline by >0.05 — validated.' if validated else 'LOPO AUC does not exceed binding baseline + 0.05 — not validated as improvement over binding.'}"
        ),
    )

    yaml_path = STRATEGIES / "pancreatic_ml_v1.yaml"
    yaml_path.write_text(
        yaml.dump(strategy_data, default_flow_style=False, sort_keys=False,
                  allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nSaved: {yaml_path}")

    # Update registry.yaml
    registry_path = STRATEGIES / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for s in registry["strategies"]:
        if s["strategy_id"] == "pancreatic_ml_v1":
            s["validated"] = validated
            s["lopo_auc"]  = round(ml_auc, 4) if ml_auc else None
            break
    registry["updated"] = "2026-04-22"
    registry_path.write_text(
        yaml.dump(registry, default_flow_style=False, sort_keys=False,
                  allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Updated: {registry_path}")

    # Save full model for cross-transfer reuse
    joblib.dump({"model": panc_full, "features": feats},
                ARTIFACTS / "stage2_pancreatic_model_v2.pkl")

    # Save results JSON
    results = dict(
        dataset="rojas_2023",
        n_rows=len(df),
        n_patients=int(df["patient_id"].nunique()),
        n_positive=int(y.sum()),
        features_used=feats,
        lopo_auc=round(ml_auc, 4) if ml_auc else None,
        lopo_auc_ci=ml_ci,
        binding_only_auc=round(bind_auc, 4) if bind_auc else None,
        rl_tcr_v1_auc=round(rl_auc, 4) if rl_auc else None,
        cross_transfer=dict(
            melanoma_to_pancreatic=round(mel_on_rojas, 4) if mel_on_rojas else None,
            pancreatic_to_melanoma=round(panc_on_mel, 4) if panc_on_mel else None,
        ),
        validated=validated,
    )
    (ARTIFACTS / "phase9_rojas_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved: {ARTIFACTS}/phase9_rojas_results.json")

    print("\nPhase 9 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
