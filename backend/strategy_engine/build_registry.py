"""
PHASE 5 — Strategy Registry.

Writes one YAML per cancer type under backend/strategy_engine/strategies/
and a registry index. melanoma_ml_v1 is marked validated (LOPO AUC 0.760).

Run: py -3.11 backend/strategy_engine/build_registry.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ARTIFACTS = Path("backend/strategy_engine/artifacts")
STRATEGIES = Path("backend/strategy_engine/strategies")
STRATEGIES.mkdir(parents=True, exist_ok=True)


def load_stage2(cancer_type: str) -> dict:
    p = ARTIFACTS / f"stage2_{cancer_type}_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"  wrote {path}")


def main() -> int:
    print("=" * 60)
    print("PHASE 5 — STRATEGY REGISTRY")
    print("=" * 60)

    stage1 = json.loads((ARTIFACTS / "stage1_results.json").read_text(encoding="utf-8"))

    # ── per-cancer-type strategy YAMLs ────────────────────────────────────
    specs = [
        dict(
            strategy_id="melanoma_ml_v1",
            cancer_type="melanoma",
            display_name="Melanoma ML v1 (XGBoost, TCR+sequence+expression)",
            model_type="xgboost",
            model_path="backend/strategy_engine/artifacts/stage2_melanoma_model.pkl",
            validated=True,
            validation_method="leave-one-patient-out (LOPO)",
            lopo_auc=0.7597,
            lopo_auc_ci=[0.7011, 0.8221],
            lopo_aupr=None,
            training_datasets=["ott_2017", "sahin_2017"],
            n_patients=19,
            n_rows=262,
            features=load_stage2("melanoma").get("features_used", []),
            notes=(
                "Strongest strategy for melanoma. TCR contact + sequence features "
                "provide +0.23 AUC lift over binding_only baseline (0.528). "
                "Does not transfer cross-cancer — use within melanoma only."
            ),
        ),
        dict(
            strategy_id="gbm_ml_v1",
            cancer_type="gbm",
            display_name="GBM ML v1 (XGBoost, expression+binding)",
            model_type="xgboost",
            model_path="backend/strategy_engine/artifacts/stage2_gbm_model.pkl",
            validated=True,
            validation_method="leave-one-patient-out (LOPO)",
            lopo_auc=0.5946,
            lopo_auc_ci=[0.5055, 0.6824],
            lopo_aupr=None,
            training_datasets=["hilf_2019", "keskin_2019"],
            n_patients=17,
            n_rows=179,
            features=load_stage2("gbm").get("features_used", []),
            notes=(
                "Binding_only baseline (0.645) outperforms this model on GBM. "
                "Expression features (rl_expression_v1) may be preferable. "
                "Use with caution; AUC CI overlaps with binding baseline."
            ),
        ),
        dict(
            strategy_id="pancreatic_ml_v1",
            cancer_type="pancreatic",
            display_name="Pancreatic ML v1 (XGBoost, limited features)",
            model_type="xgboost",
            model_path="backend/strategy_engine/artifacts/stage2_pancreatic_model.pkl",
            validated=True,
            validation_method="leave-one-patient-out (LOPO)",
            lopo_auc=0.5044,
            lopo_auc_ci=[0.3852, 0.6227],
            lopo_aupr=None,
            training_datasets=["rojas_2023"],
            n_patients=16,
            n_rows=230,
            features=load_stage2("pancreatic").get("features_used", []),
            notes=(
                "Near-chance performance. TCR features unavailable (no WT peptide). "
                "Rojas 2023 is expression-dominated — use rl_tcr_v1 or rl_expression_v1 "
                "as hand-crafted alternatives."
            ),
        ),
        dict(
            strategy_id="mixed_ml_v1",
            cancer_type="mixed",
            display_name="Mixed/TESLA ML v1 (XGBoost, multi-feature)",
            model_type="xgboost",
            model_path="backend/strategy_engine/artifacts/stage2_mixed_tesla_model.pkl",
            validated=True,
            validation_method="leave-one-patient-out (LOPO)",
            lopo_auc=0.7444,
            lopo_auc_ci=[0.6603, 0.8235],
            lopo_aupr=None,
            training_datasets=["tesla_2020"],
            n_patients=9,
            n_rows=918,
            features=load_stage2("mixed_tesla").get("features_used", []),
            notes=(
                "Good performance on mixed cancer TESLA cohort. "
                "Marginal improvement (+0.014) over rl_expression_v1 baseline (0.730). "
                "Use rl_expression_v1 as simpler equivalent."
            ),
        ),
        dict(
            strategy_id="presentation_model_v1",
            cancer_type="all",
            display_name="Stage 1 Presentation Model (NCI, XGBoost)",
            model_type="xgboost",
            model_path="backend/strategy_engine/artifacts/stage1_presentation_model.pkl",
            validated=True,
            validation_method="leave-one-patient-out (LOPO) on NCI full mutanome",
            lopo_auc=stage1.get("lopo_auc"),
            lopo_auc_ci=stage1.get("lopo_auc_ci"),
            lopo_aupr=stage1.get("lopo_aupr"),
            training_datasets=["muller_nci_full_mutanome"],
            n_patients=stage1.get("n_patients"),
            n_rows=stage1.get("n_rows"),
            features=stage1.get("features_used", []),
            notes=(
                "Predicts MHC presentation probability from binding + expression features. "
                "Trained on NCI full mutanome (292K rows, 82 positives, 56 patients). "
                "Domain gap: outputs ~0 for pre-screened clinical cohorts (expected). "
                "Use as a pre-filter, not as a standalone ranking model."
            ),
        ),
    ]

    for spec in specs:
        write_yaml(STRATEGIES / f"{spec['strategy_id']}.yaml", spec)

    # ── registry index ─────────────────────────────────────────────────────
    registry = {
        "version": "1.0",
        "updated": "2026-04-21",
        "strategies": [
            {
                "strategy_id": s["strategy_id"],
                "cancer_type": s["cancer_type"],
                "validated": s["validated"],
                "lopo_auc": s["lopo_auc"],
                "recommended_for": s["cancer_type"],
            }
            for s in specs
        ],
        "recommended_by_cancer_type": {
            "melanoma":   "melanoma_ml_v1",
            "gbm":        "gbm_ml_v1",
            "pancreatic": "pancreatic_ml_v1",
            "mixed":      "mixed_ml_v1",
            "unknown":    "melanoma_ml_v1",
        },
        "notes": (
            "melanoma_ml_v1 is the only strategy with significant LOPO improvement "
            "over binding baseline (CI does not include baseline). "
            "For cross-cancer use, prefer hand-crafted rl_tcr_v1 (Phase 4 finding)."
        ),
    }
    write_yaml(STRATEGIES / "registry.yaml", registry)

    print(f"\nPhase 5 complete. {len(specs)} strategy YAMLs + registry written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
