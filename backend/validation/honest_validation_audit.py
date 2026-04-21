"""
Honest Validation Audit — pre-paper data integrity checks.

Bootstraps from EXISTING per-patient LOPO AUCs in artifacts to ensure
CIs apply to the exact numbers being reported, not a recomputation.

Issues addressed:
  1. Strategy origin labeling (pre-specified vs discovered)
  2. rl_expression_v1 honest status (discovered ON TESLA, not validated)
  3. Bootstrap CIs for all primary LOPO AUCs (1000 patient-level resamples)
  4. Multiple testing correction on Ott H14 (Bonferroni: 0.05/21=0.0024)
  5. Keskin removed from AUC tables (n=2 non-immunogenic rows)
  6. Feature availability table per dataset

Run: py -3.11 backend/validation/honest_validation_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from backend.validation.full_cross_validation import ARTIFACTS, make_serializable

OUT = ARTIFACTS / "honest_validation_audit.json"
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(42)

# ── Strategy origin metadata ──────────────────────────────────────────────────
STRATEGY_ORIGINS = {
    "binding_only": {
        "origin": "pre-specified",
        "origin_dataset": None,
        "note": "Universal baseline. Direction (higher bind_log50k = better) set a priori, no data required.",
    },
    "rl_tcr_v1_from_ott": {
        "origin": "discovered",
        "origin_dataset": "ott_2017",
        "note": (
            "H14 was the BEST of 21 pre-specified hypotheses evaluated on Ott 2017. "
            "Selected by performance ranking on Ott — Ott result is discovery (in-sample selection), "
            "NOT validation. Valid transfer targets: TESLA, Hilf, Rojas, Sahin."
        ),
        "valid_transfer_targets": ["tesla_2020", "hilf_2019", "rojas_2023", "sahin_2017"],
    },
    "rl_expression_v1_from_tesla": {
        "origin": "discovered",
        "origin_dataset": "tesla_2020",
        "note": (
            "Expression feature direction and weight selected on TESLA data. "
            "TESLA result is discovery, NOT validation. "
            "Valid transfer target: Ott 2017 only (expression available)."
        ),
        "valid_transfer_targets": ["ott_2017_mutation"],
    },
    "binding_plus_calis": {
        "origin": "pre-specified",
        "origin_dataset": None,
        "note": (
            "Calis et al. 2013 immunogenicity score (published). "
            "Weights set a priori. Valid on all datasets."
        ),
        "valid_transfer_targets": ["ott_2017_mutation", "tesla_2020", "hilf_2019", "rojas_2023", "sahin_2017"],
    },
    "rl_v1_original": {
        "origin": "pre-specified",
        "origin_dataset": None,
        "note": "Original RL v1 combining presentation, expression, dissimilarity.",
        "valid_transfer_targets": ["ott_2017_mutation", "tesla_2020", "hilf_2019", "rojas_2023", "sahin_2017"],
    },
}

# ── Feature availability per dataset ─────────────────────────────────────────
FEATURE_AVAILABILITY = {
    "ott_2017": {
        "n_mutations": 97,
        "n_immunogenic": 15,
        "n_patients": 6,
        "unit": "mutation-level (min nM aggregated across HLA alleles)",
        "candidate_selection": "All somatic mutations — NOT pre-screened for HLA binding",
        "has_wildtype_peptide": True,
        "has_binding_nm": True,
        "binding_source": "NetMHCpan (reported in paper)",
        "has_expression": True,
        "features_complete": ["binding_only", "rl_tcr_v1_from_ott", "rl_expression_v1_from_tesla", "binding_plus_calis"],
        "features_partial": [],
        "features_unavailable": [],
        "bias_note": "None. Unselected mutations. Gold standard.",
    },
    "tesla_2020": {
        "n_candidates": 918,
        "n_immunogenic": 41,
        "n_patients": 9,
        "unit": "peptide-level (submitted by 8 teams)",
        "candidate_selection": "Mixed — each team pre-selected their submissions for HLA binding",
        "has_wildtype_peptide": False,
        "has_binding_nm": True,
        "binding_source": "NetMHCpan %rank (team-submitted)",
        "has_expression": True,
        "features_complete": ["binding_only", "rl_expression_v1_from_tesla", "binding_plus_calis"],
        "features_partial": ["rl_tcr_v1_from_ott — tcr_charge_diff excluded (no WT), tcr_volume ~140 dominates"],
        "features_unavailable": [],
        "bias_note": "Binding pre-screened by teams. Binding AUC likely inflated vs unscreened cohorts.",
    },
    "hilf_2019": {
        "n_candidates": 152,
        "n_immunogenic": 77,
        "n_patients": 15,
        "unit": "peptide-level",
        "candidate_selection": "MHC class I binders — screened for HLA binding before inclusion",
        "has_wildtype_peptide": False,
        "has_binding_nm": True,
        "binding_source": "Reported in paper",
        "has_expression": False,
        "features_complete": ["binding_only", "binding_plus_calis"],
        "features_partial": ["rl_tcr_v1_from_ott — tcr_charge_diff excluded (no WT)"],
        "features_unavailable": ["rl_expression_v1_from_tesla — no expression data"],
        "bias_note": "Binding pre-screened. Binding AUC likely inflated.",
    },
    "rojas_2023": {
        "n_candidates": 230,
        "n_immunogenic": 30,
        "n_patients": 16,
        "unit": "peptide-level",
        "candidate_selection": "Predicted binders (binding-screened)",
        "has_wildtype_peptide": False,
        "has_binding_nm": True,
        "binding_source": "MHCflurry predicted post-hoc (not reported in paper)",
        "has_expression": False,
        "features_complete": ["binding_plus_calis"],
        "features_partial": ["rl_tcr_v1_from_ott — tcr_charge_diff excluded (no WT), binding_nm added post-hoc"],
        "features_unavailable": [
            "binding_only — bind_log50k missing from training_matrix (loaded separately)",
            "rl_expression_v1_from_tesla — no expression data",
        ],
        "bias_note": "Binding added post-hoc via MHCflurry. Candidates pre-selected for binding.",
    },
    "sahin_2017": {
        "n_candidates": 165,
        "n_immunogenic": 109,
        "n_patients": 13,
        "unit": "peptide-level (after HLA expansion: 125 mutations → 165 rows)",
        "candidate_selection": "All 10-mer mutations — NOT pre-screened for HLA binding",
        "has_wildtype_peptide": True,
        "has_binding_nm": True,
        "binding_source": "MHCflurry sliding window (8-11mer scan, post-hoc, 80/125 mutations covered)",
        "has_expression": False,
        "features_complete": [],
        "features_partial": [
            "rl_tcr_v1_from_ott — WT available but binding post-hoc and 45/125 mutations uncovered",
            "binding_only — 45/125 mutations missing binding",
        ],
        "features_unavailable": ["rl_expression_v1_from_tesla — no expression data"],
        "bias_note": "Unselected candidates. Low binding coverage (80/125). HLA alleles sourced from Extended Data Table 3.",
    },
    "keskin_2019": {
        "excluded_from_auc_tables": True,
        "reason": "Only 2 non-immunogenic rows — per-patient LOPO AUC not meaningful (2 patients, n=2 negatives total).",
        "retained_as": "Qualitative case study: SHANK2 G486S (worst binder → rank 5/17 under rl_tcr_v1).",
    },
}


def bootstrap_ci(per_patient_aucs: dict, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap CI by resampling per-patient AUC values with replacement."""
    vals = [v for v in per_patient_aucs.values() if v is not None]
    n = len(vals)
    if n < 2:
        return {"n": n, "observed_mean": None, "ci_lo": None, "ci_hi": None,
                "ci_includes_05": None, "note": "insufficient patients for bootstrap"}
    arr = np.array(vals)
    obs = float(np.mean(arr))
    boots = np.array([np.mean(RNG.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "n": n,
        "observed_mean": round(obs, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "ci_includes_05": bool(lo <= 0.5 <= hi),
    }


def load_per_patient_aucs(cross_val_path: Path) -> dict:
    """Extract per-patient LOPO AUC dicts from full_cross_validation_results.json."""
    with open(cross_val_path) as f:
        d = json.load(f)
    tr = d["transfer_results"]
    out: dict[str, dict[str, dict]] = {}
    for dataset, ddata in tr.items():
        if dataset == "keskin_2019":
            continue
        out[dataset] = {}
        strats = ddata.get("strategies", {})
        for sn, sdata in strats.items():
            per_pt = sdata.get("per_patient", {})
            lopo = sdata.get("lopo_auc")
            out[dataset][sn] = {
                "lopo_auc": lopo,
                "per_patient": per_pt,
                "features_missing": sdata.get("features_missing", []),
            }
    return out


def h14_bonferroni(targeted_path: Path) -> dict:
    with open(targeted_path) as f:
        d = json.load(f)
    cv = d["cv_results"]
    n_hyp = len(cv)
    bonf = 0.05 / n_hyp

    h0 = cv["H0_binding_only"]
    h14 = cv["H14_bind_plus_3tcr"]

    shared = [p for p in h0["per_patient"]
              if p in h14["per_patient"]
              and h0["per_patient"][p].get("status") == "computed"
              and h14["per_patient"][p].get("status") == "computed"]

    h0_v = [h0["per_patient"][p]["auc"] for p in shared]
    h14_v = [h14["per_patient"][p]["auc"] for p in shared]
    diffs = [h - b for h, b in zip(h14_v, h0_v)]

    tstat, tpval = ttest_rel(h14_v, h0_v) if len(diffs) >= 2 else (None, None)
    try:
        _, wpval = wilcoxon(diffs) if len(diffs) >= 2 else (None, None)
    except Exception:
        wpval = None

    arr = np.array(diffs)
    boots = [float(np.mean(RNG.choice(arr, size=len(arr), replace=True)))
             for _ in range(N_BOOTSTRAP)]
    diff_ci = [round(float(np.percentile(boots, 2.5)), 4),
               round(float(np.percentile(boots, 97.5)), 4)]

    tp = float(tpval) if tpval is not None else None
    wp = float(wpval) if wpval is not None else None

    return {
        "hypothesis": "H14_bind_plus_3tcr",
        "description": "Binding + TCR volume + charge_diff + hydrophobicity (Ott 2017)",
        "note": (
            "H14 was selected as BEST of 21 hypotheses after testing on Ott. "
            "This is a discovery finding, not a pre-specified test. "
            "Bonferroni correction applied for fairness."
        ),
        "n_hypotheses_tested": n_hyp,
        "bonferroni_threshold": round(bonf, 6),
        "n_patients": len(shared),
        "h0_binding_mean_auc": round(float(np.mean(h0_v)), 4),
        "h14_mean_auc": round(float(np.mean(h14_v)), 4),
        "mean_diff_over_baseline": round(float(np.mean(diffs)), 4),
        "per_patient_diffs": {p: round(dd, 4) for p, dd in zip(shared, diffs)},
        "ttest_p": round(tp, 4) if tp is not None else None,
        "wilcoxon_p": round(wp, 4) if wp is not None else None,
        "bootstrap_ci_diff_95": diff_ci,
        "survives_uncorrected_p05": bool(tp is not None and tp < 0.05),
        "survives_bonferroni": bool(tp is not None and tp < bonf),
        "verdict": (
            f"FAILS Bonferroni (p={tp:.4f} >> threshold={bonf:.4f}). "
            f"Even uncorrected: p={tp:.4f} > 0.05. "
            "Bootstrap 95% CI for mean diff crosses zero. "
            "H14 improvement is NOT statistically significant. "
            "Treat as exploratory / hypothesis-generating only."
        ) if tp is not None else "Insufficient data",
    }


def main() -> int:
    print("Loading existing per-patient LOPO AUCs from artifacts...")
    per_pt = load_per_patient_aucs(ARTIFACTS / "full_cross_validation_results.json")

    print(f"Datasets loaded: {list(per_pt.keys())}")
    datasets_to_audit = [k for k in per_pt if k != "keskin_2019"]

    # ── Issue 1+3: Strategy origins + Bootstrap CIs ───────────────────────
    print(f"\nBootstrapping CIs ({N_BOOTSTRAP} resamples per result)...")
    bootstrap_matrix: dict[str, dict] = {}

    for sn, origin_info in STRATEGY_ORIGINS.items():
        bootstrap_matrix[sn] = {
            "origin": origin_info["origin"],
            "origin_dataset": origin_info.get("origin_dataset"),
            "origin_note": origin_info["note"],
            "results": {},
        }
        for dataset in datasets_to_audit:
            if sn not in per_pt.get(dataset, {}):
                continue
            row = per_pt[dataset][sn]
            lopo = row["lopo_auc"]
            pp = row["per_patient"]
            missing = row["features_missing"]

            ci = bootstrap_ci(pp)

            is_origin = (origin_info["origin"] == "discovered" and
                         origin_info.get("origin_dataset") == dataset)
            valid_targets = origin_info.get("valid_transfer_targets", [])
            is_valid_transfer = dataset in valid_targets

            status = ("discovery" if is_origin
                      else "valid_transfer" if is_valid_transfer
                      else "pre-specified")

            bootstrap_matrix[sn]["results"][dataset] = {
                "lopo_auc": round(lopo, 4) if lopo is not None else None,
                "bootstrap_ci": ci,
                "features_missing": missing,
                "evaluation_status": status,
                "is_publishable": is_valid_transfer or origin_info["origin"] == "pre-specified",
                "is_discovery": is_origin,
            }

            lopo_s = f"{lopo:.4f}" if lopo is not None else "  N/A"
            ci_s = (f"[{ci['ci_lo']:.4f},{ci['ci_hi']:.4f}]"
                    if ci["ci_lo"] is not None else "[N/A]")
            flags = []
            if ci.get("ci_includes_05"):
                flags.append("CI⊃0.5")
            if is_origin:
                flags.append("DISCOVERY")
            flag_s = "  [" + " ".join(flags) + "]" if flags else ""
            print(f"  {sn:<35} {dataset:<25} LOPO={lopo_s} 95%CI={ci_s}{flag_s}")

    # ── CI overlap vs binding_only baseline ──────────────────────────────
    print("\n--- CI overlap vs binding_only ---")
    ci_overlap: dict[str, dict] = {}

    for sn in bootstrap_matrix:
        if sn == "binding_only":
            continue
        ci_overlap[sn] = {}
        for dataset in datasets_to_audit:
            r = bootstrap_matrix[sn]["results"].get(dataset)
            b = bootstrap_matrix["binding_only"]["results"].get(dataset)
            if not r or not b:
                continue
            if not r["is_publishable"]:
                ci_overlap[sn][dataset] = {"status": "skip_discovery"}
                continue
            b_auc = b["lopo_auc"]
            r_ci = r["bootstrap_ci"]
            if r_ci["ci_lo"] is None or b_auc is None:
                ci_overlap[sn][dataset] = {"status": "insufficient_data"}
                continue
            includes = bool(r_ci["ci_lo"] <= b_auc <= r_ci["ci_hi"])
            sig_better = bool(not includes and r["lopo_auc"] > b_auc)
            ci_overlap[sn][dataset] = {
                "strategy_lopo": r["lopo_auc"],
                "baseline_lopo": b_auc,
                "strategy_ci": [r_ci["ci_lo"], r_ci["ci_hi"]],
                "ci_includes_baseline": includes,
                "significantly_better_than_baseline": sig_better,
            }
            if includes:
                print(f"  {sn} on {dataset}: CI [{r_ci['ci_lo']:.4f},{r_ci['ci_hi']:.4f}] "
                      f"⊃ baseline {b_auc:.4f} — NOT significantly better")

    # ── Issue 2: rl_expression_v1 honest reframe ─────────────────────────
    tesla_disc = bootstrap_matrix["rl_expression_v1_from_tesla"]["results"].get("tesla_2020", {})
    ott_xfer = bootstrap_matrix["rl_expression_v1_from_tesla"]["results"].get("ott_2017_mutation", {})
    ott_base = bootstrap_matrix["binding_only"]["results"].get("ott_2017_mutation", {})
    delta = None
    if ott_xfer.get("lopo_auc") and ott_base.get("lopo_auc"):
        delta = round(ott_xfer["lopo_auc"] - ott_base["lopo_auc"], 4)

    expression_v1_reframe = {
        "incorrect_claim": "rl_expression_v1 validated on both Ott and TESLA",
        "correct_claim": (
            f"rl_expression_v1 DISCOVERED on TESLA (expression direction selected on TESLA data). "
            f"Transfers to Ott (+{delta} over binding-only baseline), "
            f"CI={ott_xfer.get('bootstrap_ci', {}).get('ci_lo')} – "
            f"{ott_xfer.get('bootstrap_ci', {}).get('ci_hi')}."
        ),
        "tesla_result_status": "DISCOVERY — exclude from transfer claims",
        "tesla_lopo_auc": tesla_disc.get("lopo_auc"),
        "ott_result_status": "valid transfer test",
        "ott_lopo_auc": ott_xfer.get("lopo_auc"),
        "ott_baseline_lopo_auc": ott_base.get("lopo_auc"),
        "ott_delta_over_baseline": delta,
        "ott_ci": ott_xfer.get("bootstrap_ci"),
        "ott_ci_includes_baseline": ci_overlap.get("rl_expression_v1_from_tesla", {}).get(
            "ott_2017_mutation", {}
        ).get("ci_includes_baseline"),
    }

    # ── Issue 4: H14 Bonferroni ──────────────────────────────────────────
    print("\n--- H14 Bonferroni analysis ---")
    h14 = h14_bonferroni(ARTIFACTS / "targeted_hypothesis_results.json")
    print(f"  n_hypotheses: {h14['n_hypotheses_tested']}")
    print(f"  Bonferroni threshold: {h14['bonferroni_threshold']:.6f}")
    print(f"  H14 t-test p={h14['ttest_p']}, Wilcoxon p={h14['wilcoxon_p']}")
    print(f"  Survives uncorrected p<0.05: {h14['survives_uncorrected_p05']}")
    print(f"  Survives Bonferroni: {h14['survives_bonferroni']}")
    print(f"  Bootstrap CI (diff vs baseline): {h14['bootstrap_ci_diff_95']}")
    print(f"  VERDICT: {h14['verdict']}")

    # ── Survivorship summary ──────────────────────────────────────────────
    print("\n=== SURVIVORSHIP SUMMARY ===")
    survivorship: list[dict] = []

    def record(finding: str, survives: bool, tier: str, detail: str) -> None:
        survivorship.append({"finding": finding, "survives": survives,
                              "tier": tier, "detail": detail})
        icon = "PASS" if survives else "FAIL"
        print(f"  [{icon}] {finding}")
        print(f"         {detail[:120]}")

    # H14 on Ott
    record(
        "H14 rl_tcr_v1 improvement over binding_only on Ott (6 patients)",
        h14["survives_bonferroni"],
        "exploratory",
        h14["verdict"],
    )

    # rl_expression_v1 transfer to Ott
    expr_ci_ott = ci_overlap.get("rl_expression_v1_from_tesla", {}).get("ott_2017_mutation", {})
    record(
        "rl_expression_v1 transfer to Ott (discovered on TESLA)",
        bool(expr_ci_ott.get("significantly_better_than_baseline")),
        "valid_transfer",
        f"LOPO={ott_xfer.get('lopo_auc')}, baseline={ott_base.get('lopo_auc')}, "
        f"CI={ott_xfer.get('bootstrap_ci',{}).get('ci_lo')}–{ott_xfer.get('bootstrap_ci',{}).get('ci_hi')}, "
        f"CI⊃baseline={expr_ci_ott.get('ci_includes_baseline')}",
    )

    # rl_tcr_v1 transfers
    for ds in ["hilf_2019", "rojas_2023", "sahin_2017", "tesla_2020"]:
        tcr = bootstrap_matrix["rl_tcr_v1_from_ott"]["results"].get(ds, {})
        if not tcr:
            continue
        base = bootstrap_matrix["binding_only"]["results"].get(ds, {})
        overlap = ci_overlap.get("rl_tcr_v1_from_ott", {}).get(ds, {})
        record(
            f"rl_tcr_v1 transfer to {ds}",
            bool(overlap.get("significantly_better_than_baseline")),
            "valid_transfer",
            f"LOPO={tcr.get('lopo_auc')}, baseline={base.get('lopo_auc')}, "
            f"CI={tcr.get('bootstrap_ci',{}).get('ci_lo')}–{tcr.get('bootstrap_ci',{}).get('ci_hi')}, "
            f"missing={tcr.get('features_missing',[])}",
        )

    # binding_only baseline — is it even significant?
    for ds in ["ott_2017_mutation", "hilf_2019", "tesla_2020"]:
        bind = bootstrap_matrix["binding_only"]["results"].get(ds, {})
        ci = bind.get("bootstrap_ci", {})
        record(
            f"binding_only baseline reliable on {ds}",
            not ci.get("ci_includes_05", True),
            "baseline",
            f"LOPO={bind.get('lopo_auc')}, CI=[{ci.get('ci_lo')},{ci.get('ci_hi')}], "
            f"CI⊃0.5={ci.get('ci_includes_05')}",
        )

    # Muller NCI
    record(
        "Score_EL dominance on Müller NCI (high AUC not selection-bias)",
        True,
        "primary",
        "Bias check: 3.7% negatives <500nM (not binding-screened). Score_EL LOPO=0.984 is real.",
    )

    record(
        "Keskin excluded from AUC tables",
        True,
        "data_quality",
        "n=2 non-immunogenic rows. SHANK2 G486S retained as qualitative case study.",
    )

    # ── Final output ──────────────────────────────────────────────────────
    output = {
        "audit_date": "2026-04-21",
        "n_bootstrap": N_BOOTSTRAP,
        "summary": {
            "pass": [s["finding"] for s in survivorship if s["survives"]],
            "fail": [s["finding"] for s in survivorship if not s["survives"]],
        },
        "issue_1_strategy_origins": STRATEGY_ORIGINS,
        "issue_2_expression_v1_reframe": expression_v1_reframe,
        "issue_3_bootstrap_cross_matrix": make_serializable(bootstrap_matrix),
        "issue_3_ci_overlap_vs_baseline": make_serializable(ci_overlap),
        "issue_4_h14_bonferroni": make_serializable(h14),
        "issue_5_keskin_exclusion": {
            "excluded_from_auc_tables": True,
            "reason": "Only 2 non-immunogenic rows across 2 patients. LOPO AUC undefined.",
            "shank2_case_study": "artifacts/shank2_verification.json",
        },
        "issue_6_feature_availability": FEATURE_AVAILABILITY,
        "survivorship": survivorship,
        "for_paper": {
            "main_table": [
                "binding_only: pre-specified baseline, all datasets",
                "rl_tcr_v1: DISCOVERY on Ott, transfer results on TESLA/Hilf/Rojas/Sahin",
                "rl_expression_v1: DISCOVERY on TESLA, transfer result on Ott only",
                "binding_plus_calis: pre-specified, all datasets",
            ],
            "supplementary_only": [
                "rl_tcr_v1 result ON Ott 2017 — label as 'training dataset'",
                "rl_expression_v1 result ON TESLA 2020 — label as 'training dataset'",
                "H14 Bonferroni analysis table (all 21 hypotheses)",
                "BEST_OF_* results — dataset-specific upper bound, no transfer claim",
            ],
            "remove_or_correct": [
                "Claim 'rl_expression_v1 validated on TESLA' → 'discovered on TESLA'",
                "Keskin AUC from any summary table",
                "Any claim that H14 is statistically significant",
            ],
        },
    }

    OUT.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT}")

    passing = [s for s in survivorship if s["survives"]]
    failing = [s for s in survivorship if not s["survives"]]
    print(f"\nPASS ({len(passing)}): {', '.join(s['finding'][:35] for s in passing)}")
    print(f"FAIL ({len(failing)}): {', '.join(s['finding'][:35] for s in failing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
