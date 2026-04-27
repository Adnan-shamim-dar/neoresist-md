"""
NeoResist-MD — Figure Generation
Run from repo root: python scripts/generate_figures.py
Outputs: figures/fig1_framework.png, fig2_melanoma.png, fig3_borch.png, fig4_transfer.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT     = Path(__file__).resolve().parents[1]
ARTS     = ROOT / "backend" / "strategy_engine" / "artifacts"
OUT      = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE   = "#2166AC"
GREEN  = "#1A9850"
ORANGE = "#D73027"
GREY   = "#878787"
LIGHT  = "#BABABA"

# ── helpers ──────────────────────────────────────────────────────────────────

def _ci_err(lo: float, hi: float, mid: float) -> tuple[float, float]:
    return mid - lo, hi - mid


# ── Figure 2: Melanoma primary result + ablation ─────────────────────────────

def fig2_melanoma() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 2 — Melanoma strategy: held-out validation and feature ablation",
                 fontsize=13, fontweight="bold")

    # Panel A — strategy comparison on Sahin 2017
    strategies = [
        ("binding_only",           0.470, 0.319, 0.657),
        ("rl_tcr_v1",              0.518, 0.356, 0.691),
        ("neoguider_v2_melanoma",  0.531, 0.377, 0.702),
        ("melanoma_all_signals",   0.575, 0.511, 0.630),
    ]
    labels  = [s[0].replace("_", "\n") for s in strategies]
    aucs    = [s[1] for s in strategies]
    yerr    = np.array([[_ci_err(s[2], s[3], s[1]) for s in strategies]]).T.reshape(2, -1)
    colours = [GREY, LIGHT, LIGHT, GREEN]

    bars = ax1.bar(labels, aucs, color=colours, edgecolor="black", linewidth=0.8,
                   yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})
    ax1.axhline(0.470, color=GREY, linewidth=1.2, linestyle="--", label="Binding baseline")
    ax1.set_ylim(0.3, 0.75)
    ax1.set_ylabel("Mean patient AUC", fontsize=11)
    ax1.set_title("A   Held-out validation (Sahin 2017, n=13)", fontsize=11, loc="left")
    ax1.text(3, 0.585, "p = 0.011*", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)

    # Panel B — ablation
    features  = ["Full model", "−tcr_volume", "−aliphatic", "−expression", "−presentation"]
    abl_aucs  = [0.575, 0.490, 0.534, 0.586, 0.578]
    abl_col   = [GREEN, ORANGE, ORANGE, BLUE, BLUE]

    ax2.barh(features[::-1], abl_aucs[::-1], color=abl_col[::-1],
             edgecolor="black", linewidth=0.8)
    ax2.axvline(0.575, color=GREEN, linewidth=1.4, linestyle="--", label="Full model")
    ax2.axvline(0.470, color=GREY,  linewidth=1.0, linestyle=":",  label="Binding baseline")
    ax2.set_xlim(0.40, 0.65)
    ax2.set_xlabel("Mean patient AUC", fontsize=11)
    ax2.set_title("B   Feature ablation (Sahin 2017)", fontsize=11, loc="left")
    ax2.legend(fontsize=9)
    ax2.text(0.492, 4, "−0.086", va="center", fontsize=9, color=ORANGE, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "fig2_melanoma.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_melanoma.png")


# ── Figure 3: Borch/IMPROVE external benchmark ───────────────────────────────

def fig3_borch() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 3 — External corroboration: Borch/IMPROVE benchmark",
                 fontsize=13, fontweight="bold")

    # Panel A — melanoma
    methods  = ["binding_only", "PRIME 2.0", "LogReg\n(5-fold CV)", "HydroCore\nonly", "bind+\nHydroCore"]
    aucs     = [0.559, 0.609, 0.644, 0.650, 0.660]
    ci_lo    = [0.498, 0.556, 0.595, 0.595, 0.609]
    ci_hi    = [0.615, 0.660, 0.694, 0.699, 0.709]
    colours  = [GREY, ORANGE, LIGHT, LIGHT, GREEN]
    yerr     = np.array([[_ci_err(lo, hi, m) for lo, hi, m in zip(ci_lo, ci_hi, aucs)]]).T.reshape(2,-1)

    ax1.bar(methods, aucs, color=colours, edgecolor="black", linewidth=0.8,
            yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})
    ax1.set_ylim(0.45, 0.75)
    ax1.set_ylabel("Mean patient AUC", fontsize=11)
    ax1.set_title("A   Borch melanoma (Neye, n=23)", fontsize=11, loc="left")
    ax1.text(4, 0.668, "p = 0.067", ha="center", fontsize=9, color=GREEN)
    ax1.axhline(0.609, color=ORANGE, linewidth=1.0, linestyle="--")

    # Panel B — cancer-type specificity
    contexts = ["Melanoma", "Bladder"]
    hc_aucs  = [0.660, 0.570]
    prime    = [0.609, 0.616]
    x = np.arange(2)
    w = 0.35
    ax2.bar(x - w/2, hc_aucs, w, label="bind+HydroCore", color=GREEN,  edgecolor="black", linewidth=0.8)
    ax2.bar(x + w/2, prime,   w, label="PRIME 2.0",       color=ORANGE, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(contexts, fontsize=11)
    ax2.set_ylim(0.48, 0.72)
    ax2.set_ylabel("Mean patient AUC", fontsize=11)
    ax2.set_title("B   Cancer-type specificity", fontsize=11, loc="left")
    ax2.legend(fontsize=10)
    ax2.annotate("+0.051", xy=(0 - w/2, 0.665), ha="center", fontsize=9, color=GREEN)
    ax2.annotate("−0.046", xy=(1 - w/2, 0.576), ha="center", fontsize=9, color=ORANGE)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_borch.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_borch.png")


# ── Figure 4: Cross-cancer transfer heatmap ──────────────────────────────────

def fig4_transfer() -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Figure 4 — Bidirectional cross-cancer transfer failure",
                 fontsize=13, fontweight="bold")

    data = np.array([
        [0.575, 0.411],   # melanoma specialist: home=melanoma, away=GBM
        [0.466, 0.683],   # GBM specialist:      away=melanoma, home=GBM
    ])
    binding = np.array([0.470, 0.628])  # binding baselines per context

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.38, vmax=0.72, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Melanoma\n(Sahin)", "GBM\n(Keskin)"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Melanoma\nspecialist", "GBM\nspecialist"], fontsize=11)
    ax.set_xlabel("Test context", fontsize=11)
    ax.set_ylabel("Strategy", fontsize=11)

    for i in range(2):
        for j in range(2):
            val  = data[i, j]
            base = binding[j]
            note = "HOME" if i == j else f"Δ{val-base:+.3f} vs binding"
            ax.text(j, i, f"{val:.3f}\n({note})", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i == j else "normal",
                    color="black")

    plt.colorbar(im, ax=ax, label="Mean patient AUC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_transfer.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_transfer.png")


# ── Figure 1: Framework schematic (text-based) ───────────────────────────────

def fig1_framework() -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 1 — NeoResist-MD framework", fontsize=13, fontweight="bold")

    boxes = [
        (1.0, 2.5, "Discovery\ncohort\n(e.g. Ott 2017)", BLUE),
        (3.5, 2.5, "Autoresearch\nloop\n(222 iterations,\n>30,000 pairs)", BLUE),
        (6.0, 2.5, "Candidate\nstrategies\n(ranked by\ndiscovery AUC)", BLUE),
        (8.5, 2.5, "Held-out\nvalidation\n(e.g. Sahin 2017)\naccessed ONCE", GREEN),
    ]
    for x, y, label, colour in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.8, y - 0.9), 1.6, 1.8,
                                        boxstyle="round,pad=0.1", linewidth=1.5,
                                        edgecolor="black", facecolor=colour, alpha=0.25)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows
    for x in [1.8, 4.3, 6.8]:
        ax.annotate("", xy=(x + 0.4, 2.5), xytext=(x, 2.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    # Bottom note
    ax.text(5, 0.4,
            "Top strategy fixed before held-out access  ·  Promoted strategies crystallised to registry  ·  Full provenance logged",
            ha="center", fontsize=9, style="italic", color="grey")

    plt.tight_layout()
    plt.savefig(OUT / "fig1_framework.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_framework.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_framework()
    fig2_melanoma()
    fig3_borch()
    fig4_transfer()
    print(f"Done. Figures saved to {OUT}/")
