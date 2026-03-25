"""generate derrida-phase, f-ablation, and cross-substrate figures."""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.config import RESULTS
from src.utils.plots import setup_style, panel_label, C_POS, C_NEG, W_FULL, W_HALF, DPI

logger = logging.getLogger(__name__)


def _plot_derrida_phase(data_path: Path, out: Path):
    """scatter in (lambda, F) plane colored by derrida mu."""
    data = json.loads(data_path.read_text())
    rules = data["rules"]

    lam = np.array([r["lambda"] for r in rules])
    f = np.array([r["f"] for r in rules])
    mu = np.array([r["mu"] for r in rules])
    tier1 = np.array([r["tier1"] for r in rules])

    norm = TwoSlopeNorm(vcenter=1.0, vmin=mu.min(), vmax=mu.max())

    fig, ax = plt.subplots(figsize=(W_FULL, 4.0))

    neg_mask = ~tier1
    pos_mask = tier1

    sc_neg = ax.scatter(
        lam[neg_mask], f[neg_mask], c=mu[neg_mask], cmap="coolwarm", norm=norm,
        marker="o", s=10, alpha=0.4, edgecolors="none", rasterized=True,
        label="Non-replicating",
    )
    ax.scatter(
        lam[pos_mask], f[pos_mask], c=mu[pos_mask], cmap="coolwarm", norm=norm,
        marker="D", s=22, alpha=0.85, edgecolors="k", linewidths=0.3,
        rasterized=True, label="Tier-1 replicating",
    )

    cbar = fig.colorbar(sc_neg, ax=ax, shrink=0.85, label=r"Derrida $\mu$")
    cbar.ax.tick_params(labelsize=7)

    ax.axhline(y=0.5, color="0.5", linestyle="--", linewidth=0.8)
    ax.text(0.02, 0.52, r"$\mu \approx 1$ boundary region", fontsize=7, color="0.4")

    ax.set_xlabel(r"$\lambda$ (rule density)")
    ax.set_ylabel(r"$F$ (background stability)")
    ax.legend(fontsize=7, markerscale=1.2)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _plot_f_ablation(data_path: Path, out: Path):
    """3-panel bar chart of tier-1 rate vs F per weighting scheme."""
    data = json.loads(data_path.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.0))
    fig.subplots_adjust(wspace=0.35)

    scheme_names = ["uniform", "linear", "quadratic"]
    labels = [r"(\textbf{a}) uniform", r"(\textbf{b}) linear", r"(\textbf{c}) quadratic"]

    for ax, scheme, label in zip(axes, scheme_names, labels):
        info = data["per_scheme"][scheme]
        centers = np.array(info["f_centers"])
        rates = np.array(info["tier1_rates"])
        counts = np.array(info["bin_counts"])
        peak_f = info["peak_f"]

        bar_colors = np.where(np.abs(centers - peak_f) < 0.026, "#e74c3c", C_POS)
        ax.bar(centers, rates, width=0.045, color=bar_colors, alpha=0.85)

        nonzero = counts > 0
        for c, cnt, r in zip(centers[nonzero], counts[nonzero], rates[nonzero]):
            ax.text(c, r + 0.003, f"{cnt:,.0f}", ha="center", va="bottom",
                    fontsize=4.5, color="0.55", rotation=90)

        ax.set_xlabel(r"$F$")
        ax.set_ylabel(r"Tier-1 rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, rates.max() * 1.35 if rates.max() > 0 else 0.2)
        panel_label(ax, label, x=-0.18)

    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _plot_cross_substrate(cross_path: Path, effect_path: Path, out: Path):
    """grouped bar chart of cohen's d across substrates."""
    cross = json.loads(cross_path.read_text())
    effect = json.loads(effect_path.read_text())

    substrates = [r"$k{=}2$ Moore", r"$k{=}3$ Moore", r"$k{=}2$ vN"]
    ds = [
        effect["mass_balance"]["cohens_d"],
        cross["k3_results"]["cohens_d"],
        cross["vn_results"]["cohens_d"],
    ]
    ps = [
        effect["mass_balance"]["p_value"],
        cross["k3_results"]["p_value"],
        cross["vn_results"]["p_value"],
    ]

    sig_labels = ["***" if p < 0.001 else ("*" if p < 0.05 else "ns") for p in ps]
    colors = [C_POS if p < 0.001 else (C_NEG if p >= 0.05 else "#d4a017") for p in ps]

    fig, ax = plt.subplots(figsize=(W_HALF, 3.0))
    x = np.arange(len(substrates))
    ax.bar(x, ds, width=0.55, color=colors, alpha=0.85, edgecolor="0.3", linewidth=0.5)

    ax.axhline(y=0, color="0.4", linestyle="--", linewidth=0.8)

    for xi, d, lbl in zip(x, ds, sig_labels):
        offset = -0.08 if d < 0 else 0.05
        ax.text(xi, d + offset, lbl, ha="center", va="top" if d < 0 else "bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(substrates)
    ax.set_ylabel(r"Cohen's $d$ (mass balance)")
    ax.set_ylim(min(ds) - 0.3, 0.3)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def run(artifact: Path, **kwargs) -> dict:
    """generate derrida-phase, f-ablation, and cross-substrate figures."""
    setup_style()

    generated = []

    derrida = RESULTS / "k2-moore-derrida-phase.json"
    if derrida.exists():
        out = RESULTS / "fig-derrida-phase.png"
        _plot_derrida_phase(derrida, out)
        generated.append(out.name)
        logger.info("generated %s", out.name)

    f_ablation = RESULTS / "k2-moore-f-ablation.json"
    if f_ablation.exists():
        out = RESULTS / "fig-f-ablation.png"
        _plot_f_ablation(f_ablation, out)
        generated.append(out.name)
        logger.info("generated %s", out.name)

    cross = RESULTS / "cross-substrate-mass-balance.json"
    effect = RESULTS / "k2-moore-effect-sizes.json"
    if cross.exists() and effect.exists():
        out = RESULTS / "fig-cross-substrate-mass-balance.png"
        _plot_cross_substrate(cross, effect, out)
        generated.append(out.name)
        logger.info("generated %s", out.name)

    artifact.write_text(json.dumps({"generated": generated}))
    logger.info("new figures done: %d files", len(generated))
    return {"generated": generated}
