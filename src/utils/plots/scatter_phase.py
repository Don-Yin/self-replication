"""discrete grid phase diagram on the exact (lambda, F) lattice."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plots import setup_style, panel_label, W_FULL, DPI


def plot_lattice_scatter(census_path: Path, output_path: Path):
    """discrete grid heatmap at exact lambda values, binned F."""
    setup_style()
    data = json.loads(census_path.read_text())
    rules = data["rules"]

    lambdas = np.array([r.get("l", r.get("lambda", 0)) for r in rules])
    fs = np.array([r.get("f", r.get("f_param", 0)) for r in rules])
    tier1 = np.array([r.get("t1", r.get("tier1", False)) for r in rules])

    lam_vals = np.unique(np.round(lambdas, 6))
    n_f_bins = 22
    f_edges = np.linspace(0, 1, n_f_bins + 1)

    counts = np.full((n_f_bins, len(lam_vals)), np.nan)
    rates = np.full((n_f_bins, len(lam_vals)), np.nan)
    lam_idx = {round(v, 6): i for i, v in enumerate(lam_vals)}

    for lam, f, t1 in zip(lambdas, fs, tier1):
        li = lam_idx[round(float(lam), 6)]
        fi = min(int(f * n_f_bins), n_f_bins - 1)
        if np.isnan(counts[fi, li]):
            counts[fi, li] = 0
            rates[fi, li] = 0
        counts[fi, li] += 1
        if t1:
            rates[fi, li] += 1

    occupied = ~np.isnan(counts)
    rates[occupied] = rates[occupied] / counts[occupied]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.5))
    fig.subplots_adjust(wspace=0.38)

    log_counts = np.full_like(counts, np.nan)
    log_counts[occupied] = np.log1p(counts[occupied])

    cmap_density = plt.cm.Blues.copy()
    cmap_density.set_bad("white")
    cmap_rate = plt.cm.YlOrRd.copy()
    cmap_rate.set_bad("white")

    lam_half = 0.5 * (lam_vals[1] - lam_vals[0]) if len(lam_vals) > 1 else 0.03
    extent_x = (lam_vals[0] - lam_half, lam_vals[-1] + lam_half)
    extent_y = (0, 1)

    im0 = axes[0].imshow(log_counts, origin="lower", aspect="auto",
                         extent=[extent_x[0], extent_x[1], extent_y[0], extent_y[1]],
                         cmap=cmap_density, interpolation="nearest")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log(1{+}N)$ rules")
    cb0.ax.tick_params(labelsize=7)
    axes[0].set_xlabel(r"$\lambda$ (rule density)")
    axes[0].set_ylabel(r"$F$ (background stability)")
    panel_label(axes[0], r"(\textbf{a})")

    im1 = axes[1].imshow(rates, origin="lower", aspect="auto",
                         extent=[extent_x[0], extent_x[1], extent_y[0], extent_y[1]],
                         cmap=cmap_rate, vmin=0, vmax=1, interpolation="nearest")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Self-replication rate")
    cb1.ax.tick_params(labelsize=7)
    axes[1].set_xlabel(r"$\lambda$ (rule density)")
    axes[1].set_ylabel(r"$F$ (background stability)")
    panel_label(axes[1], r"(\textbf{b})")

    for ax in axes:
        ax.set_xlim(extent_x[0], extent_x[1])
        ax.set_ylim(extent_y[0], extent_y[1])

    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
