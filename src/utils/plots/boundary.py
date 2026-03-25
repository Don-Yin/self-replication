"""scatter plots for boundary measure comparison."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plots import setup_style, panel_label, C_POS, C_NEG, W_FULL, DPI


def plot_boundary_comparison(measures_path: Path, output_path: Path):
    """scatter plots comparing tier-1+ vs tier-1- rules on derived measures."""
    setup_style()
    data = json.loads(measures_path.read_text())
    rules = data["rules"]

    pos = [r for r in rules if r["label"] == "tier1_positive"]
    neg = [r for r in rules if r["label"] == "tier1_negative"]

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.8))

    panels = [
        (r"$\lambda$", "mass_balance", r"Mass balance", "lambda", axes[0]),
        (r"$\lambda$", "spatial_entropy", r"Spatial entropy", "lambda", axes[1]),
        ("mass_balance", "spatial_entropy", r"Spatial entropy", "mass_balance", axes[2]),
    ]

    for xkey, ykey, ylabel, xfield, ax in panels:
        xf = xfield if xfield != "mass_balance" else "mass_balance"
        ax.scatter([r[xf] for r in neg], [r[ykey] for r in neg],
                   alpha=0.25, s=8, c=C_NEG, label="Non-replicating", rasterized=True)
        ax.scatter([r[xf] for r in pos], [r[ykey] for r in pos],
                   alpha=0.5, s=8, c=C_POS, label="Tier-1 replicating", rasterized=True)
        xlabel = xkey if xkey.startswith("$") else r"Mass balance"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(markerscale=1.5)

    panel_label(axes[0], r"(\textbf{a})")
    panel_label(axes[1], r"(\textbf{b})")
    panel_label(axes[2], r"(\textbf{c})")

    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
