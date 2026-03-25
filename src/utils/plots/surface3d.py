"""3D surface plot of self-replication rate in (lambda, F) space."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

from src.utils.plots import setup_style, W_FULL, DPI


def plot_phase_surface(census_path: Path, output_path: Path, n_bins: int = 20):
    """smooth 3D surface of tier-1 rate, colored by rule density."""
    setup_style()
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})

    data = json.loads(census_path.read_text())
    rules = data["rules"]

    lambdas = np.array([r.get("l", r.get("lambda", 0)) for r in rules])
    fs = np.array([r.get("f", r.get("f_param", 0)) for r in rules])
    tier1 = np.array([r.get("t1", r.get("tier1", False)) for r in rules])

    tier1_counts = np.zeros((n_bins, n_bins))
    totals = np.zeros((n_bins, n_bins))

    for lam, f, t1 in zip(lambdas, fs, tier1):
        li = min(int(lam * n_bins), n_bins - 1)
        fi = min(int(f * n_bins), n_bins - 1)
        totals[fi, li] += 1
        if t1:
            tier1_counts[fi, li] += 1

    rates = np.where(totals > 0, tier1_counts / totals, 0)

    lam_c = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)
    f_c = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)

    rates_sm = gaussian_filter(rates, sigma=0.8)
    density_sm = gaussian_filter(np.log1p(totals), sigma=0.8)

    n_fine = 80
    lam_fine = np.linspace(lam_c[0], lam_c[-1], n_fine)
    f_fine = np.linspace(f_c[0], f_c[-1], n_fine)

    z_spline = RectBivariateSpline(f_c, lam_c, rates_sm, kx=3, ky=3)
    d_spline = RectBivariateSpline(f_c, lam_c, density_sm, kx=3, ky=3)

    L_fine, F_fine = np.meshgrid(lam_fine, f_fine)
    Z_fine = np.clip(z_spline(f_fine, lam_fine), 0, None)
    D_fine = np.clip(d_spline(f_fine, lam_fine), 0, None)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cmap_density = plt.cm.viridis
    d_norm = Normalize(vmin=D_fine.min(), vmax=D_fine.max())
    rgba = cmap_density(d_norm(D_fine))

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.05, top=0.95)

    ax.plot_surface(L_fine, F_fine, Z_fine, facecolors=rgba,
                    rstride=1, cstride=1, shade=False,
                    alpha=0.92, edgecolor="none", antialiased=True)

    ax.contour(L_fine, F_fine, Z_fine, levels=8, zdir="z", offset=0,
               cmap="Greys", alpha=0.35, linewidths=0.5)

    lc = 0.37
    f_line = np.linspace(0, 1, 30)
    ax.plot([lc] * 30, f_line, [0] * 30, "--", color="#555555", linewidth=0.8, alpha=0.6)
    ax.text(lc, 1.02, 0, r"$\lambda_c$", fontsize=7, color="#555555", ha="center")

    ax.set_xlabel(r"$\lambda$ (rule density)", labelpad=12)
    ax.set_ylabel(r"$F$ (background stability)", labelpad=12)
    ax.set_zlabel("")
    ax.text2D(0.02, 0.55, r"Self-replication rate", transform=fig.transFigure,
              fontsize=9, rotation=90, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, Z_fine.max() * 1.05)
    ax.view_init(elev=28, azim=-130)
    ax.tick_params(axis="x", which="major", labelsize=7, pad=4)
    ax.tick_params(axis="y", which="major", labelsize=7, pad=4)
    ax.tick_params(axis="z", which="major", labelsize=7, pad=5)

    sm = ScalarMappable(cmap=cmap_density, norm=d_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02,
                        label=r"$\log(1{+}N)$ rule density")
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
