"""generate all publication figures from completed experiment results."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, **kwargs) -> dict:
    """generate all figures from results json files."""
    import numpy as np
    from src.config import RESULTS
    from src.utils.plots import setup_style, panel_label, C_POS, C_NEG, C_K2, C_K3
    from src.utils.plots import C_ALL, C_LOW_F, C_MID_F, C_HIGH_F, W_FULL, W_HALF, DPI
    from src.utils.plots.boundary import plot_boundary_comparison

    setup_style()
    import matplotlib.pyplot as plt

    generated = []

    from src.utils.plots.surface3d import plot_phase_surface
    from src.utils.plots.scatter_phase import plot_lattice_scatter

    census = RESULTS / "k2-moore-census.json"
    if census.exists():
        plot_lattice_scatter(census, RESULTS / "fig-phase-diagram.png")
        generated.append("fig-phase-diagram.png")
        logger.info("  generated fig-phase-diagram.png")
        plot_phase_surface(census, RESULTS / "fig-phase-surface-3d.png", n_bins=18)
        generated.append("fig-phase-surface-3d.png")
        logger.info("  generated fig-phase-surface-3d.png")

    if census.exists():
        _plot_lambda_profiles(census, RESULTS / "fig-lambda-profiles.png")
        generated.append("fig-lambda-profiles.png")
        logger.info("  generated fig-lambda-profiles.png")

    vn_census = RESULTS / "k2-vn-census.json"
    if vn_census.exists():
        plot_lattice_scatter(vn_census, RESULTS / "fig-vn-phase-diagram.png")
        generated.append("fig-vn-phase-diagram.png")
        logger.info("  generated fig-vn-phase-diagram.png")

    measures = RESULTS / "k2-moore-boundary-measures.json"
    if measures.exists():
        plot_boundary_comparison(measures, RESULTS / "fig-boundary-scatter.png")
        generated.append("fig-boundary-scatter.png")
        logger.info("  generated fig-boundary-scatter.png")

    oinfo = RESULTS / "k2-moore-oinfo-boundary.json"
    if oinfo.exists():
        _plot_oinfo_histograms(oinfo, RESULTS / "fig-oinfo-histograms.png")
        generated.append("fig-oinfo-histograms.png")
        logger.info("  generated fig-oinfo-histograms.png")

    f_marginal = RESULTS / "k2-moore-f-marginal.json"
    if f_marginal.exists():
        _plot_f_marginal(f_marginal, RESULTS / "fig-f-marginal.png")
        generated.append("fig-f-marginal.png")
        logger.info("  generated fig-f-marginal.png")

    k3 = RESULTS / "k3-moore-sweep.json"
    if census.exists() and k3.exists():
        _plot_k2_vs_k3(census, k3, RESULTS / "fig-k2-vs-k3.png")
        generated.append("fig-k2-vs-k3.png")
        logger.info("  generated fig-k2-vs-k3.png")

    artifact.write_text(json.dumps({"generated": generated}))
    logger.info("  all figures done: %d files", len(generated))
    return {"generated": generated}


def _plot_lambda_profiles(census_path: Path, output_path: Path):
    """lambda profiles conditioned on F terciles."""
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils.plots import C_ALL, C_LOW_F, C_MID_F, C_HIGH_F, W_HALF, DPI

    data = json.loads(census_path.read_text())
    rules = data["rules"]
    lambdas = np.array([r.get("l", r.get("lambda", 0)) for r in rules])
    fs = np.array([r.get("f", r.get("f_param", 0)) for r in rules])
    tier1 = np.array([r.get("t1", r.get("tier1", False)) for r in rules])

    edges = np.linspace(0, 1, 31)
    centers = (edges[:-1] + edges[1:]) / 2
    overall = _bin_rate(lambdas, tier1, edges)

    f_terciles = np.percentile(fs[fs > 0], [33, 67]) if np.any(fs > 0) else [0.33, 0.67]
    fig, ax = plt.subplots(figsize=(W_HALF, 2.8))
    ax.plot(centers, overall, "-", color=C_ALL, linewidth=1.8, label="All rules")

    for f_lo, f_hi, color, label in [
        (0, f_terciles[0], C_LOW_F, "Low $F$"),
        (f_terciles[0], f_terciles[1], C_MID_F, "Mid $F$"),
        (f_terciles[1], 1.01, C_HIGH_F, "High $F$"),
    ]:
        mask = (fs >= f_lo) & (fs < f_hi)
        rate = _bin_rate(lambdas[mask], tier1[mask], edges)
        ax.plot(centers, rate, "--", color=color, linewidth=1.0,
                label=rf"{label} ($F \in [{f_lo:.2f},\,{f_hi:.2f}]$)")

    ax.set_xlabel(r"$\lambda$ (rule density)")
    ax.set_ylabel(r"Self-replication rate")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.35)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _plot_f_marginal(f_marginal_path: Path, output_path: Path):
    """self-replication rate vs F, conditioned on lambda terciles."""
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils.plots import C_ALL, C_LOW_F, C_MID_F, C_HIGH_F, W_HALF, DPI

    data = json.loads(f_marginal_path.read_text())
    centers = np.array(data["f_centers"])
    overall = np.array(data["overall_rate"])

    fig, ax = plt.subplots(figsize=(W_HALF, 2.8))
    ax.plot(centers, overall, "-", color=C_ALL, linewidth=1.8, label="All rules")

    for key, color, label in [
        ("low_lambda", C_LOW_F, r"Low $\lambda$"),
        ("mid_lambda", C_MID_F, r"Mid $\lambda$"),
        ("high_lambda", C_HIGH_F, r"High $\lambda$"),
    ]:
        cond = data["conditioned"][key]
        rate = np.array(cond["rate"])
        lo, hi = cond["range"]
        ax.plot(centers, rate, "--", color=color, linewidth=1.0,
                label=rf"{label} ($\lambda \in [{lo:.2f},\,{hi:.2f}]$)")

    ax.set_xlabel(r"$F$ (background stability)")
    ax.set_ylabel(r"Self-replication rate")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.35)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _plot_oinfo_histograms(oinfo_path: Path, output_path: Path):
    """O-information comparison histograms."""
    import matplotlib.pyplot as plt
    from src.utils.plots import C_POS, C_NEG, panel_label, W_FULL, DPI

    data = json.loads(oinfo_path.read_text())
    pos = [r for r in data["rules"] if r["label"] == "tier1_positive"]
    neg = [r for r in data["rules"] if r["label"] == "tier1_negative"]

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.5))
    labels_map = [
        ("oinfo", r"$\Omega$ (O-information)"),
        ("tc", r"TC (total correlation)"),
        ("dtc", r"DTC (dual total correlation)"),
    ]

    for ax, (key, xlabel), lbl in zip(axes, labels_map, [r"(\textbf{a})", r"(\textbf{b})", r"(\textbf{c})"]):
        ax.hist([r[key] for r in neg], bins=30, alpha=0.45, color=C_NEG,
                label="Non-replicating", density=True)
        ax.hist([r[key] for r in pos], bins=30, alpha=0.6, color=C_POS,
                label="Tier-1 replicating", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        panel_label(ax, lbl)

    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _plot_k2_vs_k3(k2_path: Path, k3_path: Path, output_path: Path):
    """k=2 vs k=3 comparison bar chart."""
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils.plots import C_K2, C_K3, W_HALF, DPI

    edges = np.linspace(0, 0.7, 21)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(W_HALF, 2.8))
    for path, color, label, offset in [
        (k2_path, C_K2, r"$k{=}2$ Moore", -0.009),
        (k3_path, C_K3, r"$k{=}3$ Moore", 0.009),
    ]:
        data = json.loads(path.read_text())
        lams = np.array([r.get("l", r.get("lambda", 0)) for r in data["rules"]])
        t1 = np.array([r.get("t1", r.get("tier1", False)) for r in data["rules"]])
        rate = _bin_rate(lams, t1, edges)
        ax.bar(centers + offset, rate, width=0.016, color=color, alpha=0.8, label=label)

    ax.set_xlabel(r"$\lambda$ (rule density)")
    ax.set_ylabel(r"Self-replication rate")
    ax.legend()
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.5)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def _bin_rate(values, labels, edges):
    """bin values into edges and compute label-true rate per bin."""
    import numpy as np
    n = len(edges) - 1
    counts = np.zeros(n)
    totals = np.zeros(n)
    for v, l in zip(values, labels):
        b = min(int((v - edges[0]) / (edges[-1] - edges[0]) * n), n - 1)
        b = max(0, b)
        totals[b] += 1
        if l:
            counts[b] += 1
    return np.where(totals > 0, counts / totals, 0)
