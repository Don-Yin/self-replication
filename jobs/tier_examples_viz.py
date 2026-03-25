"""visualize the 3-tier detection hierarchy with actual CA simulations."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SEED = 42
SIM_SIZE = 256
VIEW_SIZE = 64

# the known HighLife replicator (12-cell pattern that self-replicates every ~12 steps)
HIGHLIFE_REPLICATOR = np.array([
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0],
], dtype=np.int8)


def _count_components(grid: np.ndarray, min_size: int = 2) -> int:
    """count connected components with at least min_size cells."""
    from src.modules.detector import ComponentTracker
    tracker = ComponentTracker(k=2)
    labels, n = tracker.label_components(grid)
    return sum(1 for lid in range(1, n + 1) if int(np.sum(labels == lid)) >= min_size)


def _place_pattern(pattern: np.ndarray, grid_size: int) -> np.ndarray:
    """place a small pattern at the centre of an empty grid."""
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    ph, pw = pattern.shape
    cy, cx = grid_size // 2 - ph // 2, grid_size // 2 - pw // 2
    grid[cy:cy + ph, cx:cx + pw] = pattern
    return grid


def _crop_centre(grid: np.ndarray, size: int) -> np.ndarray:
    """crop a square region of given size from the grid centre."""
    h, w = grid.shape
    cy, cx = h // 2, w // 2
    half = size // 2
    return grid[cy - half:cy + half, cx - half:cx + half]


def _make_ca():
    """create a HighLife CA simulator."""
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.rule_params import HIGH_LIFE
    return FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=HIGH_LIFE)


def _render_row(axes, snapshots, timesteps, counts, row_label: str):
    """render a row of CA snapshots with component count annotations."""
    for col, (snap, t, cnt) in enumerate(zip(snapshots, timesteps, counts)):
        ax = axes[col]
        ax.imshow(snap, cmap="Greys", interpolation="nearest", vmin=0, vmax=1, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$t={t}$", fontsize=8)
        ax.text(0.5, -0.08, rf"$n_c={cnt}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)
        if col == 0:
            ax.set_ylabel(row_label, fontsize=8, labelpad=8)


def _select_snaps(all_snaps: list, interval: int, target_times: list) -> list:
    """select snapshots at target times from a list snapshotted at fixed interval."""
    return [all_snaps[t // interval] for t in target_times]


def _tier1_panels(ca):
    """generate tier-1 row: proliferation screening at short timescale."""
    ic = _place_pattern(HIGHLIFE_REPLICATOR, grid_size=SIM_SIZE)
    snaps = ca.run(ic, 160, snapshot_interval=16)
    times = [0, 32, 64, 112]
    selected = _select_snaps(snaps, 16, times)
    counts = [_count_components(s) for s in selected]
    cropped = [_crop_centre(s, VIEW_SIZE) for s in selected]
    return cropped, times, counts


def _tier2_panels(ca):
    """generate tier-2 row: extended confirmation over longer horizon."""
    ic = _place_pattern(HIGHLIFE_REPLICATOR, grid_size=SIM_SIZE)
    snaps = ca.run(ic, 160, snapshot_interval=16)
    times = [0, 48, 112, 160]
    selected = _select_snaps(snaps, 16, times)
    counts = [_count_components(s) for s in selected]
    cropped = [_crop_centre(s, VIEW_SIZE) for s in selected]
    return cropped, times, counts


def _tier3_panels(ca):
    """generate tier-3 row: seed, replicated, and perturbed outcomes."""
    seed = _place_pattern(HIGHLIFE_REPLICATOR, grid_size=SIM_SIZE)
    seed_snaps = ca.run(seed, 72, snapshot_interval=72)

    alive_ys, alive_xs = np.where(seed > 0)
    perturbed = seed.copy()
    perturbed[alive_ys[6], alive_xs[6]] = 0
    pert_snaps = ca.run(perturbed, 72, snapshot_interval=72)

    panels = [seed, seed_snaps[-1], pert_snaps[-1]]
    labels_t = ["seed", "seed $+72$", "perturbed $+72$"]
    counts = [_count_components(p) for p in panels]
    cropped = [_crop_centre(p, VIEW_SIZE) for p in panels]
    return cropped, labels_t, counts


def run(artifact: Path, **kwargs) -> dict:
    """generate figure showing the 3-tier detection hierarchy."""
    from src.utils.plots import setup_style, DPI, W_FULL
    setup_style()
    import matplotlib.pyplot as plt
    from src.config import RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)

    ca = _make_ca()

    t1_snaps, t1_times, t1_counts = _tier1_panels(ca)
    t2_snaps, t2_times, t2_counts = _tier2_panels(ca)
    t3_snaps, t3_labels, t3_counts = _tier3_panels(ca)

    fig = plt.figure(figsize=(W_FULL, 6))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.08,
                          left=0.08, right=0.98, top=0.95, bottom=0.04)

    ax_r1 = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax_r2 = [fig.add_subplot(gs[1, c]) for c in range(4)]
    ax_r3 = [fig.add_subplot(gs[2, c]) for c in range(3)]

    _render_row(ax_r1, t1_snaps, t1_times, t1_counts, "Tier 1")
    _render_row(ax_r2, t2_snaps, t2_times, t2_counts, "Tier 2")

    for col, (snap, lbl, cnt) in enumerate(zip(t3_snaps, t3_labels, t3_counts)):
        ax = ax_r3[col]
        ax.imshow(snap, cmap="Greys", interpolation="nearest", vmin=0, vmax=1, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lbl, fontsize=8)
        ax.text(0.5, -0.08, rf"$n_c={cnt}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)
        if col == 0:
            ax.set_ylabel("Tier 3", fontsize=8, labelpad=8)

    panel_labels = [("(a)", ax_r1[0]), ("(b)", ax_r2[0]), ("(c)", ax_r3[0])]
    for label, ax in panel_labels:
        ax.text(-0.22, 1.08, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig_path = RESULTS / "fig-tier-examples.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    logger.info("tier examples figure saved to %s", fig_path)

    output = {
        "figure": str(fig_path),
        "tier1_counts": t1_counts,
        "tier2_counts": t2_counts,
        "tier3_counts": t3_counts,
    }
    artifact.write_text(json.dumps(output, indent=2))
    return output
