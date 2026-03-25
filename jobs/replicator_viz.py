"""visualize example self-replicating patterns from the census."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """find and visualize tier-1 replicators as time-lapse grids."""
    import numpy as np
    from src.utils.plots import setup_style, DPI, W_FULL
    setup_style()
    import matplotlib.pyplot as plt
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer, HIGH_LIFE

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    tier1_rules = [r for r in data["rules"] if r.get("t1", r.get("tier1", False))]
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()

    examples = _select_examples(tier1_rules, all_rules)
    timesteps = [0, 32, 64, 128, 192, 256]

    n_examples = len(examples)
    fig, axes = plt.subplots(
        n_examples, 6, figsize=(W_FULL, 2.0 * n_examples),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if len(examples) == 1:
        axes = [axes]

    for row, (name, rule_table) in enumerate(examples):
        ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
        ic = make_random_ic(64, 2, 0.15, np.random.default_rng(42))
        snaps = ca.run(ic, 256, snapshot_interval=32)

        for col, (t, snap_idx) in enumerate(zip(timesteps, range(len(snaps)))):
            if snap_idx >= len(snaps):
                break
            ax = axes[row][col]
            ax.imshow(snaps[snap_idx], cmap="Greys", interpolation="nearest",
                      vmin=0, vmax=1, rasterized=True)
            ax.set_title(rf"$t={t}$", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("black")
            if col == 0:
                ax.set_ylabel(name, fontsize=7)

    fig.tight_layout(pad=1.0)
    fig.savefig(artifact.parent / "fig-replicator-examples.png", dpi=DPI)
    plt.close(fig)

    artifact.write_text(json.dumps({"examples": [e[0] for e in examples],
                                     "figure": "fig-replicator-examples.png"}))
    logger.info("  replicator visualization saved")
    return {"examples": [e[0] for e in examples]}


def _select_examples(tier1_rules: list, all_rules, n: int = 3) -> list:
    """select diverse tier-1 replicators at different lambda values."""
    from src.modules.rule_params import HIGH_LIFE

    examples = [("HighLife", HIGH_LIFE)]
    sorted_rules = sorted(tier1_rules, key=lambda r: r.get("l", r.get("lambda", 0)))
    low_lam = sorted_rules[len(sorted_rules) // 10]
    mid_lam = sorted_rules[len(sorted_rules) // 2]

    for prefix, r in [("Low", low_lam), ("Mid", mid_lam)]:
        idx = r.get("i", r.get("rule_index", 0))
        lam = r.get("l", r.get("lambda", 0))
        examples.append((rf"{prefix}-$\lambda$ ({lam:.2f})", all_rules[idx]))

    return examples
