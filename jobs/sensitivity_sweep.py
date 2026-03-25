"""detection-threshold sensitivity analysis: vary snapshot_interval and min_increases."""
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVALS = (32, 64, 128)
MIN_INCREASES_VALUES = (2, 3, 4, 5)
N_POS = 2500
N_NEG = 2500
SIM_STEPS = 512
FINEST_INTERVAL = 32
GRID_SIZE = 64
REFERENCE = {"snapshot_interval": 64, "min_increases": 3}


def _detect_tier1(snapshots: list[np.ndarray], min_increases: int) -> bool:
    """check if any canonical hash shows >= min_increases consecutive count increases."""
    from src.modules.detector import ComponentTracker

    tracker = ComponentTracker(k=2)
    hash_timeseries: dict[int, list[int]] = {}

    for snap in snapshots:
        labels, n_comp = tracker.label_components(snap)
        step_hashes: dict[int, int] = {}
        for lid in range(1, n_comp + 1):
            h = tracker.canonical_hash(snap, labels, lid)
            step_hashes[h] = step_hashes.get(h, 0) + 1
        for h, count in step_hashes.items():
            hash_timeseries.setdefault(h, []).append(count)

    return any(
        _count_increases(counts) >= min_increases
        for h, counts in hash_timeseries.items() if h != 0
    )


def _count_increases(counts: list[int]) -> int:
    """count consecutive strict increases in a time series."""
    arr = np.array(counts)
    return int(np.sum(arr[1:] > arr[:-1]))


def _subsample_snapshots(all_snapshots: list[np.ndarray], ratio: int) -> list[np.ndarray]:
    """subsample snapshots by keeping every ratio-th entry (plus the IC at index 0)."""
    return [all_snapshots[0]] + [all_snapshots[i] for i in range(ratio, len(all_snapshots), ratio)]


def _sample_rule_indices(census_rules: list[dict], rng: np.random.Generator) -> tuple:
    """sample N_POS tier-1 positive and N_NEG tier-1 negative indices from the census."""
    pos_indices = [r["i"] for r in census_rules if r["t1"]]
    neg_indices = [r["i"] for r in census_rules if r["s1"] and not r["t1"]]

    n_pos = min(N_POS, len(pos_indices))
    n_neg = min(N_NEG, len(neg_indices))
    sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False).tolist()
    sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False).tolist()
    return sampled_pos, sampled_neg


def _build_param_combos() -> list[dict]:
    """build flat list of parameter combinations."""
    return [
        {"snapshot_interval": si, "min_increases": mi}
        for si, mi in product(SNAPSHOT_INTERVALS, MIN_INCREASES_VALUES)
    ]


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """sweep detection thresholds over sampled rules and report tier-1 rates."""
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.rule_params import RuleParameterizer
    from src.modules.simulator import make_random_ic

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    census_rules = data["rules"]
    rng = np.random.default_rng(42)

    sampled_pos, sampled_neg = _sample_rule_indices(census_rules, rng)
    all_indices = sampled_pos + sampled_neg
    original_labels = np.array([True] * len(sampled_pos) + [False] * len(sampled_neg))
    n_total = len(all_indices)
    logger.info("  sampled %d pos + %d neg = %d rules", len(sampled_pos), len(sampled_neg), n_total)

    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rule_tables = parameterizer.enumerate_all()

    combos = _build_param_combos()
    subsample_ratios = {si: si // FINEST_INTERVAL for si in SNAPSHOT_INTERVALS}
    combo_hits = {(c["snapshot_interval"], c["min_increases"]): np.zeros(n_total, dtype=bool) for c in combos}

    t_start = time.time()
    for idx, rule_i in enumerate(all_indices):
        rule_table = all_rule_tables[rule_i]
        ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
        ic = make_random_ic(GRID_SIZE, 2, 0.15, np.random.default_rng(rule_i * 37 + 15))
        finest_snaps = ca.run(ic, SIM_STEPS, snapshot_interval=FINEST_INTERVAL)

        for combo in combos:
            si, mi = combo["snapshot_interval"], combo["min_increases"]
            snaps = _subsample_snapshots(finest_snaps, subsample_ratios[si])
            combo_hits[(si, mi)][idx] = _detect_tier1(snaps, mi)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            logger.info("    %d/%d (%.1f/s) | ETA: %.0f min", idx + 1, n_total, rate, (n_total - idx - 1) / rate / 60)

    results_list = []
    for combo in combos:
        si, mi = combo["snapshot_interval"], combo["min_increases"]
        hits = combo_hits[(si, mi)]
        n_tier1 = int(hits.sum())
        pos_hits = int(hits[original_labels].sum())
        neg_hits = int(hits[~original_labels].sum())
        results_list.append({
            "snapshot_interval": si,
            "min_increases": mi,
            "n_tier1": n_tier1,
            "tier1_rate": round(n_tier1 / max(n_total, 1), 6),
            "tier1_rate_among_original_pos": round(pos_hits / max(original_labels.sum(), 1), 6),
            "tier1_rate_among_original_neg": round(neg_hits / max((~original_labels).sum(), 1), 6),
        })

    elapsed = time.time() - t_start
    output = {
        "parameter_grid": [{"snapshot_interval": si, "min_increases": mi} for si, mi in product(SNAPSHOT_INTERVALS, MIN_INCREASES_VALUES)],
        "results": results_list,
        "reference": REFERENCE,
        "n_sampled": n_total,
        "n_pos": len(sampled_pos),
        "n_neg": len(sampled_neg),
        "elapsed_sec": round(elapsed, 1),
        "invariance_summary": _summarize_invariance(results_list),
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  sensitivity sweep done in %.0f min | %d combos", elapsed / 60, len(combos))
    return output


def _summarize_invariance(results: list[dict]) -> str:
    """summarize whether the island shape is preserved across parameter settings."""
    rates = [r["tier1_rate"] for r in results]
    ref = next(r["tier1_rate"] for r in results if r["snapshot_interval"] == 64 and r["min_increases"] == 3)
    spread = max(rates) - min(rates)
    ratio = max(rates) / max(min(rates), 1e-9)
    if ratio < 2.0 and spread < 0.15:
        return f"tier-1 rate stable across settings (spread={spread:.4f}, max/min ratio={ratio:.2f})"
    return f"tier-1 rate varies with settings (spread={spread:.4f}, max/min ratio={ratio:.2f}); reference rate={ref:.4f}"
