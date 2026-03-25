"""equalised census of all 1024 k=2 von Neumann rules using Moore protocol."""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DENSITIES = (0.15, 0.35)
GRID_SIZE = 64
STEPS = 256
SNAPSHOT_INTERVAL = 64


def _test_rule(i: int, rule_table, parameterizer, screener, matcher) -> dict:
    """simulate one rule at two densities, return census entry."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic

    lam = parameterizer.compute_lambda(rule_table)
    f_val = parameterizer.compute_f(rule_table)

    flagged = False
    tier1 = False
    best_snapshots = None

    for d_idx, density in enumerate(DENSITIES):
        ca = FastOuterTotalisticCA(k=2, neighborhood="von_neumann", rule_table=rule_table)
        seed = i * 10 + d_idx
        ic = make_random_ic(GRID_SIZE, 2, density, np.random.default_rng(seed))
        snapshots = ca.run(ic, STEPS, snapshot_interval=SNAPSHOT_INTERVAL)
        screen = screener.screen(snapshots)
        if screen["flagged"]:
            flagged = True
            best_snapshots = snapshots
            break

    if flagged and best_snapshots is not None:
        match_result = matcher.match(best_snapshots)
        tier1 = match_result["tier1_detected"]

    return {
        "i": int(i),
        "l": round(lam, 4),
        "f": round(f_val, 4),
        "s1": flagged,
        "t1": tier1,
    }


def run(artifact: Path, **kwargs) -> dict:
    """enumerate all 1024 k=2 vN rules with the same protocol as the Moore census."""
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher

    parameterizer = RuleParameterizer(k=2, n_neighbors=4)
    screener = Stage1Screener()
    matcher = Stage2Matcher()

    logger.info("  enumerating all k=2 von neumann rules")
    all_rules = parameterizer.enumerate_all()
    n_rules = len(all_rules)
    logger.info("  total rules: %d", n_rules)

    rule_data = []
    n_flagged = 0
    n_tier1 = 0
    t_start = time.time()

    for i in range(n_rules):
        entry = _test_rule(i, all_rules[i], parameterizer, screener, matcher)
        rule_data.append(entry)
        if entry["s1"]:
            n_flagged += 1
        if entry["t1"]:
            n_tier1 += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_rules - i - 1) / rate
            logger.info(
                "    %d/%d (%.1f/s) | flagged: %d | tier1: %d | ETA: %.0f s",
                i + 1, n_rules, rate, n_flagged, n_tier1, eta,
            )

    elapsed = time.time() - t_start
    results = {
        "neighborhood": "von_neumann",
        "k": 2,
        "n_rules": n_rules,
        "n_flagged": n_flagged,
        "n_tier1": n_tier1,
        "flag_rate": round(n_flagged / max(n_rules, 1), 6),
        "tier1_rate": round(n_tier1 / max(n_rules, 1), 6),
        "elapsed_sec": round(elapsed, 1),
        "rules": rule_data,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info(
        "  done: %d rules in %.0f s | tier1: %d (%.2f%%)",
        n_rules, elapsed, n_tier1, 100 * n_tier1 / max(n_rules, 1),
    )
    return results
