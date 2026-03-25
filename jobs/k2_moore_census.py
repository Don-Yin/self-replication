"""exhaustive census of all 262K k=2 Moore outer-totalistic rules."""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, **kwargs) -> dict:
    """enumerate all 262,144 life-like rules with fast screening + stage 2."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher

    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    screener = Stage1Screener()
    matcher = Stage2Matcher()

    logger.info("  enumerating all k=2 moore rules")
    all_rules = parameterizer.enumerate_all()
    n_rules = len(all_rules)
    logger.info("  total rules: %d", n_rules)

    rule_data = []
    n_flagged = 0
    n_tier1 = 0
    t_start = time.time()

    for i in range(n_rules):
        rule_table = all_rules[i]
        lam = parameterizer.compute_lambda(rule_table)
        f_val = parameterizer.compute_f(rule_table)

        flagged = False
        tier1 = False
        best_snapshots = None

        for density in (0.15, 0.35):
            ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
            ic = make_random_ic(64, 2, density, np.random.default_rng(i * 37 + int(density * 100)))
            snapshots = ca.run(ic, 256, snapshot_interval=64)
            screen = screener.screen(snapshots)
            if screen["flagged"]:
                flagged = True
                best_snapshots = snapshots
                break

        if flagged and best_snapshots is not None:
            n_flagged += 1
            match_result = matcher.match(best_snapshots)
            tier1 = match_result["tier1_detected"]
            if tier1:
                n_tier1 += 1

        rule_data.append({
            "i": int(i),
            "l": round(lam, 4),
            "f": round(f_val, 4),
            "s1": flagged,
            "t1": tier1,
        })

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_rules - i - 1) / rate
            logger.info(
                "    %d/%d (%.1f/s) | flagged: %d | tier1: %d | ETA: %.0f min",
                i + 1, n_rules, rate, n_flagged, n_tier1, eta / 60,
            )

    elapsed = time.time() - t_start
    results = {
        "neighborhood": "moore",
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
        "  done: %d rules in %.0f min | tier1: %d (%.2f%%)",
        n_rules, elapsed / 60, n_tier1, 100 * n_tier1 / max(n_rules, 1),
    )
    return results
