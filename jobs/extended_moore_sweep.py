"""k=2 extended Moore (|N|=25) sweep for self-replication."""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, n_per_lambda: int = 500, **kwargs) -> dict:
    """sample k=2 extended Moore rules across lambda range."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher

    parameterizer = RuleParameterizer(k=2, n_neighbors=24)
    screener = Stage1Screener()
    matcher = Stage2Matcher()

    lambda_values = np.linspace(0.05, 1 - 1 / 2, 20)
    rule_data = []
    n_total = 0
    n_tier1 = 0
    t_start = time.time()

    for lam_target in lambda_values:
        logger.info("  k=2 ext-moore lambda=%.2f: sampling %d rules", lam_target, n_per_lambda)
        rules = parameterizer.sample_at_lambda(lam_target, n_per_lambda,
                                                np.random.default_rng(int(lam_target * 10000)))

        for rule_table in rules:
            n_total += 1
            lam = parameterizer.compute_lambda(rule_table)
            f_val = parameterizer.compute_f(rule_table)
            tier1 = False

            for density in (0.15, 0.35):
                ca = FastOuterTotalisticCA(k=2, neighborhood="extended_moore", rule_table=rule_table)
                ic = make_random_ic(64, 2, density, np.random.default_rng(n_total * 37))
                snaps = ca.run(ic, 256, snapshot_interval=64)
                screen = screener.screen(snaps)
                if screen["flagged"]:
                    match = matcher.match(snaps)
                    tier1 = match["tier1_detected"]
                    break

            if tier1:
                n_tier1 += 1
            rule_data.append({"l": round(lam, 4), "f": round(f_val, 4), "t1": tier1})

    elapsed = time.time() - t_start
    results = {
        "k": 2, "neighborhood": "extended_moore", "n_neighbors": 24,
        "n_rules": n_total, "n_tier1": n_tier1,
        "tier1_rate": round(n_tier1 / max(n_total, 1), 6),
        "elapsed_sec": round(elapsed, 1), "rules": rule_data,
    }
    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  ext-moore: %d/%d tier1 (%.2f%%) in %.0f min",
                n_tier1, n_total, 100 * n_tier1 / max(n_total, 1), elapsed / 60)
    return results
