"""experiment C: k=3 outer-totalistic sweep for self-replication."""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, neighborhood: str = "moore", n_per_lambda: int = 500, **kwargs) -> dict:
    """sample k=3 outer-totalistic rules across lambda range."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher

    n_neighbors = {"von_neumann": 4, "moore": 8}[neighborhood]
    parameterizer = RuleParameterizer(k=3, n_neighbors=n_neighbors)
    screener = Stage1Screener()
    matcher = Stage2Matcher()

    lambda_values = np.linspace(0.05, 1 - 1 / 3, 20)
    rule_data = []
    n_total = 0
    n_flagged = 0
    n_tier1 = 0
    t_start = time.time()

    for lam_target in lambda_values:
        logger.info("  k=3 %s lambda=%.2f: sampling %d rules", neighborhood, lam_target, n_per_lambda)
        rules = parameterizer.sample_at_lambda(lam_target, n_per_lambda, np.random.default_rng(int(lam_target * 10000)))

        for i, rule_table in enumerate(rules):
            n_total += 1
            lam = parameterizer.compute_lambda(rule_table)
            f_val = parameterizer.compute_f(rule_table)

            flagged = False
            tier1 = False
            best_snaps = None

            for density in (0.15, 0.35):
                ca = FastOuterTotalisticCA(k=3, neighborhood=neighborhood, rule_table=rule_table)
                ic = make_random_ic(64, 3, density, np.random.default_rng(n_total * 37))
                snaps = ca.run(ic, 256, snapshot_interval=64)
                screen = screener.screen(snaps)
                if screen["flagged"]:
                    flagged = True
                    best_snaps = snaps
                    break

            if flagged and best_snaps is not None:
                n_flagged += 1
                match_result = matcher.match(best_snaps)
                tier1 = match_result["tier1_detected"]
                if tier1:
                    n_tier1 += 1

            rule_data.append({
                "l": round(lam, 4),
                "f": round(f_val, 4),
                "s1": flagged,
                "t1": tier1,
            })

    elapsed = time.time() - t_start
    results = {
        "k": 3,
        "neighborhood": neighborhood,
        "n_rules": n_total,
        "n_flagged": n_flagged,
        "n_tier1": n_tier1,
        "tier1_rate": round(n_tier1 / max(n_total, 1), 6),
        "elapsed_sec": round(elapsed, 1),
        "rules": rule_data,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  k=3 sweep: %d rules in %.0f min | tier1: %d (%.2f%%)",
                n_total, elapsed / 60, n_tier1, 100 * n_tier1 / max(n_total, 1))
    return results
