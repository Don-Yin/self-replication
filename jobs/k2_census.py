"""exhaustive census of k=2 outer-totalistic rules for self-replication."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, neighborhood: str = "von_neumann", **kwargs) -> dict:
    """enumerate all k=2 outer-totalistic rules, compute params, detect replication."""
    import numpy as np
    from src.modules.simulator import OuterTotalisticCA, make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher
    from src.config import SweepConfig

    cfg = SweepConfig()
    n_neighbors = {"von_neumann": 4, "moore": 8, "extended_moore": 24}[neighborhood]
    parameterizer = RuleParameterizer(k=2, n_neighbors=n_neighbors)
    screener = Stage1Screener()
    matcher = Stage2Matcher()

    logger.info("  enumerating k=2 %s rules", neighborhood)
    all_rules = parameterizer.enumerate_all()
    n_rules = len(all_rules)
    logger.info("  total rules: %d", n_rules)

    rule_data = []
    n_flagged = 0
    n_tier1 = 0

    for i, rule_table in enumerate(all_rules):
        lam = parameterizer.compute_lambda(rule_table)
        f_val = parameterizer.compute_f(rule_table)

        best_screen = {"flagged": False}
        best_snapshots = None

        for density in cfg.densities:
            for ic_idx in range(cfg.ics_per_density):
                ca = OuterTotalisticCA(k=2, neighborhood=neighborhood, rule_table=rule_table)
                ic = make_random_ic(64, 2, density, np.random.default_rng(ic_idx * 1000 + i))
                snapshots = ca.run(ic, 512)
                screen = screener.screen(snapshots)
                if screen["flagged"]:
                    best_screen = screen
                    best_snapshots = snapshots
                    break
            if best_screen["flagged"]:
                break

        tier1 = False
        if best_screen["flagged"] and best_snapshots is not None:
            n_flagged += 1
            match_result = matcher.match(best_snapshots)
            tier1 = match_result["tier1_detected"]
            if tier1:
                n_tier1 += 1

        entry = {
            "rule_index": int(i),
            "lambda": round(lam, 4),
            "f_param": round(f_val, 4),
            "flagged": best_screen["flagged"],
            "tier1": tier1,
        }
        rule_data.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                "    processed %d/%d | flagged: %d | tier1: %d",
                i + 1, n_rules, n_flagged, n_tier1,
            )

    results = {
        "neighborhood": neighborhood,
        "k": 2,
        "n_rules": n_rules,
        "n_flagged": n_flagged,
        "n_tier1": n_tier1,
        "flag_rate": round(n_flagged / max(n_rules, 1), 6),
        "tier1_rate": round(n_tier1 / max(n_rules, 1), 6),
        "rules": rule_data,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info(
        "  census: %d rules | flagged: %d (%.1f%%) | tier1: %d (%.1f%%)",
        n_rules, n_flagged, 100 * n_flagged / max(n_rules, 1),
        n_tier1, 100 * n_tier1 / max(n_rules, 1),
    )
    return results
