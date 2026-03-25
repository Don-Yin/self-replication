"""experiment F: verify top tier1 rules on larger grid + longer time."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, n_top: int = 100, **kwargs) -> dict:
    """re-run top tier1 rules on 256x256 grid for 8192 steps with full tier 2 test."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage2Matcher

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    tier1_rules = [r for r in data["rules"] if r["t1"]]
    tier1_rules.sort(key=lambda r: r["l"])
    selected = tier1_rules[:n_top]

    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()
    matcher = Stage2Matcher()
    results_list = []
    confirmed = 0

    for count, r in enumerate(selected):
        rule_table = all_rules[r["i"]]
        ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
        ic = make_random_ic(256, 2, 0.15, np.random.default_rng(r["i"]))
        snaps = ca.run(ic, 8192, snapshot_interval=512)
        match_result = matcher.match(snaps)

        tier1_confirmed = match_result["tier1_detected"]
        if tier1_confirmed:
            confirmed += 1

        results_list.append({
            "rule_index": r["i"],
            "lambda": r["l"],
            "f_param": r["f"],
            "tier1_256": tier1_confirmed,
            "n_replicators": len(match_result.get("tier1_replicators", [])),
        })

        if (count + 1) % 20 == 0:
            logger.info("    verified %d/%d | confirmed: %d", count + 1, len(selected), confirmed)

    output = {
        "n_tested": len(selected),
        "n_confirmed": confirmed,
        "confirmation_rate": round(confirmed / max(len(selected), 1), 4),
        "rules": results_list,
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  verification: %d/%d confirmed (%.1f%%)", confirmed, len(selected), 100 * confirmed / max(len(selected), 1))
    return output
