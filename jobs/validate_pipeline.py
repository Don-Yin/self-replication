"""validate the detection pipeline on known positive and negative rules."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, **kwargs) -> dict:
    """test pipeline on GoL, HighLife, dead rules, and chaotic rules."""
    import numpy as np
    from src.modules.simulator import OuterTotalisticCA, make_random_ic
    from src.modules.rule_params import GAME_OF_LIFE, HIGH_LIFE, RuleParameterizer
    from src.modules.detector import Stage1Screener, Stage2Matcher

    rng = np.random.default_rng(42)
    screener = Stage1Screener()
    matcher = Stage2Matcher()
    results = {}

    # test 1: Game of Life (B3/S23) - should show nontrivial dynamics
    logger.info("  testing Game of Life (B3/S23)")
    ca_gol = OuterTotalisticCA(k=2, neighborhood="moore", rule_table=GAME_OF_LIFE)
    ic = make_random_ic(64, 2, 0.35, rng)
    snapshots = ca_gol.run(ic, 512)
    screen = screener.screen(snapshots)
    match_result = matcher.match(snapshots) if screen["flagged"] else {"tier1_detected": False}
    results["gol"] = {
        "flagged": screen["flagged"],
        "indicators": {k: screen[k] for k in ("indicator_a", "indicator_b", "indicator_c")},
        "tier1": match_result["tier1_detected"],
    }
    logger.info("    GoL flagged=%s tier1=%s", screen["flagged"], match_result["tier1_detected"])

    # test 2: HighLife (B36/S23) - known replicator
    logger.info("  testing HighLife (B36/S23)")
    ca_hl = OuterTotalisticCA(k=2, neighborhood="moore", rule_table=HIGH_LIFE)
    ic = make_random_ic(64, 2, 0.15, rng)
    snapshots = ca_hl.run(ic, 1024)
    screen = screener.screen(snapshots)
    match_result = matcher.match(snapshots) if screen["flagged"] else {"tier1_detected": False}
    results["highlife"] = {
        "flagged": screen["flagged"],
        "indicators": {k: screen[k] for k in ("indicator_a", "indicator_b", "indicator_c")},
        "tier1": match_result["tier1_detected"],
    }
    logger.info("    HighLife flagged=%s tier1=%s", screen["flagged"], match_result["tier1_detected"])

    # test 3: dead rule (all entries map to 0)
    logger.info("  testing dead rule (lambda=0)")
    dead_table = np.zeros((2, 9), dtype=np.int8)
    ca_dead = OuterTotalisticCA(k=2, neighborhood="moore", rule_table=dead_table)
    ic = make_random_ic(64, 2, 0.35, rng)
    snapshots = ca_dead.run(ic, 256)
    screen = screener.screen(snapshots)
    results["dead"] = {"flagged": screen["flagged"]}
    logger.info("    dead rule flagged=%s (expect False)", screen["flagged"])

    # test 4: chaotic rule (high lambda, random table)
    logger.info("  testing chaotic rule (lambda~0.9)")
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    chaotic_rules = parameterizer.sample_at_lambda(0.9, 5, rng)
    chaotic_flagged = 0
    for i, rule in enumerate(chaotic_rules):
        ca_chaos = OuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule)
        ic = make_random_ic(64, 2, 0.35, rng)
        snapshots = ca_chaos.run(ic, 256)
        screen = screener.screen(snapshots)
        if screen["flagged"]:
            chaotic_flagged += 1
    results["chaotic"] = {"n_tested": 5, "n_flagged": chaotic_flagged}
    logger.info("    chaotic rules: %d/5 flagged", chaotic_flagged)

    # summary
    results["summary"] = {
        "gol_detected": results["gol"]["flagged"],
        "highlife_detected": results["highlife"]["flagged"],
        "dead_correctly_ignored": not results["dead"]["flagged"],
        "chaotic_flag_rate": chaotic_flagged / 5,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    return results
