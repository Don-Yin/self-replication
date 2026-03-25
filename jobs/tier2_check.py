"""tier-2 check: test whether tier-1 replicators have temporal structure."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, n_sample: int = 1000, **kwargs) -> dict:
    """run tier-2 detection on a sample of tier-1-positive rules."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage2Matcher

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    tier1_rules = [r for r in data["rules"] if r.get("t1", r.get("tier1", False))]
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()
    matcher = Stage2Matcher()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(tier1_rules), size=min(n_sample, len(tier1_rules)), replace=False)

    results_list = []
    n_tier2 = 0

    for count, idx in enumerate(sample_idx):
        r = tier1_rules[idx]
        rule_idx = r.get("i", r.get("rule_index", 0))
        rule_table = all_rules[rule_idx]
        lam = r.get("l", r.get("lambda", 0))

        ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
        ic = make_random_ic(64, 2, 0.15, np.random.default_rng(rule_idx))
        snaps = ca.run(ic, 512, snapshot_interval=64)
        match_result = matcher.match(snaps)

        tier2 = _check_tier2(match_result)
        if tier2:
            n_tier2 += 1

        results_list.append({
            "rule_index": rule_idx,
            "lambda": round(float(lam), 4),
            "tier1": True,
            "tier2": tier2,
            "n_replicators": len(match_result.get("tier1_replicators", [])),
        })

        if (count + 1) % 200 == 0:
            logger.info("    %d/%d | tier2: %d", count + 1, len(sample_idx), n_tier2)

    tier2_rate = n_tier2 / max(len(sample_idx), 1)
    output = {
        "n_sampled": len(sample_idx),
        "n_tier1": len(sample_idx),
        "n_tier2": n_tier2,
        "tier2_rate_among_tier1": round(tier2_rate, 6),
        "tier2_rate_overall": round(n_tier2 / max(data["n_rules"], 1), 6),
        "rules": results_list,
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  tier-2 check: %d/%d tier-1 rules confirmed as tier-2 (%.1f%%)",
                n_tier2, len(sample_idx), 100 * tier2_rate)
    return output


def _check_tier2(match_result: dict) -> bool:
    """check if any detected replicator has temporal structure (period >= 2)."""
    for rep in match_result.get("tier1_replicators", []):
        if rep.get("increases", 0) >= 3:
            return True
    return False
