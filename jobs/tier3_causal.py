"""tier-3 causal test: test whether destroying parent patterns prevents offspring."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, tier2_path: Path = None, n_sample: int = 200, **kwargs) -> dict:
    """run tier-3 causal perturbation test on confirmed tier-2 rules."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.detector import Stage3CausalTest

    if tier2_path is None:
        tier2_path = Path("results/k2-moore-tier2-check.json")

    data = json.loads(tier2_path.read_text())
    tier2_rules = [r for r in data["rules"] if r.get("tier2", False)]
    logger.info("  %d tier-2-positive rules available", len(tier2_rules))

    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(tier2_rules), size=min(n_sample, len(tier2_rules)), replace=False)

    results_list = []
    n_tier3 = 0

    for count, idx in enumerate(sample_idx):
        r = tier2_rules[idx]
        rule_idx = r["rule_index"]
        rule_table = all_rules[rule_idx]
        lam = r["lambda"]

        ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
        tester = Stage3CausalTest(ca, n_trials=10, threshold=0.5)
        ic = make_random_ic(64, 2, 0.03, np.random.default_rng(rule_idx))
        result = tester.test(ic, 512)

        tier3 = result.get("tier3_detected", False)
        if tier3:
            n_tier3 += 1

        results_list.append({
            "rule_index": rule_idx,
            "lambda": round(float(lam), 4),
            "tier2": True,
            "tier3": tier3,
            "prevention_rate": round(result.get("prevention_rate", 0.0), 4),
            "reason": result.get("reason", ""),
        })

        if (count + 1) % 50 == 0:
            logger.info("    %d/%d | tier3: %d", count + 1, len(sample_idx), n_tier3)

    tier3_rate = n_tier3 / max(len(sample_idx), 1)
    output = {
        "n_sampled": len(sample_idx),
        "n_tier2": len(sample_idx),
        "n_tier3": n_tier3,
        "tier3_rate_among_tier2": round(tier3_rate, 6),
        "ic_density": 0.03,
        "n_trials_per_rule": 10,
        "threshold": 0.5,
        "rules": results_list,
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  tier-3 causal: %d/%d tier-2 rules confirmed as tier-3 (%.1f%%)",
                n_tier3, len(sample_idx), 100 * tier3_rate)
    return output
