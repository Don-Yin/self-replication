"""compute mass-balance for k=3 Moore and k=2 von Neumann to test cross-substrate generalisation."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, vn_census_path: Path = None, k3_sweep_path: Path = None, **kwargs) -> dict:
    """compute mass-balance across substrates and compare tier-1+ vs tier-1-."""
    import numpy as np
    from scipy.stats import mannwhitneyu
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.measures import MassBalance

    if vn_census_path is None:
        vn_census_path = Path("results/k2-vn-census.json")
    if k3_sweep_path is None:
        k3_sweep_path = Path("results/k3-moore-sweep.json")

    mb = MassBalance()
    vn_results = _compute_vn(vn_census_path, mb)
    k3_results = _compute_k3(k3_sweep_path, mb)

    output = {"vn_results": vn_results, "k3_results": k3_results}
    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  cross-substrate mass-balance: vn d=%.2f, k3 d=%.2f",
                vn_results["cohens_d"], k3_results["cohens_d"])
    return output


def _compute_vn(census_path: Path, mb) -> dict:
    """mass-balance for all k=2 von Neumann rules, split by tier-1 label."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer

    data = json.loads(census_path.read_text())
    rules_data = data["rules"]
    parameterizer = RuleParameterizer(k=2, n_neighbors=4)
    all_rules = parameterizer.enumerate_all()

    pos_vals, neg_vals = [], []
    for r in rules_data:
        idx = r["rule_index"]
        rule_table = all_rules[idx]
        ca = FastOuterTotalisticCA(k=2, neighborhood="von_neumann", rule_table=rule_table)
        ic = make_random_ic(64, 2, 0.15, np.random.default_rng(idx))
        val = mb.compute(ca, ic, steps=64)
        if r["tier1"]:
            pos_vals.append(val)
        else:
            neg_vals.append(val)

    logger.info("  vn: %d tier1+, %d tier1-", len(pos_vals), len(neg_vals))
    return _cohens_d_result(np.array(pos_vals), np.array(neg_vals))


def _compute_k3(sweep_path: Path, mb) -> dict:
    """mass-balance for k=3 Moore rules, reproducing exact RNG seeds from k3_sweep."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer

    data = json.loads(sweep_path.read_text())
    rules_data = data["rules"]
    parameterizer = RuleParameterizer(k=3, n_neighbors=8)

    lambda_values = np.linspace(0.05, 1 - 1 / 3, 20)
    n_per_lambda = len(rules_data) // len(lambda_values)

    reproduced = []
    for lam_target in lambda_values:
        seed = int(lam_target * 10000)
        rules = parameterizer.sample_at_lambda(lam_target, n_per_lambda, np.random.default_rng(seed))
        reproduced.extend(rules)

    pos_vals, neg_vals = [], []
    for i, r in enumerate(rules_data):
        rule_table = reproduced[i]
        ca = FastOuterTotalisticCA(k=3, neighborhood="moore", rule_table=rule_table)
        ic = make_random_ic(64, 3, 0.15, np.random.default_rng(i))
        val = mb.compute(ca, ic, steps=64)
        if r["t1"]:
            pos_vals.append(val)
        else:
            neg_vals.append(val)

    logger.info("  k3: %d tier1+, %d tier1-", len(pos_vals), len(neg_vals))
    return _cohens_d_result(np.array(pos_vals), np.array(neg_vals))


def _cohens_d_result(pos: 'np.ndarray', neg: 'np.ndarray') -> dict:
    """compute Cohen's d and Mann-Whitney U between two groups."""
    import numpy as np
    from scipy.stats import mannwhitneyu

    pooled_std = np.sqrt((pos.var() + neg.var()) / 2)
    d = float((pos.mean() - neg.mean()) / max(pooled_std, 1e-12))
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")

    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "mean_pos": round(float(pos.mean()), 6),
        "mean_neg": round(float(neg.mean()), 6),
        "cohens_d": round(d, 3),
        "p_value": float(f"{p:.2e}"),
    }
