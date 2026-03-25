"""compute derived measures for rules near the self-replication phase boundary."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, n_sample: int = 500, **kwargs) -> dict:
    """sample rules near the boundary, compute info-theoretic measures."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.measures import DerridaCoefficient, MassBalance, spatial_entropy

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    rules_data = data["rules"]
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()

    tier1_pos = [r for r in rules_data if r["t1"]]
    tier1_neg = [r for r in rules_data if not r["t1"] and r["s1"]]

    rng = np.random.default_rng(42)
    pos_sample = rng.choice(len(tier1_pos), size=min(n_sample, len(tier1_pos)), replace=False)
    neg_sample = rng.choice(len(tier1_neg), size=min(n_sample, len(tier1_neg)), replace=False)

    mb = MassBalance()
    results_list = []

    for label, indices, source in [
        ("tier1_positive", pos_sample, tier1_pos),
        ("tier1_negative", neg_sample, tier1_neg),
    ]:
        logger.info("  computing measures for %s (%d rules)", label, len(indices))
        for count, idx in enumerate(indices):
            r = source[idx]
            rule_idx = r["i"]
            rule_table = all_rules[rule_idx]
            ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
            ic = make_random_ic(64, 2, 0.15, np.random.default_rng(rule_idx))

            dc = DerridaCoefficient(ca, size=32, n_perturbations=50)
            derrida = dc.compute()
            mass_bal = mb.compute(ca, ic, steps=64)
            snaps = ca.run(ic, 128, snapshot_interval=128)
            s_entropy = spatial_entropy(snaps[-1], 2)

            results_list.append({
                "rule_index": rule_idx,
                "lambda": r["l"],
                "f_param": r["f"],
                "tier1": r["t1"],
                "label": label,
                "derrida": round(derrida, 6),
                "mass_balance": round(mass_bal, 6),
                "spatial_entropy": round(s_entropy, 6),
            })

            if (count + 1) % 100 == 0:
                logger.info("    %s: %d/%d done", label, count + 1, len(indices))

    pos_measures = [r for r in results_list if r["label"] == "tier1_positive"]
    neg_measures = [r for r in results_list if r["label"] == "tier1_negative"]

    summary = {}
    for measure_name in ("derrida", "mass_balance", "spatial_entropy"):
        pos_vals = np.array([r[measure_name] for r in pos_measures])
        neg_vals = np.array([r[measure_name] for r in neg_measures])
        summary[measure_name] = {
            "tier1_pos_mean": round(float(np.mean(pos_vals)), 6),
            "tier1_pos_std": round(float(np.std(pos_vals)), 6),
            "tier1_neg_mean": round(float(np.mean(neg_vals)), 6),
            "tier1_neg_std": round(float(np.std(neg_vals)), 6),
            "difference": round(float(np.mean(pos_vals) - np.mean(neg_vals)), 6),
        }

    output = {
        "n_pos_sampled": len(pos_measures),
        "n_neg_sampled": len(neg_measures),
        "summary": summary,
        "rules": results_list,
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  boundary measures complete")
    for name, s in summary.items():
        logger.info("    %s: pos=%.4f neg=%.4f diff=%.4f",
                     name, s["tier1_pos_mean"], s["tier1_neg_mean"], s["difference"])
    return output
