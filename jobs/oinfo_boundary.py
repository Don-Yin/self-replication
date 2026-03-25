"""experiment D: O-information at the self-replication phase boundary."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, n_sample: int = 500, **kwargs) -> dict:
    """compare O-information between tier1+ and tier1- rules."""
    import numpy as np
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.simulator import make_random_ic
    from src.modules.rule_params import RuleParameterizer
    from src.modules.measures import OInformation

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    rules_data = data["rules"]
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    all_rules = parameterizer.enumerate_all()

    tier1_pos = [r for r in rules_data if r["t1"]]
    tier1_neg = [r for r in rules_data if not r["t1"] and r["s1"]]

    rng = np.random.default_rng(42)
    pos_idx = rng.choice(len(tier1_pos), size=min(n_sample, len(tier1_pos)), replace=False)
    neg_idx = rng.choice(len(tier1_neg), size=min(n_sample, len(tier1_neg)), replace=False)

    oinfo = OInformation()
    results_list = []

    for label, indices, source in [
        ("tier1_positive", pos_idx, tier1_pos),
        ("tier1_negative", neg_idx, tier1_neg),
    ]:
        logger.info("  computing O-info for %s (%d rules)", label, len(indices))
        for count, idx in enumerate(indices):
            r = source[idx]
            rule_table = all_rules[r["i"]]
            ca = FastOuterTotalisticCA(k=2, neighborhood="moore", rule_table=rule_table)
            ic = make_random_ic(32, 2, 0.15, np.random.default_rng(r["i"]))
            snaps = ca.run(ic, 256, snapshot_interval=64)
            oi = oinfo.compute(snaps[1:], 2, radius=1)

            results_list.append({
                "rule_index": r["i"],
                "lambda": r["l"],
                "f_param": r["f"],
                "tier1": r["t1"],
                "label": label,
                "oinfo": round(oi["oinfo"], 6),
                "tc": round(oi["tc"], 6),
                "dtc": round(oi["dtc"], 6),
                "h_joint": round(oi["h_joint"], 6),
                "synergy_dominated": oi["synergy_dominated"],
            })

            if (count + 1) % 100 == 0:
                logger.info("    %s: %d/%d done", label, count + 1, len(indices))

    pos_results = [r for r in results_list if r["label"] == "tier1_positive"]
    neg_results = [r for r in results_list if r["label"] == "tier1_negative"]

    summary = {}
    for key in ("oinfo", "tc", "dtc", "h_joint"):
        pv = np.array([r[key] for r in pos_results])
        nv = np.array([r[key] for r in neg_results])
        summary[key] = {
            "pos_mean": round(float(np.mean(pv)), 6),
            "pos_std": round(float(np.std(pv)), 6),
            "neg_mean": round(float(np.mean(nv)), 6),
            "neg_std": round(float(np.std(nv)), 6),
            "diff": round(float(np.mean(pv) - np.mean(nv)), 6),
        }

    output = {"n_pos": len(pos_results), "n_neg": len(neg_results), "summary": summary, "rules": results_list}
    artifact.write_text(json.dumps(output, indent=2, default=str))

    logger.info("  O-info boundary analysis complete:")
    for name, s in summary.items():
        logger.info("    %s: pos=%.4f neg=%.4f diff=%+.4f", name, s["pos_mean"], s["neg_mean"], s["diff"])
    return output
