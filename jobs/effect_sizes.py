"""compute effect sizes (Cohen's d, rank-biserial) for all boundary measures."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, boundary_path: Path = None, oinfo_path: Path = None, **kwargs) -> dict:
    """compute Cohen's d and rank-biserial r for every measure in Table 1."""
    import numpy as np
    from scipy.stats import mannwhitneyu

    if boundary_path is None:
        boundary_path = Path("results/k2-moore-boundary-measures.json")
    if oinfo_path is None:
        oinfo_path = Path("results/k2-moore-oinfo-boundary.json")

    bm = json.loads(boundary_path.read_text())
    oi = json.loads(oinfo_path.read_text())

    pos_bm = [r for r in bm["rules"] if r["label"] == "tier1_positive"]
    neg_bm = [r for r in bm["rules"] if r["label"] == "tier1_negative"]
    pos_oi = [r for r in oi["rules"] if r["label"] == "tier1_positive"]
    neg_oi = [r for r in oi["rules"] if r["label"] == "tier1_negative"]

    results = {}

    for key in ["mass_balance", "spatial_entropy"]:
        p, n = np.array([r[key] for r in pos_bm]), np.array([r[key] for r in neg_bm])
        results[key] = _compute_effects(p, n)

    for key in ["oinfo", "tc", "dtc"]:
        p, n = np.array([r[key] for r in pos_oi]), np.array([r[key] for r in neg_oi])
        results[key] = _compute_effects(p, n)

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  effect sizes: mass_balance d=%.2f, oinfo d=%.2f",
                results["mass_balance"]["cohens_d"], results["oinfo"]["cohens_d"])
    return results


def _compute_effects(pos: 'np.ndarray', neg: 'np.ndarray') -> dict:
    """compute Cohen's d, rank-biserial r, and Mann-Whitney p."""
    import numpy as np
    from scipy.stats import mannwhitneyu

    pooled_std = np.sqrt((pos.var() + neg.var()) / 2)
    d = float((pos.mean() - neg.mean()) / max(pooled_std, 1e-12))

    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    n1, n2 = len(pos), len(neg)
    r_rb = 1 - (2 * stat) / (n1 * n2)

    return {
        "pos_mean": round(float(pos.mean()), 4),
        "neg_mean": round(float(neg.mean()), 4),
        "cohens_d": round(d, 3),
        "rank_biserial_r": round(float(r_rb), 3),
        "p_value": float(f"{p:.2e}"),
        "n_pos": n1,
        "n_neg": n2,
    }
