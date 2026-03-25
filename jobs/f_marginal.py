"""F-marginal: self-replication rate as a function of F, marginalising over lambda."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """compute self-replication rate vs F, overall and conditioned on lambda terciles."""
    import numpy as np

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    rules = data["rules"]
    lambdas = np.array([r.get("l", r.get("lambda", 0)) for r in rules])
    fs = np.array([r.get("f", r.get("f_param", 0)) for r in rules])
    tier1 = np.array([r.get("t1", r.get("tier1", False)) for r in rules])

    n_bins = 25
    edges = np.linspace(0, 1, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    overall = _bin_rate(fs, tier1, edges)

    lam_terciles = np.percentile(lambdas[lambdas > 0], [33, 67])
    conditioned = {}
    for lo, hi, label in [
        (0, lam_terciles[0], "low_lambda"),
        (lam_terciles[0], lam_terciles[1], "mid_lambda"),
        (lam_terciles[1], 1.01, "high_lambda"),
    ]:
        mask = (lambdas >= lo) & (lambdas < hi)
        conditioned[label] = {
            "range": [round(float(lo), 4), round(float(hi), 4)],
            "rate": _bin_rate(fs[mask], tier1[mask], edges),
        }

    results = {
        "n_bins": n_bins,
        "f_centers": centers,
        "overall_rate": overall,
        "lambda_terciles": [round(float(v), 4) for v in lam_terciles],
        "conditioned": conditioned,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  F-marginal computed (%d bins)", n_bins)
    return results


def _bin_rate(values, labels, edges):
    """bin values and compute label-true rate per bin."""
    import numpy as np
    n = len(edges) - 1
    counts = np.zeros(n)
    totals = np.zeros(n)
    for v, l in zip(values, labels):
        b = min(int((v - edges[0]) / (edges[-1] - edges[0]) * n), n - 1)
        b = max(0, b)
        totals[b] += 1
        if l:
            counts[b] += 1
    return np.where(totals > 0, counts / totals, 0).tolist()
