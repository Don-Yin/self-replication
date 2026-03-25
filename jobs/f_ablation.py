"""F-weighting ablation: recompute F under uniform, linear, quadratic schemes."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

WEIGHT_SCHEMES = ("uniform", "linear", "quadratic")
N_F_BINS = 20


def compute_f_weighted(rule_table: np.ndarray, weight_scheme: str, s_max: int) -> float:
    """compute F using the specified weighting scheme over quiescent row."""
    score = 0.0
    total_weight = 0.0
    for s in range(s_max + 1):
        w = _weight(s, s_max, weight_scheme)
        total_weight += w
        if rule_table[0, s] != 0:
            score += w
    return score / max(total_weight, 1e-12)


def _weight(s: int, s_max: int, scheme: str) -> float:
    """compute weight for neighbor sum s under the given scheme."""
    ratio = s / max(s_max, 1)
    match scheme:
        case "uniform":
            return 1.0
        case "linear":
            return 1.0 - ratio
        case "quadratic":
            return (1.0 - ratio) ** 2


def _bin_tier1_rate(lambdas: np.ndarray, fs: np.ndarray, tier1: np.ndarray) -> dict:
    """bin rules into (lambda, F) grid and compute tier-1 rate per F bin."""
    f_edges = np.linspace(0, 1, N_F_BINS + 1)
    f_centers = ((f_edges[:-1] + f_edges[1:]) / 2).tolist()

    f_indices = np.clip(((fs - f_edges[0]) / (f_edges[-1] - f_edges[0]) * N_F_BINS).astype(int), 0, N_F_BINS - 1)
    totals = np.bincount(f_indices, minlength=N_F_BINS).astype(float)
    hits = np.bincount(f_indices, weights=tier1.astype(float), minlength=N_F_BINS)
    rates = np.where(totals > 0, hits / totals, 0).tolist()

    peak_bin = int(np.argmax(rates))
    return {
        "f_centers": f_centers,
        "tier1_rates": rates,
        "bin_counts": totals.astype(int).tolist(),
        "peak_f": round(f_centers[peak_bin], 4),
        "peak_rate": round(rates[peak_bin], 4),
    }


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """recompute F under 3 weighting schemes and measure tier-1 rate invariance."""
    from src.modules.rule_params import RuleParameterizer

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    data = json.loads(census_path.read_text())
    census_rules = data["rules"]
    tier1_labels = np.array([r.get("t1", r.get("tier1", False)) for r in census_rules])
    census_lambdas = np.array([r.get("l", r.get("lambda", 0)) for r in census_rules])

    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    logger.info("  enumerating all k=2 moore rules for F recomputation")
    all_rules = parameterizer.enumerate_all()
    n_rules = len(all_rules)
    logger.info("  total rules: %d", n_rules)

    per_scheme = {}
    peak_fs = []

    for scheme in WEIGHT_SCHEMES:
        fs = np.array([compute_f_weighted(all_rules[i], scheme, parameterizer.max_sum) for i in range(n_rules)])
        binned = _bin_tier1_rate(census_lambdas, fs, tier1_labels)
        per_scheme[scheme] = binned
        peak_fs.append(binned["peak_f"])
        logger.info("  %s: peak F=%.3f, peak rate=%.3f", scheme, binned["peak_f"], binned["peak_rate"])

    peak_spread = round(max(peak_fs) - min(peak_fs), 4)
    invariant = peak_spread < 0.15

    results = {
        "n_rules": n_rules,
        "n_f_bins": N_F_BINS,
        "weight_schemes": list(WEIGHT_SCHEMES),
        "per_scheme": per_scheme,
        "summary": {
            "peak_fs": {s: per_scheme[s]["peak_f"] for s in WEIGHT_SCHEMES},
            "peak_rates": {s: per_scheme[s]["peak_rate"] for s in WEIGHT_SCHEMES},
            "peak_spread": peak_spread,
            "qualitatively_invariant": invariant,
        },
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  F ablation done | invariant=%s | spread=%.4f", invariant, peak_spread)
    return results
