"""compute derrida mu across the (lambda, F) plane for phase diagram overlay."""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _rule_table_from_index(idx: int, k: int, table_shape: tuple[int, int]) -> np.ndarray:
    """reconstruct a rule table from its enumeration index."""
    n_entries = table_shape[0] * table_shape[1]
    flat = np.zeros(n_entries, dtype=np.int8)
    val = idx
    for j in range(n_entries):
        flat[j] = val % k
        val //= k
    return flat.reshape(table_shape)


def _sample_uniform_by_lambda(
    parameterizer, n_per_level: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """sample rules uniformly across all lambda levels."""
    n_levels = parameterizer.table_size + 1
    sampled = []
    for level in range(n_levels):
        lam = level / parameterizer.table_size
        batch = parameterizer.sample_at_lambda(lam, n_per_level, rng)
        sampled.extend(batch)
    return sampled


def _sample_tier1_rules(
    census_rules: list[dict], n_sample: int, k: int, table_shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, dict]]:
    """randomly sample tier-1 rules from the census, returning (table, record) pairs."""
    tier1 = [r for r in census_rules if r["t1"]]
    indices = rng.choice(len(tier1), size=min(n_sample, len(tier1)), replace=False)
    return [(_rule_table_from_index(tier1[i]["i"], k, table_shape), tier1[i]) for i in indices]


def _compute_single(ca_cls, k: int, rule_table: np.ndarray) -> float:
    """build a CA from a rule table and compute its derrida mu."""
    from src.modules.measures import DerridaCoefficient
    ca = ca_cls(k=k, neighborhood="moore", rule_table=rule_table)
    dc = DerridaCoefficient(ca, size=32, n_perturbations=100)
    return dc.compute()


def _build_summary(results: list[dict]) -> dict:
    """compute summary statistics: tier1+ vs tier1- means, by-lambda-bin profile."""
    pos = [r["mu"] for r in results if r["tier1"]]
    neg = [r["mu"] for r in results if not r["tier1"]]
    pos_arr, neg_arr = np.array(pos), np.array(neg)

    lambda_bins = np.linspace(0, 1, 11)
    profile = []
    for i in range(len(lambda_bins) - 1):
        lo, hi = lambda_bins[i], lambda_bins[i + 1]
        in_bin = [r["mu"] for r in results if lo <= r["lambda"] < hi]
        profile.append({
            "lambda_lo": round(lo, 2),
            "lambda_hi": round(hi, 2),
            "n": len(in_bin),
            "mu_mean": round(float(np.mean(in_bin)), 6) if in_bin else None,
            "mu_std": round(float(np.std(in_bin)), 6) if in_bin else None,
        })

    return {
        "tier1_pos_mu_mean": round(float(np.mean(pos_arr)), 6) if len(pos) else None,
        "tier1_pos_mu_std": round(float(np.std(pos_arr)), 6) if len(pos) else None,
        "tier1_neg_mu_mean": round(float(np.mean(neg_arr)), 6) if len(neg) else None,
        "tier1_neg_mu_std": round(float(np.std(neg_arr)), 6) if len(neg) else None,
        "by_lambda_bin": profile,
    }


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """compute derrida mu for ~1000 rules stratified across (lambda, F)."""
    from src.modules.fast_sim import FastOuterTotalisticCA
    from src.modules.rule_params import RuleParameterizer

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    census = json.loads(census_path.read_text())
    parameterizer = RuleParameterizer(k=2, n_neighbors=8)
    rng = np.random.default_rng(42)

    logger.info("sampling ~800 uniform rules across lambda levels")
    n_levels = parameterizer.table_size + 1
    n_per_level = 800 // n_levels
    uniform_tables = _sample_uniform_by_lambda(parameterizer, n_per_level, rng)

    logger.info("sampling ~200 tier-1 rules from census")
    tier1_pairs = _sample_tier1_rules(
        census["rules"], 200, 2, parameterizer.table_shape, rng,
    )

    results = []
    logger.info("computing derrida mu for %d uniform rules", len(uniform_tables))
    for count, table in enumerate(uniform_tables):
        lam = parameterizer.compute_lambda(table)
        f_val = parameterizer.compute_f(table)
        mu = _compute_single(FastOuterTotalisticCA, 2, table)
        results.append({
            "rule_index": None,
            "lambda": round(lam, 6),
            "f": round(f_val, 6),
            "mu": round(mu, 6),
            "tier1": False,
        })
        if (count + 1) % 100 == 0:
            logger.info("  uniform: %d/%d done", count + 1, len(uniform_tables))

    logger.info("computing derrida mu for %d tier-1 rules", len(tier1_pairs))
    for count, (table, record) in enumerate(tier1_pairs):
        lam = parameterizer.compute_lambda(table)
        f_val = parameterizer.compute_f(table)
        mu = _compute_single(FastOuterTotalisticCA, 2, table)
        results.append({
            "rule_index": record["i"],
            "lambda": round(lam, 6),
            "f": round(f_val, 6),
            "mu": round(mu, 6),
            "tier1": True,
        })
        if (count + 1) % 50 == 0:
            logger.info("  tier-1: %d/%d done", count + 1, len(tier1_pairs))

    summary = _build_summary(results)
    output = {
        "n_uniform": len(uniform_tables),
        "n_tier1": len(tier1_pairs),
        "n_total": len(results),
        "summary": summary,
        "rules": results,
    }

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("derrida phase diagram complete: %d rules", len(results))
    logger.info("  tier1+ mu=%.4f, tier1- mu=%.4f",
                summary["tier1_pos_mu_mean"] or 0, summary["tier1_neg_mu_mean"] or 0)
    return output
