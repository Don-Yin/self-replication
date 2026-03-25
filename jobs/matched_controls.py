"""lambda/F-matched control analysis for boundary measures and O-information."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(artifact: Path, boundary_path: Path = None, oinfo_path: Path = None, **kwargs) -> dict:
    """re-test mass-balance and O-info with lambda/F-matched negative controls."""
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

    matched_bm = _match_controls(pos_bm, neg_bm)
    matched_oi = _match_controls(pos_oi, neg_oi)

    results = {"matched_boundary": {}, "matched_oinfo": {}, "permutation": {}}

    for key, pos, neg_matched in [
        ("mass_balance", pos_bm, matched_bm),
        ("spatial_entropy", pos_bm, matched_bm),
    ]:
        p_vals = [r[key] for r in pos]
        n_vals = [r[key] for r in neg_matched]
        stat, p = mannwhitneyu(p_vals, n_vals, alternative="two-sided")
        d = _cohens_d(p_vals, n_vals)
        results["matched_boundary"][key] = {
            "pos_mean": round(float(np.mean(p_vals)), 4),
            "neg_matched_mean": round(float(np.mean(n_vals)), 4),
            "p_matched": float(f"{p:.2e}"),
            "cohens_d": round(float(d), 3),
            "n_matched": len(neg_matched),
        }

    for key in ["oinfo", "tc", "dtc"]:
        p_vals = [r[key] for r in pos_oi]
        n_vals = [r[key] for r in matched_oi]
        stat, p = mannwhitneyu(p_vals, n_vals, alternative="two-sided")
        d = _cohens_d(p_vals, n_vals)
        results["matched_oinfo"][key] = {
            "pos_mean": round(float(np.mean(p_vals)), 4),
            "neg_matched_mean": round(float(np.mean(n_vals)), 4),
            "p_matched": float(f"{p:.2e}"),
            "cohens_d": round(float(d), 3),
            "n_matched": len(matched_oi),
        }

    perm_results = _permutation_test(pos_bm, neg_bm, "mass_balance", n_perms=10000)
    results["permutation"]["mass_balance"] = perm_results
    perm_results_oi = _permutation_test(pos_oi, neg_oi, "oinfo", n_perms=10000)
    results["permutation"]["oinfo"] = perm_results_oi

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  matched controls: mass_balance p=%s (d=%.2f), oinfo p=%s (d=%.2f)",
                results["matched_boundary"]["mass_balance"]["p_matched"],
                results["matched_boundary"]["mass_balance"]["cohens_d"],
                results["matched_oinfo"]["oinfo"]["p_matched"],
                results["matched_oinfo"]["oinfo"]["cohens_d"])
    return results


def _match_controls(pos_rules: list, neg_rules: list) -> list:
    """for each positive rule, find nearest negative rule in (lambda, F) space."""
    import numpy as np
    neg_lf = np.array([(r["lambda"], r.get("f_param", r.get("f", 0))) for r in neg_rules])
    matched = []
    used = set()
    for p in pos_rules:
        p_lf = np.array([p["lambda"], p.get("f_param", p.get("f", 0))])
        dists = np.sqrt(np.sum((neg_lf - p_lf) ** 2, axis=1))
        for idx in np.argsort(dists):
            if idx not in used:
                used.add(idx)
                matched.append(neg_rules[idx])
                break
    return matched


def _cohens_d(a: list, b: list) -> float:
    """compute Cohen's d effect size."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / max(pooled_std, 1e-12)


def _permutation_test(pos: list, neg: list, key: str, n_perms: int = 10000) -> dict:
    """permutation test shuffling labels within matched pairs."""
    import numpy as np
    all_vals = [r[key] for r in pos] + [r[key] for r in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    observed = abs(np.mean([r[key] for r in pos]) - np.mean([r[key] for r in neg]))
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(labels)
        vals = np.array(all_vals)
        diff = abs(vals[perm == 1].mean() - vals[perm == 0].mean())
        if diff >= observed:
            count += 1
    return {"observed_diff": round(float(observed), 4), "p_perm": round(count / n_perms, 6)}
