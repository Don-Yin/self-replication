"""download ASAL open-endedness scores and compute correlation with self-replication."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ASAL_URL_PRIMARY = "https://pub.sakana.ai/asal/data/sweep_gol.npz"
ASAL_URL_FALLBACK = "https://raw.githubusercontent.com/SakanaAI/asal/main/data/sweep_gol.npz"
GOL_TABLE_FLAT = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def run(artifact: Path, census_path: Path = None, **kwargs) -> dict:
    """correlate ASAL open-endedness scores with tier-1 self-replication labels."""
    import numpy as np

    if census_path is None:
        census_path = Path("results/k2-moore-census.json")

    gol_index = _compute_gol_index()
    cache_path = Path("data/asal-sweep-gol.npz")
    scores = _load_asal_scores(cache_path)

    if scores is None:
        output = _failure_result(gol_index)
        artifact.write_text(json.dumps(output, indent=2, default=str))
        logger.warning("  asal comparison: download failed, wrote placeholder result")
        return output

    data = json.loads(census_path.read_text())
    rules_data = data["rules"]
    n_rules = len(rules_data)

    if len(scores) != n_rules:
        logger.warning("  asal score count %d != census rule count %d", len(scores), n_rules)

    n_usable = min(len(scores), n_rules)
    labels = np.array([rules_data[i]["t1"] for i in range(n_usable)], dtype=bool)
    asal = scores[:n_usable]

    encoding_verified = _verify_encoding(scores, gol_index)
    output = _compute_correlations(asal, labels, gol_index, encoding_verified, n_usable)

    artifact.write_text(json.dumps(output, indent=2, default=str))
    logger.info("  asal comparison: spearman rho=%.3f, point-biserial r=%.3f",
                output["spearman_rho"], output["point_biserial_r"])
    return output


def _compute_gol_index() -> int:
    """compute the integer index of Game of Life in our k=2 Moore encoding."""
    index = 0
    for j, val in enumerate(GOL_TABLE_FLAT):
        index += val * (2 ** j)
    return index


def _load_asal_scores(cache_path: Path) -> 'np.ndarray | None':
    """download ASAL scores, caching to disk; return None on failure."""
    import numpy as np

    if cache_path.exists():
        npz = np.load(str(cache_path))
        key = list(npz.files)[0]
        return npz[key].ravel()

    downloaded = _download_file(ASAL_URL_PRIMARY, cache_path)
    if not downloaded:
        downloaded = _download_file(ASAL_URL_FALLBACK, cache_path)
    if not downloaded:
        return None

    npz = np.load(str(cache_path))
    key = list(npz.files)[0]
    return npz[key].ravel()


def _download_file(url: str, dest: Path) -> bool:
    """download url to dest via curl subprocess; return True on success."""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  downloading %s", url)
    result = subprocess.run(
        ["curl", "-fsSL", "-o", str(dest), url],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        logger.warning("  curl failed for %s: %s", url, result.stderr.decode().strip())
        return False
    return dest.exists()


def _verify_encoding(scores: 'np.ndarray', gol_index: int) -> bool:
    """check that GoL index has a non-trivial score, suggesting correct alignment."""
    if gol_index >= len(scores):
        return False
    return float(scores[gol_index]) > 0


def _compute_correlations(
    asal: 'np.ndarray', labels: 'np.ndarray', gol_index: int,
    encoding_verified: bool, n_rules: int,
) -> dict:
    """compute Spearman and point-biserial correlations."""
    import numpy as np
    from scipy.stats import spearmanr, pointbiserialr

    rho, sp = spearmanr(asal, labels.astype(float))
    rpb, pp = pointbiserialr(labels.astype(int), asal)

    return {
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(f"{sp:.2e}"),
        "point_biserial_r": round(float(rpb), 4),
        "point_biserial_p": float(f"{pp:.2e}"),
        "n_rules": n_rules,
        "encoding_verified": encoding_verified,
        "gol_index_ours": gol_index,
        "gol_index_theirs": None,
    }


def _failure_result(gol_index: int) -> dict:
    """placeholder result when ASAL data cannot be downloaded."""
    return {
        "spearman_rho": None,
        "spearman_p": None,
        "point_biserial_r": None,
        "point_biserial_p": None,
        "n_rules": 0,
        "encoding_verified": False,
        "gol_index_ours": gol_index,
        "gol_index_theirs": None,
        "download_failed": True,
    }
