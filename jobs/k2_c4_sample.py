"""experiment B: sample rotationally-symmetric (C4) k=2 Moore rules."""
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run(artifact: Path, n_rules: int = 10000, **kwargs) -> dict:
    """sample C4-symmetric k=2 Moore rules and test for self-replication."""
    from src.modules.general_sim import GeneralBinaryCA
    from src.modules.simulator import make_random_ic
    from src.modules.detector import Stage1Screener, Stage2Matcher

    screener = Stage1Screener()
    matcher = Stage2Matcher()
    rng = np.random.default_rng(42)

    orbits = _compute_c4_orbits()
    n_orbits = len(orbits)
    logger.info("  C4 orbits: %d (rule space: 2^%d)", n_orbits, n_orbits)

    rule_data = []
    n_flagged = 0
    n_tier1 = 0
    t_start = time.time()

    for i in range(n_rules):
        rule_bits = rng.integers(0, 2, size=n_orbits)
        rule_table = _bits_to_rule_table(rule_bits, orbits)
        lam = float(np.sum(rule_table) / 512)

        flagged = False
        tier1 = False
        best_snaps = None

        for density in (0.15, 0.35):
            ca = GeneralBinaryCA(rule_table)
            ic = make_random_ic(64, 2, density, np.random.default_rng(i * 37 + int(density * 100)))
            snaps = ca.run(ic, 256, snapshot_interval=64)
            screen = screener.screen(snaps)
            if screen["flagged"]:
                flagged = True
                best_snaps = snaps
                break

        if flagged and best_snaps is not None:
            n_flagged += 1
            match_result = matcher.match(best_snaps)
            tier1 = match_result["tier1_detected"]
            if tier1:
                n_tier1 += 1

        rule_data.append({"l": round(lam, 4), "s1": flagged, "t1": tier1})

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            logger.info("    %d/%d (%.1f/s) | flagged: %d | tier1: %d",
                        i + 1, n_rules, rate, n_flagged, n_tier1)

    elapsed = time.time() - t_start
    results = {
        "rule_class": "C4_rotationally_symmetric",
        "k": 2,
        "neighborhood": "moore",
        "n_orbits": n_orbits,
        "n_rules": n_rules,
        "n_flagged": n_flagged,
        "n_tier1": n_tier1,
        "tier1_rate": round(n_tier1 / max(n_rules, 1), 6),
        "elapsed_sec": round(elapsed, 1),
        "rules": rule_data,
    }

    artifact.write_text(json.dumps(results, indent=2, default=str))
    logger.info("  C4 sample: %d rules in %.0f min | tier1: %d (%.2f%%)",
                n_rules, elapsed / 60, n_tier1, 100 * n_tier1 / max(n_rules, 1))
    return results


def _compute_c4_orbits() -> list[list[int]]:
    """compute the C4 orbits of the 512 binary Moore neighbourhood configs."""
    import numpy as np

    def rotate_config(bits: np.ndarray) -> np.ndarray:
        """rotate a 3x3 grid 90 degrees clockwise."""
        return np.rot90(bits.reshape(3, 3), -1).ravel()

    seen = set()
    orbits = []
    for config_id in range(512):
        if config_id in seen:
            continue
        bits = np.array([(config_id >> b) & 1 for b in range(9)], dtype=np.int8)
        orbit = set()
        current = bits.copy()
        for _ in range(4):
            idx = sum(int(current[b]) << b for b in range(9))
            orbit.add(idx)
            current = rotate_config(current)
        orbits.append(sorted(orbit))
        seen.update(orbit)
    return orbits


def _bits_to_rule_table(bits: np.ndarray, orbits: list[list[int]]) -> np.ndarray:
    """convert orbit bits to a full 512-entry rule table."""
    import numpy as np
    table = np.zeros(512, dtype=np.int8)
    for bit_val, orbit in zip(bits, orbits):
        for config_id in orbit:
            table[config_id] = int(bit_val)
    return table
