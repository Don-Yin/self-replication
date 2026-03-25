"""self-replication detection pipeline: screening, template matching, causal test."""
import numpy as np
from numba import njit


class ComponentTracker:
    """track connected components and their canonical hashes across snapshots."""

    def __init__(self, k: int):
        """initialize tracker for k-state CA."""
        self.k = k

    def label_components(self, grid: np.ndarray) -> tuple[np.ndarray, int]:
        """label connected components via flood fill on non-zero cells."""
        return _label_components(grid)

    def canonical_hash(self, grid: np.ndarray, label_grid: np.ndarray, label_id: int) -> int:
        """compute rotation/reflection invariant hash of a single component."""
        ys, xs = np.where(label_grid == label_id)
        if len(ys) == 0:
            return 0
        min_y, min_x = ys.min(), xs.min()
        coords = np.column_stack((ys - min_y, xs - min_x))
        states = grid[ys, xs]
        return _canonical_hash(coords, states)


class Stage1Screener:
    """population dynamics screening with multi-indicator OR filter."""

    def screen(self, snapshots: list[np.ndarray], min_component_size: int = 2) -> dict:
        """return screening indicators from a sequence of grid snapshots."""
        component_counts = []
        alive_counts = []
        hash_census: dict[int, int] = {}

        tracker = ComponentTracker(k=2)

        for t, snap in enumerate(snapshots):
            alive_counts.append(int(np.count_nonzero(snap)))
            labels, n_components = tracker.label_components(snap)
            component_counts.append(n_components)
            if t == 0:
                continue
            for lid in range(1, n_components + 1):
                comp_size = int(np.sum(labels == lid))
                if comp_size < min_component_size:
                    continue
                h = tracker.canonical_hash(snap, labels, lid)
                hash_census[h] = hash_census.get(h, 0) + 1

        indicator_a = _monotonic_growth(component_counts[1:], min_increases=3)
        indicator_b = any(v >= 3 for v in hash_census.values())
        indicator_c = _nontrivial_dynamics(alive_counts[1:], snapshots[0].size)

        return {
            "flagged": indicator_a or indicator_b,
            "indicator_a": indicator_a,
            "indicator_b": indicator_b,
            "indicator_c": indicator_c,
            "component_counts": component_counts,
            "alive_counts": alive_counts,
            "top_hashes": sorted(hash_census.items(), key=lambda x: -x[1])[:10],
        }


class Stage2Matcher:
    """template matching for tier 1 and tier 2 self-replication."""

    def match(self, snapshots: list[np.ndarray]) -> dict:
        """detect proliferating patterns across snapshots."""
        tracker = ComponentTracker(k=2)
        hash_timeseries: dict[int, list[int]] = {}

        for t, snap in enumerate(snapshots):
            labels, n_comp = tracker.label_components(snap)
            step_hashes: dict[int, int] = {}
            for lid in range(1, n_comp + 1):
                h = tracker.canonical_hash(snap, labels, lid)
                step_hashes[h] = step_hashes.get(h, 0) + 1
            for h, count in step_hashes.items():
                hash_timeseries.setdefault(h, []).append(count)

        tier1_replicators = []
        for h, counts in hash_timeseries.items():
            if h == 0:
                continue
            increases = sum(1 for i in range(1, len(counts)) if counts[i] > counts[i - 1])
            if increases >= 3:
                tier1_replicators.append({"hash": h, "counts": counts, "increases": increases})

        return {
            "tier1_detected": len(tier1_replicators) > 0,
            "tier1_replicators": tier1_replicators,
            "n_unique_patterns": len(hash_timeseries),
        }


class Stage3CausalTest:
    """causal perturbation test: isolate parent pattern and test if it self-replicates."""

    def __init__(self, ca, n_trials: int = 10, threshold: float = 0.5):
        """initialize with CA simulator, trial count, and confirmation threshold."""
        self.ca = ca
        self.n_trials = n_trials
        self.threshold = threshold

    def test(self, grid: np.ndarray, steps: int) -> dict:
        """isolate a replicating pattern, test causal dependence via cell deletion."""
        base_snaps = self.ca.run(grid, steps, snapshot_interval=64)
        matcher = Stage2Matcher()
        base_result = matcher.match(base_snaps)

        if not base_result["tier1_detected"]:
            return {"tier3_detected": False, "reason": "no tier1 in base run"}

        seed = self._extract_seed(base_snaps, base_result, grid.shape)
        if seed is None:
            return {"tier3_detected": False, "reason": "could not isolate seed pattern"}

        seed_snaps = self.ca.run(seed, steps, snapshot_interval=64)
        seed_result = matcher.match(seed_snaps)
        if not seed_result["tier1_detected"]:
            return {"tier3_detected": False, "reason": "seed does not replicate in isolation"}

        alive_ys, alive_xs = np.where(seed > 0)
        n_cells = len(alive_ys)
        prevented = 0
        trials_run = min(self.n_trials, n_cells)

        for trial in range(trials_run):
            perturbed = seed.copy()
            perturbed[alive_ys[trial % n_cells], alive_xs[trial % n_cells]] = 0
            pert_snaps = self.ca.run(perturbed, steps, snapshot_interval=64)
            pert_result = matcher.match(pert_snaps)
            if not pert_result["tier1_detected"]:
                prevented += 1

        prevention_rate = prevented / max(trials_run, 1)
        return {
            "tier3_detected": prevention_rate >= self.threshold,
            "prevention_rate": round(prevention_rate, 4),
            "trials": trials_run,
            "seed_cells": n_cells,
            "seed_replicates_alone": True,
        }

    def _extract_seed(self, snaps: list, result: dict, grid_shape: tuple):
        """extract a replicating component from early snapshot onto empty grid."""
        tracker = ComponentTracker(k=2)
        rep_hashes = {r["hash"] for r in result["tier1_replicators"]}

        for snap in snaps[1:3]:
            labels, n_comp = tracker.label_components(snap)
            for lid in range(1, n_comp + 1):
                h = tracker.canonical_hash(snap, labels, lid)
                if h not in rep_hashes:
                    continue
                size = int(np.sum(labels == lid))
                if size < 2 or size > 100:
                    continue
                ys, xs = np.where(labels == lid)
                seed = np.zeros(grid_shape, dtype=np.int8)
                cy, cx = grid_shape[0] // 2, grid_shape[1] // 2
                oy, ox = ys.mean().astype(int), xs.mean().astype(int)
                for y, x in zip(ys, xs):
                    ny, nx = y - oy + cy, x - ox + cx
                    if 0 <= ny < grid_shape[0] and 0 <= nx < grid_shape[1]:
                        seed[ny, nx] = snap[y, x]
                return seed
        return None


def _monotonic_growth(counts: list[int], min_increases: int) -> bool:
    """check if counts show at least min_increases strict increases."""
    increases = sum(1 for i in range(1, len(counts)) if counts[i] > counts[i - 1])
    return increases >= min_increases


def _nontrivial_dynamics(alive_counts: list[int], grid_size: int) -> bool:
    """check alive counts neither die to zero nor saturate."""
    if not alive_counts:
        return False
    final = alive_counts[-1]
    return 0 < final < grid_size * 0.8


@njit(cache=True)
def _label_components(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """flood-fill connected component labeling for non-zero cells."""
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    stack = np.empty((h * w, 2), dtype=np.int32)

    for y in range(h):
        for x in range(w):
            if grid[y, x] > 0 and labels[y, x] == 0:
                current_label += 1
                top = 0
                stack[top, 0] = y
                stack[top, 1] = x
                labels[y, x] = current_label
                while top >= 0:
                    cy = stack[top, 0]
                    cx = stack[top, 1]
                    top -= 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny = cy + dy
                            nx = cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if grid[ny, nx] > 0 and labels[ny, nx] == 0:
                                    labels[ny, nx] = current_label
                                    top += 1
                                    stack[top, 0] = ny
                                    stack[top, 1] = nx
    return labels, current_label


def _canonical_hash(coords: np.ndarray, states: np.ndarray) -> int:
    """compute rotation/reflection invariant hash from relative coordinates and states."""
    transforms = [
        coords,
        np.column_stack((-coords[:, 1], coords[:, 0])),
        np.column_stack((-coords[:, 0], -coords[:, 1])),
        np.column_stack((coords[:, 1], -coords[:, 0])),
        np.column_stack((coords[:, 0], -coords[:, 1])),
        np.column_stack((-coords[:, 0], coords[:, 1])),
        np.column_stack((coords[:, 1], coords[:, 0])),
        np.column_stack((-coords[:, 1], -coords[:, 0])),
    ]
    min_hash = None
    for t in transforms:
        t_shifted = t - t.min(axis=0)
        order = np.lexsort((t_shifted[:, 1], t_shifted[:, 0]))
        key = tuple((*t_shifted[i], states[i]) for i in order)
        h = hash(key)
        if min_hash is None or h < min_hash:
            min_hash = h
    return min_hash
