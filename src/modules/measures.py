"""information-theoretic derived measures for CA rules."""
import numpy as np
from numba import njit


class DerridaCoefficient:
    """compute the derrida parameter mu for a CA rule via perturbation spreading."""

    def __init__(self, ca, size: int = 32, t_steps: int = 10, n_perturbations: int = 100):
        """initialize with a CA and simulation parameters."""
        self.ca = ca
        self.size = size
        self.t_steps = t_steps
        self.n_perturbations = n_perturbations

    def compute(self) -> float:
        """return mu: mean(delta_{t+1}/delta_t) averaged over perturbations and time steps."""
        rng = np.random.default_rng(42)
        ic = _random_binary_grid(self.size, rng)
        base_snaps = self.ca.run(ic, self.t_steps, snapshot_interval=1)
        ratios = []
        for _ in range(self.n_perturbations):
            perturbed = ic.copy()
            y, x = rng.integers(0, self.size), rng.integers(0, self.size)
            perturbed[y, x] = (perturbed[y, x] + 1) % self.ca.k
            pert_snaps = self.ca.run(perturbed, self.t_steps, snapshot_interval=1)
            _collect_ratios(base_snaps, pert_snaps, self.size, ratios)
        return float(np.mean(ratios)) if ratios else 0.0


def _random_binary_grid(size: int, rng: np.random.Generator) -> np.ndarray:
    """generate a random binary grid at 50% density."""
    return rng.integers(0, 2, size=(size, size), dtype=np.int8)


def _collect_ratios(
    base_snaps: list[np.ndarray],
    pert_snaps: list[np.ndarray],
    size: int,
    ratios: list[float],
) -> None:
    """append delta_{t+1}/delta_t ratios where delta_t > 0."""
    n_cells = size * size
    deltas = [np.sum(base_snaps[t] != pert_snaps[t]) / n_cells for t in range(len(base_snaps))]
    for t in range(len(deltas) - 1):
        if deltas[t] > 0:
            ratios.append(deltas[t + 1] / deltas[t])


class MassBalance:
    """compute how close a rule is to being number-conserving."""

    def compute(self, ca, grid: np.ndarray, steps: int = 64) -> float:
        """average absolute mass change per step, normalized by grid size."""
        snapshots = ca.run(grid, steps)
        masses = [float(np.sum(s)) for s in snapshots]
        changes = [abs(masses[i + 1] - masses[i]) for i in range(len(masses) - 1)]
        if not changes:
            return 0.0
        return float(np.mean(changes) / grid.size)


class PatternPersistence:
    """fraction of initial conditions that produce long-lived structures."""

    def compute(self, ca, size: int, k: int, n_trials: int = 16, steps: int = 256) -> float:
        """run n_trials random ICs and count how many survive to final step."""
        from src.modules.simulator import make_random_ic
        survived = 0
        for i in range(n_trials):
            ic = make_random_ic(size, k, 0.15, np.random.default_rng(i * 777))
            snapshots = ca.run(ic, steps)
            final_alive = np.count_nonzero(snapshots[-1])
            if 0 < final_alive < size * size * 0.8:
                survived += 1
        return float(survived / n_trials)


@njit(cache=True)
def local_entropy_grid(grid: np.ndarray, k: int, radius: int = 1) -> np.ndarray:
    """compute local entropy at each cell over a (2r+1)x(2r+1) patch."""
    h, w = grid.shape
    result = np.zeros((h, w), dtype=np.float64)
    patch_size = (2 * radius + 1) ** 2

    for y in range(h):
        for x in range(w):
            counts = np.zeros(k, dtype=np.int32)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = (y + dy) % h
                    nx = (x + dx) % w
                    counts[grid[ny, nx]] += 1
            entropy = 0.0
            for s in range(k):
                if counts[s] > 0:
                    p = counts[s] / patch_size
                    entropy -= p * np.log2(p)
            result[y, x] = entropy
    return result


def spatial_entropy(grid: np.ndarray, k: int) -> float:
    """mean local entropy across the grid."""
    eg = local_entropy_grid(grid, k, radius=1)
    return float(np.mean(eg))


class OInformation:
    """O-information estimated from ensemble of spatial patches across the grid."""

    def compute(self, snapshots: list[np.ndarray], k: int, radius: int = 1) -> dict:
        """estimate O-information from patch statistics across multiple snapshots."""
        n = (2 * radius + 1) ** 2
        all_patches = []
        for grid in snapshots:
            patches = _extract_patches(grid, radius)
            all_patches.append(patches)
        patches = np.concatenate(all_patches, axis=0)

        marginal_entropies = np.zeros(n)
        for i in range(n):
            counts = np.bincount(patches[:, i], minlength=k)
            p = counts / counts.sum()
            marginal_entropies[i] = -np.sum(p[p > 0] * np.log2(p[p > 0]))

        joint_ids = _pack_patch_ids(patches, k, n)
        unique, counts = np.unique(joint_ids, return_counts=True)
        p_joint = counts / counts.sum()
        h_joint = -np.sum(p_joint * np.log2(p_joint))

        leave_one_out_entropies = np.zeros(n)
        for i in range(n):
            cols = [j for j in range(n) if j != i]
            sub = patches[:, cols]
            sub_ids = _pack_patch_ids(sub, k, n - 1)
            unique_s, counts_s = np.unique(sub_ids, return_counts=True)
            p_s = counts_s / counts_s.sum()
            leave_one_out_entropies[i] = -np.sum(p_s * np.log2(p_s))

        tc = float(np.sum(marginal_entropies) - h_joint)
        dtc = float(np.sum(leave_one_out_entropies) - (n - 1) * h_joint)
        oinfo = tc - dtc

        return {
            "oinfo": round(oinfo, 6),
            "tc": round(tc, 6),
            "dtc": round(dtc, 6),
            "h_joint": round(h_joint, 6),
            "synergy_dominated": oinfo < 0,
        }


def _extract_patches(grid: np.ndarray, radius: int) -> np.ndarray:
    """extract all (2r+1)x(2r+1) patches from grid as rows of a matrix."""
    h, w = grid.shape
    n = (2 * radius + 1) ** 2
    padded = np.pad(grid, radius, mode="wrap")
    patches = np.empty((h * w, n), dtype=np.int32)
    idx = 0
    for y in range(h):
        for x in range(w):
            patch = padded[y:y + 2 * radius + 1, x:x + 2 * radius + 1].ravel()
            patches[idx] = patch
            idx += 1
    return patches


def _pack_patch_ids(patches: np.ndarray, k: int, n: int) -> np.ndarray:
    """encode each patch row as a single integer for fast counting."""
    ids = np.zeros(patches.shape[0], dtype=np.int64)
    for i in range(n):
        ids = ids * k + patches[:, i]
    return ids
