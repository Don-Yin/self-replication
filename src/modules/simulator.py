"""outer-totalistic cellular automata simulator for arbitrary k and neighborhood."""
import numpy as np
from numba import njit

from src.config import NEIGHBORHOOD_OFFSETS


class OuterTotalisticCA:
    """2D outer-totalistic CA with periodic boundaries."""

    def __init__(self, k: int, neighborhood: str, rule_table: np.ndarray):
        """initialize with k states, neighborhood type, and rule table."""
        self.k = k
        self.neighborhood = neighborhood
        offsets = NEIGHBORHOOD_OFFSETS[neighborhood]
        self.offsets = np.array(offsets, dtype=np.int32)
        self.n_neighbors = len(offsets) - 1
        self.rule_table = rule_table
        self._validate_rule_table()

    def _validate_rule_table(self):
        """check rule table has correct shape: (k, max_neighbor_sum + 1)."""
        max_sum = self.n_neighbors * (self.k - 1)
        expected_shape = (self.k, max_sum + 1)
        assert self.rule_table.shape == expected_shape, (
            f"rule table shape {self.rule_table.shape} != expected {expected_shape}"
        )

    def run(self, grid: np.ndarray, steps: int) -> list[np.ndarray]:
        """run simulation, return list of grids at checkpoint intervals."""
        return _run_simulation(
            grid.astype(np.int8),
            self.rule_table.astype(np.int8),
            self.offsets,
            self.k,
            steps,
        )


@njit(cache=True)
def _run_simulation(
    grid: np.ndarray,
    rule_table: np.ndarray,
    offsets: np.ndarray,
    k: int,
    steps: int,
) -> list[np.ndarray]:
    """numba-accelerated CA simulation with periodic boundaries."""
    h, w = grid.shape
    snapshots = [grid.copy()]
    new_grid = np.empty_like(grid)

    for step in range(steps):
        for y in range(h):
            for x in range(w):
                center = grid[y, x]
                neighbor_sum = 0
                for oi in range(offsets.shape[0]):
                    dy, dx = offsets[oi, 0], offsets[oi, 1]
                    if dy == 0 and dx == 0:
                        continue
                    ny = (y + dy) % h
                    nx = (x + dx) % w
                    neighbor_sum += grid[ny, nx]
                new_grid[y, x] = rule_table[center, neighbor_sum]
            # end x
        # end y
        grid = new_grid.copy()

        if (step + 1) % 64 == 0:
            snapshots.append(grid.copy())

    return snapshots


def make_random_ic(size: int, k: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """generate a random initial condition at given density."""
    grid = np.zeros((size, size), dtype=np.int8)
    n_alive = int(size * size * density)
    positions = rng.choice(size * size, size=n_alive, replace=False)
    states = rng.integers(1, k, size=n_alive).astype(np.int8) if k > 2 else np.ones(n_alive, dtype=np.int8)
    grid.ravel()[positions] = states
    return grid
