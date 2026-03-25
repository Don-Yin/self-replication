"""general (non-totalistic) binary CA simulator using numba."""
import numpy as np
from numba import njit


class GeneralBinaryCA:
    """2D binary CA with arbitrary 512-entry Moore neighbourhood lookup table."""

    def __init__(self, rule_table: np.ndarray):
        """initialize with 512-entry lookup table mapping neighbourhood configs to output."""
        assert rule_table.shape == (512,), f"expected shape (512,), got {rule_table.shape}"
        self.rule_table = rule_table.astype(np.int8)
        self.k = 2

    def run(self, grid: np.ndarray, steps: int, snapshot_interval: int = 64) -> list[np.ndarray]:
        """run simulation with periodic boundaries."""
        return _run_general(grid.astype(np.int8), self.rule_table, steps, snapshot_interval)


@njit(cache=True)
def _run_general(grid: np.ndarray, lut: np.ndarray, steps: int, snap_interval: int) -> list[np.ndarray]:
    """numba-accelerated general binary CA simulation."""
    h, w = grid.shape
    snapshots = [grid.copy()]
    new_grid = np.empty_like(grid)

    for step in range(steps):
        for y in range(h):
            for x in range(w):
                config = 0
                bit = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny = (y + dy) % h
                        nx = (x + dx) % w
                        config |= grid[ny, nx] << bit
                        bit += 1
                new_grid[y, x] = lut[config]
        grid = new_grid.copy()

        if (step + 1) % snap_interval == 0:
            snapshots.append(grid.copy())

    return snapshots
