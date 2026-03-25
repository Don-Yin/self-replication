"""fast CA simulation using scipy convolution for outer-totalistic rules."""
import numpy as np
from scipy.signal import fftconvolve


class FastOuterTotalisticCA:
    """fast 2D outer-totalistic CA using convolution for neighbor sum."""

    def __init__(self, k: int, neighborhood: str, rule_table: np.ndarray):
        """initialize with rule table and precomputed kernel."""
        self.k = k
        self.rule_table = rule_table.astype(np.int8)
        self.kernel = _make_neighbor_kernel(neighborhood)
        n_neighbors = int(self.kernel.sum())
        self.max_sum = n_neighbors * (k - 1)
        self._flat_lut = self._build_flat_lut()

    def _build_flat_lut(self) -> np.ndarray:
        """flatten rule table to 1D: lut[center * (max_sum+1) + neighbor_sum] = new_state."""
        lut = np.zeros(self.k * (self.max_sum + 1), dtype=np.int8)
        for c in range(self.k):
            for s in range(self.max_sum + 1):
                if s < self.rule_table.shape[1]:
                    lut[c * (self.max_sum + 1) + s] = self.rule_table[c, s]
        return lut

    def run(self, grid: np.ndarray, steps: int, snapshot_interval: int = 64) -> list[np.ndarray]:
        """run simulation using fft convolution for neighbor sums."""
        grid = grid.astype(np.int8)
        snapshots = [grid.copy()]
        kernel = self.kernel.astype(np.float32)
        max_sum_plus_1 = self.max_sum + 1
        lut = self._flat_lut

        for step in range(steps):
            neighbor_sum = _periodic_convolve(grid.astype(np.float32), kernel, grid.shape)
            neighbor_sum = np.clip(neighbor_sum, 0, self.max_sum).astype(np.int32)
            indices = grid.astype(np.int32) * max_sum_plus_1 + neighbor_sum
            grid = lut[indices]

            if (step + 1) % snapshot_interval == 0:
                snapshots.append(grid.copy())

        return snapshots


def _periodic_convolve(grid: np.ndarray, kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """convolve with periodic boundary conditions via padding."""
    h, w = shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(grid, ((ph, ph), (pw, pw)), mode="wrap")
    result = fftconvolve(padded, kernel, mode="same")
    return np.round(result[ph:ph + h, pw:pw + w]).astype(np.int32)


def _make_neighbor_kernel(neighborhood: str) -> np.ndarray:
    """create convolution kernel for neighbors only (center=0)."""
    match neighborhood:
        case "von_neumann":
            k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        case "moore":
            k = np.ones((3, 3), dtype=np.float32)
            k[1, 1] = 0
        case "extended_moore":
            k = np.ones((5, 5), dtype=np.float32)
            k[2, 2] = 0
        case _:
            raise ValueError(f"unknown neighborhood: {neighborhood}")
    return k
