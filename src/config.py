"""paths, constants, hyperparameters for the self-replication phase diagram."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class SweepConfig:
    """parameter sweep specification."""
    densities: tuple[float, ...] = (0.15, 0.35)
    ics_per_density: int = 4


NEIGHBORHOOD_OFFSETS: dict[str, list[tuple[int, int]]] = {
    "von_neumann": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    "moore": [
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ],
    "extended_moore": [
        (dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
    ],
}


def ensure_dirs() -> None:
    """create all output directories if they do not exist."""
    for d in (DATA, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
