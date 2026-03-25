"""compute lambda, F parameter, and rule sampling for outer-totalistic CA."""
import numpy as np


class RuleParameterizer:
    """compute and control rule-table statistics for outer-totalistic CA."""

    def __init__(self, k: int, n_neighbors: int):
        """initialize for k-state CA with n_neighbors neighbors (excluding center)."""
        self.k = k
        self.n_neighbors = n_neighbors
        self.max_sum = n_neighbors * (k - 1)
        self.table_shape = (k, self.max_sum + 1)
        self.table_size = k * (self.max_sum + 1)

    def compute_lambda(self, rule_table: np.ndarray) -> float:
        """fraction of non-quiescent entries in the rule table."""
        return float(np.count_nonzero(rule_table) / self.table_size)

    def compute_f(self, rule_table: np.ndarray) -> float:
        """adapted sakai-kanno F: weighted quiescent-background-breaking score.

        F_OT = sum_s w(s) * I[T(0, s) != 0] / sum_s w(s)
        where w(s) = (1 - s / S_max)^2, s is the neighbour sum,
        and I[T(0, s) != 0] indicates the rule maps a quiescent center
        with neighbour sum s to a non-quiescent state.
        high F means the rule aggressively breaks quiescent regions.
        """
        score = 0.0
        total_weight = 0.0
        for s in range(self.max_sum + 1):
            w = (1.0 - s / max(self.max_sum, 1)) ** 2
            total_weight += w
            if rule_table[0, s] != 0:
                score += w
        return score / max(total_weight, 1e-12)

    def enumerate_all(self) -> np.ndarray:
        """enumerate all possible outer-totalistic rules (only feasible for small spaces)."""
        n_entries = self.table_size
        n_rules = self.k ** n_entries
        assert n_rules <= 300_000, f"rule space too large: {self.k}^{n_entries} = {n_rules}"
        rules = np.zeros((n_rules, *self.table_shape), dtype=np.int8)
        for i in range(n_rules):
            val = i
            flat = np.zeros(n_entries, dtype=np.int8)
            for j in range(n_entries):
                flat[j] = val % self.k
                val //= self.k
            rules[i] = flat.reshape(self.table_shape)
        return rules

    def sample_at_lambda(
        self, target_lambda: float, n_rules: int, rng: np.random.Generator
    ) -> np.ndarray:
        """sample random rules conditioned on approximate lambda."""
        n_nonzero = max(1, int(round(target_lambda * self.table_size)))
        rules = np.zeros((n_rules, *self.table_shape), dtype=np.int8)
        for i in range(n_rules):
            flat = np.zeros(self.table_size, dtype=np.int8)
            positions = rng.choice(self.table_size, size=n_nonzero, replace=False)
            flat[positions] = rng.integers(1, self.k, size=n_nonzero).astype(np.int8)
            rules[i] = flat.reshape(self.table_shape)
        return rules

    def sample_at_lambda_f(
        self,
        target_lambda: float,
        target_f: float,
        n_rules: int,
        rng: np.random.Generator,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """sample rules at controlled (lambda, F) via rejection sampling."""
        collected = []
        tolerance = 0.1
        for _ in range(max_attempts * n_rules):
            if len(collected) >= n_rules:
                break
            rule = self.sample_at_lambda(target_lambda, 1, rng)[0]
            f_val = self.compute_f(rule)
            if abs(f_val - target_f) < tolerance:
                collected.append(rule)
        if len(collected) < n_rules:
            extra = self.sample_at_lambda(target_lambda, n_rules - len(collected), rng)
            collected.extend(extra)
        return np.array(collected[:n_rules])


def life_like_rule(birth: set[int], survival: set[int]) -> np.ndarray:
    """create a k=2 moore outer-totalistic rule from birth/survival sets."""
    table = np.zeros((2, 9), dtype=np.int8)
    for s in range(9):
        table[0, s] = 1 if s in birth else 0
        table[1, s] = 1 if s in survival else 0
    return table


GAME_OF_LIFE = life_like_rule(birth={3}, survival={2, 3})
HIGH_LIFE = life_like_rule(birth={3, 6}, survival={2, 3})
