"""Ranges R_k, the pair congruence bound, the lag budgets and propagation.

Brute-force checks of the ranges, the pair congruence bound and the
propagation theorem against enumeration (evidence, not proofs).
"""

import numpy as np
from helpers import completions, random_node

from qcbb.labs import bound


def test_lag_ranges_equal_the_enumerated_autocorrelation_values() -> None:
    rng = np.random.default_rng(0)
    for _ in range(600):
        node = random_node(int(rng.integers(2, 14)), rng)
        _, c = completions(node)
        for k, (lo, hi, g) in enumerate(bound.lag_ranges(node), start=1):
            reach = set(range(lo, hi + 1, g)) if g else {lo}
            assert reach == set(c[:, k - 1].tolist())


def test_pair_congruence_bound_is_valid_and_exact_at_leaves() -> None:
    rng = np.random.default_rng(1)
    for _ in range(600):
        node = random_node(int(rng.integers(2, 14)), rng)
        n = node.size
        x, c = completions(node)
        lb, stats = bound.statistics(node)
        for k in range(1, (n + 1) // 2):
            assert stats[k, 4] <= (c[:, k - 1] ** 2 + c[:, n - k - 1] ** 2).min()
        e = (c * c).sum(axis=1)
        assert lb <= e.min()
        assert bound.lower_bound(x[0]) == e[0]


def test_lag_budgets_and_propagation_keep_every_completion_below_t() -> None:
    rng = np.random.default_rng(2)
    forced = 0
    for _ in range(1500):
        node = random_node(int(rng.integers(2, 14)), rng)
        x, c = completions(node)
        e = (c * c).sum(axis=1)
        threshold = int(e.min()) + int(rng.choice([-4, 0, 0, 4, 8, 16, 40]))
        good = e <= threshold
        assert np.all(c[good] ** 2 <= bound.budgets(node, threshold))
        child = bound.propagate(node, threshold)
        if child is None:
            assert not good.any()
            continue
        assert np.all(child[node != 0] == node[node != 0])
        assert np.all((child == 0) | (x[good] == child))
        forced += int(np.count_nonzero(child) - np.count_nonzero(node))
    assert forced > 0
