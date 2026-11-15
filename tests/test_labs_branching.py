"""Conflict values for LABS: lag budgets, matrix P, outermost pair and ties."""

import numpy as np
from helpers import WorstSampler

from qcbb import branching_variable, conflict_values
from qcbb.labs import LabsProblem, NodeHamiltonian


def test_outermost_pair_is_eq9_on_prefix_suffix_nodes() -> None:
    """With nu = 1, eq. 9 over all free spins of a prefix-suffix node picks an end."""
    for n in range(3, 41):
        for a in range(n):
            for r in range(n - a):
                if a + r == 0:
                    continue
                node = np.zeros(n, dtype=np.int8)
                node[:a], node[n - r :] = 1, -1
                chosen = []
                for mode in ("all", "outermost"):
                    problem = LabsProblem(n, candidates=mode)  # type: ignore[arg-type]
                    gamma = conflict_values(problem, node, None, 0)
                    chosen.append(branching_variable(problem, node, gamma))
                assert chosen[0] == chosen[1]
                if a == r:
                    p = problem.correspondence(node)
                    assert np.array_equal(p[:, 0], p[:, -1])


def test_correspondence_violations_candidates_and_tie_rule() -> None:
    spins = np.zeros(8, dtype=np.int8)
    spins[[0, 1, 7]] = 1
    problem = LabsProblem(8, candidates="outermost")
    p = problem.correspondence(spins)
    assert p.shape == (7, 5)
    assert p[0].tolist() == [1, 0, 0, 0, 1]
    assert problem.candidates(spins).tolist() == [0, 4]
    assert problem.tie_key(spins, 5) < problem.tie_key(spins, 2)  # 3 fixed: right
    samples = WorstSampler().sample([NodeHamiltonian(spins)], shots=1)[0]
    x = problem.complete(spins, samples.spins)
    v = problem.violations(spins, x, threshold=20)
    assert v.shape == (7, 1) and v.dtype == np.bool_
