"""Conflict values and the branching variable (QCBB eqs. 5-9).

The problem defines its constraints through V (which samples violate which
constraint) and P (which free variables appear in which constraint). With the
node's samples, completed to full assignments, and their counts omega_l,
q = sum omega_l:

    V_kl  = 1 if sample l violates constraint k           (eq. 5)
    nu_k  = (1/q) sum_l V_kl omega_l                      (eq. 6)
    P_ki  = 1 if free variable i appears in constraint k  (eq. 7)
    gamma = nu P                                          (eq. 8)
    branch on argmax_i gamma_i over the candidate set     (eq. 9)

A node without samples uses nu_k = 1. None of this enters a bound or a prune,
so the choice never affects correctness.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qcbb.problem import Problem
from qcbb.sampler import SampleSet

Floats = NDArray[np.float64]


def conflict_values(
    problem: Problem[Any], node: Any, samples: SampleSet | None, threshold: int
) -> Floats:
    """Return gamma of eq. 8 for the free variables of a node.

    Args:
        problem: The problem.
        node: The node.
        samples: The node's samples, or None or an empty set (then nu_k = 1).
        threshold: Threshold T, used by the constraints.

    Returns:
        gamma, one value per free variable, shape (F,).
    """
    p = problem.correspondence(node)
    if samples is None or samples.counts.sum() == 0:
        nu = np.ones(p.shape[0])
    else:
        v = problem.violations(node, problem.complete(node, samples.spins), threshold)
        nu = (v @ samples.counts) / samples.counts.sum()
    gamma: Floats = nu @ p
    return gamma


def branching_variable(problem: Problem[Any], node: Any, gamma: Floats) -> int:
    """Return the branching variable, eq. 9 over the problem's candidate set.

    Args:
        problem: The problem.
        node: The node.
        gamma: Conflict values of the free variables, shape (F,).

    Returns:
        The chosen variable; ties go to the smallest `tie_key`.
    """
    free = problem.free(node)
    pool = problem.candidates(node)
    top = pool[gamma[pool] == gamma[pool].max()]
    return min((int(free[i]) for i in top), key=lambda v: problem.tie_key(node, v))
