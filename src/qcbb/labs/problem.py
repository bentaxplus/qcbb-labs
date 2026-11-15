"""LABS as a QCBB problem.

A node is a partial sequence (0 = free spin). The conditions of the
correctness theorem come from `bound` (the pair congruence bound and the
budget propagation), `energy.lattice_step` (the lattice) and `symmetry`
(the class roots and the reversal rule); the children of a node fix one
spin. The constraints for the conflict values are the lag budgets (paper,
Sec. 3.2). The node loop itself is `qcbb.solve`.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qcbb.labs import bound, symmetry
from qcbb.labs.energy import autocorrelations, energy, lattice_step
from qcbb.labs.hamiltonian import NodeHamiltonian
from qcbb.problem import Ints, Spins

Candidates = Literal["all", "outermost"]


class LabsProblem:
    """Low autocorrelation binary sequences of length n.

    Args:
        n: Sequence length, n >= 2.
        boundary: Number m_0 of fixed spins at each end of a class root.
        candidates: "outermost" applies eq. 9 to the leftmost and rightmost
            free spin; "all" to every free spin. Both satisfy condition (v).
    """

    def __init__(
        self, n: int, boundary: int = 7, candidates: Candidates = "outermost"
    ) -> None:
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self.lattice_step = lattice_step(n)
        self.boundary = min(boundary, n // 2)
        self.candidate_set = candidates

    def master(self) -> Spins:
        """Return the master node, the empty partial sequence."""
        return np.zeros(self.n, dtype=np.int8)

    def class_roots(self) -> list[Spins]:
        """Return the class roots that pass the reversal rule."""
        if self.boundary < 2:
            root = self.master()
            root[:2] = 1
            return [root]
        roots = symmetry.class_roots(self.n, self.boundary)
        return [r for r in roots if self.passes_symmetry_rule(r)]

    def hamiltonian(self, node: Spins) -> NodeHamiltonian:
        """Return the node Hamiltonian."""
        return NodeHamiltonian(node)

    def complete(self, node: Spins, z: Spins) -> Spins:
        """Return the full sequence with the free spins set to z."""
        return NodeHamiltonian(node).complete(z)

    def objective(self, x: Spins) -> int:
        """Return the energy E(x)."""
        return energy(x)

    def free(self, node: Spins) -> Ints:
        """Return the free positions in increasing order."""
        return np.flatnonzero(node == 0)

    def fix(self, node: Spins, var: int, value: int) -> Spins:
        """Return the child with spin `var` fixed to `value`."""
        child = node.copy()
        child[var] = value
        return child

    def lower_bound(self, node: Spins) -> int:
        """Return the integer lower bound LB of the node (pair congruence)."""
        return bound.lower_bound(node)

    def propagate(self, node: Spins, threshold: int) -> Spins | None:
        """Return the node with forced spins fixed, or None if E <= T is impossible."""
        return bound.propagate(node, threshold)

    def passes_symmetry_rule(self, node: Spins) -> bool:
        """Apply the reversal rule once both end blocks are fixed."""
        m = self.boundary
        if m < 2 or np.any(node[:m] == 0) or np.any(node[self.n - m :] == 0):
            return True
        return symmetry.passes_reversal_rule(node, m)

    def violations(self, node: Spins, x: Spins, threshold: int) -> NDArray[np.bool_]:
        """Return V of eq. 5: V[k - 1, l] is true if sequence l breaks lag budget k.

        Args:
            node: The node, whose lag budgets at T are the constraints.
            x: Full sequences, shape (m, n).
            threshold: T.

        Returns:
            V, shape (n - 1, m).
        """
        c = autocorrelations(x)
        v: NDArray[np.bool_] = (c * c > bound.budgets(node, threshold)).T
        return v

    def correspondence(self, node: Spins) -> NDArray[np.int8]:
        """Return the correspondence matrix P of eq. 7.

        P[k - 1, i] = 1 if the i-th free spin has a fixed partner at lag k,
        that is, if it enters the linear part of C_k.

        Args:
            node: Partial sequence, 0 for free spins, shape (n,).

        Returns:
            P, shape (n - 1, F).
        """
        n = self.n
        free = np.flatnonzero(node == 0)
        fixed = node != 0
        k = np.arange(1, n)[:, None]
        left = free[None, :] - k
        right = free[None, :] + k
        has_left = (left >= 0) & fixed[np.clip(left, 0, n - 1)]
        has_right = (right < n) & fixed[np.clip(right, 0, n - 1)]
        p: NDArray[np.int8] = (has_left | has_right).astype(np.int8)
        return p

    def candidates(self, node: Spins) -> Ints:
        """Return the local indices eligible for eq. 9.

        "outermost" keeps the leftmost and rightmost free spin. On a node whose
        fixed spins form a prefix and a suffix, with nu = 1, eq. 9 over all
        free spins picks one of these two (paper, proposition on nodes with a fixed
        prefix and suffix).
        """
        f = int(np.count_nonzero(node == 0))
        if self.candidate_set == "outermost":
            return np.unique(np.array([0, f - 1], dtype=np.int64))
        return np.arange(f, dtype=np.int64)

    def tie_key(self, node: Spins, var: int) -> tuple[int, int]:
        """Prefer the spin nearest to an end; then left if the fixed count is even."""
        left_first = (self.n - int(np.count_nonzero(node == 0))) % 2 == 0
        return min(var, self.n - 1 - var), var if left_first else -var
