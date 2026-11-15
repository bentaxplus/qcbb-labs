"""The node Hamiltonian: the LABS energy with some spins fixed.

A node fixes a set Phi of spins and leaves the set U free, F = |U|. With the
fixed spins substituted, each autocorrelation reads

    C_k(z) = c_k + sum_{i in U} h_ki z_i + sum_{i, i+k in U} z_i z_{i+k},

and the node Hamiltonian is H_node(z) = sum_k C_k(z)^2 over z in {-1, +1}^F
(paper, eq. 5). H_node(z) is the energy of the completed sequence, so sample
energies are exact energies. This is the subproblem construction of QCBB
(eqs. 10-16) for a polynomial instead of linear rows. With no spin fixed it
is the master Hamiltonian H = sum_k C_k^2 on n qubits.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from qcbb.labs.energy import energies
from qcbb.problem import Ints, Monomial, Spins


@dataclass(frozen=True, eq=False)
class NodeHamiltonian:
    """The energy of all completions of a partial sequence.

    Attributes:
        spins: Partial sequence, shape (n,): +1 or -1 for fixed spins, 0 for free.
    """

    spins: Spins

    @classmethod
    def master(cls, n: int) -> "NodeHamiltonian":
        """Return the master Hamiltonian of length n (no spin fixed)."""
        return cls(np.zeros(n, dtype=np.int8))

    @property
    def n(self) -> int:
        """Sequence length."""
        return int(self.spins.size)

    @cached_property
    def free(self) -> NDArray[np.int64]:
        """Positions of the free spins, in increasing order (local index order)."""
        return np.flatnonzero(self.spins == 0)

    @property
    def size(self) -> int:
        """Number of free spins F, the number of qubits."""
        return int(self.free.size)

    def complete(self, z: Spins) -> Spins:
        """Return the full sequences with the free spins set to z.

        Args:
            z: Free-spin values, shape (F,) or (m, F).

        Returns:
            Full sequences, shape (n,) or (m, n).
        """
        z = np.asarray(z, dtype=np.int8)
        s = np.broadcast_to(self.spins, (*z.shape[:-1], self.n)).copy()
        s[..., self.free] = z
        return s

    def energies(self, z: Spins) -> Ints:
        """Return the exact energies H_node(z) of free-spin assignments.

        Args:
            z: Free-spin values, shape (m, F).

        Returns:
            Energies of the completed sequences, shape (m,).
        """
        return energies(self.complete(z))

    def monomials(self) -> tuple[int, dict[Monomial, int]]:
        """Expand H_node into Ising monomials over the free spins.

        Monomials use local indices 0, ..., F - 1 (positions in `free`). The
        products z_i^2 = 1 are reduced, so each monomial has 1 to 4 distinct
        spins.

        Returns:
            The constant term and a map from monomial to integer coefficient.
        """
        local = np.full(self.n, -1, dtype=np.int64)
        local[self.free] = np.arange(self.size)
        return expand(self._lag_terms(k, local) for k in range(1, self.n))

    def _lag_terms(self, k: int, local: Ints) -> dict[Monomial, int]:
        """Return C_k as a map from monomial (local indices) to coefficient."""
        s = self.spins
        terms: defaultdict[Monomial, int] = defaultdict(int)
        for i in range(self.n - k):
            a, b = int(s[i]), int(s[i + k])
            if a and b:
                terms[()] += a * b
            elif a:
                terms[(int(local[i + k]),)] += a
            elif b:
                terms[(int(local[i]),)] += b
            else:
                terms[(int(local[i]), int(local[i + k]))] += 1
        return {m: w for m, w in terms.items() if w != 0}


def expand(lags: Iterable[dict[Monomial, int]]) -> tuple[int, dict[Monomial, int]]:
    """Return sum_k C_k^2 as Ising monomials, given each C_k as monomials.

    Products of spins are reduced with z_i^2 = 1, so the product of two
    monomials is their symmetric difference.

    Args:
        lags: C_k for each lag k, as maps from monomial to coefficient.

    Returns:
        The constant term and a map from monomial to integer coefficient.
    """
    total: defaultdict[Monomial, int] = defaultdict(int)
    for terms in lags:
        for t1, w1 in terms.items():
            for t2, w2 in terms.items():
                total[tuple(sorted(set(t1) ^ set(t2)))] += w1 * w2
    constant = total.pop((), 0)
    return constant, {m: w for m, w in total.items() if w != 0}
