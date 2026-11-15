"""Simulated annealing on node Hamiltonians.

This is the SA backend of the paper. It works on the full
sequence with the fixed spins clamped. Flipping spin j changes each C_k by
dC_k = -2 s_j (s_{j-k} + s_{j+k}) (terms outside the sequence are zero), so
dE = sum_k dC_k (2 C_k + dC_k) costs O(n) per flip. Each shot is one
independent annealing run and returns the best state it visited.
"""

from collections.abc import Sequence

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from qcbb.labs.hamiltonian import NodeHamiltonian
from qcbb.problem import Hamiltonian, Spins
from qcbb.sampler import SampleSet, collect


class AnnealingSampler:
    """Metropolis simulated annealing with a geometric inverse-temperature schedule.

    Args:
        sweeps: Number of sweeps over the free spins per shot.
        beta_start: Inverse temperature of the first sweep.
        beta_end: Inverse temperature of the last sweep.
    """

    def __init__(
        self, sweeps: int = 1000, beta_start: float = 0.02, beta_end: float = 2.0
    ) -> None:
        self.betas = np.geomspace(beta_start, beta_end, sweeps)

    def sample(
        self,
        problems: Sequence[Hamiltonian],
        shots: int,
        seed: int | None = None,
    ) -> list[SampleSet]:
        """Anneal each node Hamiltonian `shots` times.

        Args:
            problems: LABS node Hamiltonians to sample.
            shots: Number of annealing runs per problem.
            seed: Seed for reproducible runs; random if None.

        Returns:
            One sample set per problem.

        Raises:
            TypeError: If a problem is not a LABS node Hamiltonian.
        """
        base = np.random.SeedSequence(seed).generate_state(1)[0] & 0x3FFFFFFF
        out = []
        for idx, p in enumerate(problems):
            if not isinstance(p, NodeHamiltonian):
                raise TypeError("AnnealingSampler needs a LABS NodeHamiltonian")
            bits = _anneal(p.spins, p.free, self.betas, shots, int(base) + idx * shots)
            out.append(collect(p, bits))
        return out


@njit(cache=True)
def _flip(s: Spins, c: NDArray[np.int64], j: int, apply: bool) -> int:
    """Return dE of flipping spin j; also flip it and update C if `apply`.

    Lags 1..m have both partners inside the sequence; the larger lags have one,
    on the side farther from spin j.
    """
    n = s.size
    f = -2 * s[j]
    m = min(j, n - 1 - j)
    de = 0
    for k in range(1, m + 1):
        dc = f * (s[j - k] + s[j + k])
        de += dc * (2 * c[k - 1] + dc)
        if apply:
            c[k - 1] += dc
    for k in range(m + 1, j + 1):
        dc = f * s[j - k]
        de += dc * (2 * c[k - 1] + dc)
        if apply:
            c[k - 1] += dc
    for k in range(m + 1, n - j):
        dc = f * s[j + k]
        de += dc * (2 * c[k - 1] + dc)
        if apply:
            c[k - 1] += dc
    if apply:
        s[j] = -s[j]
    return de


@njit(parallel=True, cache=True)
def _anneal(
    spins: Spins,
    free: NDArray[np.int64],
    betas: NDArray[np.float64],
    shots: int,
    seed: int,
) -> Spins:
    """Run `shots` independent annealing runs and return their best free-spin states."""
    n = spins.size
    f = free.size
    out = np.empty((shots, f), dtype=np.int8)
    for r in prange(shots):
        np.random.seed(seed + r)
        s = spins.copy()
        for j in free:
            s[j] = 1 if np.random.random() < 0.5 else -1
        c = np.zeros(n - 1, dtype=np.int64)
        for k in range(1, n):
            for i in range(n - k):
                c[k - 1] += s[i] * s[i + k]
        e = 0
        for k in range(n - 1):
            e += c[k] * c[k]
        best_e = e
        best = s[free].copy()
        for beta in betas:
            for j in free:
                de = _flip(s, c, j, False)
                if de <= 0 or np.random.random() < np.exp(-beta * de):
                    _flip(s, c, j, True)
                    e += de
                    if e < best_e:
                        best_e = e
                        best[:] = s[free]
        out[r] = best
    return out
