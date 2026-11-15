"""Skew-symmetric restriction of the master Hamiltonian (odd n, sampler only).

For n = 2q - 1 a sequence is skew-symmetric if s_{q+l} = (-1)^l s_{q-l} for
l = 1, ..., q - 1 (1-based, center q). Then C_k = 0 for every odd k, and the
sequence is fixed by its first q spins. The restricted energy is a quartic
Ising Hamiltonian on q = (n + 1) / 2 qubits with about 6.5 times fewer CNOTs per
QAOA layer than the master.

The restriction supplies incumbent candidates only. Many odd-length optima are
skew-symmetric, but not all (n = 19, 23, 25, 31, 33, 35, 37, 61, 63, 65 among
19..65), so the restriction is used inside the sampler: restricted samples are
completed to n spins and returned as ordinary master samples, by default
merged with samples of the master itself. It never enters a bound or a prune.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from qcbb.labs.energy import energies
from qcbb.labs.hamiltonian import NodeHamiltonian, expand
from qcbb.problem import Hamiltonian, Ints, Monomial, Spins
from qcbb.sampler import Sampler, SampleSet, collect


@dataclass(frozen=True, eq=False)
class SkewHamiltonian:
    """The LABS energy of skew-symmetric sequences of odd length n.

    Attributes:
        n: Sequence length, odd.
    """

    n: int

    @property
    def size(self) -> int:
        """Number of free spins q = (n + 1) / 2."""
        return (self.n + 1) // 2

    @cached_property
    def source(self) -> tuple[Ints, Ints]:
        """Return (index, sign) with s_i = sign_i z_{index_i}."""
        i = np.arange(self.n)
        d = i - (self.size - 1)
        index = np.where(d <= 0, i, self.size - 1 - d)
        sign = np.where((d > 0) & (d % 2 == 1), -1, 1)
        return index.astype(np.int64), sign.astype(np.int64)

    def complete(self, z: Spins) -> Spins:
        """Return the skew-symmetric sequences of free-spin values z, shape (m, n)."""
        index, sign = self.source
        return (sign * np.asarray(z, dtype=np.int8)[..., index]).astype(np.int8)

    def energies(self, z: Spins) -> Ints:
        """Return the exact energies of free-spin assignments, shape (m,)."""
        return energies(self.complete(z))

    def monomials(self) -> tuple[int, dict[Monomial, int]]:
        """Expand the restricted energy into Ising monomials over the q spins."""
        index, sign = self.source
        lags = []
        for k in range(1, self.n):
            terms: defaultdict[Monomial, int] = defaultdict(int)
            for i in range(self.n - k):
                a, b = int(index[i]), int(index[i + k])
                terms[() if a == b else (min(a, b), max(a, b))] += int(
                    sign[i] * sign[i + k]
                )
            lags.append({m: w for m, w in terms.items() if w != 0})
        return expand(lags)


class SkewSampler:
    """Sample the master Hamiltonian of odd n through its skew restriction.

    Every other Hamiltonian, and the master when q exceeds `max_size`, is
    passed to the inner sampler unchanged.

    Args:
        inner: Sampler that accepts any Hamiltonian (e.g. `QaoaSampler`).
        max_size: Largest q restricted; no limit if None.
        with_master: Also sample the master and merge both sample sets, so the
            incumbent is never worse than without the restriction.
    """

    def __init__(
        self, inner: Sampler, max_size: int | None = None, with_master: bool = True
    ) -> None:
        self.inner = inner
        self.max_size = max_size
        self.with_master = with_master

    def sample(
        self,
        problems: Sequence[Hamiltonian],
        shots: int,
        seed: int | None = None,
    ) -> list[SampleSet]:
        """Sample each problem; odd-length masters are sampled restricted.

        Args:
            problems: Node Hamiltonians to sample.
            shots: Number of samples per problem.
            seed: Seed for reproducible runs.

        Returns:
            One sample set per problem, as samples of the given Hamiltonian.
        """
        skews = {
            i: s for i, h in enumerate(problems) if (s := self._restrict(h)) is not None
        }
        batch = [skews.get(i, h) for i, h in enumerate(problems)]
        extra = [problems[i] for i in skews] if self.with_master else []
        sets = self.inner.sample([*batch, *extra], shots, seed)
        out = sets[: len(problems)]
        for j, (i, skew) in enumerate(skews.items()):
            rows = [skew.complete(np.repeat(out[i].spins, out[i].counts, axis=0))]
            if self.with_master:
                m = sets[len(problems) + j]
                rows.append(np.repeat(m.spins, m.counts, axis=0))
            out[i] = collect(problems[i], np.concatenate(rows))
        return out

    def _restrict(self, h: Hamiltonian) -> SkewHamiltonian | None:
        """Return the skew restriction of an odd-length master, else None."""
        if not isinstance(h, NodeHamiltonian) or h.size != h.n or h.n % 2 == 0:
            return None
        skew = SkewHamiltonian(h.n)
        if self.max_size is not None and skew.size > self.max_size:
            return None
        return skew
